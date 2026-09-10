"""
SignalHQ nightly data builder.

Pulls end-of-day options data from Yahoo Finance (via the unofficial `yfinance`
library — free, no API key, but not officially supported by Yahoo and can break
or rate-limit without warning) and computes the fields the SignalHQ UI expects,
writing them to data.json at the repo root.

IMPORTANT — read before trusting the numbers:
- `iv` (implied volatility) comes directly from Yahoo's option chain.
- `delta` is computed here via Black-Scholes, assuming 0% dividend yield and a
  flat risk-free rate (RISK_FREE_RATE below). Real delta from a broker may differ.
- `pot` (probability of touch) uses the common trader heuristic pot ≈ 2 × |delta|,
  not a rigorous barrier-option calculation. Treat it as a rough guide.
- `ivr` (IV Rank) is NOT true IV rank (which needs a year of historical *option*
  IV data, which isn't freely available). It's a proxy built from the percentile
  of recent 20-day realized volatility vs. the past year — correlated with real
  IV rank but not the same number your broker would show.
- `score` (composite rating, 0-10) is an illustrative weighted blend of the above.
  It is not a validated trading signal. Adjust the weights in `composite_score()`
  to match what you actually care about.
- Strategy/strike selection targets a ~0.20 delta short leg, a common informal
  "20-delta" premium-selling convention — not personalized to any risk tolerance.

This script is a starting point, not a finished quant model. Treat every number
it produces as directional, not authoritative, and verify anything before
acting on it.
"""

import json
import math
import re
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.stats import norm

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed — run: pip install -r scripts/requirements.txt", file=sys.stderr)
    raise

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    print("vaderSentiment not installed — run: pip install -r scripts/requirements.txt", file=sys.stderr)
    raise

# VADER's stock lexicon is tuned for general/social text and badly misreads financial
# headlines — e.g. out of the box it scores "faces lawsuit... shares tumble" as
# slightly POSITIVE, because words like "tumble" and "beats" aren't in its default
# dictionary. This augments it with common financial-news vocabulary so it actually
# reads headlines the way a finance-literate person would. Not exhaustive — extend
# FINANCE_LEXICON below if you notice it missing common terms.
FINANCE_LEXICON = {
    # positive
    "beat": 3.0, "beats": 3.0, "beating": 3.0, "exceeded": 2.8, "exceeds": 2.8,
    "upgrade": 2.5, "upgraded": 2.5, "upgrades": 2.5, "outperform": 2.5,
    "raises": 1.8, "raised": 1.8, "surge": 3.0, "surged": 3.0, "surges": 3.0,
    "rally": 2.5, "rallied": 2.5, "rallies": 2.5, "soar": 3.2, "soared": 3.2, "soars": 3.2,
    "bullish": 2.5, "buyback": 1.8, "record high": 2.8, "accelerate": 1.5,
    "accelerated": 1.5, "breakthrough": 2.5, "guidance raised": 2.5,
    "strong demand": 2.2, "blowout": 3.0, "jumps": 2.2, "jumped": 2.2,
    # negative
    "miss": -3.0, "misses": -3.0, "missed": -3.0, "downgrade": -2.5, "downgraded": -2.5,
    "downgrades": -2.5, "cut": -1.8, "cuts": -1.8, "plunge": -3.2, "plunged": -3.2,
    "plunges": -3.2, "tumble": -3.0, "tumbled": -3.0, "tumbles": -3.0,
    "slump": -2.5, "slumped": -2.5, "bearish": -2.5, "lawsuit": -2.2,
    "investigation": -2.5, "recall": -2.2, "layoffs": -2.5, "bankruptcy": -3.5,
    "default": -3.0, "delisted": -3.0, "fraud": -3.5, "scandal": -3.0,
    "warning": -1.8, "weak demand": -2.2, "slowdown": -1.8, "guidance cut": -2.8,
    "sinks": -2.5, "sank": -2.5, "slides": -1.8,
}

_sentiment_analyzer = SentimentIntensityAnalyzer()
_sentiment_analyzer.lexicon.update(FINANCE_LEXICON)

# --- Configuration -----------------------------------------------------------

def load_tickers():
    """
    Reads the watchlist from tickers.json (repo root) so it can be edited without
    touching this script — either by hand on GitHub, or via the "Manage Tickers"
    panel in the app, which generates ready-to-paste JSON for this file.
    Falls back to a small built-in default set if the file is missing or invalid,
    so a bad edit here can't break the nightly run entirely.
    """
    default_tickers = ["AAPL", "MSFT", "NVDA", "XOM", "JPM", "SPY", "META", "TSLA", "AMD"]
    default_etfs = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "GLD"]
    try:
        with open("tickers.json") as f:
            cfg = json.load(f)
        tickers = cfg.get("tickers") or default_tickers
        etfs = set(cfg.get("etfs") or default_etfs)
        return [t.strip().upper() for t in tickers if t.strip()], etfs
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"tickers.json missing or invalid ({e}) — using built-in defaults", file=sys.stderr)
        return default_tickers, set(default_etfs)


TICKERS, KNOWN_ETFS = load_tickers()

TARGET_DTE_MIN = 7          # was 5, then 14 originally — narrowed to 7-45 so the
                             # picker searches a real window and optimizes within it,
                             # rather than drifting into 0-6 DTE territory this nightly
                             # EOD tool shouldn't be making picks for anyway.
TARGET_DTE_MAX = 45
TARGET_SHORT_DELTA = 0.20   # informal "20-delta" premium-selling target
RISK_FREE_RATE = 0.045      # flat approximation; update periodically
OUTPUT_PATH = "data.json"
MAX_PLAUSIBLE_ROC = 35      # raw period ROC (%) sanity ceiling — deliberately NOT applied to
                             # the annualized figure, since annualizing amplifies short-DTE
                             # trades by up to 365/DTE (60x+ at 6 DTE), which used to make
                             # legitimate short-dated premium look "implausible" and get
                             # rejected. Raw ROC (premium/collateral, pre-annualization) means
                             # the same thing regardless of DTE, so it's the correct thing to
                             # sanity-check for thin/wide-market quotes.


# --- Math helpers --------------------------------------------------------------

def bs_delta(spot, strike, dte_days, iv, option_type, r=RISK_FREE_RATE):
    """Black-Scholes delta. option_type: 'call' or 'put'. Assumes 0% dividend yield."""
    if dte_days <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    t = dte_days / 365.0
    d1 = (math.log(spot / strike) + (r + 0.5 * iv ** 2) * t) / (iv * math.sqrt(t))
    if option_type == "call":
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1)


def safe_float(v, default=0.0):
    """float() that treats NaN (and bad input) as `default` instead of propagating NaN.
    Needed because Python's `x or default` idiom does NOT catch NaN — NaN is truthy —
    and NaN silently passes any `<= 0` / `> 0` comparison (all NaN comparisons are False).
    Both of those gaps let real Yahoo data (which frequently has NaN IV/price fields
    on illiquid strikes) sail past guards that looked like they should have caught it.
    """
    try:
        f = float(v)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def probability_of_touch(delta):
    """Rough trader heuristic, not a rigorous barrier-option calculation."""
    if delta is None or math.isnan(delta):
        return 50  # neutral fallback rather than crashing the whole ticker
    return min(100, round(abs(delta) * 2 * 100))


def compute_ema(closes, span):
    return closes.ewm(span=span, adjust=False).mean()


def compute_atr(history, period=14):
    high, low, close = history["High"], history["Low"], history["Close"]
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    return tr.rolling(period).mean().iloc[-1]


def iv_rank_proxy(history, window=252, vol_window=20):
    """
    Proxy for IV rank using realized volatility percentile, since a year of
    historical *implied* volatility isn't freely available. Correlated with
    real IV rank but not equivalent to it.
    Returns (percentile, current_realized_vol_pct) — the second value is used
    separately to compare against the option's actual IV (see classify_vol_regime).
    """
    closes = history["Close"].tail(window + vol_window)
    log_returns = np.log(closes / closes.shift(1)).dropna()
    realized_vol = log_returns.rolling(vol_window).std() * math.sqrt(252)
    realized_vol = realized_vol.dropna()
    if len(realized_vol) < 20:
        return 50, None  # not enough history yet — neutral fallback
    current = realized_vol.iloc[-1]
    percentile = (realized_vol < current).sum() / len(realized_vol) * 100
    return round(percentile), round(current * 100, 1)


def classify_vol_regime(iv_pct, realized_vol_pct):
    """
    Compares an option's implied volatility against the stock's own recent
    realized volatility. All of SignalHQ's current strategies are premium
    SELLING strategies (short put, covered call, credit spreads), which do
    better when IV is "rich" relative to what the stock has actually been
    doing — you're being paid more than the recent movement would justify.
    "Cheap" IV doesn't make a selling strategy wrong, but the edge is thinner.
    Returns (regime_label, ratio) — ratio is IV/RV, None if RV unavailable.
    """
    if realized_vol_pct is None or realized_vol_pct <= 0 or iv_pct is None or iv_pct <= 0:
        return "Unknown", None
    ratio = round(iv_pct / realized_vol_pct, 2)
    if ratio >= 1.15:
        return "Rich", ratio
    if ratio <= 0.85:
        return "Cheap", ratio
    return "Fair", ratio


def composite_score(ann_profit, pot, ivr):
    """Illustrative 0-10 blend — adjust weights to match your priorities."""
    profit_component = min(10, max(0, ann_profit / 5))      # ~50% ann. profit -> 10
    safety_component = min(10, max(0, (100 - pot) / 10))     # lower POT -> higher score
    ivr_component = min(10, max(0, ivr / 10))
    score = 0.45 * profit_component + 0.35 * safety_component + 0.20 * ivr_component
    return round(min(10, max(1, score)), 1)


def chain_diagnostics(df, spot=None):
    """
    Describes what a chain actually contained, for logging when strike-picking
    fails. The distinction matters a lot: 0 rows means Yahoo likely blocked or
    rate-limited the request (a known risk running yfinance from shared CI IP
    ranges); rows present but no valid IV means a stale/garbage snapshot; rows
    with valid IV but still no match means the target delta genuinely wasn't
    available that day, which is a data problem, not a request problem;
    strikes wildly inconsistent with spot means a stale/un-adjusted chain,
    most commonly following a real stock split the chain hasn't caught up to
    (confirmed against ServiceNow/NOW after its Dec 2025 5-for-1 split).
    """
    if df is None or len(df) == 0:
        return "chain came back with 0 rows — likely blocked/rate-limited by Yahoo, not a data-quality issue"
    ivs = df["impliedVolatility"].apply(lambda v: safe_float(v, default=float("nan")))
    valid = ivs[(ivs > 0) & (~ivs.isna())]
    if len(valid) == 0:
        return f"{len(df)} rows but none had valid IV — likely a stale/blocked Yahoo snapshot"
    if spot and spot > 0:
        strikes = df["strike"].apply(lambda v: safe_float(v, default=0.0))
        sane = strikes[(strikes / spot >= 0.25) & (strikes / spot <= 4.0)]
        if len(sane) == 0:
            return (f"{len(df)} rows with valid IV, but every strike is wildly inconsistent with "
                    f"spot ${spot:.2f} (strike range {strikes.min():.0f}-{strikes.max():.0f}) — "
                    f"likely a stale/un-adjusted chain, check for a recent stock split")
    return f"{len(df)} rows, {len(valid)} with valid IV (range {valid.min():.2f}-{valid.max():.2f})"


def pick_strike_by_delta(chain_df, spot, dte_days, target_delta, option_type):
    """Return the chain row whose computed delta is closest to target_delta."""
    best_row, best_diff = None, None
    for _, row in chain_df.iterrows():
        iv = safe_float(row.get("impliedVolatility"), default=0.0)
        if iv <= 0:
            continue
        strike = safe_float(row.get("strike"), default=0.0)
        if strike <= 0:
            continue
        # Sanity guard: a ~20-delta strike should always land within a fairly
        # narrow band of spot under any realistic market condition. A strike
        # wildly outside that band (e.g. 5-10x spot) means the options chain
        # itself is bad data — most commonly a real stock split that yfinance's
        # free/unofficial chain hasn't caught up to adjusting for yet, so the
        # strikes still reflect pre-split contract prices while `spot` (from
        # price history) correctly reflects the post-split price. Confirmed
        # this exact scenario against ServiceNow (NOW): a real 5-for-1 split
        # effective Dec 18, 2025 left the chain showing ~$1200 strikes against
        # a genuine ~$134 spot — an 8.96x mismatch a real 0.20-delta pick would
        # never produce. Skip rather than publish an internally-inconsistent
        # trade (strike and breakeven implying two different stock prices).
        if not (0.25 <= strike / spot <= 4.0):
            continue
        delta = bs_delta(spot, strike, dte_days, iv, option_type)
        if math.isnan(delta):
            continue
        diff = abs(abs(delta) - target_delta)
        if best_diff is None or diff < best_diff:
            best_diff, best_row = diff, (row, delta)
    return best_row  # (row, delta) or None


def mid_price(row):
    bid, ask = safe_float(row.get("bid")), safe_float(row.get("ask"))
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    return safe_float(row.get("lastPrice"))


# --- Per-ticker trade construction --------------------------------------------

def rank_expirations(expirations, today):
    """
    Returns expiration candidates as (exp_str, dte) tuples, closest-to-target-window
    first. A list rather than a single pick, because a chosen expiration's chain can
    turn out to be unusable (e.g. every relevant strike has NaN IV that day — this
    happens on real Yahoo data more often than you'd expect) — the caller can then
    fall back to the next-best expiration instead of giving up on the ticker entirely.
    """
    target_mid = (TARGET_DTE_MIN + TARGET_DTE_MAX) / 2
    in_window, outside_window = [], []
    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if dte <= 0:
            continue
        diff = abs(dte - target_mid)
        (in_window if TARGET_DTE_MIN <= dte <= TARGET_DTE_MAX else outside_window).append((diff, exp_str, dte))
    in_window.sort(key=lambda x: x[0])
    outside_window.sort(key=lambda x: x[0])
    return [(exp_str, dte) for _, exp_str, dte in (in_window + outside_window)]


MAX_CANDIDATES_TO_EVALUATE = 8  # how many expirations within the target window to
                                 # actually price and compare before picking whichever
                                 # produces the best annualized profit. Bounded so a
                                 # name with unusually many listed expirations doesn't
                                 # blow up the number of chain fetches per ticker.


def evaluate_expiration_candidate(tk, strat, side, spot, cand_exp, cand_dte, atr):
    """
    Builds a complete, fully-priced trade for ONE specific expiration, so multiple
    expirations can be compared against each other on actual economics (annualized
    profit) rather than just picking whichever is closest to a fixed target date.
    Returns (result_dict, None) on success or (None, reason_string) on failure —
    result_dict covers everything that varies per-expiration; the caller fills in
    the per-ticker fields (news, earnings, composite score) that don't depend on
    which expiration ends up winning.
    """
    try:
        chain = tk.option_chain(cand_exp)
    except Exception as e:
        return None, f"chain fetch failed: {e}"
    calls, puts = chain.calls, chain.puts

    fields, reason = try_strategy_pick(strat, calls, puts, spot, cand_dte)
    if not fields:
        return None, reason

    premium = fields["premium"]
    collateral = fields["collateral"]
    if premium <= 0 or collateral <= 0:
        return None, "unusable premium/collateral"

    roc = round((premium / collateral) * 100, 2)
    ann_profit = round(roc * (365 / cand_dte), 1)

    # sanity guard: check the RAW period ROC, not the annualized figure — annualizing
    # amplifies short-DTE trades by 365/dte (60x+ at 6 DTE), so a flat cap on the
    # annualized number would reject perfectly legitimate short-dated premium just
    # for being short-dated. Raw ROC means the same thing regardless of DTE.
    if roc > MAX_PLAUSIBLE_ROC:
        return None, f"implausible raw ROC ({roc}%), likely a thin/wide-market quote"

    total_call_oi = calls["openInterest"].fillna(0).sum()
    total_put_oi = puts["openInterest"].fillna(0).sum()
    total_call_vol = calls["volume"].fillna(0).sum()
    total_put_vol = puts["volume"].fillna(0).sum()
    pc_oi = round(total_put_oi / total_call_oi, 2) if total_call_oi else 0
    pc_vol = round(total_put_vol / total_call_vol, 2) if total_call_vol else 0

    iv = fields["iv"]
    daily_return = round(premium * 100 / cand_dte, 2)
    pot = probability_of_touch(bs_delta(spot, fields["strike_for_pot"], cand_dte, iv / 100 if iv else 0.3, "put" if side == "bull" else "call"))
    margin_of_safety = bool(atr and abs(spot - fields["strike_for_pot"]) >= atr)
    exp_label = datetime.strptime(cand_exp, "%Y-%m-%d").strftime("%b %-d") if sys.platform != "win32" else datetime.strptime(cand_exp, "%Y-%m-%d").strftime("%b %d").replace(" 0", " ")

    return {
        "exp": exp_label,
        "dte": cand_dte,
        "pot": pot,
        "ap": ann_profit,
        "dailyReturn": daily_return,
        "roc": roc,
        "marginOfSafety": margin_of_safety,
        "delta": round(fields["delta_for_output"], 2),
        "iv": round(iv, 1),
        "premium": round(premium, 2),
        "breakeven": round(fields["breakeven"], 2),
        "maxLoss": fields["max_loss"],
        "pcOI": pc_oi,
        "pcVol": pc_vol,
        "strike": fields["strike_label"],
    }, None
MAX_NEWS_HEADLINES = 3       # how many recent headlines to pull and score per ticker


def fetch_news_and_sentiment(tk, ticker_symbol):
    """
    Pulls recent headlines via yfinance's free .news property and scores them with
    a finance-augmented VADER analyzer. This is lexicon-based sentiment on
    headlines only — not full-article analysis, and not an LLM reading the story
    for context. Treat it as a rough "does the recent press read positive or
    negative" gauge, not a rigorous signal.

    IMPORTANT: yfinance's .news is NOT reliably scoped to the requested ticker —
    since March 2024 it can return general Yahoo Finance homepage/trending stories
    mixed in with genuinely ticker-specific ones (documented upstream:
    github.com/ranaroussi/yfinance/issues/1956). Trusting it blindly means a
    completely unrelated headline (e.g. a Costco story under an Amazon trade) can
    end up attached to the wrong stock.

    Yahoo's own news items carry a `relatedTickers` field naming which symbols
    an article is actually about — used as ground truth when present. But in
    practice that field is often just not populated at all, so when it's missing
    this falls back to checking whether the ticker symbol or company brand name
    appears in the headline. That fallback matches on NORMALIZED text (all
    whitespace/punctuation stripped from both the candidate name and the
    headline) rather than an exact substring — confirmed necessary in practice:
    Yahoo's info can list a company as "JP Morgan Chase & Co." (with a space),
    while headlines write it as "JPMorgan" (no space) — a literal substring
    match fails there even though the two obviously refer to the same company.

    Returns (avg_compound_score, label, headlines_list). Fails soft — a ticker
    with no news, or if Yahoo's news endpoint has a bad day, just gets neutral
    sentiment and an empty headline list rather than sinking the whole ticker.
    """
    FILLER_WORDS = {"inc", "incorporated", "corp", "corporation", "co", "ltd", "limited",
                     "plc", "llc", "holdings", "holding", "group", "company", "com", "net",
                     "org", "the", "and"}

    def normalize(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())

    try:
        raw_news = tk.news or []
    except Exception as e:
        print(f"    {ticker_symbol} news: tk.news raised an exception: {e}")
        raw_news = []

    if not raw_news:
        print(f"    {ticker_symbol} news: tk.news returned 0 raw items (Yahoo may be rate-limiting/blocking, "
              f"or this ticker genuinely has no recent news — can't tell which without this line)")

    # Best-effort brand-name candidates, used only for the text-match fallback
    # below — a failure here just means the fallback relies on the ticker
    # symbol alone. Builds normalized candidates from the company's short/long
    # name: the first meaningful word alone ("JPMorgan"), and the first two
    # words concatenated ("JP"+"Morgan" -> "jpmorgan"), covering both
    # single-word and space-separated brand name styles.
    brand_candidates = []
    try:
        info = tk.info or {}
        raw_name = (info.get("shortName") or info.get("longName") or "").strip()
        words = re.findall(r"[A-Za-z0-9']+", raw_name)
        brand_words = [w for w in words if w.lower() not in FILLER_WORDS]
        if brand_words:
            brand_candidates.append(normalize(brand_words[0]))
            if len(brand_words) >= 2:
                brand_candidates.append(normalize(brand_words[0] + brand_words[1]))
        brand_candidates = [c for c in brand_candidates if len(c) >= 3]
        if raw_news and not brand_candidates:
            print(f"    {ticker_symbol} news: tk.info had no usable shortName/longName (company-name fallback unavailable)")
    except Exception as e:
        if raw_news:
            print(f"    {ticker_symbol} news: tk.info raised an exception: {e} (company-name fallback unavailable)")

    def is_relevant(title, related_upper):
        if related_upper:
            return ticker_symbol.upper() in related_upper
        title_lower = title.lower()
        if re.search(r"\b" + re.escape(ticker_symbol.lower()) + r"\b", title_lower):
            return True
        norm_title = normalize(title)
        if any(cand in norm_title for cand in brand_candidates):
            return True
        return False

    headlines = []
    scores = []
    rejected_examples = []
    for item in raw_news:
        if len(headlines) >= MAX_NEWS_HEADLINES:
            break
        # yfinance's news item shape has shifted across versions; handle both
        # the flat dict style and the newer nested {"content": {...}} style.
        content = item.get("content", item)

        title = content.get("title") or content.get("headline")
        if not title:
            continue

        related = content.get("relatedTickers") or item.get("relatedTickers") or []
        related_upper = [str(r).upper() for r in related]
        if not is_relevant(title, related_upper):
            if len(rejected_examples) < 3:
                reason = f"relatedTickers={related_upper}" if related_upper else "no relatedTickers, no text match"
                rejected_examples.append(f'"{title[:50]}..." ({reason})')
            continue

        publisher = (content.get("provider") or {}).get("displayName") if isinstance(content.get("provider"), dict) else content.get("publisher")
        link = (content.get("canonicalUrl") or {}).get("url") if isinstance(content.get("canonicalUrl"), dict) else content.get("link")
        compound = _sentiment_analyzer.polarity_scores(title)["compound"]
        scores.append(compound)
        headlines.append({
            "title": title,
            "publisher": publisher or "Unknown source",
            "link": link or "",
            "sentiment": round(compound, 2),
        })

    if raw_news and not headlines:
        print(f"    {ticker_symbol} news: {len(raw_news)} raw items came back but ALL were filtered out as irrelevant. Examples:")
        for ex in rejected_examples:
            print(f"      - {ex}")

    if not scores:
        return 0.0, "Neutral", []

    avg = sum(scores) / len(scores)
    if avg >= 0.15:
        label = "Positive"
    elif avg <= -0.15:
        label = "Negative"
    else:
        label = "Neutral"
    return round(avg, 2), label, headlines


def try_strategy_pick(strat, calls, puts, spot, dte):
    """
    Attempts to pick strikes for `strat` against one expiration's chain.
    Returns (fields_dict, None) on success, or (None, reason_string) on failure —
    the reason gets logged by the caller, and used to try the next expiration
    candidate rather than silently giving up on the whole ticker.
    """
    if strat == "Short Put":
        picked_row = pick_strike_by_delta(puts, spot, dte, TARGET_SHORT_DELTA, "put")
        if not picked_row:
            return None, f"no put near target delta — {chain_diagnostics(puts, spot)}"
        row, delta = picked_row
        premium = mid_price(row)
        strike = float(row["strike"])
        return {
            "premium": premium, "strike_for_pot": strike, "collateral": strike,
            "breakeven": strike - premium, "max_loss": round((strike - premium) * 100, 2),
            "strike_label": f"${strike:.0f} P", "iv": safe_float(row.get("impliedVolatility")) * 100,
            "delta_for_output": delta,
        }, None

    if strat == "Covered Call":
        picked_row = pick_strike_by_delta(calls, spot, dte, TARGET_SHORT_DELTA, "call")
        if not picked_row:
            return None, f"no call near target delta — {chain_diagnostics(calls, spot)}"
        row, delta = picked_row
        premium = mid_price(row)
        strike = float(row["strike"])
        return {
            "premium": premium, "strike_for_pot": strike, "collateral": spot,
            "breakeven": spot - premium, "max_loss": round((spot - premium) * 100, 2),
            "strike_label": f"${strike:.0f} C", "iv": safe_float(row.get("impliedVolatility")) * 100,
            "delta_for_output": delta,
        }, None

    if strat == "Short Call":
        picked_row = pick_strike_by_delta(calls, spot, dte, TARGET_SHORT_DELTA, "call")
        if not picked_row:
            return None, f"no call near target delta — {chain_diagnostics(calls, spot)}"
        row, delta = picked_row
        premium = mid_price(row)
        strike = float(row["strike"])
        return {
            "premium": premium, "strike_for_pot": strike, "collateral": strike,  # rough proxy; true naked-call risk is undefined
            "breakeven": strike + premium, "max_loss": round(strike * 100, 2),  # illustrative cap, not a real max-loss figure
            "strike_label": f"${strike:.0f} C", "iv": safe_float(row.get("impliedVolatility")) * 100,
            "delta_for_output": delta,
        }, None

    if strat == "Bull Put Spread":
        short_row = pick_strike_by_delta(puts, spot, dte, TARGET_SHORT_DELTA, "put")
        if not short_row:
            return None, f"no put near target delta for short leg — {chain_diagnostics(puts, spot)}"
        s_row, s_delta = short_row
        short_strike = float(s_row["strike"])
        lower_strikes = puts[puts["strike"] < short_strike].sort_values("strike", ascending=False)
        if lower_strikes.empty:
            return None, "no further-OTM strike available for the long leg"
        long_row = lower_strikes.iloc[min(1, len(lower_strikes) - 1)]
        long_strike = float(long_row["strike"])
        premium = mid_price(s_row) - mid_price(long_row)
        width = short_strike - long_strike
        return {
            "premium": premium, "strike_for_pot": short_strike, "collateral": width,
            "breakeven": short_strike - premium, "max_loss": round((width - premium) * 100, 2),
            "strike_label": f"${short_strike:.0f}/{long_strike:.0f}", "iv": safe_float(s_row.get("impliedVolatility")) * 100,
            "delta_for_output": s_delta,
        }, None

    # Bear Call Spread
    short_row = pick_strike_by_delta(calls, spot, dte, TARGET_SHORT_DELTA, "call")
    if not short_row:
        return None, f"no call near target delta for short leg — {chain_diagnostics(calls, spot)}"
    s_row, s_delta = short_row
    short_strike = float(s_row["strike"])
    higher_strikes = calls[calls["strike"] > short_strike].sort_values("strike")
    if higher_strikes.empty:
        return None, "no further-OTM strike available for the long leg"
    long_row = higher_strikes.iloc[min(1, len(higher_strikes) - 1)]
    long_strike = float(long_row["strike"])
    premium = mid_price(s_row) - mid_price(long_row)
    width = long_strike - short_strike
    return {
        "premium": premium, "strike_for_pot": short_strike, "collateral": width,
        "breakeven": short_strike + premium, "max_loss": round((width - premium) * 100, 2),
        "strike_label": f"${short_strike:.0f}/{long_strike:.0f}", "iv": safe_float(s_row.get("impliedVolatility")) * 100,
        "delta_for_output": s_delta,
    }, None


def build_trade_for_ticker(ticker_symbol, index):
    try:
        tk = yf.Ticker(ticker_symbol)
        history = tk.history(period="1y")
        if history.empty:
            print(f"  skip {ticker_symbol}: no price history")
            return None

        spot = float(history["Close"].iloc[-1])
        if math.isnan(spot) or spot <= 0:
            # the most recent bar is occasionally incomplete/NaN right after close —
            # a NaN spot silently poisons every single strike's delta calculation
            # downstream (spot<=0 doesn't catch NaN; NaN just propagates through the
            # math with no error), which looks like "no strike matched" across the
            # entire chain rather than the actual, single-point root cause. Try
            # falling back a day before giving up.
            if len(history) >= 2:
                spot = float(history["Close"].iloc[-2])
            if math.isnan(spot) or spot <= 0:
                print(f"  skip {ticker_symbol}: spot price is invalid/NaN (most recent close data looks broken)")
                return None

        ema8 = compute_ema(history["Close"], 8).iloc[-1]
        ema20 = compute_ema(history["Close"], 20).iloc[-1]
        uptrend = ema8 > ema20
        near_ema = (abs(spot - ema8) / spot < 0.015) or (abs(spot - ema20) / spot < 0.015)
        atr = compute_atr(history)
        ivr, realized_vol = iv_rank_proxy(history)

        today = datetime.now(timezone.utc).date()
        expirations = tk.options
        if not expirations:
            print(f"  skip {ticker_symbol}: no options listed")
            return None

        expiration_candidates = rank_expirations(expirations, today)
        if not expiration_candidates:
            print(f"  skip {ticker_symbol}: no usable expiration (all listed dates are in the past or unparsable)")
            return None

        is_etf = ticker_symbol in KNOWN_ETFS

        # strategy selection: uptrend -> bullish rotation, downtrend -> bearish rotation
        if uptrend:
            strat = ["Short Put", "Covered Call", "Bull Put Spread"][index % 3]
            side = "bull"
        else:
            strat = ["Short Call", "Bear Call Spread"][index % 2]
            side = "bear"

        # Evaluate every expiration candidate within the target window (up to the
        # cap), and keep whichever produces the best annualized profit — instead
        # of just taking the first usable one. This is also why different tickers
        # naturally land on different expiration dates now, rather than every name
        # converging on "whichever Friday is closest to 30 days out": each ticker's
        # own IV term structure and chain liquidity determines its own best pick.
        best = None
        failure_reasons = []
        evaluated_log = []  # every candidate's outcome, logged regardless of win/loss —
                             # needed to see WHY a ticker keeps landing on the same
                             # expiration: genuinely winning on merit vs. every
                             # alternative failing validation outright.
        for cand_exp, cand_dte in expiration_candidates[:MAX_CANDIDATES_TO_EVALUATE]:
            result, reason = evaluate_expiration_candidate(tk, strat, side, spot, cand_exp, cand_dte, atr)
            if result:
                evaluated_log.append(f"{cand_exp}({cand_dte}d)=ap:{result['ap']}%")
                if best is None or result["ap"] > best["ap"]:
                    best = result
            else:
                evaluated_log.append(f"{cand_exp}({cand_dte}d)=FAILED:{reason}")
                failure_reasons.append(f"{cand_exp} ({cand_dte}d): {reason}")

        print(f"    {ticker_symbol} [{strat}] evaluated {len(evaluated_log)} candidate(s): {' | '.join(evaluated_log)}")

        if not best:
            tried = len(failure_reasons)
            print(f"  skip {ticker_symbol}: {strat} unusable across {tried} expiration(s) tried — {'; '.join(failure_reasons)}")
            return None

        dte = best["dte"]

        # earnings within the chosen expiration's window? also capture days-until
        # regardless of window, since "next earnings in 4 days" is useful context
        # even for a trade that isn't flagged as earnings-risky.
        earnings_soon = False
        days_to_earnings = None
        try:
            edates = tk.get_earnings_dates(limit=4)
            if edates is not None and not edates.empty:
                for dt in edates.index:
                    d = dt.date() if hasattr(dt, "date") else dt
                    delta_days = (d - today).days
                    if delta_days >= 0 and (days_to_earnings is None or delta_days < days_to_earnings):
                        days_to_earnings = delta_days
                    if 0 <= delta_days <= dte:
                        earnings_soon = True
        except Exception:
            pass  # earnings calendar not always available — leave as None/False

        news_sentiment, news_sentiment_label, news_headlines = fetch_news_and_sentiment(tk, ticker_symbol)

        score = composite_score(best["ap"], best["pot"], ivr)
        vol_regime, iv_rv_ratio = classify_vol_regime(best["iv"], realized_vol)

        return {
            "sym": ticker_symbol,
            "strat": strat,
            "side": side,
            "isETF": is_etf,
            "strike": best["strike"],
            "exp": best["exp"],
            "dte": best["dte"],
            "pot": best["pot"],
            "ap": best["ap"],
            "ivr": ivr,
            "dailyReturn": best["dailyReturn"],
            "roc": best["roc"],
            "score": score,
            "buy": bool(uptrend),
            "sell": bool(not uptrend),
            "ema": bool(near_ema),
            "earningsSoon": earnings_soon,
            "daysToEarnings": days_to_earnings,
            "marginOfSafety": best["marginOfSafety"],
            "delta": best["delta"],
            "iv": best["iv"],
            "premium": best["premium"],
            "breakeven": best["breakeven"],
            "maxLoss": best["maxLoss"],
            "pcOI": best["pcOI"],
            "pcVol": best["pcVol"],
            "newsSentiment": news_sentiment,
            "newsSentimentLabel": news_sentiment_label,
            "newsHeadlines": news_headlines,
            "volRegime": vol_regime,
            "ivRvRatio": iv_rv_ratio,
            "realizedVol": realized_vol,
        }

    except Exception as e:
        print(f"  skip {ticker_symbol}: {e}")
        return None


def compute_rsi(closes, period=14):
    """Standard 14-day RSI, using an EWM approximation of Wilder's smoothing."""
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def classify_trend(ema8, ema20, ema50):
    """Simple moving-average-alignment trend classification — not predictive,
    just describes the current shape of the average, same spirit as the
    options side's EMA8/20 buy/sell signal."""
    if ema8 > ema20 > ema50:
        return "Strong Uptrend"
    if ema8 > ema20:
        return "Uptrend"
    if ema8 < ema20 < ema50:
        return "Strong Downtrend"
    if ema8 < ema20:
        return "Downtrend"
    return "Neutral"


def equity_composite_score(trend, rsi, chg_6m):
    """Illustrative 0-10 blend for a plain stock/ETF buy-candidate view — same
    honesty caveat as the options side's composite_score: this is a heuristic
    blend I picked, not a validated signal. Adjust the weights to your taste."""
    trend_map = {"Strong Uptrend": 10, "Uptrend": 7, "Neutral": 5, "Downtrend": 3, "Strong Downtrend": 0}
    trend_component = trend_map.get(trend, 5)

    # reward "healthy" momentum (45-65 RSI), penalize extreme overbought/oversold
    if rsi is None:
        rsi_component = 5
    elif 45 <= rsi <= 65:
        rsi_component = 8
    elif rsi > 80 or rsi < 20:
        rsi_component = 2
    else:
        rsi_component = 5

    momentum_component = min(10, max(0, 5 + (chg_6m or 0) / 4))  # +20% over 6mo -> 10

    score = 0.45 * trend_component + 0.30 * rsi_component + 0.25 * momentum_component
    return round(min(10, max(0, score)), 1)


def build_equity_snapshot(ticker_symbol, is_etf):
    """
    Evaluates a ticker as a plain stock/ETF buy candidate — the same free
    yfinance data source as the options pipeline above, but scored on
    equity-appropriate criteria (trend, momentum, valuation, dividend yield)
    instead of options Greeks. Returns a dict, or None if the ticker's data
    is unusable.

    IMPORTANT — same spirit as the options docstring at the top of this file:
    - `peRatio` / `dividendYield` / `marketCap` come from yfinance's .info,
      which is frequently incomplete, especially for ETFs — expense ratio,
      holdings, and AUM aren't reliably available for free at all, so they're
      not included here. A missing value means "not reported," not "zero."
    - `rsi` is a standard 14-day RSI (Wilder-style), a momentum indicator, not
      a prediction.
    - `trend` is a moving-average-alignment heuristic (8/20/50-day EMA).
    - `equityScore` is an illustrative weighted blend — see
      `equity_composite_score()` above to change what it weighs.
    """
    try:
        tk = yf.Ticker(ticker_symbol)
        history = tk.history(period="1y")
        if history.empty or len(history) < 60:
            print(f"  skip {ticker_symbol} (equity): insufficient price history")
            return None

        closes = history["Close"]
        price = float(closes.iloc[-1])
        if math.isnan(price) or price <= 0:
            print(f"  skip {ticker_symbol} (equity): invalid current price")
            return None

        high_52wk = float(history["High"].max())
        low_52wk = float(history["Low"].min())
        pct_from_high = round((price - high_52wk) / high_52wk * 100, 1) if high_52wk > 0 else None
        pct_from_low = round((price - low_52wk) / low_52wk * 100, 1) if low_52wk > 0 else None

        ema8 = compute_ema(closes, 8).iloc[-1]
        ema20 = compute_ema(closes, 20).iloc[-1]
        ema50 = compute_ema(closes, 50).iloc[-1] if len(closes) >= 50 else ema20
        trend = classify_trend(ema8, ema20, ema50)

        rsi_series = compute_rsi(closes)
        last_rsi = rsi_series.iloc[-1]
        rsi = round(float(last_rsi), 1) if not math.isnan(last_rsi) else None

        def pct_change(days_back):
            if len(closes) <= days_back:
                return None
            past = closes.iloc[-days_back - 1]
            if past is None or math.isnan(past) or past <= 0:
                return None
            return round((price - past) / past * 100, 1)

        chg_1m = pct_change(21)
        chg_3m = pct_change(63)
        chg_6m = pct_change(126)
        chg_1y = pct_change(min(252, len(closes) - 1))

        vol_ratio = None
        if "Volume" in history.columns:
            avg_vol_30 = history["Volume"].tail(30).mean()
            latest_vol = history["Volume"].iloc[-1]
            if avg_vol_30 and avg_vol_30 > 0 and not math.isnan(avg_vol_30) and not math.isnan(latest_vol):
                vol_ratio = round(latest_vol / avg_vol_30, 2)

        # best-effort fundamentals — often missing, especially for ETFs
        pe_ratio, dividend_yield, market_cap = None, None, None
        try:
            info = tk.info or {}
            raw_pe = info.get("trailingPE")
            pe_ratio = round(raw_pe, 1) if isinstance(raw_pe, (int, float)) else None
            raw_dy = info.get("dividendYield")
            # yfinance's dividendYield is already a percentage value (e.g. 0.73
            # meaning 0.73%), NOT a fraction needing *100 — confirmed against
            # real known yields after an earlier version of this code multiplied
            # by 100 and produced "AAPL: 34%", "JPM: 167%" etc. Sanity-clamp
            # anyway: a legitimate stock/ETF yield above ~20% is essentially
            # never real (usually a units mismatch or a data glitch), so treat
            # it as unreliable rather than display an obviously-wrong number.
            if isinstance(raw_dy, (int, float)) and 0 <= raw_dy <= 20:
                dividend_yield = round(raw_dy, 2)
            market_cap = info.get("marketCap")
        except Exception:
            pass  # fundamentals unavailable — leave as None, not zero

        # reuse the same news/sentiment pipeline as the options side — already
        # hardened against yfinance's .news not being reliably ticker-scoped
        news_sentiment, news_sentiment_label, news_headlines = fetch_news_and_sentiment(tk, ticker_symbol)

        score = equity_composite_score(trend, rsi, chg_6m)

        return {
            "sym": ticker_symbol,
            "isETF": is_etf,
            "price": round(price, 2),
            "high52wk": round(high_52wk, 2),
            "low52wk": round(low_52wk, 2),
            "pctFromHigh": pct_from_high,
            "pctFromLow": pct_from_low,
            "trend": trend,
            "rsi": rsi,
            "chg1m": chg_1m,
            "chg3m": chg_3m,
            "chg6m": chg_6m,
            "chg1y": chg_1y,
            "volRatio": vol_ratio,
            "peRatio": pe_ratio,
            "dividendYield": dividend_yield,
            "marketCap": market_cap,
            "score": score,
            "newsSentiment": news_sentiment,
            "newsSentimentLabel": news_sentiment_label,
            "newsHeadlines": news_headlines,
        }
    except Exception as e:
        print(f"  skip {ticker_symbol} (equity): {e}")
        return None


def main():
    trades = []
    equities = []
    for i, ticker in enumerate(TICKERS):
        print(f"Fetching {ticker}...")
        trade = build_trade_for_ticker(ticker, i)
        if trade:
            trades.append(trade)

        equity = build_equity_snapshot(ticker, ticker in KNOWN_ETFS)
        if equity:
            equities.append(equity)

    if not trades and not equities:
        print("No trades or equities were built — leaving existing data.json untouched.", file=sys.stderr)
        sys.exit(1)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "yfinance (unofficial, free, EOD)",
        "trades": trades,
        "equities": equities,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(trades)} trades and {len(equities)} equity snapshots to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
