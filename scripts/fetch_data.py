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
from zoneinfo import ZoneInfo

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
TARGET_LONG_DELTA = 0.45    # long call/put target delta — near-the-money; balances
                             # cost against probability of profit, instead of either
                             # a cheap far-OTM "lottery ticket" or an expensive
                             # deep-ITM stock-replacement play
RISK_FREE_RATE = 0.045      # flat approximation; update periodically
OUTPUT_PATH = "data.json"

# --- Portfolio hedge candidate ---------------------------------------------
# Every current strategy in this file sells premium (short vol) — a portfolio
# built entirely from them is exposed to the same bad day: a sudden move that
# spikes IV and moves the underlying against several short strikes at once.
# The common, simple offset is a small, cheap, far-OTM index put bought as
# tail-risk insurance — it's not trying to make money, it's there to pay off
# specifically when everything else is hurting at the same time. QQQ (not a
# single name) so this doesn't overlap with any one ticker's own short strikes.
HEDGE_TICKER = "QQQ"
HEDGE_TARGET_DTE_MIN = 30    # longer-dated than the 7-45d strategy window on
HEDGE_TARGET_DTE_MAX = 60    # purpose — a hedge you're re-checking every EOD run
                             # doesn't need to be rolled as often as a premium-
                             # selling trade does, and a 30-60d put decays slower.
HEDGE_TARGET_DELTA = 0.15    # further OTM than the 0.20-delta short-strike
                             # convention above — cheaper per contract, which
                             # matters since this is meant to cost a small,
                             # known amount, not be a large directional bet.
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


def probability_of_profit(delta):
    """
    Rough proxy for probability of finishing beyond a point AT expiration — used
    for long option breakeven, where a HIGHER number is favorable (the opposite
    reading from probability_of_touch's short-strike-touch risk metric above;
    see the "potIsProfitProb" flag on Long Call/Long Put trades, which tells the
    frontend which meaning applies).

    Deliberately NOT doubled like probability_of_touch: that ×2 approximates
    probability of touching a barrier ANY TIME before expiration, a different
    (larger) quantity than probability of finishing beyond a point AT
    expiration — delta itself is the standard rough proxy for the latter.
    """
    if delta is None or math.isnan(delta):
        return 50  # neutral fallback rather than crashing the whole ticker
    return min(100, round(abs(delta) * 100))


def compute_ema(closes, span):
    return closes.ewm(span=span, adjust=False).mean()


def _try_fast_info_price(tk):
    """
    Best-effort fresher price via yfinance's fast_info, which taps a
    different, live-quote-oriented endpoint than history()'s historical-
    chart data (see resolve_current_price() below for why that distinction
    matters). fast_info's exact attribute/key names have shifted across
    yfinance versions and requirements.txt only pins a lower bound
    (yfinance>=0.2.40), so this tries several known spellings defensively
    rather than assuming one — returns None (never raises) if none work,
    so a yfinance version this wasn't written against just falls back to
    the caller's existing history()-based price instead of crashing.
    """
    try:
        fi = tk.fast_info
    except Exception:
        return None
    for key in ("last_price", "lastPrice", "regular_market_price", "regularMarketPrice"):
        val = None
        try:
            val = fi[key]
        except Exception:
            val = getattr(fi, key, None)
        fval = safe_float(val, default=None)
        if fval is not None and fval > 0:
            return fval
    return None


def resolve_current_price(tk, history, ticker_symbol, context, now_et=None):
    """
    `history["Close"].iloc[-1]` (the price source every caller in this file
    uses) can lag the real most-recent close by a full trading day — this
    isn't theoretical, it's confirmed against real data: a run at 11:19pm ET
    on 2026-09-15 (long after that day's 4pm close, well past normal
    settlement) returned SPY at $760.88 and QQQ at $709.18 — both exactly
    2026-09-14's close, not 2026-09-15's real close ($757.39 / $704.54, per
    stockanalysis.com's published history). That's Yahoo's own historical-
    chart data lagging its live-quote data, not something a caching
    parameter here controls — a shorter period= on the same history() call
    hits the same backend pipeline and would show the same lag.

    Heuristic: if it's a weekday evening (past 5pm ET — safely past close
    and normal settlement) and the most recent daily bar isn't from today,
    treat the close as suspect and try fast_info, which isn't affected by
    the same lag, as a fresher cross-check. This is a heuristic, not an
    exhaustive fix (it won't catch every possible staleness window, e.g.
    one spanning a long holiday weekend) — it directly targets the exact
    failure mode confirmed above rather than trying to be a full market
    calendar. Returns (price, is_stale): is_stale is True when this is
    still using the (known-suspect) history() price because fast_info
    wasn't available or didn't look any better — callers can use that to
    flag the number rather than presenting it as confidently current.

    `now_et` is exposed purely so tests can pin "now" instead of depending
    on the real clock — production callers should always leave it as None.
    """
    price = float(history["Close"].iloc[-1])
    last_bar_date = history.index[-1]
    try:
        last_bar_date = last_bar_date.date()
    except AttributeError:
        pass  # already a plain date, or an unexpected index type — comparison below just won't match, which is safe (treated as not-stale)

    if now_et is None:
        now_et = datetime.now(ZoneInfo("America/New_York"))
    looks_stale = now_et.weekday() < 5 and now_et.hour >= 17 and last_bar_date < now_et.date()
    if not looks_stale:
        return price, False

    fresher = _try_fast_info_price(tk)
    if fresher is not None and price > 0 and abs(fresher - price) / price > 0.001:
        print(f"  {ticker_symbol} ({context}): history() close ({price}) looks stale "
              f"(last bar {last_bar_date}, but it's {now_et:%H:%M} ET) — using fast_info "
              f"price ({fresher}) instead")
        return fresher, False

    print(f"  {ticker_symbol} ({context}): history() close ({price}) looks stale "
          f"(last bar {last_bar_date}, but it's {now_et:%H:%M} ET) and no better fast_info "
          f"price was available — using it anyway, flagged as stale")
    return price, True


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


def composite_score_long_option(pot_profit, ivr, breakeven_move_pct):
    """
    Illustrative 0-10 blend for Long Call/Long Put — scored differently from
    composite_score() above because none of its inputs carry over cleanly:
    there's no annualized-profit figure (unlimited/floor-at-zero upside can't
    be reduced to one number without inventing a price target), `pot` here
    already means probability of PROFIT so higher is better (the opposite of
    composite_score's touch-probability reading), and cheap IV (low ivr) is
    what a BUYER wants — the inverse of composite_score's ivr_component,
    which rewards rich premium for a seller.
    """
    profit_prob_component = min(10, max(0, pot_profit / 10))
    cheap_iv_component = min(10, max(0, (100 - ivr) / 10))
    move_component = min(10, max(0, 10 - (breakeven_move_pct or 0) / 2))  # smaller
                                                                            # required
                                                                            # move -> higher
    score = 0.40 * profit_prob_component + 0.30 * cheap_iv_component + 0.30 * move_component
    return round(min(10, max(1, score)), 1)


def composite_score_calendar(ivr, front_iv, back_iv):
    """
    Rough illustrative 0-10 blend for Calendar Spread — the least-grounded of
    the scores in this file, since a calendar's real edge depends on future
    IV/theta dynamics this EOD, no-forecasting pipeline doesn't model (see
    build_calendar_trade()'s docstring). Two weak signals only: cheap IV Rank
    (entry-cost proxy, same preference every debit strategy here gets), and
    whether the front (sold) leg's IV is rich relative to the back (bought)
    leg's — the classic favorable calendar setup — defaulting to neutral when
    either IV reading is missing. Treat this score even more skeptically than
    the others in this file.
    """
    cheap_component = min(10, max(0, (100 - ivr) / 10))
    term_structure_component = 5  # neutral default when IVs aren't usable
    if front_iv and back_iv and back_iv > 0:
        ratio = front_iv / back_iv
        term_structure_component = min(10, max(0, (ratio - 0.8) * 25))  # >1.0 favorable
    score = 0.5 * cheap_component + 0.5 * term_structure_component
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

    # Long Call/Long Put have uncapped (call) or floor-at-zero (put) upside — a
    # "% return on capital" figure is either undefined or requires inventing a
    # made-up price target, so these two skip roc/ap entirely. breakevenMovePct
    # (below) is the honest substitute: the % move actually required to profit.
    is_uncapped_debit = strat in ("Long Call", "Long Put")
    if is_uncapped_debit:
        if premium >= spot:
            return None, f"premium (${premium:.2f}) implausibly >= spot (${spot:.2f}), likely a thin/wide-market quote"
        roc = None
        ann_profit = None
    elif strat == "Butterfly":
        # collateral/premium ARE the net debit here (see try_strategy_pick) —
        # but ROC has to come from max PROFIT, not premium/collateral (that
        # formula assumes premium = credit received, which isn't true for a
        # debit trade), so it's computed explicitly instead of reusing the
        # premium-selling formula below.
        max_profit_per_share = fields["max_profit_per_share"]
        if max_profit_per_share <= 0:
            return None, "no room for profit after costs (net debit exceeds wing width)"
        roc = round((max_profit_per_share / collateral) * 100, 2)
        ann_profit = round(roc * (365 / cand_dte), 1)
        # Butterflies legitimately run much higher MAX-ROI multiples than
        # premium-selling strategies (cheap debit, wide payoff) — a several-
        # hundred-percent max (not typical) return is normal here, not a data
        # error, so the credit-strategy ceiling doesn't apply as-is. A much
        # higher one still catches genuinely broken quotes.
        if roc > MAX_PLAUSIBLE_ROC * 5:
            return None, f"implausible max ROC ({roc}%), likely a thin/wide-market quote"
    else:
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
    # $/day only means "income per day" for a credit strategy — for a debit
    # strategy premium is money PAID, so this is left unset rather than
    # published under a label that implies the opposite of what it means.
    daily_return = None if (is_uncapped_debit or strat == "Butterfly") else round(premium * 100 / cand_dte, 2)

    if strat == "Iron Condor":
        # Two strikes at risk instead of one — use the WORSE (higher) of the
        # two individual touch probabilities as the overall risk reading,
        # since the position is only as safe as its more-threatened wing.
        pot_put = probability_of_touch(bs_delta(spot, fields["put_short_strike"], cand_dte, iv / 100 if iv else 0.3, "put"))
        pot_call = probability_of_touch(bs_delta(spot, fields["call_short_strike"], cand_dte, iv / 100 if iv else 0.3, "call"))
        pot = max(pot_put, pot_call)
        margin_of_safety = bool(atr and min(
            abs(spot - fields["put_short_strike"]), abs(spot - fields["call_short_strike"])
        ) >= atr)
        breakeven_out = fields["breakeven_low"]  # single-field fallback; breakevenLow/High carry the real range below
        breakeven_low_out = fields["breakeven_low"]
        breakeven_high_out = fields["breakeven_high"]
        breakeven_move_pct_out = None
        pot_is_profit_prob = False
    elif strat == "Butterfly":
        # Same "worse of the two boundaries" reading Iron Condor uses just
        # above — pot means probability of BREACHING a breakeven (finishing
        # outside the profit zone), higher = more likely to lose, same
        # direction as every non-debit-long strategy in this file. NOT
        # inverted like Long Call/Long Put below.
        pot_low = probability_of_touch(bs_delta(spot, fields["breakeven_low"], cand_dte, iv / 100 if iv else 0.3, "put"))
        pot_high = probability_of_touch(bs_delta(spot, fields["breakeven_high"], cand_dte, iv / 100 if iv else 0.3, "call"))
        pot = max(pot_low, pot_high)
        margin_of_safety = bool(atr and min(
            abs(spot - fields["breakeven_low"]), abs(spot - fields["breakeven_high"])
        ) >= atr)
        breakeven_out = fields["breakeven_low"]
        breakeven_low_out = fields["breakeven_low"]
        breakeven_high_out = fields["breakeven_high"]
        breakeven_move_pct_out = None
        pot_is_profit_prob = False
    elif is_uncapped_debit:
        option_type = "call" if strat == "Long Call" else "put"
        breakeven_delta = bs_delta(spot, fields["breakeven"], cand_dte, iv / 100 if iv else 0.3, option_type)
        pot = probability_of_profit(breakeven_delta)
        margin_of_safety = bool(atr and abs(spot - fields["strike_for_pot"]) >= atr)
        breakeven_out = round(fields["breakeven"], 2)
        breakeven_low_out = None
        breakeven_high_out = None
        move = (fields["breakeven"] - spot) if strat == "Long Call" else (spot - fields["breakeven"])
        breakeven_move_pct_out = round(move / spot * 100, 2)
        pot_is_profit_prob = True
    else:
        pot = probability_of_touch(bs_delta(spot, fields["strike_for_pot"], cand_dte, iv / 100 if iv else 0.3, "put" if side == "bull" else "call"))
        margin_of_safety = bool(atr and abs(spot - fields["strike_for_pot"]) >= atr)
        breakeven_out = round(fields["breakeven"], 2)
        breakeven_low_out = None
        breakeven_high_out = None
        breakeven_move_pct_out = None
        pot_is_profit_prob = False

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
        "breakeven": breakeven_out,
        "breakevenLow": breakeven_low_out,
        "breakevenHigh": breakeven_high_out,
        "breakevenMovePct": breakeven_move_pct_out,
        "maxLoss": fields["max_loss"],
        "pcOI": pc_oi,
        "pcVol": pc_vol,
        "strike": fields["strike_label"],
        "hedge": fields.get("hedge"),
        "debitStrategy": strat in ("Long Call", "Long Put", "Butterfly"),
        "potIsProfitProb": pot_is_profit_prob,
    }, None


def build_calendar_trade(tk, spot, expiration_candidates, atr):
    """
    Calendar Spread: sell a near-term ATM call, buy a longer-dated ATM call at
    the SAME strike. A theta/term-structure play, not a directional one — it
    profits from time decay differential between the two legs and/or an IV
    pickup in the back month, not from the stock moving.

    Needs TWO separate chain fetches (front + back expiration) instead of the
    one every other strategy in this file uses, so it's meaningfully more
    yfinance requests per pick — worth knowing if you're tuning how often the
    nightly workflow runs.

    Deliberately NOT modeled here: max profit and a breakeven. Both depend on
    what the back-month leg is worth AT front-month expiration, which depends
    on implied volatility at that future date — something this EOD, no-
    forecasting pipeline has no honest way to estimate. Rather than publish a
    plausible-looking but fundamentally made-up number, this only reports
    what's actually knowable today: the net debit paid, which doubles as
    calendars' standard textbook max loss (if held to front expiration and
    unwound worthless). Treat the missing max-profit/breakeven fields as a
    real gap, not an oversight — chart a real payoff diagram before trading
    one of these, not just this app's numbers.
    """
    # front leg: nearest expiration with at least the usual minimum DTE (avoid
    # an already-negligible front premium); back leg: first candidate at least
    # 3 weeks past the front, so there's a real time-value gap between legs.
    usable = sorted(
        ((exp, dte) for exp, dte in expiration_candidates if dte >= TARGET_DTE_MIN),
        key=lambda pair: pair[1],
    )
    if len(usable) < 2:
        return None, "not enough listed expirations for a front+back calendar pair"
    front_exp, front_dte = usable[0]
    back_candidates = [(exp, dte) for exp, dte in usable if dte >= front_dte + 21]
    if not back_candidates:
        return None, "no back-month expiration at least 3 weeks past the front month"
    back_exp, back_dte = back_candidates[0]

    try:
        front_chain = tk.option_chain(front_exp)
        back_chain = tk.option_chain(back_exp)
    except Exception as e:
        return None, f"chain fetch failed: {e}"

    front_calls, front_puts = front_chain.calls, front_chain.puts
    front_sorted = front_calls.assign(_dist=(front_calls["strike"] - spot).abs()).sort_values("_dist")
    front_row = None
    for _, r in front_sorted.iterrows():
        if safe_float(r.get("impliedVolatility"), 0) > 0 and safe_float(r.get("strike"), 0) > 0:
            front_row = r
            break
    if front_row is None:
        return None, f"no usable at-the-money front-month call — {chain_diagnostics(front_calls, spot)}"
    strike = float(front_row["strike"])

    back_matches = back_chain.calls[back_chain.calls["strike"] == strike]
    if back_matches.empty:
        return None, f"back-month chain has no matching ${strike:.0f} strike"
    back_row = back_matches.iloc[0]
    if safe_float(back_row.get("impliedVolatility"), 0) <= 0:
        return None, f"back-month ${strike:.0f} strike has no valid IV — {chain_diagnostics(back_chain.calls, spot)}"

    front_premium = mid_price(front_row)  # collected — this leg is sold
    back_premium = mid_price(back_row)    # paid — this leg is bought
    net_debit = back_premium - front_premium
    if net_debit <= 0:
        return None, "back-month leg isn't pricier than front-month — no real debit to pay (unusual/stale quote)"

    front_iv = safe_float(front_row.get("impliedVolatility")) * 100
    back_iv = safe_float(back_row.get("impliedVolatility")) * 100
    front_delta = bs_delta(spot, strike, front_dte, front_iv / 100 if front_iv else 0.3, "call")

    total_call_oi = front_calls["openInterest"].fillna(0).sum()
    total_put_oi = front_puts["openInterest"].fillna(0).sum()
    total_call_vol = front_calls["volume"].fillna(0).sum()
    total_put_vol = front_puts["volume"].fillna(0).sum()
    pc_oi = round(total_put_oi / total_call_oi, 2) if total_call_oi else 0
    pc_vol = round(total_put_vol / total_call_vol, 2) if total_call_vol else 0

    def label(exp_str):
        d = datetime.strptime(exp_str, "%Y-%m-%d")
        return d.strftime("%b %-d") if sys.platform != "win32" else d.strftime("%b %d").replace(" 0", " ")

    return {
        "exp": label(front_exp),
        "backExp": label(back_exp),
        "dte": front_dte,
        "backDte": back_dte,
        "pot": None,
        "ap": None,
        "dailyReturn": None,
        "roc": None,
        # Near-the-money IS the point for a calendar, so "margin of safety"
        # here really means "well-centered on spot", not "far enough away" —
        # the opposite sense from every other strategy's use of this flag.
        "marginOfSafety": bool(atr and abs(spot - strike) < atr),
        "delta": round(front_delta, 2),
        "iv": round(front_iv, 1),
        "backIv": round(back_iv, 1),
        "premium": round(net_debit, 2),
        "breakeven": None,
        "breakevenLow": None,
        "breakevenHigh": None,
        "breakevenMovePct": None,
        "maxLoss": round(net_debit * 100, 2),
        "pcOI": pc_oi,
        "pcVol": pc_vol,
        "strike": f"${strike:.0f} C",
        "hedge": None,
        "debitStrategy": True,
        "potIsProfitProb": False,
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


def compute_collar_hedge(puts, spot, dte, call_premium):
    """
    Suggests a protective put that would turn a Covered Call into a collar.
    This is genuinely different math from compute_naked_hedge — a covered
    call's risk is the STOCK declining, not option assignment, so the "max
    loss" here is the gap between spot and the protective put's strike, net
    of the combined premium (call collected, put paid). Reuses the same
    delta-targeted strike-picking already used for the call leg, for the
    protective put too, rather than a separate ad-hoc rule.
    """
    picked = pick_strike_by_delta(puts, spot, dte, TARGET_SHORT_DELTA, "put")
    if not picked:
        return None
    row, _ = picked
    put_strike = float(row["strike"])
    put_cost = mid_price(row)
    net_credit_per_share = call_premium - put_cost
    # floored at 0: a negative result means the net credit alone exceeds the
    # spot-to-put gap, i.e. the worst case is still a net gain, not a loss —
    # simpler to show "$0 max loss" than a confusing negative "loss" figure
    max_loss = max(0.0, (spot - put_strike - net_credit_per_share) * 100)
    return {
        "hedgeStrike": put_strike,
        "hedgeCost": round(put_cost, 2),
        "cappedMaxLoss": round(max_loss, 2),
    }


def try_strategy_pick(strat, calls, puts, spot, dte):
    """
    Attempts to pick strikes for `strat` against one expiration's chain.
    Returns (fields_dict, None) on success, or (None, reason_string) on failure —
    the reason gets logged by the caller, and used to try the next expiration
    candidate rather than silently giving up on the whole ticker.
    """
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
            "hedge": compute_collar_hedge(puts, spot, dte, premium),
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

    if strat == "Iron Condor":
        # Both sides reuse the exact same leg-selection rule as the standalone
        # Bull Put Spread / Bear Call Spread above (2nd-next further-OTM
        # strike) — an iron condor IS those two spreads, run together on the
        # same ticker and expiration, not a different construction method.
        put_short_row = pick_strike_by_delta(puts, spot, dte, TARGET_SHORT_DELTA, "put")
        if not put_short_row:
            return None, f"no put near target delta for condor's put side — {chain_diagnostics(puts, spot)}"
        ps_row, ps_delta = put_short_row
        put_short_strike = float(ps_row["strike"])
        lower_strikes = puts[puts["strike"] < put_short_strike].sort_values("strike", ascending=False)
        if lower_strikes.empty:
            return None, "no further-OTM strike available for condor's put long leg"
        put_long_row = lower_strikes.iloc[min(1, len(lower_strikes) - 1)]
        put_long_strike = float(put_long_row["strike"])
        put_premium = mid_price(ps_row) - mid_price(put_long_row)
        put_width = put_short_strike - put_long_strike

        call_short_row = pick_strike_by_delta(calls, spot, dte, TARGET_SHORT_DELTA, "call")
        if not call_short_row:
            return None, f"no call near target delta for condor's call side — {chain_diagnostics(calls, spot)}"
        cs_row, cs_delta = call_short_row
        call_short_strike = float(cs_row["strike"])
        higher_strikes = calls[calls["strike"] > call_short_strike].sort_values("strike")
        if higher_strikes.empty:
            return None, "no further-OTM strike available for condor's call long leg"
        call_long_row = higher_strikes.iloc[min(1, len(higher_strikes) - 1)]
        call_long_strike = float(call_long_row["strike"])
        call_premium = mid_price(cs_row) - mid_price(call_long_row)
        call_width = call_long_strike - call_short_strike

        total_premium = put_premium + call_premium
        # The stock can't simultaneously be above the call spread AND below
        # the put spread at expiration — only one side can ever be breached —
        # so max loss uses the WORSE single-side width, not the sum of both.
        worst_width = max(put_width, call_width)
        avg_iv = (safe_float(ps_row.get("impliedVolatility")) + safe_float(cs_row.get("impliedVolatility"))) / 2 * 100

        return {
            "premium": total_premium, "collateral": worst_width,
            "max_loss": round((worst_width - total_premium) * 100, 2),
            "strike_label": f"${put_long_strike:.0f}/{put_short_strike:.0f}/{call_short_strike:.0f}/{call_long_strike:.0f}",
            "iv": avg_iv,
            "delta_for_output": ps_delta + cs_delta,  # net delta -- roughly market-neutral by construction
            "put_short_strike": put_short_strike, "call_short_strike": call_short_strike,
            "breakeven_low": round(put_short_strike - total_premium, 2),
            "breakeven_high": round(call_short_strike + total_premium, 2),
            # Iron Condor is already defined-risk on both sides by construction —
            # nothing to hedge, same as the standalone spreads above.
            "hedge": None,
        }, None

    if strat == "Bear Call Spread":
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

    if strat == "Long Call":
        picked_row = pick_strike_by_delta(calls, spot, dte, TARGET_LONG_DELTA, "call")
        if not picked_row:
            return None, f"no call near target delta — {chain_diagnostics(calls, spot)}"
        row, delta = picked_row
        premium = mid_price(row)
        strike = float(row["strike"])
        return {
            "premium": premium, "strike_for_pot": strike, "collateral": premium,
            "breakeven": strike + premium, "max_loss": round(premium * 100, 2),
            "strike_label": f"${strike:.0f} C", "iv": safe_float(row.get("impliedVolatility")) * 100,
            "delta_for_output": delta,
        }, None

    if strat == "Long Put":
        picked_row = pick_strike_by_delta(puts, spot, dte, TARGET_LONG_DELTA, "put")
        if not picked_row:
            return None, f"no put near target delta — {chain_diagnostics(puts, spot)}"
        row, delta = picked_row
        premium = mid_price(row)
        strike = float(row["strike"])
        return {
            "premium": premium, "strike_for_pot": strike, "collateral": premium,
            "breakeven": strike - premium, "max_loss": round(premium * 100, 2),
            "strike_label": f"${strike:.0f} P", "iv": safe_float(row.get("impliedVolatility")) * 100,
            "delta_for_output": delta,
        }, None

    if strat == "Butterfly":
        # Centered ATM (closest strike to spot), wings one "next available"
        # strike further out on each side — same "2nd next" convention the
        # spreads above use, applied symmetrically on both sides. Built with
        # calls only: a call butterfly and a put butterfly at the same three
        # strikes have virtually identical payoffs by put-call parity, so
        # pricing both would cost extra requests for no real benefit.
        sorted_by_dist = calls.assign(_dist=(calls["strike"] - spot).abs()).sort_values("_dist")
        body_row = None
        for _, r in sorted_by_dist.iterrows():
            if safe_float(r.get("impliedVolatility"), 0) > 0 and safe_float(r.get("strike"), 0) > 0:
                body_row = r
                break
        if body_row is None:
            return None, f"no usable at-the-money call for butterfly body — {chain_diagnostics(calls, spot)}"
        body_strike = float(body_row["strike"])

        lower = calls[calls["strike"] < body_strike].sort_values("strike", ascending=False)
        upper = calls[calls["strike"] > body_strike].sort_values("strike")
        if lower.empty or upper.empty:
            return None, "no further-OTM strikes available on both sides for butterfly wings"
        lower_row = lower.iloc[min(1, len(lower) - 1)]
        upper_row = upper.iloc[min(1, len(upper) - 1)]
        lower_strike = float(lower_row["strike"])
        upper_strike = float(upper_row["strike"])

        lower_width = body_strike - lower_strike
        upper_width = upper_strike - body_strike
        # Reject meaningfully lopsided wings (thin chain / missing strikes)
        # rather than publish a butterfly whose two sides have very different
        # widths — the wider side's math (max profit, breakeven) gets murkier
        # the more asymmetric it is, and a clean symmetric butterfly is the
        # whole point of the strategy.
        if lower_width <= 0 or upper_width <= 0 or max(lower_width, upper_width) / min(lower_width, upper_width) > 1.5:
            return None, f"butterfly wings too asymmetric (lower ${lower_width:.2f} vs upper ${upper_width:.2f}) — thin chain"

        body_price = mid_price(body_row)
        lower_price = mid_price(lower_row)
        upper_price = mid_price(upper_row)
        net_debit = (lower_price + upper_price) - 2 * body_price
        if net_debit <= 0:
            return None, "net debit came out zero/negative — likely a stale/crossed quote"

        width = min(lower_width, upper_width)  # conservative: size max profit off
                                                 # the narrower side if not perfectly
                                                 # symmetric
        max_profit_per_share = width - net_debit

        lower_iv = safe_float(lower_row.get("impliedVolatility"), 0.3)
        body_iv = safe_float(body_row.get("impliedVolatility"), 0.3)
        upper_iv = safe_float(upper_row.get("impliedVolatility"), 0.3)
        # Net position delta (lower - 2*body + upper), not just the body call's
        # ~0.50 delta — a well-centered butterfly's real directional exposure
        # is close to zero by construction, same "net delta" framing Iron
        # Condor already uses above.
        net_delta = (
            bs_delta(spot, lower_strike, dte, lower_iv, "call")
            - 2 * bs_delta(spot, body_strike, dte, body_iv, "call")
            + bs_delta(spot, upper_strike, dte, upper_iv, "call")
        )

        return {
            "premium": net_debit, "collateral": net_debit,
            "max_profit_per_share": max_profit_per_share,
            "breakeven_low": round(lower_strike + net_debit, 2),
            "breakeven_high": round(upper_strike - net_debit, 2),
            "max_loss": round(net_debit * 100, 2),
            "strike_label": f"${lower_strike:.0f}/{body_strike:.0f}/{upper_strike:.0f}",
            "iv": body_iv * 100,
            "delta_for_output": net_delta,
            "hedge": None,
        }, None

    return None, f"unknown strategy '{strat}'"


def build_trade_for_ticker(ticker_symbol, index):
    try:
        tk = yf.Ticker(ticker_symbol)
        history = tk.history(period="1y")
        if history.empty:
            print(f"  skip {ticker_symbol}: no price history")
            return None

        spot, spot_is_stale = resolve_current_price(tk, history, ticker_symbol, "options")
        if math.isnan(spot) or spot <= 0:
            # the most recent bar is occasionally incomplete/NaN right after close —
            # a NaN spot silently poisons every single strike's delta calculation
            # downstream (spot<=0 doesn't catch NaN; NaN just propagates through the
            # math with no error), which looks like "no strike matched" across the
            # entire chain rather than the actual, single-point root cause. Try
            # falling back a day before giving up.
            if len(history) >= 2:
                spot = float(history["Close"].iloc[-2])
                spot_is_stale = True  # admittedly using an even older bar now
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

        # strategy selection: 1-in-4 tickers try a neutral, range-bound bet
        # regardless of trend (rotating across Iron Condor / Butterfly /
        # Calendar Spread) since none of those are directional the way the
        # rest are. The remaining tickers still split by trend: uptrend ->
        # bullish rotation (now including Long Call), downtrend -> bearish
        # rotation (now including Long Put).
        if index % 4 == 0:
            strat = ["Iron Condor", "Butterfly", "Calendar Spread"][index % 3]
            side = "neutral"
        elif uptrend:
            strat = ["Covered Call", "Bull Put Spread", "Long Call"][index % 3]
            side = "bull"
        else:
            strat = ["Bear Call Spread", "Long Put"][index % 2]
            side = "bear"

        if strat == "Calendar Spread":
            # Calendars span two expirations at once, so they can't go through
            # the single-best-expiration loop below — build_calendar_trade
            # picks its own front/back pair internally.
            best, reason = build_calendar_trade(tk, spot, expiration_candidates, atr)
            if not best:
                print(f"  skip {ticker_symbol}: Calendar Spread unusable — {reason}")
                return None
        else:
            # Evaluate every expiration candidate within the target window (up to the
            # cap), and keep whichever produces the best annualized profit — instead
            # of just taking the first usable one. This is also why different tickers
            # naturally land on different expiration dates now, rather than every name
            # converging on "whichever Friday is closest to 30 days out": each ticker's
            # own IV term structure and chain liquidity determines its own best pick.
            #
            # Long Call/Long Put have no "ap" at all (see evaluate_expiration_candidate)
            # — for those two, rank candidates by probability of profit instead, the
            # only comparable-across-candidates number they do produce.
            def _rank_key(result):
                return result["ap"] if result["ap"] is not None else result["pot"]

            best = None
            failure_reasons = []
            evaluated_log = []  # every candidate's outcome, logged regardless of win/loss —
                                 # needed to see WHY a ticker keeps landing on the same
                                 # expiration: genuinely winning on merit vs. every
                                 # alternative failing validation outright.
            for cand_exp, cand_dte in expiration_candidates[:MAX_CANDIDATES_TO_EVALUATE]:
                result, reason = evaluate_expiration_candidate(tk, strat, side, spot, cand_exp, cand_dte, atr)
                if result:
                    log_val = f"ap:{result['ap']}%" if result['ap'] is not None else f"potProfit:{result['pot']}%"
                    evaluated_log.append(f"{cand_exp}({cand_dte}d)={log_val}")
                    if best is None or _rank_key(result) > _rank_key(best):
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

        if strat == "Calendar Spread":
            score = composite_score_calendar(ivr, best.get("iv"), best.get("backIv"))
        elif strat in ("Long Call", "Long Put"):
            score = composite_score_long_option(best["pot"], ivr, best["breakevenMovePct"])
        else:
            score = composite_score(best["ap"], best["pot"], ivr)
        vol_regime, iv_rv_ratio = classify_vol_regime(best["iv"], realized_vol)

        return {
            "sym": ticker_symbol,
            "strat": strat,
            "side": side,
            "isETF": is_etf,
            "spot": round(spot, 2),
            "spotPriceStale": spot_is_stale,
            "strike": best["strike"],
            "hedge": best["hedge"],
            "exp": best["exp"],
            "dte": best["dte"],
            "backExp": best.get("backExp"),
            "backDte": best.get("backDte"),
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
            "backIv": best.get("backIv"),
            "premium": best["premium"],
            "breakeven": best["breakeven"],
            "breakevenLow": best["breakevenLow"],
            "breakevenHigh": best["breakevenHigh"],
            "breakevenMovePct": best.get("breakevenMovePct"),
            "maxLoss": best["maxLoss"],
            "pcOI": best["pcOI"],
            "pcVol": best["pcVol"],
            "debitStrategy": best.get("debitStrategy", False),
            "potIsProfitProb": best.get("potIsProfitProb", False),
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


# --- On-demand single-ticker lookup ------------------------------------------
# Added 2026-09-19 so the UI can offer a "look up any symbol" box on top of
# the fixed nightly watchlist above, triggered on demand (see run_lookup()
# and the --ticker CLI flag at the bottom of this file). Reuses every pricing/
# validation helper the batch path above uses — the only real difference is
# how the strategy gets picked.

def pick_lookup_strategy(uptrend, near_ema):
    """
    Picks ONE strategy for an on-demand lookup, from trend alone. This is
    deliberately NOT the nightly batch's `index % 4` / `index % 3` rotation in
    build_trade_for_ticker() above — that rotation exists purely to keep the
    fixed WATCHLIST diversified across strategy types across many tickers, and
    has no meaning applied to a single ad-hoc symbol (a user typing in one
    ticker doesn't have a "list position" to rotate on).

    Also deliberately narrower than "evaluate every strategy and keep the
    best": each strategy evaluated costs up to MAX_CANDIDATES_TO_EVALUATE
    separate options-chain fetches against Yahoo's free, rate-limit-prone
    feed (see the module docstring and chain_diagnostics() above) — trying
    several strategies per click would multiply that cost every time someone
    uses the lookup box. One well-established, defined-risk pick per trend
    bucket keeps a single lookup roughly as expensive as one ticker in the
    nightly batch, not several.
    """
    if near_ema:
        return "Iron Condor", "neutral"
    if uptrend:
        return "Bull Put Spread", "bull"
    return "Bear Call Spread", "bear"


def build_lookup_trade(ticker_symbol):
    """
    On-demand equivalent of build_trade_for_ticker() above, for a single
    symbol typed into the UI rather than a slot in the fixed watchlist.
    Returns (trade_dict, None) on success, or (None, reason_string) on
    failure — unlike the batch path (which only logs skip reasons to stderr
    and silently omits the ticker from data.json), the reason here is shown
    directly to whoever typed the symbol in, so it needs to read as an
    explanation, not a log line.
    """
    try:
        tk = yf.Ticker(ticker_symbol)
        history = tk.history(period="1y")
        if history.empty:
            return None, "No price history found for this symbol — double-check the ticker."

        spot, spot_is_stale = resolve_current_price(tk, history, ticker_symbol, "lookup")
        if math.isnan(spot) or spot <= 0:
            if len(history) >= 2:
                spot = float(history["Close"].iloc[-2])
                spot_is_stale = True
            if math.isnan(spot) or spot <= 0:
                return None, "This symbol's current price looks invalid — Yahoo's data may be broken for it right now."

        ema8 = compute_ema(history["Close"], 8).iloc[-1]
        ema20 = compute_ema(history["Close"], 20).iloc[-1]
        uptrend = bool(ema8 > ema20)
        near_ema = bool((abs(spot - ema8) / spot < 0.015) or (abs(spot - ema20) / spot < 0.015))
        atr = compute_atr(history)
        ivr, realized_vol = iv_rank_proxy(history)

        today = datetime.now(timezone.utc).date()
        expirations = tk.options
        if not expirations:
            return None, "This symbol doesn't have listed options."

        expiration_candidates = rank_expirations(expirations, today)
        if not expiration_candidates:
            return None, "No usable (future-dated) options expiration is listed for this symbol."

        strat, side = pick_lookup_strategy(uptrend, near_ema)
        is_etf = ticker_symbol in KNOWN_ETFS

        def _rank_key(result):
            return result["ap"] if result["ap"] is not None else result["pot"]

        best = None
        failure_reasons = []
        for cand_exp, cand_dte in expiration_candidates[:MAX_CANDIDATES_TO_EVALUATE]:
            result, reason = evaluate_expiration_candidate(tk, strat, side, spot, cand_exp, cand_dte, atr)
            if result:
                if best is None or _rank_key(result) > _rank_key(best):
                    best = result
            else:
                failure_reasons.append(f"{cand_exp} ({cand_dte}d): {reason}")

        if not best:
            reasons = "; ".join(failure_reasons[:3]) if failure_reasons else "no usable expirations"
            return None, f"Couldn't build a {strat} for this symbol right now — {reasons}"

        today_iso = today
        earnings_soon = False
        days_to_earnings = None
        try:
            edates = tk.get_earnings_dates(limit=4)
            if edates is not None and not edates.empty:
                for dt in edates.index:
                    d = dt.date() if hasattr(dt, "date") else dt
                    delta_days = (d - today_iso).days
                    if delta_days >= 0 and (days_to_earnings is None or delta_days < days_to_earnings):
                        days_to_earnings = delta_days
                    if 0 <= delta_days <= best["dte"]:
                        earnings_soon = True
        except Exception:
            pass

        news_sentiment, news_sentiment_label, news_headlines = fetch_news_and_sentiment(tk, ticker_symbol)

        if strat == "Calendar Spread":
            score = composite_score_calendar(ivr, best.get("iv"), best.get("backIv"))
        elif strat in ("Long Call", "Long Put"):
            score = composite_score_long_option(best["pot"], ivr, best["breakevenMovePct"])
        else:
            score = composite_score(best["ap"], best["pot"], ivr)
        vol_regime, iv_rv_ratio = classify_vol_regime(best["iv"], realized_vol)

        return {
            "sym": ticker_symbol,
            "strat": strat,
            "side": side,
            "isETF": is_etf,
            "spot": round(spot, 2),
            "spotPriceStale": spot_is_stale,
            "strike": best["strike"],
            "hedge": best["hedge"],
            "exp": best["exp"],
            "dte": best["dte"],
            "backExp": best.get("backExp"),
            "backDte": best.get("backDte"),
            "pot": best["pot"],
            "ap": best["ap"],
            "ivr": ivr,
            "dailyReturn": best["dailyReturn"],
            "roc": best["roc"],
            "score": score,
            "buy": uptrend,
            "sell": not uptrend,
            "ema": near_ema,
            "earningsSoon": earnings_soon,
            "daysToEarnings": days_to_earnings,
            "marginOfSafety": best["marginOfSafety"],
            "delta": best["delta"],
            "iv": best["iv"],
            "backIv": best.get("backIv"),
            "premium": best["premium"],
            "breakeven": best["breakeven"],
            "breakevenLow": best["breakevenLow"],
            "breakevenHigh": best["breakevenHigh"],
            "breakevenMovePct": best.get("breakevenMovePct"),
            "maxLoss": best["maxLoss"],
            "pcOI": best["pcOI"],
            "pcVol": best["pcVol"],
            "debitStrategy": best.get("debitStrategy", False),
            "potIsProfitProb": best.get("potIsProfitProb", False),
            "newsSentiment": news_sentiment,
            "newsSentimentLabel": news_sentiment_label,
            "newsHeadlines": news_headlines,
            "volRegime": vol_regime,
            "ivRvRatio": iv_rv_ratio,
            "realizedVol": realized_vol,
        }, None

    except Exception as e:
        return None, f"Unexpected error while building a trade for this symbol: {e}"


LOOKUP_OUTPUT_PATH = "lookup.json"


def run_lookup(raw_symbol):
    """
    Entry point for `--ticker SYMBOL`. Always writes lookup.json — even on
    failure — so the frontend polling for a result never waits forever on a
    file that never changes; a "status": "error" response is itself the
    answer. Never raises: an on-demand run failing loudly would still leave
    the workflow's commit step with nothing new to commit, which is the same
    "frontend polls forever" problem from the other direction.
    """
    ticker_symbol = re.sub(r"[^A-Za-z0-9.\-]", "", (raw_symbol or "")).strip().upper()
    output = {
        "requested_ticker": ticker_symbol,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "yfinance (unofficial, free, EOD)",
        "status": "error",
        "error": None,
        "trade": None,
        "equity": None,
    }

    if not ticker_symbol or not (1 <= len(ticker_symbol) <= 10):
        output["error"] = "Enter a valid ticker symbol (letters/numbers, up to 10 characters)."
        with open(LOOKUP_OUTPUT_PATH, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Lookup rejected: invalid symbol {raw_symbol!r}")
        return

    print(f"On-demand lookup: {ticker_symbol}")
    try:
        trade, trade_error = build_lookup_trade(ticker_symbol)
    except Exception as e:
        trade, trade_error = None, f"Unexpected error: {e}"

    try:
        equity = build_equity_snapshot(ticker_symbol, ticker_symbol in KNOWN_ETFS)
    except Exception as e:
        print(f"  lookup equity snapshot failed for {ticker_symbol}: {e}")
        equity = None

    output["trade"] = trade
    output["equity"] = equity

    if trade is None and equity is None:
        output["status"] = "error"
        output["error"] = trade_error or "Couldn't find usable data for this symbol — check it's a valid, actively-traded ticker."
        print(f"  lookup failed: {output['error']}")
    else:
        output["status"] = "ok"
        # Surface a soft warning even on partial success (e.g. equity data came
        # back but no usable options trade) rather than silently dropping it.
        if trade is None:
            output["error"] = trade_error

    with open(LOOKUP_OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote lookup.json for {ticker_symbol} (status={output['status']})")


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
        price, price_is_stale = resolve_current_price(tk, history, ticker_symbol, "equity")
        if math.isnan(price) or price <= 0:
            # same fallback as build_trade_for_ticker above — the most recent
            # bar is occasionally incomplete/NaN right after close. Confirmed
            # in production: every equity snapshot failed with "invalid
            # current price" in the exact same run where every options trade
            # for the same tickers succeeded, because that function already
            # had this fallback and this one didn't.
            if len(closes) >= 2:
                price = float(closes.iloc[-2])
                price_is_stale = True  # admittedly using an even older bar now
            if math.isnan(price) or price <= 0:
                print(f"  skip {ticker_symbol} (equity): current price is invalid/NaN (most recent close data looks broken)")
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
            "priceStale": price_is_stale,
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


def build_hedge_candidate():
    """
    Suggests one small tail-risk hedge — a single OTM QQQ put — to sit
    alongside the strategies above, all of which sell premium (short vol) and
    so share the same bad-day exposure: a sudden move that spikes IV and goes
    against several short strikes at once. Deliberately minimal: this returns
    one contract's terms and cost, not a recommended size or an auto-sized
    position — how many (if any) to actually buy is a sizing decision the
    frontend leaves to the person, same as every other number in this file.
    See the HEDGE_* constants above for the selection rules. Returns a dict,
    or None if QQQ's chain isn't usable right now — the frontend treats a
    missing "hedge" key as "not shown," not an error.
    """
    try:
        tk = yf.Ticker(HEDGE_TICKER)
        history = tk.history(period="5d")
        if history.empty:
            print(f"  skip hedge candidate: no price history for {HEDGE_TICKER}")
            return None
        spot = float(history["Close"].iloc[-1])
        if math.isnan(spot) or spot <= 0:
            print(f"  skip hedge candidate: {HEDGE_TICKER} spot price invalid")
            return None

        expirations = tk.options
        if not expirations:
            print(f"  skip hedge candidate: no options listed for {HEDGE_TICKER}")
            return None

        # Same closest-to-target-window ranking as rank_expirations() above,
        # but against HEDGE_TARGET_DTE_MIN/MAX rather than the strategy
        # picker's 7-45d window — kept as a local copy rather than
        # parameterizing the shared helper, so this doesn't risk changing
        # behavior for every existing strategy.
        today = datetime.now(timezone.utc).date()
        target_mid = (HEDGE_TARGET_DTE_MIN + HEDGE_TARGET_DTE_MAX) / 2
        in_window, outside_window = [], []
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if dte <= 0:
                continue
            diff = abs(dte - target_mid)
            bucket = in_window if HEDGE_TARGET_DTE_MIN <= dte <= HEDGE_TARGET_DTE_MAX else outside_window
            bucket.append((diff, exp_str, dte))
        in_window.sort(key=lambda x: x[0])
        outside_window.sort(key=lambda x: x[0])
        candidates = [(exp_str, dte) for _, exp_str, dte in (in_window + outside_window)]
        if not candidates:
            print(f"  skip hedge candidate: no usable {HEDGE_TICKER} expiration")
            return None

        # Fall back to the next-ranked expiration if a chain turns out
        # unusable (all-NaN IV on a bad data day) — same pattern the
        # strategy picker uses instead of giving up on the first failure.
        for exp_str, dte in candidates[:5]:
            try:
                chain = tk.option_chain(exp_str)
            except Exception as e:
                print(f"  hedge candidate: {HEDGE_TICKER} {exp_str} chain fetch failed: {e}")
                continue
            puts = chain.puts
            if puts is None or puts.empty:
                continue
            picked = pick_strike_by_delta(puts, spot, dte, HEDGE_TARGET_DELTA, "put")
            if not picked:
                continue
            row, delta = picked
            premium = mid_price(row)
            if premium <= 0:
                continue
            iv = safe_float(row.get("impliedVolatility"), default=0.0) * 100
            strike = safe_float(row.get("strike"), default=0.0)
            exp_label = (datetime.strptime(exp_str, "%Y-%m-%d").strftime("%b %-d") if sys.platform != "win32"
                         else datetime.strptime(exp_str, "%Y-%m-%d").strftime("%b %d").replace(" 0", " "))
            return {
                "symbol": HEDGE_TICKER,
                "spot": round(spot, 2),
                "strike": round(strike, 2),
                "otmPct": round((spot - strike) / spot * 100, 1),
                "exp": exp_label,
                "expDate": exp_str,
                "dte": dte,
                "delta": round(delta, 2),
                "iv": round(iv, 1),
                "premium": round(premium, 2),
                "costPerContract": round(premium * 100, 2),
            }

        print(f"  skip hedge candidate: no usable put found across {HEDGE_TICKER} candidates")
        return None
    except Exception as e:
        print(f"  skip hedge candidate: {e}")
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

    print(f"Fetching hedge candidate ({HEDGE_TICKER})...")
    hedge = build_hedge_candidate()
    if not hedge:
        # Not fatal — the frontend just hides the hedge section when this key
        # is absent, same as any other optional field in this file.
        print(f"  hedge candidate unavailable this run — omitting from {OUTPUT_PATH}")

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "yfinance (unofficial, free, EOD)",
        "trades": trades,
        "equities": equities,
        "hedge": hedge,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(trades)} trades, {len(equities)} equity snapshots, "
          f"and {'a' if hedge else 'no'} hedge candidate to {OUTPUT_PATH}")


if __name__ == "__main__":
    # `--ticker SYMBOL` runs the on-demand single-symbol lookup path instead
    # of the normal full-watchlist batch — see run_lookup() above. No flag
    # (the normal nightly/hourly invocation) behaves exactly as before.
    if "--ticker" in sys.argv:
        idx = sys.argv.index("--ticker")
        symbol_arg = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else ""
        run_lookup(symbol_arg)
    else:
        main()
