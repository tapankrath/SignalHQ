"""
VantEdgeAI universe builder.

Screens Yahoo Finance (via yfinance's built-in screener) for large US-listed
stocks and writes the top slice of them to universe.json at the repo root.
scripts/fetch_data.py reads that file on every run and screens those names in
addition to the hand-pinned list in tickers.json.

Why this is a separate, slower-cadence job: market cap moves slowly, so the
universe only needs refreshing once a day — the hourly data run just reads the
cached file and never pays for the screen itself.

Selection rule (edit the constants below):
  1. US-listed (NASDAQ / NYSE), common stock, market cap above MIN_MARKET_CAP.
  2. Ranked by average daily dollar volume (price x 3-month average volume) —
     a proxy for how tradeable the options are — with market cap as tiebreak.
     Market cap alone would just give you the same 50 mega-caps forever; it's
     the size filter, not the ranking.
  3. The top UNIVERSE_SIZE survive. Two share classes of one company (GOOG /
     GOOGL, BRK-A / BRK-B) only take one slot.

If the screen fails or looks incomplete, the existing universe.json is left
untouched and this script exits non-zero, so a bad day never shrinks the list.

Like the rest of this repo, this rides on Yahoo's unofficial endpoints: free,
no API key, and liable to change or rate-limit without warning.
"""

import json
import os
import re
import sys
from datetime import datetime, timezone

try:
    import yfinance as yf
    from yfinance import EquityQuery
except ImportError:
    print("yfinance (>=1.0) not installed — run: pip install -r scripts/requirements.txt", file=sys.stderr)
    raise

# --- Configuration -----------------------------------------------------------
MIN_MARKET_CAP = 50_000_000_000   # $50B
UNIVERSE_SIZE = 50                # how many dynamic names to keep
EXCHANGES = ("NMS", "NYQ")        # Yahoo codes for NASDAQ and NYSE
PAGE_SIZE = 250                   # Yahoo's screener page cap
MAX_PAGES = 4                     # safety stop; >$50B is only a few hundred names
MIN_ACCEPTABLE_RESULTS = 20       # fewer than this = treat the screen as broken
OUTPUT_PATH = "universe.json"

# Same company, multiple listed share classes — keep only the more liquid one.
SHARE_CLASS_GROUPS = [
    {"GOOG", "GOOGL"},
    {"BRK-A", "BRK-B"},
    {"NWS", "NWSA"},
    {"FOX", "FOXA"},
]

SYMBOL_RE = re.compile(r"^[A-Z]{1,5}(-[A-Z])?$")


def fetch_quotes():
    """Pages through the screener, largest market cap first."""
    query = EquityQuery("and", [
        EquityQuery("gt", ["intradaymarketcap", MIN_MARKET_CAP]),
        EquityQuery("is-in", ["exchange", *EXCHANGES]),
    ])
    quotes = []
    for page in range(MAX_PAGES):
        result = yf.screen(
            query,
            offset=page * PAGE_SIZE,
            size=PAGE_SIZE,
            sortField="intradaymarketcap",
            sortAsc=False,
        )
        batch = (result or {}).get("quotes") or []
        quotes.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
    return quotes


def _num(v):
    try:
        f = float(v)
        return f if f == f and f > 0 else None  # drops NaN, zero and negatives
    except (TypeError, ValueError):
        return None


def clean(quotes):
    """Keeps plain US common stocks with a usable symbol and market cap."""
    rows, seen = [], set()
    for q in quotes:
        sym = (q.get("symbol") or "").strip().upper()
        if not SYMBOL_RE.match(sym) or sym in seen:
            continue
        qtype = (q.get("quoteType") or "EQUITY").upper()
        if qtype != "EQUITY":
            continue
        cap = _num(q.get("marketCap"))
        if cap is None or cap < MIN_MARKET_CAP:
            continue  # re-check client-side; the server filter uses intraday cap
        price = _num(q.get("regularMarketPrice"))
        avg_vol = _num(q.get("averageDailyVolume3Month"))
        dollar_vol = price * avg_vol if price and avg_vol else None
        seen.add(sym)
        rows.append({
            "symbol": sym,
            "name": q.get("shortName") or q.get("longName") or sym,
            "marketCap": cap,
            "avgDollarVolume": dollar_vol,
        })
    return rows


def rank(rows):
    """Dollar volume first (rows without it sort last), market cap as tiebreak."""
    return sorted(
        rows,
        key=lambda r: (r["avgDollarVolume"] is not None, r["avgDollarVolume"] or 0, r["marketCap"]),
        reverse=True,
    )


def drop_duplicate_share_classes(ranked):
    """`ranked` is best-first, so the first member of a group we meet is the keeper."""
    drop = set()
    for group in SHARE_CLASS_GROUPS:
        members = [r["symbol"] for r in ranked if r["symbol"] in group]
        drop.update(members[1:])
    return [r for r in ranked if r["symbol"] not in drop]


def main():
    try:
        quotes = fetch_quotes()
    except Exception as e:
        print(f"Screener call failed ({type(e).__name__}: {e}) — leaving existing {OUTPUT_PATH} untouched.", file=sys.stderr)
        sys.exit(1)

    rows = clean(quotes)
    print(f"Screener returned {len(quotes)} quotes; {len(rows)} usable above ${MIN_MARKET_CAP/1e9:.0f}B.")
    if len(rows) < MIN_ACCEPTABLE_RESULTS:
        print(f"Only {len(rows)} usable results (< {MIN_ACCEPTABLE_RESULTS}) — screen looks broken or partial; "
              f"leaving existing {OUTPUT_PATH} untouched.", file=sys.stderr)
        sys.exit(1)

    top = drop_duplicate_share_classes(rank(rows))[:UNIVERSE_SIZE]

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "yfinance screener (unofficial, free)",
        "criteria": {
            "min_market_cap": MIN_MARKET_CAP,
            "size": UNIVERSE_SIZE,
            "exchanges": list(EXCHANGES),
            "ranked_by": "avg daily dollar volume (3-month)",
            "candidates_above_min_cap": len(rows),
        },
        "tickers": [r["symbol"] for r in top],
        "details": top,
    }

    tmp_path = OUTPUT_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(output, f, indent=2)
    os.replace(tmp_path, OUTPUT_PATH)  # atomic — a crash mid-write can't leave a half-file

    print(f"Wrote {len(top)} tickers to {OUTPUT_PATH}:")
    print("  " + ", ".join(output["tickers"]))


if __name__ == "__main__":
    main()
