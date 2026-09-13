"""
India equity screener — Stocks & ETFs only, no options.

Reuses the exact same equity-scoring functions as fetch_data.py (trend,
RSI, momentum, composite score) via direct import, so US and India results
are computed with identical methodology rather than two implementations
that could quietly drift apart over time. Only the ticker list, output
file, and market/currency tags differ.

Options screening is deliberately NOT included here — confirmed via direct
testing (a throwaway GitHub Actions run) that yfinance, the free/
unofficial data source this whole app runs on, returns zero option
expirations for Indian tickers, including the most heavily-traded index
options (Nifty 50, Bank Nifty). The data simply isn't available through
this source at all, so there's no options pipeline to build here — this
isn't a scoped-down version of one, it's a genuinely different, smaller
feature (equity screening only).
"""
import json
import sys
from datetime import datetime, timezone

from fetch_data import build_equity_snapshot

OUTPUT_PATH = "data_india.json"

# Popular large-cap NSE names to start with, per "start with some popular
# tickers, add more later." Spans banking, IT, energy, consumer, auto, and
# pharma — a reasonable starting set, not meant to be exhaustive. Add more
# .NS-suffixed symbols here as the list grows.
INDIA_TICKERS = [
    "RELIANCE.NS",    # Reliance Industries
    "TCS.NS",         # Tata Consultancy Services
    "HDFCBANK.NS",    # HDFC Bank
    "ICICIBANK.NS",   # ICICI Bank
    "INFY.NS",        # Infosys
    "SBIN.NS",        # State Bank of India
    "BHARTIARTL.NS",  # Bharti Airtel
    "ITC.NS",         # ITC
    "LT.NS",          # Larsen & Toubro
    "KOTAKBANK.NS",   # Kotak Mahindra Bank
    "HINDUNILVR.NS",  # Hindustan Unilever
    "AXISBANK.NS",    # Axis Bank
    "BAJFINANCE.NS",  # Bajaj Finance
    "MARUTI.NS",      # Maruti Suzuki
    "ASIANPAINT.NS",  # Asian Paints
    "SUNPHARMA.NS",   # Sun Pharmaceutical
    "TITAN.NS",       # Titan Company
    "WIPRO.NS",       # Wipro
    "NTPC.NS",        # NTPC
    "TATAMOTORS.NS",  # Tata Motors
]

# None of the starting list are ETFs. Add .NS-suffixed ETF symbols here
# as the list grows (e.g. "NIFTYBEES.NS" for the Nifty 50 ETF).
KNOWN_INDIA_ETFS = set()


def main():
    equities = []
    for ticker in INDIA_TICKERS:
        print(f"Fetching {ticker}...")
        equity = build_equity_snapshot(ticker, ticker in KNOWN_INDIA_ETFS)
        if equity:
            equities.append(equity)

    if not equities:
        print("No equities were built — leaving existing data_india.json untouched.", file=sys.stderr)
        sys.exit(1)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "yfinance (unofficial, free, EOD)",
        "market": "IN",
        "currency": "INR",
        "equities": equities,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(equities)} equity snapshots to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
