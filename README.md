# SignalHQ

Ready-made options trades and stock/ETF screens, evaluated by outlook, risk, reward, and
probability — built on end-of-day market data, refreshed automatically once a day via a
free GitHub Action.

## 1. Create the repo

1. On GitHub, click **New repository**. Name it whatever you like (e.g. `signalhq`).
   Keep it **Public** — GitHub Pages on a free account only serves public repos.
2. Don't initialize with a README (you already have one here).

## 2. Upload the files

Keep this exact folder structure when you upload:

```
your-repo/
├── index.html
├── manifest.json
├── service-worker.js
├── data.json
├── tickers.json
├── .nojekyll
├── README.md
├── icons/
│   ├── icon-192.png
│   ├── icon-512.png
│   ├── apple-touch-icon.png
│   └── favicon.png
├── scripts/
│   ├── fetch_data.py
│   └── requirements.txt
└── .github/
    └── workflows/
        └── update-data.yml
```

GitHub's drag-and-drop "Upload files" button doesn't always preserve subfolders. The
`.github/workflows/` path specifically needs to exist exactly as shown, or the Action
won't be picked up — the safest way to create nested folders through the web UI is
**Add file → Create new file**, then type the full path (e.g.
`.github/workflows/update-data.yml`) as the filename; GitHub creates the folders for
you. Do the same for `scripts/fetch_data.py`.

## 3. Turn on GitHub Pages

1. In the repo, go to **Settings → Pages**.
2. Under **Build and deployment**, set **Source** to `Deploy from a branch`.
3. Branch: `main`, folder: `/ (root)`. Save.
4. GitHub gives you a URL like `https://yourusername.github.io/signalhq/` — it takes
   a minute or two to go live after the first deploy.

## 4. Add it to your iPhone home screen

1. Open the GitHub Pages URL in **Safari** on your iPhone (must be Safari, not Chrome —
   iOS only allows installing home-screen apps from Safari).
2. Tap the **Share** icon (square with an arrow) in the toolbar.
3. Scroll down and tap **Add to Home Screen**.
4. Confirm the name (defaults to "SignalHQ") and tap **Add**.

It'll now launch full-screen from your home screen, no Safari address bar.

## 5. Turn on the nightly data refresh

The repo ships with `.github/workflows/update-data.yml`, which is already scheduled to
run weekday evenings after market close. Nothing extra to configure — but GitHub
disables scheduled (`cron`) workflows automatically if a repo goes 60 days with no
activity, so if data ever stops updating, check the **Actions** tab and re-enable it
there.

**To test it right away instead of waiting for the schedule:** go to your repo's
**Actions** tab → click **Update EOD options data** in the left sidebar → **Run workflow**
→ **Run workflow** again to confirm. It takes a minute or two; when it finishes, `data.json`
in your repo will have a real `generated_at` timestamp and real trade data instead of the
seed placeholder.

## Updating the site later

Any time you push new commits to `main` (or re-upload changed files through the GitHub
UI), GitHub Pages redeploys automatically within a minute or two. If you change
`index.html`, `manifest.json`, or any icon, bump `CACHE_NAME` in `service-worker.js`
(e.g. `signalhq-v1` → `signalhq-v2`) so phones that already installed the app pick
up the change instead of serving a stale cached copy. `data.json` is exempt from this —
the service worker always fetches it fresh over the network.

## How the data pipeline works

- `scripts/fetch_data.py` pulls prices and options chains from Yahoo Finance via the
  `yfinance` library (free, no API key — but unofficial and not guaranteed stable by
  Yahoo; it can break or get rate-limited without warning).
- It picks an expiration ~5–55 days out per ticker (weekly-and-up premium; not 0-1 DTE, since
  this is a nightly EOD batch tool and that short a window needs pricing closer to real-time
  than an overnight snapshot can responsibly give), selects strikes near a ~0.20 delta
  (a common informal "20-delta" premium-selling convention), and computes the fields the
  UI displays.
- `.github/workflows/update-data.yml` runs that script on GitHub's servers (which have
  full internet access, unlike this sandbox) on a schedule, and commits the resulting
  `data.json` back to the repo automatically.
- `index.html` fetches `data.json` on load instead of using any hardcoded data.

**Read the docstring at the top of `scripts/fetch_data.py` before trusting the numbers.**
Several fields are approximations, clearly labeled as such in the code:
- `iv` is real, straight from Yahoo's option chain.
- `delta` is computed here via Black-Scholes (0% dividend yield assumed) — a real
  broker's delta may differ slightly.
- `pot` (probability of touch) uses the rough trader heuristic `2 × |delta|`, not a
  rigorous calculation.
- `ivr` (IV Rank) is a realized-volatility-percentile proxy, not true IV rank (which
  needs a year of historical *option* IV data that isn't freely available).
- `score` (composite rating) is an illustrative weighted blend — adjust the weights in
  `composite_score()` in the script to match what you actually care about.
- `newsSentiment` / `newsSentimentLabel` come from VADER, a lexicon-based sentiment
  scorer, run only on headline text (not full articles) via a finance-vocabulary
  augmented lexicon (`FINANCE_LEXICON` in the script — VADER's default dictionary badly
  misreads financial language out of the box; e.g. it originally scored "shares tumble
  after lawsuit" as *positive*). This is pattern-matching on words, not an LLM reading
  the story for context — treat it as a rough gauge of recent press tone, not analysis.
- `daysToEarnings` / `earningsSoon` come from Yahoo's earnings calendar. Full
  earnings-call transcript analysis isn't included — that's realistically a paid-API
  feature (AlphaVantage/Finnhub premium tiers), not something free/EOD tooling can do.

To change which tickers get screened, edit `tickers.json` at the repo root — either
directly on GitHub, or use the **Tickers** button in the app itself, which lets you
add/remove symbols with a `+` button and generates the exact updated file content plus
a direct link to paste it into. The app can't commit to your repo on its own (that
would need a GitHub access token embedded in a public page, which isn't safe), so it's
a one-paste-then-commit flow rather than fully automatic — but no more hand-editing
Python syntax. `scripts/fetch_data.py` reads this file at the start of every run, with
a small built-in fallback list if `tickers.json` is ever missing or malformed, so a bad
edit there can't break the nightly job entirely.

## Where this goes next

- **Better probability/IV-rank math**: swap the approximations above for a real options
  analytics library or paid data provider (ORATS, EODHD) if the estimates aren't good
  enough for how you're using this.
- **More strategies per ticker**: right now each ticker gets exactly one strategy pick
  per run; the script could generate several candidates per ticker instead of one.
- **AI Trade Analysis**: "Copy Filters" currently copies your filter selections as plain
  text — it could be extended to build a fuller prompt including the visible results.
