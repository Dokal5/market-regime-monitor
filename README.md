# Stock Momentum Tracker MVP

Local Python MVP that downloads one year of daily adjusted close and volume data with `yfinance`, calculates ticker-level momentum metrics, and writes clean CSV outputs.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The script creates:

- `outputs/ticker_momentum.csv`
- `outputs/industry_momentum.csv`
- `outputs/watchlist_alerts.csv`
- `outputs/data_quality.csv`
- `outputs/update_health.csv`
- `outputs/index.html`
- `outputs/history/YYYY-MM-DD/ticker_momentum.csv`
- `outputs/history/YYYY-MM-DD/industry_momentum.csv`
- `outputs/history/industry_rotation_history.csv`
- `outputs/journal/YYYY-MM-DD.md`
- `outputs/journal/latest.md`

Open `outputs/index.html` in a browser to view the static dashboard. The dashboard has embedded data from the Python run and does not require a server, framework, or build step.

## Check local and online sync status

Run this command to verify whether the local repository, GitHub `origin/main`, GitHub raw output files, and GitHub Pages dashboard are aligned:

```bash
python scripts/status_check.py
```

The check prints stable key/value lines such as `LOCAL_SYNCED`, `RAW_SYNCED`, `PAGES_SYNCED`, `LOCAL_LATEST_MARKET_DATE`, `ONLINE_LATEST_MARKET_DATE`, and `ACTION`.

Exit codes:

- `0`: local outputs, GitHub raw outputs, and GitHub Pages outputs are aligned
- `1`: the check completed, but a mismatch was found
- `2`: the check could not complete because git, network, or required local files failed

When `LOCAL_SYNCED=false` and `ACTION=git pull --ff-only origin main`, sync the local project with:

```bash
git pull --ff-only origin main
```

## Update tickers.csv

Edit `tickers.csv` with these columns:

```csv
ticker,company_name,industry_group,leader_type,industry_quality_score
AAPL,Apple Inc.,Technology Hardware,non_leader,3
```

Each row should include:

- `ticker`: stock ticker symbol, such as `AAPL`
- `company_name`: display name used in CSVs and the dashboard
- `industry_group`: peer group used for industry rankings and relative strength
- `leader_type`: optional deterministic research metadata; allowed values are `core_leader`, `challenger`, `infrastructure_leader`, `emerging_leader`, `specialist`, and `non_leader`
- `industry_quality_score`: optional deterministic research metadata from `1` to `5`

Ticker rows are retained even when a ticker has missing or unavailable market data. In those cases, `data_points` is `0` and metric values are blank.

If `leader_type` or `industry_quality_score` is missing, the loader fills `non_leader` and `3`.

## Update watchlist.csv

Edit `watchlist.csv` whenever the tickers you care about change. This file is the source of truth for watchlist alerts in the local run and the GitHub Actions daily run. It is intentionally not a position file, so it does not need share count, cost basis, or portfolio weight.

```csv
ticker,theme,notes
VRT,AI Infrastructure,watch for weakening momentum
```

Only `ticker` is required. `theme` and `notes` are optional context fields for your own tracking.

Each run writes `outputs/watchlist_alerts.csv` and adds a `追蹤名單轉換提醒` card to the Daily Brief. Alert levels are:

- `red`: open-before-market review; the ticker has risk warning, weak 5/10 day momentum versus industry, or a falling-knife state
- `orange`: watch for transition; the ticker is starting to lag its industry
- `yellow`: monitor; one caution flag exists but the full weakness rule is not triggered
- `green`: no current momentum warning
- `unknown`: the ticker is not in `tickers.csv` or lacks generated market data

To keep the online dashboard synced with the current watchlist, update `watchlist.csv`, commit it, and push it to `main`. A GitHub Actions run starts when `watchlist.csv` changes on `main`, regenerates the outputs, and deploys the updated dashboard. The scheduled daily run also reads the latest watchlist file and commits refreshed outputs under `outputs/`.

Before editing locally, run `git pull --ff-only origin main` so your local files match the latest GitHub Actions commit. After the next triggered or scheduled run, `python scripts/status_check.py` checks `watchlist_alerts.csv` along with the dashboard, ticker, industry, update health, and latest journal outputs.

## Data quality

Market data comes from Yahoo Finance through `yfinance`. This is useful for research, watchlists, and market observation, but it is not an institutional-grade data feed.

Each run writes `outputs/data_quality.csv` and appends these fields to `outputs/ticker_momentum.csv`:

- `data_status`: `ok`, `missing`, `stale`, or `limited_history`
- `data_quality_note`: short explanation for the status

`missing` means no usable daily data was downloaded. `stale` means a ticker has data, but its latest date is older than the latest market date found in the current run. `limited_history` means fewer than 60 daily bars are available, so long-term price position metrics should be read conservatively.

## Update health

Each run writes `outputs/update_health.csv` with one row describing the operational health of the latest update. This is not a trading or investment signal; it only checks whether the output was generated recently and whether the data feed looked complete.

Key fields:

- `run_context`: `local` or `github_actions`
- `github_run_url`: link to the GitHub Actions run when available
- `latest_market_date`: latest market date found in the generated ticker data
- `market_data_age_days`: number of days between the New York generation date and `latest_market_date`
- `success_rate`: share of tickers with usable market data
- `update_health_status`: `healthy`, `warning`, or `unknown`
- `update_health_note`: concise reason for the status

`warning` is used when market data is more than 3 days old, when missing or stale ticker data appears, or when data success rate falls below 98%. `limited_history_count` is displayed as context but does not by itself make the update unhealthy.

## Metrics

Ticker output includes:

- 3, 5, 10, and 20 day returns
- 3, 5, and 20 day average volume
- Relative volume, calculated as latest volume divided by 20 day average volume
- 5, 10, and 20 day moving averages
- Max drawdown over the last 10 trading days
- Up days in the last 10 trading days
- Early, confirmed, and strong momentum signal flags
- Risk warning flag
- Relative strength versus industry, calculated as ticker 10 day return minus industry average 10 day return
- Distance from the 20 day moving average
- Distance from the 52 week high
- Position in the 52 week range

Industry output groups by `industry_group` and reports the mean of each ticker metric, plus ticker counts.

`max_drawdown_10d` is written as a negative percentage return. The `risk_warning` flag is true when the 10 day drawdown is worse than `-8%`, or when the latest adjusted close is more than 15% above the 20 day moving average.

The dashboard is a static HTML file with embedded data from the Python run. It does not require a framework, server, or build step.

The dashboard also includes a Daily Brief section near the top of the page. The brief condenses existing generated data into five deterministic cards covering data status, market theme, rotation changes, research candidates, and risk focus. It does not add new indicators, call any AI API, or provide investment advice.

Industry output also includes breadth fields:

- `positive_5d_pct`: share of industry tickers with `return_5d > 0`
- `positive_10d_pct`: share of industry tickers with `return_10d > 0`
- `confirmed_signal_pct`: share of industry tickers with confirmed momentum
- `strong_signal_pct`: share of industry tickers with strong momentum
- `high_relative_volume_pct`: share of industry tickers with `relative_volume > 1.2`
- `breadth_score`: weighted breadth score using 20% positive 5 day, 25% positive 10 day, 25% confirmed, 20% strong, and 10% high relative volume

Industry output also includes trend intelligence fields:

- `rotation_score`: improvement in industry rank over the last 5 historical trading dates, where positive means the industry moved up.
- `momentum_persistence`: consecutive historical trading dates the industry has stayed in the top 3 by average 10 day return.
- `momentum_acceleration`: current average 5 day return minus the previous snapshot's average 5 day return.
- `momentum_exhaustion_warning`: true when average 10 day return is still strong, but average 3 day return has weakened meaningfully or relative volume is below `0.8`.

Industry output also includes `industry_regime`, a deterministic trend-state classification used by the Leader Accumulation Filter. Possible values are `momentum_leader`, `early_recovery`, `neutral`, and `weak`.

Phase 1 of the causality layer keeps observable signals separate from possible explanations:

- `industry_risk_flag`: preserves risk context separately from trend state. Possible values are `none`, `momentum_exhaustion`, `narrow_leadership`, `late_cycle_momentum`, and `data_limited`.
- `rotation_type`: a deterministic classification aid from `INDUSTRY_ROTATION_TYPE_MAP`, defaulting to `unclear`; it does not infer complex causes from price and volume alone.
- `causal_hypothesis`: a review hypothesis derived from `rotation_type`, not a proven cause.
- `evidence_status`: indicates whether the current system's observable market evidence is `observed`, `inferred`, `needs_review`, or `unsupported`. `unsupported` is reserved for future rules.

Observable price, volume, relative strength, breadth, rotation, and trend signals are evidence only. They do not prove deeper causes such as earnings revisions, policy support, rates, risk appetite, ETF flows, or short covering.

## Leader Accumulation Filter

The Leader Accumulation Filter is deterministic research support. It does not call any AI API and is not investment advice.

Ticker output appends these fields:

- `leader_type`
- `industry_quality_score`
- `industry_regime`
- `industry_risk_flag`
- `rotation_type`
- `causal_hypothesis`
- `evidence_status`
- `distance_from_20d_ma`
- `distance_from_52w_high`
- `position_in_52w_range`
- `short_term_price_zone`
- `long_term_price_zone`
- `price_zone`
- `current_state`
- `watch_status`

`price_zone`, `short_term_price_zone`, and `long_term_price_zone` are technical price position fields. They are not valuation fields, and the system does not make valuation claims without valuation data.

Meaningful `research_candidate` output requires curated `leader_type` and `industry_quality_score` metadata before the filter becomes useful. Existing tickers default to `non_leader` and score `3`, so `research_candidate` can be empty until that metadata is maintained.

The filter only evaluates higher-quality leader metadata when the industry regime is `momentum_leader` or `early_recovery`. Risk handling has priority in `watch_status`, so unstable or negatively flagged names are not hidden behind industry eligibility.

## Project structure

`main.py` is the orchestration entry point. The implementation lives in `src/`:

- `config.py`: paths, output names, lookback settings, and signal thresholds
- `data_loader.py`: ticker input cleanup and `yfinance` downloads
- `metrics.py`: ticker-level return, volume, moving average, drawdown, and up-day calculations
- `signals.py`: ticker momentum signals, risk warning, and relative strength
- `industry.py`: industry-level aggregation and confirmed signal percentages
- `history.py`: dated snapshots, rotation history, and trend intelligence fields
- `leader_filter.py`: industry regimes, price zones, current state, and watch status
- `data_quality.py`: source transparency, missing/stale checks, and data quality CSV output
- `update_health.py`: run context, data freshness, and update health CSV output
- `daily_brief.py`: deterministic daily summary shared by the dashboard and journal
- `journal.py`: deterministic daily Markdown journal generation
- `dashboard.py`: static HTML dashboard generation
- `io_utils.py`: shared CSV writing helpers

## Historical snapshots

Each run saves a dated copy of the current ticker and industry outputs under `outputs/history/YYYY-MM-DD/`, using the latest market date in the downloaded data. If a snapshot directory for that date already exists, it is left unchanged. The script rebuilds `outputs/history/industry_rotation_history.csv` from all dated snapshots.

`industry_rotation_history.csv` tracks each industry's daily rank by average 10 day return, average 10 day return, and confirmed signal percentage. The dashboard uses this file for the `Industry Rotation Trend` section. Trend tables populate once there are at least two historical snapshot dates.

## Daily journal

Each run writes a deterministic Markdown journal under `outputs/journal/`, using the latest market date from the generated ticker data:

- `outputs/journal/YYYY-MM-DD.md`: dated journal for that market date
- `outputs/journal/latest.md`: copy of the latest generated journal

The journal summarizes the current snapshot, leading industries, breadth leaders, relative strength stocks, early momentum candidates, and risk warnings. Its system interpretation is rule based, uses only generated output data, and does not call any AI API.

The journal also includes a `Daily Brief` section before `Market Snapshot`. This section matches the dashboard brief and is a concise deterministic summary, not a recommendation.

## GitHub Pages

This repo includes GitHub Actions workflows to publish `outputs/` as a static GitHub Pages site.

To enable Pages:

1. Push the repository to GitHub.
2. Open the repository settings.
3. Go to `Pages`.
4. Under `Build and deployment`, set `Source` to `GitHub Actions`.
5. Run the `Deploy GitHub Pages` workflow manually, or wait for the next update workflow.

The Pages site serves `outputs/index.html` as the dashboard.

## Scheduled updates

`.github/workflows/daily-update.yml` runs every weekday at `21:30 UTC`, which is after the regular US market close in both EST and EDT. The job sets `TZ=America/New_York`, installs Python 3.11 and `requirements.txt`, runs `python main.py`, commits changed files under `outputs/`, and deploys `outputs/` to GitHub Pages.

The workflow also supports manual runs with `workflow_dispatch`. It uses no API keys and does not require repository secrets.

## Disclaimer

This project is for research and tracking only. It is not investment advice, financial advice, or a securities transaction recommendation.

## Leader metadata 維護流程（建議）

為了讓 `research_candidate` 從「可顯示」進一步成為「可決策參考」，建議把 `tickers.csv` 的 `leader_type` 與 `industry_quality_score` 維護拆成固定流程：

1. 建立核心觀察池（每個重點產業先 3-8 檔），優先覆蓋半導體、資安、雲端軟體、AI 基礎設施、太空/衛星、能源儲存。
2. 每週固定檢查一次：
   - 公司是否仍屬於原產業敘事與競爭位置
   - `leader_type` 是否需要升降級（例如 `challenger` ↔ `core_leader`）
   - `industry_quality_score` 是否需要調整（1-5）
3. 每次財報季後做一次深度校準，避免 metadata 與市場結構脫節。
4. 若無把握，先維持保守分級（`specialist` / `challenger`），避免過度標記為 `core_leader`。

實務上可先把「近 20 日內曾出現在 Daily Brief 強勢動能或早期動能清單」的標的納入優先維護，逐步擴展到全 universe。
