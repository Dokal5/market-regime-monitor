# Stock Momentum Tracker MVP

Local Python MVP that downloads six months of daily adjusted close and volume data with `yfinance`, calculates ticker-level momentum metrics, and writes clean CSV outputs.

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
- `outputs/index.html`
- `outputs/history/YYYY-MM-DD/ticker_momentum.csv`
- `outputs/history/YYYY-MM-DD/industry_momentum.csv`
- `outputs/history/industry_rotation_history.csv`

Open `outputs/index.html` in a browser to view the static dashboard. The dashboard has embedded data from the Python run and does not require a server, framework, or build step.

## Update tickers.csv

Edit `tickers.csv` with these columns:

```csv
ticker,company_name,industry_group
AAPL,Apple Inc.,Technology Hardware
```

Each row should include:

- `ticker`: stock ticker symbol, such as `AAPL`
- `company_name`: display name used in CSVs and the dashboard
- `industry_group`: peer group used for industry rankings and relative strength

Ticker rows are retained even when a ticker has missing or unavailable market data. In those cases, `data_points` is `0` and metric values are blank.

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

Industry output groups by `industry_group` and reports the mean of each ticker metric, plus ticker counts.

`max_drawdown_10d` is written as a negative percentage return. The `risk_warning` flag is true when the 10 day drawdown is worse than `-8%`, or when the latest adjusted close is more than 15% above the 20 day moving average.

The dashboard is a static HTML file with embedded data from the Python run. It does not require a framework, server, or build step.

Industry output also includes trend intelligence fields:

- `rotation_score`: improvement in industry rank over the last 5 historical trading dates, where positive means the industry moved up.
- `momentum_persistence`: consecutive historical trading dates the industry has stayed in the top 3 by average 10 day return.
- `momentum_acceleration`: current average 5 day return minus the previous snapshot's average 5 day return.
- `momentum_exhaustion_warning`: true when average 10 day return is still strong, but average 3 day return has weakened meaningfully or relative volume is below `0.8`.

## Historical snapshots

Each run saves a dated copy of the current ticker and industry outputs under `outputs/history/YYYY-MM-DD/`, using the latest market date in the downloaded data. The script rebuilds `outputs/history/industry_rotation_history.csv` from all dated snapshots.

`industry_rotation_history.csv` tracks each industry's daily rank by average 10 day return, average 10 day return, and confirmed signal percentage. The dashboard uses this file for the `Industry Rotation Trend` section. Trend tables populate once there are at least two historical snapshot dates.

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

This project is for research and tracking only. It is not investment advice, financial advice, or a recommendation to buy or sell securities.
