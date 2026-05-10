from __future__ import annotations

import pandas as pd

from src.config import (
    DASHBOARD_OUTPUT_PATH,
    INDUSTRY_OUTPUT_PATH,
    INDUSTRY_ROTATION_HISTORY_PATH,
    TICKER_OUTPUT_PATH,
    TICKERS_PATH,
)
from src.dashboard import write_dashboard
from src.data_loader import clean_tickers, download_market_data
from src.history import (
    add_industry_trend_columns,
    build_industry_rotation_history,
    get_snapshot_date,
    write_daily_snapshot,
)
from src.industry import add_breadth_columns, build_industry_output
from src.io_utils import write_csv
from src.journal import write_journal
from src.metrics import build_ticker_output
from src.signals import add_signal_columns


def main() -> None:
    if not TICKERS_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {TICKERS_PATH}")

    tickers = clean_tickers(pd.read_csv(TICKERS_PATH))
    downloaded_data = download_market_data(tickers["ticker"].tolist())
    ticker_output = build_ticker_output(tickers, downloaded_data)
    industry_output = build_industry_output(ticker_output)
    ticker_output = add_signal_columns(ticker_output, industry_output)
    industry_output = add_breadth_columns(industry_output, ticker_output)
    snapshot_date = get_snapshot_date(ticker_output)
    industry_output = add_industry_trend_columns(industry_output, snapshot_date)

    write_csv(ticker_output, TICKER_OUTPUT_PATH)
    write_csv(industry_output, INDUSTRY_OUTPUT_PATH)
    snapshot_dir = write_daily_snapshot(ticker_output, industry_output, snapshot_date)
    rotation_history = build_industry_rotation_history()
    write_csv(rotation_history, INDUSTRY_ROTATION_HISTORY_PATH)
    write_dashboard(ticker_output, industry_output, rotation_history, DASHBOARD_OUTPUT_PATH)
    journal_path, latest_journal_path = write_journal(ticker_output, industry_output, snapshot_date)

    print(f"Wrote {TICKER_OUTPUT_PATH}")
    print(f"Wrote {INDUSTRY_OUTPUT_PATH}")
    print(f"Wrote {snapshot_dir}")
    print(f"Wrote {INDUSTRY_ROTATION_HISTORY_PATH}")
    print(f"Wrote {DASHBOARD_OUTPUT_PATH}")
    print(f"Wrote {journal_path}")
    print(f"Wrote {latest_journal_path}")


if __name__ == "__main__":
    main()
