from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import DATA_QUALITY_COLUMNS, DATA_QUALITY_EXPORT_COLUMNS, LIMITED_HISTORY_MIN_DATA_POINTS


def latest_market_date(ticker_output: pd.DataFrame) -> str | None:
    if ticker_output.empty or "latest_date" not in ticker_output.columns:
        return None

    data_points = pd.to_numeric(ticker_output.get("data_points", 0), errors="coerce").fillna(0)
    latest_dates = ticker_output.loc[data_points > 0, "latest_date"].dropna().astype(str)
    if latest_dates.empty:
        return None
    return str(latest_dates.max())


def classify_data_quality(row: pd.Series, market_date: str | None) -> tuple[str, str]:
    data_points = pd.to_numeric(row.get("data_points"), errors="coerce")
    data_points_value = 0 if pd.isna(data_points) else int(data_points)
    latest_date_value = row.get("latest_date")
    latest_date = "" if pd.isna(latest_date_value) else str(latest_date_value)

    if data_points_value <= 0:
        return "missing", "未取得可用日線資料；此標的不參與有效動能比較。"

    if market_date and latest_date and latest_date < market_date:
        return "stale", f"最新資料日期 {latest_date} 早於本次市場日期 {market_date}。"

    if data_points_value < LIMITED_HISTORY_MIN_DATA_POINTS:
        return "limited_history", f"僅有 {data_points_value} 筆日線資料；長週期位置需保守解讀。"

    return "ok", "資料日期與本次市場日期一致。"


def add_data_quality_columns(ticker_output: pd.DataFrame) -> pd.DataFrame:
    base_columns = [column for column in ticker_output.columns if column not in DATA_QUALITY_COLUMNS]
    output_columns = base_columns + DATA_QUALITY_COLUMNS
    if ticker_output.empty:
        return pd.DataFrame(columns=output_columns)

    tickers = ticker_output.copy()
    market_date = latest_market_date(tickers)
    classifications = tickers.apply(lambda row: classify_data_quality(row, market_date), axis=1)
    tickers["data_status"] = [status for status, _ in classifications]
    tickers["data_quality_note"] = [note for _, note in classifications]
    return tickers[output_columns]


def build_data_quality_output(ticker_output: pd.DataFrame) -> pd.DataFrame:
    for column in DATA_QUALITY_EXPORT_COLUMNS:
        if column not in ticker_output.columns:
            ticker_output[column] = None
    return ticker_output[DATA_QUALITY_EXPORT_COLUMNS].copy()


def build_data_quality_summary(ticker_output: pd.DataFrame) -> dict[str, Any]:
    total_tickers = int(len(ticker_output))
    data_points = pd.to_numeric(ticker_output.get("data_points", 0), errors="coerce").fillna(0)
    status_counts = ticker_output.get("data_status", pd.Series(dtype=str)).fillna("missing").value_counts()
    tickers_with_data = int((data_points > 0).sum())
    success_rate = tickers_with_data / total_tickers if total_tickers else 0
    return {
        "data_source": "Yahoo Finance via yfinance",
        "latest_market_date": latest_market_date(ticker_output),
        "total_tickers": total_tickers,
        "tickers_with_data": tickers_with_data,
        "success_rate": success_rate,
        "ok_count": int(status_counts.get("ok", 0)),
        "missing_count": int(status_counts.get("missing", 0)),
        "stale_count": int(status_counts.get("stale", 0)),
        "limited_history_count": int(status_counts.get("limited_history", 0)),
    }
