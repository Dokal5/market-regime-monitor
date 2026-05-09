from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.config import INPUT_COLUMNS, INTERNAL_COLUMNS, METRIC_COLUMNS
from src.data_loader import get_ticker_frame


def pct_return(series: pd.Series, days: int) -> float:
    if len(series) <= days:
        return math.nan

    latest = series.iloc[-1]
    previous = series.iloc[-days - 1]
    if pd.isna(latest) or pd.isna(previous) or previous == 0:
        return math.nan
    return (latest / previous) - 1


def max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return math.nan

    running_max = series.cummax()
    drawdowns = (series / running_max) - 1
    return drawdowns.min()


def finite_or_none(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value


def calculate_metrics(prices: pd.DataFrame) -> dict[str, Any]:
    if prices.empty:
        return {column: None for column in [*INTERNAL_COLUMNS, *METRIC_COLUMNS]}

    close = prices["adjusted_close"]
    volume = prices["Volume"]
    latest_price = close.iloc[-1] if not close.empty else math.nan
    latest_volume = volume.iloc[-1] if not volume.empty else math.nan
    avg_volume_20d = volume.tail(20).mean() if len(volume) >= 20 else math.nan

    metrics = {
        "latest_price": latest_price,
        "return_3d": pct_return(close, 3),
        "return_5d": pct_return(close, 5),
        "return_10d": pct_return(close, 10),
        "return_20d": pct_return(close, 20),
        "avg_volume_3d": volume.tail(3).mean() if len(volume) >= 3 else math.nan,
        "avg_volume_5d": volume.tail(5).mean() if len(volume) >= 5 else math.nan,
        "avg_volume_20d": avg_volume_20d,
        "relative_volume": latest_volume / avg_volume_20d if avg_volume_20d and not pd.isna(avg_volume_20d) else math.nan,
        "ma_5d": close.tail(5).mean() if len(close) >= 5 else math.nan,
        "ma_10d": close.tail(10).mean() if len(close) >= 10 else math.nan,
        "ma_20d": close.tail(20).mean() if len(close) >= 20 else math.nan,
        "max_drawdown_10d": max_drawdown(close.tail(10)) if len(close) >= 10 else math.nan,
        "up_days_10d": int((close.diff().tail(10) > 0).sum()) if len(close) >= 11 else math.nan,
    }
    return {key: finite_or_none(value) for key, value in metrics.items()}


def build_ticker_output(tickers: pd.DataFrame, downloaded_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for ticker_info in tickers.to_dict("records"):
        ticker = ticker_info["ticker"]
        prices = get_ticker_frame(downloaded_data, ticker)
        metrics = calculate_metrics(prices)

        rows.append(
            {
                **ticker_info,
                "latest_date": prices.index[-1].date().isoformat() if not prices.empty else None,
                "data_points": int(len(prices)),
                **metrics,
            }
        )

    output_columns = INPUT_COLUMNS + ["latest_date", "data_points"] + INTERNAL_COLUMNS + METRIC_COLUMNS
    return pd.DataFrame(rows, columns=output_columns)
