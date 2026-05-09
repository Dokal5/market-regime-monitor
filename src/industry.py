from __future__ import annotations

import pandas as pd

from src.config import (
    BREADTH_COLUMNS,
    BREADTH_CONFIRMED_SIGNAL_WEIGHT,
    BREADTH_HIGH_RELATIVE_VOLUME_WEIGHT,
    BREADTH_POSITIVE_10D_WEIGHT,
    BREADTH_POSITIVE_5D_WEIGHT,
    BREADTH_STRONG_SIGNAL_WEIGHT,
    HIGH_RELATIVE_VOLUME_THRESHOLD,
    METRIC_COLUMNS,
)


def build_industry_output(ticker_output: pd.DataFrame) -> pd.DataFrame:
    if ticker_output.empty:
        return pd.DataFrame(columns=["industry_group", "ticker_count", "tickers_with_data", *METRIC_COLUMNS])

    numeric_columns = ["data_points", *METRIC_COLUMNS]
    numeric_tickers = ticker_output.copy()
    for column in numeric_columns:
        numeric_tickers[column] = pd.to_numeric(numeric_tickers[column], errors="coerce")

    grouped = numeric_tickers.groupby("industry_group", dropna=False)
    industry = grouped[METRIC_COLUMNS].mean(numeric_only=True).reset_index()
    industry.insert(1, "ticker_count", grouped["ticker"].count().values)
    industry.insert(2, "tickers_with_data", grouped["data_points"].apply(lambda values: int((values > 0).sum())).values)
    return industry


def add_breadth_columns(industry_output: pd.DataFrame, ticker_output: pd.DataFrame) -> pd.DataFrame:
    output_columns = list(industry_output.columns) + [
        column for column in BREADTH_COLUMNS if column not in industry_output.columns
    ]
    if industry_output.empty:
        return pd.DataFrame(columns=output_columns)

    tickers = ticker_output.copy()
    for column in ["return_5d", "return_10d", "relative_volume"]:
        tickers[column] = pd.to_numeric(tickers[column], errors="coerce")

    for column in ["confirmed_momentum_signal", "strong_momentum_signal"]:
        tickers[column] = tickers.get(column, False)
        tickers[column] = tickers[column].fillna(False).astype(bool)

    breadth = (
        tickers.groupby("industry_group", dropna=False)
        .agg(
            positive_5d_pct=("return_5d", lambda values: float((values > 0).sum() / len(values)) if len(values) else 0),
            positive_10d_pct=("return_10d", lambda values: float((values > 0).sum() / len(values)) if len(values) else 0),
            confirmed_signal_pct=(
                "confirmed_momentum_signal",
                lambda values: float(values.sum() / len(values)) if len(values) else 0,
            ),
            strong_signal_pct=(
                "strong_momentum_signal",
                lambda values: float(values.sum() / len(values)) if len(values) else 0,
            ),
            high_relative_volume_pct=(
                "relative_volume",
                lambda values: float((values > HIGH_RELATIVE_VOLUME_THRESHOLD).sum() / len(values)) if len(values) else 0,
            ),
        )
        .reset_index()
    )
    breadth["breadth_score"] = (
        BREADTH_POSITIVE_5D_WEIGHT * breadth["positive_5d_pct"]
        + BREADTH_POSITIVE_10D_WEIGHT * breadth["positive_10d_pct"]
        + BREADTH_CONFIRMED_SIGNAL_WEIGHT * breadth["confirmed_signal_pct"]
        + BREADTH_STRONG_SIGNAL_WEIGHT * breadth["strong_signal_pct"]
        + BREADTH_HIGH_RELATIVE_VOLUME_WEIGHT * breadth["high_relative_volume_pct"]
    )

    industry = industry_output.merge(breadth, on="industry_group", how="left")
    for column in BREADTH_COLUMNS:
        industry[column] = pd.to_numeric(industry[column], errors="coerce")
    return industry[output_columns]


def calculate_confirmed_by_industry(ticker_output: pd.DataFrame) -> pd.DataFrame:
    columns = ["industry_group", "ticker_count", "tickers_with_data", "confirmed_count", "confirmed_signal_pct"]
    if ticker_output.empty:
        return pd.DataFrame(columns=columns)

    tickers = ticker_output.copy()
    if "data_points" not in tickers.columns:
        tickers["data_points"] = 0
    tickers["data_points"] = pd.to_numeric(tickers["data_points"], errors="coerce").fillna(0)
    tickers["confirmed_momentum_signal"] = tickers.get("confirmed_momentum_signal", False)
    tickers["confirmed_momentum_signal"] = tickers["confirmed_momentum_signal"].fillna(False).astype(bool)

    confirmed = (
        tickers.groupby("industry_group", dropna=False)
        .agg(
            ticker_count=("ticker", "count"),
            tickers_with_data=("data_points", lambda values: int((values > 0).sum())),
            confirmed_count=("confirmed_momentum_signal", "sum"),
        )
        .reset_index()
    )
    confirmed["confirmed_signal_pct"] = confirmed["confirmed_count"] / confirmed["ticker_count"]
    return confirmed[columns]
