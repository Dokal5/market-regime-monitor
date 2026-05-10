from __future__ import annotations

import pandas as pd

from src.config import (
    CONFIRMED_MOMENTUM_MIN_RETURN_10D,
    CONFIRMED_MOMENTUM_MIN_RETURN_5D,
    CONFIRMED_MOMENTUM_MIN_UP_DAYS_10D,
    EARLY_MOMENTUM_10D_RETURN_DIVISOR,
    EARLY_MOMENTUM_MIN_RETURN_3D,
    EARLY_MOMENTUM_MIN_RETURN_5D,
    INPUT_COLUMNS,
    METRIC_COLUMNS,
    OPTIONAL_TICKER_COLUMNS,
    PRICE_POSITION_COLUMNS,
    RISK_DRAWDOWN_THRESHOLD,
    RISK_EXTENSION_MULTIPLE,
    SIGNAL_COLUMNS,
    STRONG_MOMENTUM_RELATIVE_VOLUME_THRESHOLD,
)


def add_signal_columns(ticker_output: pd.DataFrame, industry_output: pd.DataFrame) -> pd.DataFrame:
    if ticker_output.empty:
        return pd.DataFrame(
            columns=INPUT_COLUMNS
            + ["latest_date", "data_points"]
            + METRIC_COLUMNS
            + SIGNAL_COLUMNS
            + OPTIONAL_TICKER_COLUMNS
            + PRICE_POSITION_COLUMNS
        )

    signals = ticker_output.copy()
    industry_returns = industry_output[["industry_group", "return_10d"]].rename(
        columns={"return_10d": "industry_return_10d"}
    )
    signals = signals.merge(industry_returns, on="industry_group", how="left")

    numeric_columns = [
        "return_3d",
        "return_5d",
        "return_10d",
        "relative_volume",
        "ma_5d",
        "ma_10d",
        "ma_20d",
        "max_drawdown_10d",
        "up_days_10d",
        "latest_price",
        "industry_return_10d",
    ]
    for column in numeric_columns:
        signals[column] = pd.to_numeric(signals[column], errors="coerce")

    signals["relative_strength_vs_industry"] = signals["return_10d"] - signals["industry_return_10d"]

    signals["early_momentum_signal"] = (
        (signals["return_3d"] > EARLY_MOMENTUM_MIN_RETURN_3D)
        & (signals["return_5d"] > EARLY_MOMENTUM_MIN_RETURN_5D)
        & (signals["return_5d"] > (signals["return_10d"] / EARLY_MOMENTUM_10D_RETURN_DIVISOR))
    ).fillna(False)

    signals["confirmed_momentum_signal"] = (
        (signals["return_5d"] > CONFIRMED_MOMENTUM_MIN_RETURN_5D)
        & (signals["return_10d"] > CONFIRMED_MOMENTUM_MIN_RETURN_10D)
        & (signals["ma_5d"] > signals["ma_10d"])
        & (signals["up_days_10d"] >= CONFIRMED_MOMENTUM_MIN_UP_DAYS_10D)
    ).fillna(False)

    signals["strong_momentum_signal"] = (
        signals["confirmed_momentum_signal"]
        & (signals["return_10d"] > signals["industry_return_10d"])
        & (signals["relative_volume"] > STRONG_MOMENTUM_RELATIVE_VOLUME_THRESHOLD)
    ).fillna(False)

    signals["risk_warning"] = (
        (signals["max_drawdown_10d"] < RISK_DRAWDOWN_THRESHOLD)
        | (signals["latest_price"] > signals["ma_20d"] * RISK_EXTENSION_MULTIPLE)
    ).fillna(False)

    output_columns = (
        INPUT_COLUMNS
        + ["latest_date", "data_points"]
        + METRIC_COLUMNS
        + SIGNAL_COLUMNS
        + OPTIONAL_TICKER_COLUMNS
        + PRICE_POSITION_COLUMNS
    )
    return signals[output_columns]
