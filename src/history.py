from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from src.config import (
    HISTORY_DIR,
    INDUSTRY_TREND_COLUMNS,
    MOMENTUM_EXHAUSTION_STRONG_10D_RETURN_THRESHOLD,
    MOMENTUM_EXHAUSTION_WEAK_3D_RETURN_DIVISOR,
    RELATIVE_VOLUME_LOW_THRESHOLD,
    ROTATION_HISTORY_COLUMNS,
    TOP_INDUSTRY_RANK_THRESHOLD,
)
from src.industry import calculate_confirmed_by_industry
from src.io_utils import write_csv


def get_snapshot_date(ticker_output: pd.DataFrame) -> str:
    if "latest_date" in ticker_output.columns:
        latest_dates = ticker_output["latest_date"].dropna().astype(str)
        latest_dates = latest_dates[latest_dates != ""]
        if len(latest_dates):
            return max(latest_dates)

    return pd.Timestamp.now(tz="America/New_York").date().isoformat()


def rank_industry_frame(industry_output: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    history_columns = ["date", "industry_group", "industry_rank", "return_3d", "return_5d", "return_10d", "relative_volume"]
    if industry_output.empty:
        return pd.DataFrame(columns=history_columns)

    ranked = industry_output.copy()
    for column in ["return_3d", "return_5d", "return_10d", "relative_volume"]:
        if column in ranked.columns:
            ranked[column] = pd.to_numeric(ranked[column], errors="coerce")
        else:
            ranked[column] = math.nan

    ranked["industry_rank"] = ranked["return_10d"].rank(method="min", ascending=False, na_option="bottom")
    ranked.insert(0, "date", snapshot_date)
    return ranked[history_columns]


def load_industry_snapshot_history(before_date: str | None = None) -> pd.DataFrame:
    history_columns = ["date", "industry_group", "industry_rank", "return_3d", "return_5d", "return_10d", "relative_volume"]
    if not HISTORY_DIR.exists():
        return pd.DataFrame(columns=history_columns)

    snapshots = []
    for snapshot_dir in sorted(HISTORY_DIR.iterdir()):
        if not snapshot_dir.is_dir():
            continue
        if before_date is not None and snapshot_dir.name >= before_date:
            continue

        industry_path = snapshot_dir / "industry_momentum.csv"
        if not industry_path.exists():
            continue

        try:
            industry_snapshot = pd.read_csv(industry_path)
        except Exception as exc:
            print(f"Warning: could not read industry history snapshot {snapshot_dir.name}: {exc}")
            continue

        snapshots.append(rank_industry_frame(industry_snapshot, snapshot_dir.name))

    if not snapshots:
        return pd.DataFrame(columns=history_columns)

    return pd.concat(snapshots, ignore_index=True)


def calculate_momentum_persistence(history: pd.DataFrame, industry_group: str, current_date: str) -> int:
    industry_history = history[history["industry_group"] == industry_group].sort_values("date", ascending=False)
    count = 0
    for row in industry_history.to_dict("records"):
        if row["date"] > current_date:
            continue
        rank = row.get("industry_rank")
        if pd.isna(rank) or rank > TOP_INDUSTRY_RANK_THRESHOLD:
            break
        count += 1
    return count


def add_industry_trend_columns(industry_output: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    output_columns = list(industry_output.columns) + [
        column for column in INDUSTRY_TREND_COLUMNS if column not in industry_output.columns
    ]
    if industry_output.empty:
        return pd.DataFrame(columns=output_columns)

    industry = industry_output.copy()
    for column in ["return_3d", "return_5d", "return_10d", "relative_volume"]:
        industry[column] = pd.to_numeric(industry[column], errors="coerce")

    historical_industries = load_industry_snapshot_history(before_date=snapshot_date)
    current_industries = rank_industry_frame(industry, snapshot_date)
    combined_history = pd.concat([historical_industries, current_industries], ignore_index=True)
    combined_history["industry_rank"] = pd.to_numeric(combined_history["industry_rank"], errors="coerce")
    combined_history["return_5d"] = pd.to_numeric(combined_history["return_5d"], errors="coerce")
    dates = sorted(combined_history["date"].dropna().astype(str).unique())

    current_ranks = current_industries[["industry_group", "industry_rank"]].rename(
        columns={"industry_rank": "current_industry_rank"}
    )
    industry = industry.merge(current_ranks, on="industry_group", how="left")

    if dates:
        start_date = dates[max(0, len(dates) - 5)]
        start_ranks = combined_history[combined_history["date"] == start_date][
            ["industry_group", "industry_rank"]
        ].rename(columns={"industry_rank": "start_industry_rank"})
        industry = industry.merge(start_ranks, on="industry_group", how="left")
        industry["rotation_score"] = industry["start_industry_rank"] - industry["current_industry_rank"]
    else:
        industry["rotation_score"] = math.nan

    previous_dates = [date for date in dates if date < snapshot_date]
    if previous_dates:
        previous_date = previous_dates[-1]
        previous_returns = combined_history[combined_history["date"] == previous_date][
            ["industry_group", "return_5d"]
        ].rename(columns={"return_5d": "previous_return_5d"})
        industry = industry.merge(previous_returns, on="industry_group", how="left")
        industry["momentum_acceleration"] = industry["return_5d"] - industry["previous_return_5d"]
    else:
        industry["momentum_acceleration"] = math.nan

    industry["momentum_persistence"] = industry["industry_group"].apply(
        lambda industry_group: calculate_momentum_persistence(combined_history, industry_group, snapshot_date)
    )

    strong_10d_return = industry["return_10d"] > MOMENTUM_EXHAUSTION_STRONG_10D_RETURN_THRESHOLD
    weak_3d_return = (industry["return_3d"] < 0) | (
        industry["return_3d"] < (industry["return_10d"] / MOMENTUM_EXHAUSTION_WEAK_3D_RETURN_DIVISOR)
    )
    low_relative_volume = industry["relative_volume"] < RELATIVE_VOLUME_LOW_THRESHOLD
    industry["momentum_exhaustion_warning"] = (strong_10d_return & (weak_3d_return | low_relative_volume)).fillna(False)

    industry["rotation_score"] = pd.to_numeric(industry["rotation_score"], errors="coerce")
    industry["momentum_persistence"] = pd.to_numeric(industry["momentum_persistence"], errors="coerce").fillna(0).astype(int)
    industry = industry.drop(columns=["current_industry_rank", "start_industry_rank", "previous_return_5d"], errors="ignore")
    return industry[output_columns]


def build_rotation_snapshot(
    ticker_output: pd.DataFrame, industry_output: pd.DataFrame, snapshot_date: str
) -> pd.DataFrame:
    if industry_output.empty:
        return pd.DataFrame(columns=ROTATION_HISTORY_COLUMNS)

    rotation = industry_output[["industry_group", "return_10d"]].copy()
    rotation["average_10d_return"] = pd.to_numeric(rotation["return_10d"], errors="coerce")
    rotation = rotation.drop(columns=["return_10d"])
    rotation["industry_rank"] = rotation["average_10d_return"].rank(
        method="min", ascending=False, na_option="bottom"
    )

    confirmed = calculate_confirmed_by_industry(ticker_output)[["industry_group", "confirmed_signal_pct"]]
    rotation = rotation.merge(confirmed, on="industry_group", how="left")
    rotation.insert(0, "date", snapshot_date)
    rotation = rotation[ROTATION_HISTORY_COLUMNS].sort_values(["industry_rank", "industry_group"], na_position="last")
    return rotation


def write_daily_snapshot(ticker_output: pd.DataFrame, industry_output: pd.DataFrame, snapshot_date: str) -> Path:
    snapshot_dir = HISTORY_DIR / snapshot_date
    write_csv(ticker_output, snapshot_dir / "ticker_momentum.csv")
    write_csv(industry_output, snapshot_dir / "industry_momentum.csv")
    return snapshot_dir


def build_industry_rotation_history() -> pd.DataFrame:
    if not HISTORY_DIR.exists():
        return pd.DataFrame(columns=ROTATION_HISTORY_COLUMNS)

    snapshots = []
    for snapshot_dir in sorted(HISTORY_DIR.iterdir()):
        if not snapshot_dir.is_dir():
            continue

        ticker_path = snapshot_dir / "ticker_momentum.csv"
        industry_path = snapshot_dir / "industry_momentum.csv"
        if not ticker_path.exists() or not industry_path.exists():
            continue

        try:
            ticker_snapshot = pd.read_csv(ticker_path)
            industry_snapshot = pd.read_csv(industry_path)
        except Exception as exc:
            print(f"Warning: could not read history snapshot {snapshot_dir.name}: {exc}")
            continue

        snapshots.append(build_rotation_snapshot(ticker_snapshot, industry_snapshot, snapshot_dir.name))

    if not snapshots:
        return pd.DataFrame(columns=ROTATION_HISTORY_COLUMNS)

    history = pd.concat(snapshots, ignore_index=True)
    history["industry_rank"] = pd.to_numeric(history["industry_rank"], errors="coerce").astype("Int64")
    history["average_10d_return"] = pd.to_numeric(history["average_10d_return"], errors="coerce")
    history["confirmed_signal_pct"] = pd.to_numeric(history["confirmed_signal_pct"], errors="coerce")
    return history.sort_values(["date", "industry_rank", "industry_group"], na_position="last")
