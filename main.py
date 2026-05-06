from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


BASE_DIR = Path(__file__).resolve().parent
TICKERS_PATH = BASE_DIR / "tickers.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
TICKER_OUTPUT_PATH = OUTPUT_DIR / "ticker_momentum.csv"
INDUSTRY_OUTPUT_PATH = OUTPUT_DIR / "industry_momentum.csv"
DASHBOARD_OUTPUT_PATH = OUTPUT_DIR / "index.html"
HISTORY_DIR = OUTPUT_DIR / "history"
INDUSTRY_ROTATION_HISTORY_PATH = HISTORY_DIR / "industry_rotation_history.csv"

INPUT_COLUMNS = ["ticker", "company_name", "industry_group"]
METRIC_COLUMNS = [
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "avg_volume_3d",
    "avg_volume_5d",
    "avg_volume_20d",
    "relative_volume",
    "ma_5d",
    "ma_10d",
    "ma_20d",
    "max_drawdown_10d",
    "up_days_10d",
]
INTERNAL_COLUMNS = ["latest_price"]
SIGNAL_COLUMNS = [
    "early_momentum_signal",
    "confirmed_momentum_signal",
    "strong_momentum_signal",
    "risk_warning",
    "relative_strength_vs_industry",
]
ROTATION_HISTORY_COLUMNS = [
    "date",
    "industry_group",
    "industry_rank",
    "average_10d_return",
    "confirmed_signal_pct",
]
INDUSTRY_TREND_COLUMNS = [
    "rotation_score",
    "momentum_persistence",
    "momentum_acceleration",
    "momentum_exhaustion_warning",
]


def clean_tickers(raw_tickers: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in INPUT_COLUMNS if column not in raw_tickers.columns]
    if missing_columns:
        raise ValueError(f"tickers.csv is missing required columns: {', '.join(missing_columns)}")

    tickers = raw_tickers[INPUT_COLUMNS].copy()
    tickers["ticker"] = tickers["ticker"].astype(str).str.strip().str.upper()
    tickers["company_name"] = tickers["company_name"].fillna("").astype(str).str.strip()
    tickers["industry_group"] = tickers["industry_group"].fillna("Unknown").astype(str).str.strip()
    tickers = tickers[tickers["ticker"] != ""].drop_duplicates(subset=["ticker"])
    return tickers.reset_index(drop=True)


def download_market_data(ticker_symbols: list[str]) -> dict[str, pd.DataFrame]:
    market_data = {}
    if not ticker_symbols:
        return market_data

    for ticker in ticker_symbols:
        try:
            market_data[ticker] = yf.download(
                tickers=ticker,
                period="6mo",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            print(f"Warning: could not download data for {ticker}: {exc}")
            market_data[ticker] = pd.DataFrame()

    return market_data


def get_ticker_frame(downloaded_data: dict[str, pd.DataFrame], ticker: str) -> pd.DataFrame:
    ticker_data = downloaded_data.get(ticker, pd.DataFrame()).copy()
    if ticker_data.empty:
        return pd.DataFrame()

    if isinstance(ticker_data.columns, pd.MultiIndex):
        if ticker in ticker_data.columns.get_level_values(0):
            ticker_data = ticker_data[ticker].copy()
        elif ticker in ticker_data.columns.get_level_values(1):
            ticker_data = ticker_data.xs(ticker, axis=1, level=1, drop_level=True).copy()
        else:
            ticker_data.columns = ticker_data.columns.get_level_values(-1)

    close_column = "Adj Close" if "Adj Close" in ticker_data.columns else "Close"
    required_columns = [close_column, "Volume"]
    if any(column not in ticker_data.columns for column in required_columns):
        return pd.DataFrame()

    prices = ticker_data[required_columns].rename(columns={close_column: "adjusted_close"})
    prices = prices.dropna(subset=["adjusted_close", "Volume"])
    prices = prices[prices["Volume"].notna()]
    return prices.sort_index()


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


def add_signal_columns(ticker_output: pd.DataFrame, industry_output: pd.DataFrame) -> pd.DataFrame:
    if ticker_output.empty:
        return pd.DataFrame(
            columns=INPUT_COLUMNS + ["latest_date", "data_points"] + METRIC_COLUMNS + SIGNAL_COLUMNS
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
        (signals["return_3d"] > 0)
        & (signals["return_5d"] > 0)
        & (signals["return_5d"] > (signals["return_10d"] / 2))
    ).fillna(False)

    signals["confirmed_momentum_signal"] = (
        (signals["return_5d"] > 0)
        & (signals["return_10d"] > 0)
        & (signals["ma_5d"] > signals["ma_10d"])
        & (signals["up_days_10d"] >= 6)
    ).fillna(False)

    signals["strong_momentum_signal"] = (
        signals["confirmed_momentum_signal"]
        & (signals["return_10d"] > signals["industry_return_10d"])
        & (signals["relative_volume"] > 1.2)
    ).fillna(False)

    signals["risk_warning"] = (
        (signals["max_drawdown_10d"] < -0.08)
        | (signals["latest_price"] > signals["ma_20d"] * 1.15)
    ).fillna(False)

    output_columns = INPUT_COLUMNS + ["latest_date", "data_points"] + METRIC_COLUMNS + SIGNAL_COLUMNS
    return signals[output_columns]


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


def write_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, float_format="%.6f")


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
        if pd.isna(rank) or rank > 3:
            break
        count += 1
    return count


def add_industry_trend_columns(industry_output: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    output_columns = list(industry_output.columns) + [column for column in INDUSTRY_TREND_COLUMNS if column not in industry_output.columns]
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

    strong_10d_return = industry["return_10d"] > 0.05
    weak_3d_return = (industry["return_3d"] < 0) | (industry["return_3d"] < (industry["return_10d"] / 3))
    low_relative_volume = industry["relative_volume"] < 0.8
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


def dataframe_records(data: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(data.to_json(orient="records", double_precision=10))


def sorted_records(data: pd.DataFrame, sort_column: str, limit: int | None = None) -> list[dict[str, Any]]:
    if data.empty:
        return []

    sorted_data = data.sort_values(sort_column, ascending=False, na_position="last")
    if limit is not None:
        sorted_data = sorted_data.head(limit)
    return dataframe_records(sorted_data)


def build_rotation_trends(rotation_history: pd.DataFrame) -> dict[str, Any]:
    empty_trends = {
        "date_count": 0,
        "start_date": None,
        "current_date": None,
        "five_day_start_date": None,
        "gaining_rank": [],
        "losing_rank": [],
        "improving_5d": [],
    }
    if rotation_history.empty:
        return empty_trends

    rotation = rotation_history.copy()
    rotation["date"] = rotation["date"].astype(str)
    rotation["industry_rank"] = pd.to_numeric(rotation["industry_rank"], errors="coerce")
    rotation["average_10d_return"] = pd.to_numeric(rotation["average_10d_return"], errors="coerce")
    rotation["confirmed_signal_pct"] = pd.to_numeric(rotation["confirmed_signal_pct"], errors="coerce")
    dates = sorted(rotation["date"].dropna().unique())

    if not dates:
        return empty_trends

    trends = {
        **empty_trends,
        "date_count": int(len(dates)),
        "start_date": dates[0],
        "current_date": dates[-1],
        "five_day_start_date": dates[max(0, len(dates) - 5)],
    }
    if len(dates) < 2:
        return trends

    def compare_dates(start_date: str, current_date: str) -> pd.DataFrame:
        start = rotation[rotation["date"] == start_date][
            ["industry_group", "industry_rank", "average_10d_return", "confirmed_signal_pct"]
        ].rename(
            columns={
                "industry_rank": "start_rank",
                "average_10d_return": "start_average_10d_return",
                "confirmed_signal_pct": "start_confirmed_signal_pct",
            }
        )
        current = rotation[rotation["date"] == current_date][
            ["industry_group", "industry_rank", "average_10d_return", "confirmed_signal_pct"]
        ].rename(
            columns={
                "industry_rank": "current_rank",
                "average_10d_return": "current_average_10d_return",
                "confirmed_signal_pct": "current_confirmed_signal_pct",
            }
        )
        comparison = current.merge(start, on="industry_group", how="inner")
        comparison.insert(0, "current_date", current_date)
        comparison.insert(0, "start_date", start_date)
        comparison["rank_change"] = comparison["start_rank"] - comparison["current_rank"]
        comparison["average_10d_return_change"] = (
            comparison["current_average_10d_return"] - comparison["start_average_10d_return"]
        )
        comparison["confirmed_signal_pct_change"] = (
            comparison["current_confirmed_signal_pct"] - comparison["start_confirmed_signal_pct"]
        )
        return comparison

    comparison = compare_dates(dates[0], dates[-1])
    trend_columns = [
        "industry_group",
        "start_date",
        "current_date",
        "start_rank",
        "current_rank",
        "rank_change",
        "current_average_10d_return",
        "average_10d_return_change",
        "current_confirmed_signal_pct",
        "confirmed_signal_pct_change",
    ]

    gaining = comparison[comparison["rank_change"] > 0].sort_values(
        ["rank_change", "average_10d_return_change"], ascending=[False, False]
    )
    losing = comparison[comparison["rank_change"] < 0].sort_values(
        ["rank_change", "average_10d_return_change"], ascending=[True, True]
    )

    five_day_comparison = compare_dates(trends["five_day_start_date"], dates[-1])
    improving_5d = five_day_comparison[five_day_comparison["average_10d_return_change"] > 0].sort_values(
        ["average_10d_return_change", "rank_change"], ascending=[False, False]
    )

    trends["gaining_rank"] = dataframe_records(gaining[trend_columns].head(10))
    trends["losing_rank"] = dataframe_records(losing[trend_columns].head(10))
    trends["improving_5d"] = dataframe_records(improving_5d[trend_columns].head(10))
    return trends


def build_dashboard_data(
    ticker_output: pd.DataFrame, industry_output: pd.DataFrame, rotation_history: pd.DataFrame
) -> dict[str, Any]:
    tickers = ticker_output.copy()
    industries = industry_output.copy()

    ticker_numeric_columns = [
        "data_points",
        "return_3d",
        "return_5d",
        "return_10d",
        "return_20d",
        "relative_volume",
        "ma_5d",
        "ma_10d",
        "ma_20d",
        "max_drawdown_10d",
        "up_days_10d",
        "relative_strength_vs_industry",
    ]
    for column in ticker_numeric_columns:
        if column in tickers.columns:
            tickers[column] = pd.to_numeric(tickers[column], errors="coerce")

    for column in ["early_momentum_signal", "confirmed_momentum_signal", "strong_momentum_signal", "risk_warning"]:
        if column in tickers.columns:
            tickers[column] = tickers[column].fillna(False).astype(bool)

    for column in INDUSTRY_TREND_COLUMNS:
        if column not in industries.columns:
            industries[column] = False if column == "momentum_exhaustion_warning" else math.nan

    numeric_industry_columns = [
        "ticker_count",
        "tickers_with_data",
        *METRIC_COLUMNS,
        "rotation_score",
        "momentum_persistence",
        "momentum_acceleration",
    ]
    for column in numeric_industry_columns:
        if column in industries.columns:
            industries[column] = pd.to_numeric(industries[column], errors="coerce")
    if "momentum_exhaustion_warning" in industries.columns:
        industries["momentum_exhaustion_warning"] = (
            industries["momentum_exhaustion_warning"].fillna(False).astype(str).str.lower().isin(["true", "1"])
        )

    confirmed_by_industry = calculate_confirmed_by_industry(tickers)
    confirmed_by_industry = confirmed_by_industry.sort_values(
        ["confirmed_signal_pct", "confirmed_count", "ticker_count"],
        ascending=[False, False, False],
        na_position="last",
    )

    industry_momentum_columns = [
        "industry_group",
        "ticker_count",
        "tickers_with_data",
        "return_10d",
        "return_5d",
        "return_20d",
        "relative_volume",
    ]
    industry_trend_columns = [
        "industry_group",
        "ticker_count",
        "tickers_with_data",
        "return_3d",
        "return_5d",
        "return_10d",
        "relative_volume",
        "rotation_score",
        "momentum_persistence",
        "momentum_acceleration",
        "momentum_exhaustion_warning",
        "confirmed_signal_pct",
    ]

    stock_columns = [
        "ticker",
        "company_name",
        "industry_group",
        "latest_date",
        "return_3d",
        "return_5d",
        "return_10d",
        "return_20d",
        "relative_volume",
        "max_drawdown_10d",
        "up_days_10d",
        "relative_strength_vs_industry",
    ]

    tradable_tickers = tickers[tickers["data_points"] > 0] if "data_points" in tickers.columns else tickers
    latest_dates = tradable_tickers["latest_date"].dropna().astype(str) if "latest_date" in tradable_tickers.columns else []
    industry_trends = industries.merge(
        confirmed_by_industry[["industry_group", "confirmed_signal_pct"]], on="industry_group", how="left"
    )
    if "momentum_exhaustion_warning" in industry_trends.columns:
        industry_trends["momentum_exhaustion_warning"] = (
            industry_trends["momentum_exhaustion_warning"].fillna(False).astype(str).str.lower().isin(["true", "1"])
        )
    industries_with_trend_history = industry_trends[industry_trends["momentum_acceleration"].notna()]
    strongest_improving = industries_with_trend_history[
        (industries_with_trend_history["rotation_score"] > 0)
        | (industries_with_trend_history["momentum_acceleration"] > 0)
    ].sort_values(["rotation_score", "momentum_acceleration", "return_10d"], ascending=[False, False, False])
    strongest_persistent = industry_trends[industry_trends["momentum_persistence"] > 0].sort_values(
        ["momentum_persistence", "return_10d", "confirmed_signal_pct"], ascending=[False, False, False]
    )
    exhaustion = industry_trends[industry_trends["momentum_exhaustion_warning"]].sort_values(
        ["return_10d", "relative_volume"], ascending=[False, True], na_position="last"
    )
    momentum_recovery = industries_with_trend_history[
        (industries_with_trend_history["momentum_acceleration"] > 0)
        & (industries_with_trend_history["return_5d"] > 0)
        & (industries_with_trend_history["return_10d"] > 0)
        & (industries_with_trend_history["rotation_score"] >= 0)
    ].sort_values(["momentum_acceleration", "return_5d", "rotation_score"], ascending=[False, False, False])

    return {
        "summary": {
            "latest_date": max(latest_dates) if len(latest_dates) else None,
            "total_tickers": int(len(tickers)),
            "tickers_with_data": int((tickers["data_points"] > 0).sum()) if "data_points" in tickers.columns else 0,
            "early_count": int(tickers["early_momentum_signal"].sum()) if "early_momentum_signal" in tickers.columns else 0,
            "confirmed_count": int(tickers["confirmed_momentum_signal"].sum())
            if "confirmed_momentum_signal" in tickers.columns
            else 0,
            "strong_count": int(tickers["strong_momentum_signal"].sum()) if "strong_momentum_signal" in tickers.columns else 0,
            "risk_count": int(tickers["risk_warning"].sum()) if "risk_warning" in tickers.columns else 0,
        },
        "industry_momentum": dataframe_records(
            industries[industry_momentum_columns].sort_values("return_10d", ascending=False, na_position="last")
        )
        if not industries.empty
        else [],
        "industry_confirmed": dataframe_records(confirmed_by_industry),
        "industry_trend_intelligence": {
            "strongest_improving": dataframe_records(strongest_improving[industry_trend_columns].head(10)),
            "strongest_persistent": dataframe_records(strongest_persistent[industry_trend_columns].head(10)),
            "momentum_exhaustion": dataframe_records(exhaustion[industry_trend_columns].head(10)),
            "momentum_recovery": dataframe_records(momentum_recovery[industry_trend_columns].head(10)),
        },
        "top_relative_strength": sorted_records(
            tradable_tickers[stock_columns], "relative_strength_vs_industry", limit=10
        )
        if not tradable_tickers.empty
        else [],
        "early_candidates": sorted_records(
            tradable_tickers[tradable_tickers["early_momentum_signal"]][stock_columns], "return_5d"
        )
        if not tradable_tickers.empty
        else [],
        "strong_candidates": sorted_records(
            tradable_tickers[tradable_tickers["strong_momentum_signal"]][stock_columns],
            "relative_strength_vs_industry",
        )
        if not tradable_tickers.empty
        else [],
        "risk_warnings": dataframe_records(
            tradable_tickers[tradable_tickers["risk_warning"]][stock_columns].sort_values(
                ["max_drawdown_10d", "relative_volume"], ascending=[True, False], na_position="last"
            )
        )
        if not tradable_tickers.empty
        else [],
        "rotation_trend": build_rotation_trends(rotation_history),
    }


def build_dashboard_html(dashboard_data: dict[str, Any]) -> str:
    data_json = json.dumps(dashboard_data, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stock Momentum Tracker</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f4;
      --surface: #ffffff;
      --surface-soft: #eef4f2;
      --ink: #17211d;
      --muted: #65736e;
      --line: #d9e0dc;
      --line-strong: #b8c4be;
      --green: #047857;
      --red: #b91c1c;
      --amber: #b45309;
      --blue: #1d4ed8;
      --teal: #0f766e;
      --shadow: 0 10px 30px rgba(23, 33, 29, 0.08);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 15px;
      line-height: 1.45;
    }

    header {
      border-bottom: 1px solid var(--line);
      background: var(--surface);
    }

    .header-inner,
    main {
      width: min(1360px, calc(100% - 32px));
      margin: 0 auto;
    }

    .header-inner {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 24px;
      padding: 24px 0 18px;
    }

    h1,
    h2,
    p {
      margin: 0;
    }

    h1 {
      font-size: 28px;
      font-weight: 750;
      letter-spacing: 0;
    }

    .subhead {
      margin-top: 5px;
      color: var(--muted);
      font-size: 14px;
    }

    .timestamp {
      color: var(--muted);
      font-size: 13px;
      text-align: right;
      white-space: nowrap;
    }

    main {
      padding: 22px 0 40px;
    }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(6, minmax(130px, 1fr));
      gap: 10px;
      margin-bottom: 22px;
    }

    .summary-tile {
      min-height: 78px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      padding: 13px 14px;
      box-shadow: var(--shadow);
    }

    .summary-label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }

    .summary-value {
      margin-top: 8px;
      font-size: 26px;
      font-weight: 760;
    }

    .dashboard-section {
      border-top: 1px solid var(--line-strong);
      padding: 22px 0 8px;
    }

    .section-heading {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 10px;
    }

    h2 {
      font-size: 18px;
      font-weight: 740;
      letter-spacing: 0;
    }

    h3 {
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 760;
      letter-spacing: 0;
      text-transform: uppercase;
    }

    .row-count {
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }

    .table-wrap {
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      box-shadow: var(--shadow);
    }

    .rotation-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }

    .intelligence-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }

    .rotation-block table {
      min-width: 620px;
    }

    .intelligence-block table {
      min-width: 720px;
    }

    table {
      width: 100%;
      min-width: 820px;
      border-collapse: collapse;
    }

    th,
    td {
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      vertical-align: middle;
      white-space: nowrap;
    }

    th {
      background: var(--surface-soft);
      color: var(--muted);
      font-size: 12px;
      font-weight: 760;
      text-transform: uppercase;
    }

    tr:last-child td {
      border-bottom: 0;
    }

    tbody tr:hover {
      background: #f9fbfa;
    }

    .numeric {
      text-align: right;
      font-variant-numeric: tabular-nums;
    }

    .rank {
      width: 54px;
      color: var(--muted);
      text-align: right;
    }

    .ticker {
      color: var(--blue);
      font-weight: 760;
    }

    .company {
      max-width: 280px;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .positive {
      color: var(--green);
      font-weight: 720;
    }

    .negative {
      color: var(--red);
      font-weight: 720;
    }

    .warning {
      color: var(--amber);
      font-weight: 720;
    }

    .empty-state {
      border: 1px dashed var(--line-strong);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.65);
      color: var(--muted);
      padding: 16px;
    }

    @media (max-width: 920px) {
      .header-inner {
        align-items: start;
        flex-direction: column;
      }

      .timestamp {
        text-align: left;
      }

      .summary-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      .section-heading {
        align-items: start;
        flex-direction: column;
        gap: 4px;
      }

      .rotation-grid {
        grid-template-columns: 1fr;
      }

      .intelligence-grid {
        grid-template-columns: 1fr;
      }
    }

    @media (max-width: 520px) {
      .header-inner,
      main {
        width: min(100% - 20px, 1360px);
      }

      h1 {
        font-size: 24px;
      }

      .summary-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <header>
    <div class="header-inner">
      <div>
        <h1>Stock Momentum Tracker</h1>
        <p class="subhead">Static signal dashboard generated from the latest local CSV run.</p>
      </div>
      <p class="timestamp" id="generated-date"></p>
    </div>
  </header>

  <main>
    <section class="summary-grid" id="summary-grid" aria-label="Momentum summary"></section>

    <section class="dashboard-section">
      <div class="section-heading">
        <h2>Industry momentum ranking by average 10 day return</h2>
        <span class="row-count" data-count-for="industry-momentum"></span>
      </div>
      <div class="table-wrap">
        <table data-table="industry-momentum"></table>
      </div>
      <p class="empty-state" data-empty-for="industry-momentum" hidden>No industry rows.</p>
    </section>

    <section class="dashboard-section">
      <div class="section-heading">
        <h2>Industries ranked by percentage of tickers with confirmed_momentum_signal</h2>
        <span class="row-count" data-count-for="industry-confirmed"></span>
      </div>
      <div class="table-wrap">
        <table data-table="industry-confirmed"></table>
      </div>
      <p class="empty-state" data-empty-for="industry-confirmed" hidden>No industry signal rows.</p>
    </section>

    <section class="dashboard-section">
      <div class="section-heading">
        <h2>Industry Rotation Trend</h2>
        <span class="row-count" id="rotation-history-status"></span>
      </div>
      <div class="rotation-grid">
        <div class="rotation-block">
          <h3>Industries gaining rank over time</h3>
          <div class="table-wrap">
            <table data-table="rotation-gaining"></table>
          </div>
          <p class="empty-state" data-empty-for="rotation-gaining" hidden>No gaining industries yet.</p>
        </div>
        <div class="rotation-block">
          <h3>Industries losing rank over time</h3>
          <div class="table-wrap">
            <table data-table="rotation-losing"></table>
          </div>
          <p class="empty-state" data-empty-for="rotation-losing" hidden>No losing industries yet.</p>
        </div>
        <div class="rotation-block">
          <h3>Strongest improving industries over last 5 trading days</h3>
          <div class="table-wrap">
            <table data-table="rotation-improving-5d"></table>
          </div>
          <p class="empty-state" data-empty-for="rotation-improving-5d" hidden>No 5 day improvement yet.</p>
        </div>
      </div>
    </section>

    <section class="dashboard-section">
      <div class="section-heading">
        <h2>Industry Trend Intelligence</h2>
        <span class="row-count" id="trend-intelligence-status"></span>
      </div>
      <div class="intelligence-grid">
        <div class="intelligence-block">
          <h3>Strongest improving industries</h3>
          <div class="table-wrap">
            <table data-table="trend-strongest-improving"></table>
          </div>
          <p class="empty-state" data-empty-for="trend-strongest-improving" hidden>No improving industries yet.</p>
        </div>
        <div class="intelligence-block">
          <h3>Strongest persistent industries</h3>
          <div class="table-wrap">
            <table data-table="trend-strongest-persistent"></table>
          </div>
          <p class="empty-state" data-empty-for="trend-strongest-persistent" hidden>No persistent top 3 industries yet.</p>
        </div>
        <div class="intelligence-block">
          <h3>Possible momentum exhaustion industries</h3>
          <div class="table-wrap">
            <table data-table="trend-momentum-exhaustion"></table>
          </div>
          <p class="empty-state" data-empty-for="trend-momentum-exhaustion" hidden>No exhaustion warnings.</p>
        </div>
        <div class="intelligence-block">
          <h3>Newest momentum recovery industries</h3>
          <div class="table-wrap">
            <table data-table="trend-momentum-recovery"></table>
          </div>
          <p class="empty-state" data-empty-for="trend-momentum-recovery" hidden>No recovery industries yet.</p>
        </div>
      </div>
    </section>

    <section class="dashboard-section">
      <div class="section-heading">
        <h2>Top 10 stocks by relative_strength_vs_industry</h2>
        <span class="row-count" data-count-for="top-relative-strength"></span>
      </div>
      <div class="table-wrap">
        <table data-table="top-relative-strength"></table>
      </div>
      <p class="empty-state" data-empty-for="top-relative-strength" hidden>No ticker rows.</p>
    </section>

    <section class="dashboard-section">
      <div class="section-heading">
        <h2>Early momentum candidates</h2>
        <span class="row-count" data-count-for="early-candidates"></span>
      </div>
      <div class="table-wrap">
        <table data-table="early-candidates"></table>
      </div>
      <p class="empty-state" data-empty-for="early-candidates" hidden>No early momentum candidates.</p>
    </section>

    <section class="dashboard-section">
      <div class="section-heading">
        <h2>Strong momentum candidates</h2>
        <span class="row-count" data-count-for="strong-candidates"></span>
      </div>
      <div class="table-wrap">
        <table data-table="strong-candidates"></table>
      </div>
      <p class="empty-state" data-empty-for="strong-candidates" hidden>No strong momentum candidates.</p>
    </section>

    <section class="dashboard-section">
      <div class="section-heading">
        <h2>Risk warning list</h2>
        <span class="row-count" data-count-for="risk-warnings"></span>
      </div>
      <div class="table-wrap">
        <table data-table="risk-warnings"></table>
      </div>
      <p class="empty-state" data-empty-for="risk-warnings" hidden>No current risk warnings.</p>
    </section>
  </main>

  <script id="dashboard-data" type="application/json">__DASHBOARD_DATA__</script>
  <script>
    const dashboardData = JSON.parse(document.getElementById("dashboard-data").textContent);

    const tableConfigs = {
      "industry-momentum": {
        rows: dashboardData.industry_momentum,
        columns: [
          { key: "__rank", label: "Rank", type: "rank" },
          { key: "industry_group", label: "Industry" },
          { key: "return_10d", label: "Avg 10D", type: "percent" },
          { key: "return_5d", label: "Avg 5D", type: "percent" },
          { key: "return_20d", label: "Avg 20D", type: "percent" },
          { key: "relative_volume", label: "Rel Vol", type: "number", digits: 2 },
          { key: "tickers_with_data", label: "With Data", type: "integer" },
          { key: "ticker_count", label: "Tickers", type: "integer" }
        ]
      },
      "industry-confirmed": {
        rows: dashboardData.industry_confirmed,
        columns: [
          { key: "__rank", label: "Rank", type: "rank" },
          { key: "industry_group", label: "Industry" },
          { key: "confirmed_signal_pct", label: "Confirmed %", type: "percent" },
          { key: "confirmed_count", label: "Confirmed", type: "integer" },
          { key: "tickers_with_data", label: "With Data", type: "integer" },
          { key: "ticker_count", label: "Tickers", type: "integer" }
        ]
      },
      "rotation-gaining": {
        rows: dashboardData.rotation_trend.gaining_rank,
        columns: [
          { key: "industry_group", label: "Industry" },
          { key: "start_rank", label: "From", type: "integer" },
          { key: "current_rank", label: "Now", type: "integer" },
          { key: "rank_change", label: "Rank Chg", type: "signedInteger" },
          { key: "current_average_10d_return", label: "Avg 10D", type: "percent" },
          { key: "average_10d_return_change", label: "10D Chg", type: "signedPercent" }
        ]
      },
      "rotation-losing": {
        rows: dashboardData.rotation_trend.losing_rank,
        columns: [
          { key: "industry_group", label: "Industry" },
          { key: "start_rank", label: "From", type: "integer" },
          { key: "current_rank", label: "Now", type: "integer" },
          { key: "rank_change", label: "Rank Chg", type: "signedInteger" },
          { key: "current_average_10d_return", label: "Avg 10D", type: "percent" },
          { key: "average_10d_return_change", label: "10D Chg", type: "signedPercent" }
        ]
      },
      "rotation-improving-5d": {
        rows: dashboardData.rotation_trend.improving_5d,
        columns: [
          { key: "industry_group", label: "Industry" },
          { key: "average_10d_return_change", label: "10D Chg", type: "signedPercent" },
          { key: "current_average_10d_return", label: "Avg 10D", type: "percent" },
          { key: "rank_change", label: "Rank Chg", type: "signedInteger" },
          { key: "current_rank", label: "Rank", type: "integer" },
          { key: "current_confirmed_signal_pct", label: "Confirmed", type: "percent" }
        ]
      },
      "trend-strongest-improving": {
        rows: dashboardData.industry_trend_intelligence.strongest_improving,
        columns: [
          { key: "industry_group", label: "Industry" },
          { key: "rotation_score", label: "Rotation", type: "signedInteger" },
          { key: "momentum_acceleration", label: "Accel", type: "signedPercent" },
          { key: "return_10d", label: "Avg 10D", type: "percent" },
          { key: "return_5d", label: "Avg 5D", type: "percent" },
          { key: "relative_volume", label: "Rel Vol", type: "number", digits: 2 }
        ]
      },
      "trend-strongest-persistent": {
        rows: dashboardData.industry_trend_intelligence.strongest_persistent,
        columns: [
          { key: "industry_group", label: "Industry" },
          { key: "momentum_persistence", label: "Top 3 Days", type: "integer" },
          { key: "return_10d", label: "Avg 10D", type: "percent" },
          { key: "confirmed_signal_pct", label: "Confirmed", type: "percent" },
          { key: "rotation_score", label: "Rotation", type: "signedInteger" },
          { key: "momentum_acceleration", label: "Accel", type: "signedPercent" }
        ]
      },
      "trend-momentum-exhaustion": {
        rows: dashboardData.industry_trend_intelligence.momentum_exhaustion,
        columns: [
          { key: "industry_group", label: "Industry" },
          { key: "return_10d", label: "Avg 10D", type: "percent" },
          { key: "return_3d", label: "Avg 3D", type: "percent" },
          { key: "relative_volume", label: "Rel Vol", type: "number", digits: 2 },
          { key: "momentum_acceleration", label: "Accel", type: "signedPercent" },
          { key: "momentum_persistence", label: "Top 3 Days", type: "integer" }
        ]
      },
      "trend-momentum-recovery": {
        rows: dashboardData.industry_trend_intelligence.momentum_recovery,
        columns: [
          { key: "industry_group", label: "Industry" },
          { key: "momentum_acceleration", label: "Accel", type: "signedPercent" },
          { key: "rotation_score", label: "Rotation", type: "signedInteger" },
          { key: "return_5d", label: "Avg 5D", type: "percent" },
          { key: "return_10d", label: "Avg 10D", type: "percent" },
          { key: "relative_volume", label: "Rel Vol", type: "number", digits: 2 }
        ]
      },
      "top-relative-strength": {
        rows: dashboardData.top_relative_strength,
        columns: [
          { key: "__rank", label: "Rank", type: "rank" },
          { key: "ticker", label: "Ticker", type: "ticker" },
          { key: "company_name", label: "Company", type: "company" },
          { key: "industry_group", label: "Industry" },
          { key: "relative_strength_vs_industry", label: "Rel Strength", type: "percent" },
          { key: "return_10d", label: "10D", type: "percent" },
          { key: "relative_volume", label: "Rel Vol", type: "number", digits: 2 },
          { key: "up_days_10d", label: "Up Days", type: "integer" }
        ]
      },
      "early-candidates": {
        rows: dashboardData.early_candidates,
        columns: [
          { key: "ticker", label: "Ticker", type: "ticker" },
          { key: "company_name", label: "Company", type: "company" },
          { key: "industry_group", label: "Industry" },
          { key: "return_3d", label: "3D", type: "percent" },
          { key: "return_5d", label: "5D", type: "percent" },
          { key: "return_10d", label: "10D", type: "percent" },
          { key: "relative_strength_vs_industry", label: "Rel Strength", type: "percent" },
          { key: "relative_volume", label: "Rel Vol", type: "number", digits: 2 }
        ]
      },
      "strong-candidates": {
        rows: dashboardData.strong_candidates,
        columns: [
          { key: "ticker", label: "Ticker", type: "ticker" },
          { key: "company_name", label: "Company", type: "company" },
          { key: "industry_group", label: "Industry" },
          { key: "relative_strength_vs_industry", label: "Rel Strength", type: "percent" },
          { key: "return_10d", label: "10D", type: "percent" },
          { key: "relative_volume", label: "Rel Vol", type: "number", digits: 2 },
          { key: "up_days_10d", label: "Up Days", type: "integer" },
          { key: "max_drawdown_10d", label: "Max DD", type: "percent" }
        ]
      },
      "risk-warnings": {
        rows: dashboardData.risk_warnings,
        columns: [
          { key: "ticker", label: "Ticker", type: "ticker" },
          { key: "company_name", label: "Company", type: "company" },
          { key: "industry_group", label: "Industry" },
          { key: "max_drawdown_10d", label: "Max DD", type: "warningPercent" },
          { key: "return_10d", label: "10D", type: "percent" },
          { key: "return_20d", label: "20D", type: "percent" },
          { key: "relative_volume", label: "Rel Vol", type: "number", digits: 2 },
          { key: "up_days_10d", label: "Up Days", type: "integer" }
        ]
      }
    };

    function isMissing(value) {
      return value === null || value === undefined || Number.isNaN(value);
    }

    function formatPercent(value) {
      if (isMissing(value)) return "";
      return `${(value * 100).toFixed(2)}%`;
    }

    function formatSignedPercent(value) {
      if (isMissing(value)) return "";
      const formatted = formatPercent(value);
      return value > 0 ? `+${formatted}` : formatted;
    }

    function formatNumber(value, digits = 2) {
      if (isMissing(value)) return "";
      return Number(value).toLocaleString(undefined, {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits
      });
    }

    function formatInteger(value) {
      if (isMissing(value)) return "";
      return Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 });
    }

    function formatSignedInteger(value) {
      if (isMissing(value)) return "";
      const formatted = formatInteger(value);
      return value > 0 ? `+${formatted}` : formatted;
    }

    function formatCell(value, column, rowIndex) {
      if (column.type === "rank") return String(rowIndex + 1);
      if (column.type === "percent" || column.type === "warningPercent") return formatPercent(value);
      if (column.type === "signedPercent") return formatSignedPercent(value);
      if (column.type === "number") return formatNumber(value, column.digits ?? 2);
      if (column.type === "integer") return formatInteger(value);
      if (column.type === "signedInteger") return formatSignedInteger(value);
      return isMissing(value) ? "" : String(value);
    }

    function classForCell(value, column) {
      const classes = [];
      if (["rank", "number", "integer", "signedInteger", "percent", "signedPercent", "warningPercent"].includes(column.type)) {
        classes.push("numeric");
      }
      if (column.type === "rank") classes.push("rank");
      if (column.type === "ticker") classes.push("ticker");
      if (column.type === "company") classes.push("company");
      if ((column.type === "percent" || column.type === "signedPercent" || column.type === "warningPercent" || column.type === "signedInteger") && !isMissing(value)) {
        if (value > 0) classes.push("positive");
        if (value < 0) classes.push(column.type === "warningPercent" ? "warning" : "negative");
      }
      return classes.join(" ");
    }

    function renderSummary() {
      const summary = dashboardData.summary;
      const grid = document.getElementById("summary-grid");
      const tiles = [
        ["Tickers", summary.total_tickers],
        ["With Data", summary.tickers_with_data],
        ["Early", summary.early_count],
        ["Confirmed", summary.confirmed_count],
        ["Strong", summary.strong_count],
        ["Risk", summary.risk_count]
      ];

      document.getElementById("generated-date").textContent = summary.latest_date
        ? `Latest market date: ${summary.latest_date}`
        : "Latest market date unavailable";

      for (const [label, value] of tiles) {
        const tile = document.createElement("div");
        tile.className = "summary-tile";

        const labelEl = document.createElement("div");
        labelEl.className = "summary-label";
        labelEl.textContent = label;

        const valueEl = document.createElement("div");
        valueEl.className = "summary-value";
        valueEl.textContent = formatInteger(value);

        tile.append(labelEl, valueEl);
        grid.appendChild(tile);
      }
    }

    function renderTable(id, config) {
      const rows = config.rows || [];
      const table = document.querySelector(`[data-table="${id}"]`);
      const empty = document.querySelector(`[data-empty-for="${id}"]`);
      const count = document.querySelector(`[data-count-for="${id}"]`);
      if (count) {
        count.textContent = `${rows.length} row${rows.length === 1 ? "" : "s"}`;
      }

      if (!rows.length) {
        table.parentElement.hidden = true;
        empty.hidden = false;
        return;
      }

      const thead = document.createElement("thead");
      const headerRow = document.createElement("tr");
      for (const column of config.columns) {
        const th = document.createElement("th");
        th.textContent = column.label;
        if (["rank", "number", "integer", "signedInteger", "percent", "signedPercent", "warningPercent"].includes(column.type)) {
          th.className = "numeric";
        }
        headerRow.appendChild(th);
      }
      thead.appendChild(headerRow);

      const tbody = document.createElement("tbody");
      rows.forEach((row, rowIndex) => {
        const tr = document.createElement("tr");
        for (const column of config.columns) {
          const value = column.key === "__rank" ? rowIndex : row[column.key];
          const td = document.createElement("td");
          td.className = classForCell(value, column);
          td.textContent = formatCell(value, column, rowIndex);
          tr.appendChild(td);
        }
        tbody.appendChild(tr);
      });

      table.replaceChildren(thead, tbody);
    }

    renderSummary();
    document.getElementById("rotation-history-status").textContent =
      `${dashboardData.rotation_trend.date_count} historical date${dashboardData.rotation_trend.date_count === 1 ? "" : "s"}`;
    const intelligenceRows = Object.values(dashboardData.industry_trend_intelligence)
      .reduce((total, rows) => total + rows.length, 0);
    document.getElementById("trend-intelligence-status").textContent =
      `${intelligenceRows} signal row${intelligenceRows === 1 ? "" : "s"}`;
    for (const [id, config] of Object.entries(tableConfigs)) {
      renderTable(id, config);
    }
  </script>
</body>
</html>
"""
    return html.replace("__DASHBOARD_DATA__", data_json)


def write_dashboard(
    ticker_output: pd.DataFrame, industry_output: pd.DataFrame, rotation_history: pd.DataFrame, path: Path
) -> None:
    dashboard_data = build_dashboard_data(ticker_output, industry_output, rotation_history)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_dashboard_html(dashboard_data), encoding="utf-8")


def main() -> None:
    if not TICKERS_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {TICKERS_PATH}")

    tickers = clean_tickers(pd.read_csv(TICKERS_PATH))
    downloaded_data = download_market_data(tickers["ticker"].tolist())
    ticker_output = build_ticker_output(tickers, downloaded_data)
    industry_output = build_industry_output(ticker_output)
    ticker_output = add_signal_columns(ticker_output, industry_output)
    snapshot_date = get_snapshot_date(ticker_output)
    industry_output = add_industry_trend_columns(industry_output, snapshot_date)

    write_csv(ticker_output, TICKER_OUTPUT_PATH)
    write_csv(industry_output, INDUSTRY_OUTPUT_PATH)
    snapshot_dir = write_daily_snapshot(ticker_output, industry_output, snapshot_date)
    rotation_history = build_industry_rotation_history()
    write_csv(rotation_history, INDUSTRY_ROTATION_HISTORY_PATH)
    write_dashboard(ticker_output, industry_output, rotation_history, DASHBOARD_OUTPUT_PATH)

    print(f"Wrote {TICKER_OUTPUT_PATH}")
    print(f"Wrote {INDUSTRY_OUTPUT_PATH}")
    print(f"Wrote {snapshot_dir}")
    print(f"Wrote {INDUSTRY_ROTATION_HISTORY_PATH}")
    print(f"Wrote {DASHBOARD_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
