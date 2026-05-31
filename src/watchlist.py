from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import WATCHLIST_ALERT_COLUMNS, WATCHLIST_COLUMNS


def is_missing(value: Any) -> bool:
    return value is None or bool(pd.isna(value))


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ["true", "1", "yes"]


def format_percent(value: Any) -> str:
    if is_missing(value):
        return "n/a"
    return f"{float(value) * 100:+.2f}%"


def empty_watchlist() -> pd.DataFrame:
    return pd.DataFrame(columns=WATCHLIST_COLUMNS)


def empty_alerts() -> pd.DataFrame:
    return pd.DataFrame(columns=WATCHLIST_ALERT_COLUMNS)


def clean_watchlist(watchlist_input: pd.DataFrame) -> pd.DataFrame:
    if watchlist_input.empty:
        return empty_watchlist()
    if "ticker" not in watchlist_input.columns:
        raise ValueError("watchlist.csv must include a ticker column")

    watchlist = watchlist_input.copy()
    for column in WATCHLIST_COLUMNS:
        if column not in watchlist.columns:
            watchlist[column] = pd.NA

    watchlist["ticker"] = watchlist["ticker"].fillna("").astype(str).str.strip().str.upper()
    watchlist = watchlist[watchlist["ticker"] != ""].copy()
    for column in ["theme", "notes"]:
        watchlist[column] = watchlist[column].fillna("").astype(str)

    return watchlist[WATCHLIST_COLUMNS]


def load_watchlist(path: Path) -> pd.DataFrame:
    if not path.exists():
        return empty_watchlist()
    return clean_watchlist(pd.read_csv(path))


def prepare_tickers(ticker_output: pd.DataFrame) -> pd.DataFrame:
    tickers = ticker_output.copy()
    if "ticker" not in tickers.columns:
        tickers["ticker"] = pd.Series(dtype=str)
    tickers["ticker"] = tickers["ticker"].fillna("").astype(str).str.strip().str.upper()
    numeric_columns = [
        "data_points",
        "return_5d",
        "return_10d",
        "relative_strength_vs_industry",
        "relative_volume",
    ]
    for column in numeric_columns:
        if column not in tickers.columns:
            tickers[column] = pd.NA
        tickers[column] = pd.to_numeric(tickers[column], errors="coerce")
    for column in ["risk_warning", "early_momentum_signal", "strong_momentum_signal"]:
        if column not in tickers.columns:
            tickers[column] = False
        tickers[column] = tickers[column].fillna(False).map(to_bool)
    for column, default in [
        ("company_name", ""),
        ("industry_group", ""),
        ("latest_date", ""),
        ("watch_status", ""),
        ("current_state", ""),
        ("industry_regime", ""),
        ("industry_risk_flag", "none"),
        ("data_status", "missing"),
    ]:
        if column not in tickers.columns:
            tickers[column] = default
        tickers[column] = tickers[column].fillna(default).astype(str)
    return tickers


def prepare_industries(industry_output: pd.DataFrame) -> pd.DataFrame:
    industries = industry_output.copy()
    for column in ["return_5d", "return_10d", "breadth_score", "rotation_score", "momentum_acceleration"]:
        if column not in industries.columns:
            industries[column] = pd.NA
        industries[column] = pd.to_numeric(industries[column], errors="coerce")
    if "momentum_exhaustion_warning" not in industries.columns:
        industries["momentum_exhaustion_warning"] = False
    industries["momentum_exhaustion_warning"] = industries["momentum_exhaustion_warning"].fillna(False).map(to_bool)
    for column, default in [("industry_group", ""), ("industry_regime", ""), ("industry_risk_flag", "none")]:
        if column not in industries.columns:
            industries[column] = default
        industries[column] = industries[column].fillna(default).astype(str)
    return industries


def build_replacement_industries(industries: pd.DataFrame, limit: int = 3) -> list[str]:
    if industries.empty:
        return []

    median_breadth = industries["breadth_score"].median()
    candidates = industries[
        (industries["return_5d"] > 0)
        & (industries["return_10d"] > 0)
        & (industries["industry_regime"] != "weak")
        & (~industries["momentum_exhaustion_warning"])
    ].copy()
    if not is_missing(median_breadth):
        candidates = candidates[candidates["breadth_score"] >= median_breadth]
    if candidates.empty:
        candidates = industries.copy()

    ranked = candidates.sort_values(
        ["breadth_score", "return_10d", "rotation_score", "momentum_acceleration"],
        ascending=[False, False, False, False],
        na_position="last",
    )
    return ranked["industry_group"].dropna().astype(str).head(limit).tolist()


def build_replacement_candidates(tickers: pd.DataFrame, current_ticker: str, limit: int = 5) -> list[str]:
    if tickers.empty:
        return []

    candidates = tickers[
        (tickers["ticker"] != current_ticker)
        & (tickers["data_points"] > 0)
        & (tickers["data_status"] == "ok")
        & (~tickers["risk_warning"])
        & (tickers["return_5d"] > 0)
        & (tickers["return_10d"] > 0)
        & (tickers["relative_strength_vs_industry"] > 0)
    ].copy()
    if candidates.empty:
        candidates = tickers[
            (tickers["ticker"] != current_ticker)
            & (tickers["data_points"] > 0)
            & (tickers["data_status"] == "ok")
            & (~tickers["risk_warning"])
            & (tickers["return_10d"] > 0)
        ].copy()
    if candidates.empty:
        return []

    ranked = candidates.sort_values(
        ["strong_momentum_signal", "early_momentum_signal", "relative_strength_vs_industry", "return_10d"],
        ascending=[False, False, False, False],
        na_position="last",
    )
    return [
        f"{row.ticker} ({row.industry_group})"
        for row in ranked[["ticker", "industry_group"]].head(limit).itertuples(index=False)
    ]


def classify_alert(row: pd.Series) -> tuple[str, str, list[str]]:
    if is_missing(row.get("company_name")) or not str(row.get("company_name") or "").strip():
        return "unknown", "add_to_tickers", ["追蹤 ticker 不在 tickers.csv 動能資料中"]

    reasons: list[str] = []
    return_5d = row.get("return_5d")
    return_10d = row.get("return_10d")
    relative_strength = row.get("relative_strength_vs_industry")
    risk_warning = to_bool(row.get("risk_warning"))
    current_state = str(row.get("current_state") or "")
    watch_status = str(row.get("watch_status") or "")
    industry_regime = str(row.get("industry_regime") or "")
    industry_risk_flag = str(row.get("industry_risk_flag") or "none")

    if risk_warning:
        reasons.append("已有 risk_warning")
    if pd.notna(return_5d) and float(return_5d) < 0:
        reasons.append(f"5 日報酬 {format_percent(return_5d)}")
    if pd.notna(return_10d) and float(return_10d) < 0:
        reasons.append(f"10 日報酬 {format_percent(return_10d)}")
    if pd.notna(relative_strength) and float(relative_strength) < 0:
        reasons.append(f"相對產業 {format_percent(relative_strength)}")
    if current_state == "falling_knife":
        reasons.append("current_state=falling_knife")
    if watch_status == "avoid_for_now":
        reasons.append("watch_status=avoid_for_now")
    if industry_regime in ["weak", "neutral"]:
        reasons.append(f"產業狀態 {industry_regime}")
    if industry_risk_flag not in ["", "none"]:
        reasons.append(f"產業風險 {industry_risk_flag}")

    weak_price = (
        pd.notna(return_5d)
        and pd.notna(return_10d)
        and pd.notna(relative_strength)
        and float(return_5d) < 0
        and float(return_10d) < 0
        and float(relative_strength) < 0
    )
    if risk_warning or weak_price or current_state == "falling_knife":
        return "red", "review_replacement", reasons
    if pd.notna(relative_strength) and float(relative_strength) < 0 and (
        (pd.notna(return_5d) and float(return_5d) < 0) or (pd.notna(return_10d) and float(return_10d) < 0)
    ):
        return "orange", "watch_transition", reasons
    if reasons:
        return "yellow", "monitor", reasons
    return "green", "watch_ok", ["追蹤 ticker 動能與產業狀態暫無明顯警示"]


def build_watchlist_alerts(
    watchlist_input: pd.DataFrame,
    ticker_output: pd.DataFrame,
    industry_output: pd.DataFrame,
) -> pd.DataFrame:
    watchlist = clean_watchlist(watchlist_input)
    if watchlist.empty:
        return empty_alerts()

    tickers = prepare_tickers(ticker_output)
    industries = prepare_industries(industry_output)
    ticker_columns = [
        "ticker",
        "company_name",
        "industry_group",
        "latest_date",
        "data_points",
        "return_5d",
        "return_10d",
        "relative_strength_vs_industry",
        "risk_warning",
        "watch_status",
        "current_state",
        "industry_regime",
        "industry_risk_flag",
        "data_status",
    ]
    alerts = watchlist.merge(tickers[ticker_columns], on="ticker", how="left")
    replacement_industries = ", ".join(build_replacement_industries(industries))

    rows: list[dict[str, Any]] = []
    for _, row in alerts.iterrows():
        alert_level, action, reasons = classify_alert(row)
        current_ticker = str(row.get("ticker") or "")
        replacement_candidates = ", ".join(build_replacement_candidates(tickers, current_ticker))
        rows.append(
            {
                "ticker": current_ticker,
                "theme": row.get("theme"),
                "notes": row.get("notes"),
                "company_name": row.get("company_name"),
                "industry_group": row.get("industry_group"),
                "latest_date": row.get("latest_date"),
                "alert_level": alert_level,
                "action": action,
                "alert_reason": "；".join(reasons),
                "replacement_industries": replacement_industries,
                "replacement_candidates": replacement_candidates,
                "return_5d": row.get("return_5d"),
                "return_10d": row.get("return_10d"),
                "relative_strength_vs_industry": row.get("relative_strength_vs_industry"),
                "risk_warning": row.get("risk_warning"),
                "watch_status": row.get("watch_status"),
                "current_state": row.get("current_state"),
                "industry_regime": row.get("industry_regime"),
                "industry_risk_flag": row.get("industry_risk_flag"),
            }
        )

    return pd.DataFrame(rows, columns=WATCHLIST_ALERT_COLUMNS)
