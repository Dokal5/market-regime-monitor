from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    BREADTH_COLUMNS,
    CAUSAL_HYPOTHESIS_COLUMN,
    EVIDENCE_STATUS_COLUMN,
    INDUSTRY_REGIME_COLUMN,
    INDUSTRY_RISK_FLAG_COLUMN,
    INDUSTRY_TREND_COLUMNS,
    METRIC_COLUMNS,
    PEG_COLUMNS,
    ROTATION_TYPE_COLUMN,
)
from src.data_quality import build_data_quality_summary
from src.daily_brief import build_daily_brief
from src.industry import calculate_confirmed_by_industry
from src.update_health import build_update_health_output


def dataframe_records(data: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(data.to_json(orient="records", double_precision=10))


def sorted_records(data: pd.DataFrame, sort_column: str, limit: int | None = None) -> list[dict[str, Any]]:
    if data.empty:
        return []

    sorted_data = data.sort_values(sort_column, ascending=False, na_position="last")
    if limit is not None:
        sorted_data = sorted_data.head(limit)
    return dataframe_records(sorted_data)


def build_momentum_map(
    tickers: pd.DataFrame,
    industries: pd.DataFrame,
    watchlist_alerts: pd.DataFrame | None = None,
) -> dict[str, Any]:
    industry_columns = [
        "industry_group",
        "return_10d",
        "return_5d",
        "breadth_score",
        INDUSTRY_REGIME_COLUMN,
        INDUSTRY_RISK_FLAG_COLUMN,
        ROTATION_TYPE_COLUMN,
    ]
    holding_columns = [
        "ticker",
        "company_name",
        "industry_group",
        "return_10d",
        "relative_strength_vs_industry",
        "relative_volume",
        "watch_status",
        "alert_level",
        INDUSTRY_REGIME_COLUMN,
        INDUSTRY_RISK_FLAG_COLUMN,
    ]
    empty_map = {
        "industry_bars": [],
        "holding_alignment": [],
        "summary": {
            "strong_industry_holding_count": 0,
            "lagging_holding_count": 0,
            "priority_review_count": 0,
            "momentum_exposure_gap_count": 0,
        },
        "momentum_exposure_gaps": [],
    }
    if industries.empty:
        return empty_map

    alerts = watchlist_alerts.copy() if watchlist_alerts is not None else pd.DataFrame()
    if not alerts.empty:
        alerts["ticker"] = alerts.get("ticker", "").fillna("").astype(str).str.upper()
        alerts["alert_level"] = alerts.get("alert_level", "").fillna("").astype(str)

    holdings = pd.DataFrame(columns=holding_columns)
    if not alerts.empty and not tickers.empty:
        ticker_details = tickers.drop(columns=["alert_level"], errors="ignore").copy()
        holding_tickers = alerts[["ticker", "alert_level"]].drop_duplicates(subset=["ticker"])
        holdings = holding_tickers.merge(ticker_details, on="ticker", how="left")
        for column in holding_columns:
            if column not in holdings.columns:
                holdings[column] = math.nan if column in ["return_10d", "relative_strength_vs_industry", "relative_volume"] else ""
        holdings = holdings[holding_columns].copy()
        for column in ["return_10d", "relative_strength_vs_industry", "relative_volume"]:
            holdings[column] = pd.to_numeric(holdings[column], errors="coerce")
        for column in ["ticker", "company_name", "industry_group", "watch_status", "alert_level", INDUSTRY_REGIME_COLUMN, INDUSTRY_RISK_FLAG_COLUMN]:
            holdings[column] = holdings[column].fillna("").astype(str)

    holdings_by_industry: dict[str, list[str]] = {}
    if not holdings.empty:
        for industry_group, group in holdings.groupby("industry_group", dropna=False):
            tickers_for_industry = sorted(group["ticker"].dropna().astype(str).tolist())
            holdings_by_industry[str(industry_group)] = tickers_for_industry

    industry_bars = industries[industry_columns].copy()
    industry_bars = industry_bars.sort_values("return_10d", ascending=False, na_position="last")
    industry_bars["holding_tickers"] = industry_bars["industry_group"].astype(str).map(
        lambda industry_group: holdings_by_industry.get(industry_group, [])
    )
    industry_bars["watchlist_tickers"] = industry_bars["holding_tickers"]

    strong_industries = industry_bars[
        industry_bars[INDUSTRY_REGIME_COLUMN].isin(["momentum_leader", "early_recovery"])
    ].copy()
    holding_industries = set()
    if not holdings.empty:
        holding_industries = set(holdings["industry_group"].dropna().astype(str))

    exposure_gaps = strong_industries[
        ~strong_industries["industry_group"].astype(str).isin(holding_industries)
    ].copy()
    if not exposure_gaps.empty:
        exposure_gaps["reason"] = "strong_or_recovering_industry_without_watchlist_exposure"

    lagging_count = 0
    priority_count = 0
    strong_holding_count = 0
    if not holdings.empty:
        lagging_count = int((holdings["relative_strength_vs_industry"] < -0.10).sum())
        priority_count = int(holdings["alert_level"].isin(["red", "orange"]).sum())
        strong_holding_count = int(
            holdings[INDUSTRY_REGIME_COLUMN].isin(["momentum_leader", "early_recovery"]).sum()
        )

    return {
        "industry_bars": dataframe_records(industry_bars),
        "holding_alignment": dataframe_records(holdings),
        "summary": {
            "strong_industry_holding_count": strong_holding_count,
            "lagging_holding_count": lagging_count,
            "priority_review_count": priority_count,
            "momentum_exposure_gap_count": int(len(exposure_gaps)),
        },
        "momentum_exposure_gaps": dataframe_records(
            exposure_gaps[
                [
                    "industry_group",
                    "return_10d",
                    "breadth_score",
                    INDUSTRY_REGIME_COLUMN,
                    INDUSTRY_RISK_FLAG_COLUMN,
                    "reason",
                ]
            ]
        ),
    }


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
    ticker_output: pd.DataFrame,
    industry_output: pd.DataFrame,
    rotation_history: pd.DataFrame,
    update_health_output: pd.DataFrame | None = None,
    watchlist_alerts: pd.DataFrame | None = None,
) -> dict[str, Any]:
    tickers = ticker_output.copy()
    industries = industry_output.copy()

    ticker_numeric_columns = [
        "data_points",
        "return_3d",
        "return_5d",
        "return_10d",
        "return_20d",
        "return_1m",
        "return_3m",
        "return_6m",
        "return_1y",
        "latest_volume",
        "avg_volume_20d",
        "relative_volume",
        "ma_5d",
        "ma_10d",
        "ma_20d",
        "max_drawdown_10d",
        "up_days_10d",
        "relative_strength_vs_industry",
        "industry_quality_score",
        "distance_from_20d_ma",
        "distance_from_52w_high",
        "position_in_52w_range",
        "forward_pe",
        "earnings_growth",
        "peg_ratio",
    ]
    for column in ticker_numeric_columns:
        if column in tickers.columns:
            tickers[column] = pd.to_numeric(tickers[column], errors="coerce")
    for column in ["forward_pe", "earnings_growth", "peg_ratio"]:
        if column not in tickers.columns:
            tickers[column] = math.nan

    for column in ["early_momentum_signal", "confirmed_momentum_signal", "strong_momentum_signal", "risk_warning"]:
        if column not in tickers.columns:
            tickers[column] = False
        tickers[column] = tickers[column].fillna(False).astype(bool)
    for column, default in [
        ("leader_type", "non_leader"),
        (INDUSTRY_REGIME_COLUMN, "neutral"),
        (INDUSTRY_RISK_FLAG_COLUMN, "none"),
        (ROTATION_TYPE_COLUMN, "unclear"),
        (CAUSAL_HYPOTHESIS_COLUMN, "unclear"),
        (EVIDENCE_STATUS_COLUMN, "needs_review"),
        ("short_term_price_zone", "neutral"),
        ("long_term_price_zone", "neutral"),
        ("price_zone", "neutral"),
        ("current_state", "sideways_base"),
        ("watch_status", "avoid_for_now"),
        ("data_status", "missing"),
        ("data_quality_note", ""),
        ("peg_rating", "unavailable"),
        ("peg_status", "missing_pe"),
        ("peg_note", ""),
    ]:
        if column not in tickers.columns:
            tickers[column] = default
        tickers[column] = tickers[column].fillna(default).astype(str)

    for column in INDUSTRY_TREND_COLUMNS:
        if column not in industries.columns:
            industries[column] = False if column == "momentum_exhaustion_warning" else math.nan
    for column in BREADTH_COLUMNS:
        if column not in industries.columns:
            industries[column] = math.nan
    if INDUSTRY_REGIME_COLUMN not in industries.columns:
        industries[INDUSTRY_REGIME_COLUMN] = "neutral"
    industries[INDUSTRY_REGIME_COLUMN] = industries[INDUSTRY_REGIME_COLUMN].fillna("neutral").astype(str)
    for column, default in [
        (INDUSTRY_RISK_FLAG_COLUMN, "none"),
        (ROTATION_TYPE_COLUMN, "unclear"),
        (CAUSAL_HYPOTHESIS_COLUMN, "unclear"),
        (EVIDENCE_STATUS_COLUMN, "needs_review"),
    ]:
        if column not in industries.columns:
            industries[column] = default
        industries[column] = industries[column].fillna(default).astype(str)

    numeric_industry_columns = [
        "ticker_count",
        "tickers_with_data",
        *METRIC_COLUMNS,
        *BREADTH_COLUMNS,
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
        "return_1m",
        "return_3m",
        "return_6m",
        "return_1y",
        "relative_volume",
        INDUSTRY_REGIME_COLUMN,
        INDUSTRY_RISK_FLAG_COLUMN,
        ROTATION_TYPE_COLUMN,
    ]
    industry_breadth_columns = [
        "industry_group",
        "ticker_count",
        "tickers_with_data",
        "return_5d",
        "return_10d",
        "positive_5d_pct",
        "positive_10d_pct",
        "confirmed_signal_pct",
        "strong_signal_pct",
        "high_relative_volume_pct",
        "breadth_score",
        INDUSTRY_REGIME_COLUMN,
        INDUSTRY_RISK_FLAG_COLUMN,
        ROTATION_TYPE_COLUMN,
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
        INDUSTRY_REGIME_COLUMN,
        INDUSTRY_RISK_FLAG_COLUMN,
        ROTATION_TYPE_COLUMN,
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
        "return_1m",
        "return_3m",
        "return_6m",
        "return_1y",
        "relative_volume",
        "max_drawdown_10d",
        "up_days_10d",
        "relative_strength_vs_industry",
        "early_momentum_signal",
        "risk_warning",
    ]
    leader_stock_columns = [
        "ticker",
        "company_name",
        "industry_group",
        INDUSTRY_REGIME_COLUMN,
        INDUSTRY_RISK_FLAG_COLUMN,
        ROTATION_TYPE_COLUMN,
        CAUSAL_HYPOTHESIS_COLUMN,
        EVIDENCE_STATUS_COLUMN,
        "leader_type",
        "industry_quality_score",
        "watch_status",
        "current_state",
        "short_term_price_zone",
        "long_term_price_zone",
        "price_zone",
        "return_5d",
        "return_10d",
        "return_20d",
        "return_1m",
        "return_3m",
        "return_6m",
        "return_1y",
        "relative_strength_vs_industry",
        "relative_volume",
        "max_drawdown_10d",
        "distance_from_20d_ma",
        "distance_from_52w_high",
        "position_in_52w_range",
        "risk_warning",
        *PEG_COLUMNS,
    ]
    leader_industry_columns = [
        "industry_group",
        INDUSTRY_REGIME_COLUMN,
        INDUSTRY_RISK_FLAG_COLUMN,
        ROTATION_TYPE_COLUMN,
        CAUSAL_HYPOTHESIS_COLUMN,
        EVIDENCE_STATUS_COLUMN,
        "return_5d",
        "return_10d",
        "breadth_score",
        "confirmed_signal_pct",
        "momentum_persistence",
        "momentum_exhaustion_warning",
    ]
    constituent_columns = [
        "ticker",
        "company_name",
        "industry_group",
        "latest_date",
        "data_points",
        "data_status",
        "leader_type",
        "industry_quality_score",
        "watch_status",
        "current_state",
        "return_5d",
        "return_10d",
        "return_20d",
        "return_1m",
        "return_3m",
        "return_6m",
        "return_1y",
        "relative_strength_vs_industry",
        "latest_volume",
        "avg_volume_20d",
        "relative_volume",
        *PEG_COLUMNS,
        "confirmed_momentum_signal",
        "strong_momentum_signal",
        "risk_warning",
    ]
    data_quality_columns = [
        "ticker",
        "company_name",
        "industry_group",
        "latest_date",
        "data_points",
        "data_status",
        "data_quality_note",
    ]

    tradable_tickers = tickers[tickers["data_points"] > 0] if "data_points" in tickers.columns else tickers
    latest_dates = tradable_tickers["latest_date"].dropna().astype(str) if "latest_date" in tradable_tickers.columns else []
    industry_trends = industries.copy()
    if "confirmed_signal_pct" not in industry_trends.columns or industry_trends["confirmed_signal_pct"].isna().all():
        industry_trends = industry_trends.drop(columns=["confirmed_signal_pct"], errors="ignore")
        industry_trends = industry_trends.merge(
            confirmed_by_industry[["industry_group", "confirmed_signal_pct"]], on="industry_group", how="left"
        )
    if "momentum_exhaustion_warning" in industry_trends.columns:
        industry_trends["momentum_exhaustion_warning"] = (
            industry_trends["momentum_exhaustion_warning"].fillna(False).astype(str).str.lower().isin(["true", "1"])
        )
    breadth_industries = industries[industry_breadth_columns].copy() if not industries.empty else pd.DataFrame()
    if not breadth_industries.empty:
        median_return_10d = breadth_industries["return_10d"].median()
        median_breadth_score = breadth_industries["breadth_score"].median()
        ranked_breadth = breadth_industries.sort_values("breadth_score", ascending=False, na_position="last")
        high_return_weak_breadth = breadth_industries[
            (breadth_industries["return_10d"] > median_return_10d)
            & (breadth_industries["breadth_score"] < median_breadth_score)
        ].sort_values(["return_10d", "breadth_score"], ascending=[False, True], na_position="last")
        moderate_return_improving_breadth = breadth_industries[
            (breadth_industries["return_10d"] > 0)
            & (breadth_industries["return_10d"] <= median_return_10d)
            & (breadth_industries["positive_5d_pct"] > breadth_industries["positive_10d_pct"])
        ].sort_values(["positive_5d_pct", "breadth_score", "return_5d"], ascending=[False, False, False])
    else:
        ranked_breadth = pd.DataFrame(columns=industry_breadth_columns)
        high_return_weak_breadth = pd.DataFrame(columns=industry_breadth_columns)
        moderate_return_improving_breadth = pd.DataFrame(columns=industry_breadth_columns)
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
    leader_research_candidates = tradable_tickers[
        tradable_tickers["watch_status"] == "research_candidate"
    ].sort_values(
        ["industry_quality_score", "relative_strength_vs_industry", "return_10d"],
        ascending=[False, False, False],
        na_position="last",
    )
    leader_wait_for_stabilization = tradable_tickers[
        tradable_tickers["watch_status"] == "wait_for_stabilization"
    ].sort_values(
        ["industry_quality_score", "return_5d", "relative_volume"],
        ascending=[False, False, False],
        na_position="last",
    )
    leader_too_extended = tradable_tickers[tradable_tickers["watch_status"] == "too_extended"].sort_values(
        ["distance_from_20d_ma", "position_in_52w_range", "return_10d"],
        ascending=[False, False, False],
        na_position="last",
    )
    not_eligible_industries = industries[industries[INDUSTRY_REGIME_COLUMN].isin(["neutral", "weak"])].sort_values(
        ["return_10d", "breadth_score"], ascending=[False, False], na_position="last"
    )
    industry_constituents = {}
    if not tickers.empty:
        for industry_group, industry_tickers in tickers.groupby("industry_group", dropna=False):
            ranked_tickers = industry_tickers.sort_values(
                ["data_points", "return_10d", "relative_volume", "ticker"],
                ascending=[False, False, False, True],
                na_position="last",
            )
            industry_constituents[str(industry_group)] = dataframe_records(ranked_tickers[constituent_columns])
    data_quality_summary = build_data_quality_summary(tickers)
    data_quality_issues = tickers[tickers["data_status"].isin(["missing", "stale"])].sort_values(
        ["data_status", "ticker"], ascending=[True, True], na_position="last"
    )
    if update_health_output is None:
        update_health_output = build_update_health_output(tickers)
    update_health_records = dataframe_records(update_health_output) if not update_health_output.empty else []
    update_health = update_health_records[0] if update_health_records else {}
    watchlist_alert_records = dataframe_records(watchlist_alerts) if watchlist_alerts is not None and not watchlist_alerts.empty else []

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
        "momentum_map": build_momentum_map(tickers, industries, watchlist_alerts),
        "daily_brief": build_daily_brief(tickers, industries, update_health_output, watchlist_alerts),
        "watchlist_alerts": watchlist_alert_records,
        "data_quality": {
            "summary": data_quality_summary,
            "issue_tickers": dataframe_records(data_quality_issues[data_quality_columns]),
        },
        "update_health": update_health,
        "industry_momentum": dataframe_records(
            industries[industry_momentum_columns].sort_values("return_10d", ascending=False, na_position="last")
        )
        if not industries.empty
        else [],
        "industry_confirmed": dataframe_records(confirmed_by_industry),
        "industry_breadth": {
            "ranked": dataframe_records(ranked_breadth[industry_breadth_columns]),
            "high_return_weak_breadth": dataframe_records(high_return_weak_breadth[industry_breadth_columns].head(10)),
            "moderate_return_improving_breadth": dataframe_records(
                moderate_return_improving_breadth[industry_breadth_columns].head(10)
            ),
        },
        "industry_trend_intelligence": {
            "strongest_improving": dataframe_records(strongest_improving[industry_trend_columns].head(10)),
            "strongest_persistent": dataframe_records(strongest_persistent[industry_trend_columns].head(10)),
            "momentum_exhaustion": dataframe_records(exhaustion[industry_trend_columns].head(10)),
            "momentum_recovery": dataframe_records(momentum_recovery[industry_trend_columns].head(10)),
        },
        "leader_accumulation": {
            "research_candidates": dataframe_records(leader_research_candidates[leader_stock_columns].head(20)),
            "wait_for_stabilization": dataframe_records(leader_wait_for_stabilization[leader_stock_columns].head(20)),
            "too_extended_leaders": dataframe_records(leader_too_extended[leader_stock_columns].head(20)),
            "not_eligible_industries": dataframe_records(not_eligible_industries[leader_industry_columns]),
        },
        "industry_constituents": industry_constituents,
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
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>股票動能追蹤</title>
  <style>
    html {
      scroll-behavior: smooth;
    }

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

    .dashboard-nav {
      position: sticky;
      top: 0;
      z-index: 20;
      margin: -8px 0 18px;
      padding: 8px 0;
      background: linear-gradient(180deg, rgba(246, 248, 247, 0.98), rgba(246, 248, 247, 0.92));
      backdrop-filter: blur(8px);
    }

    .nav-panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.94);
      padding: 10px;
      box-shadow: var(--shadow);
    }

    .section-nav-select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--surface);
      color: var(--ink);
      padding: 9px 10px;
      font: inherit;
    }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(6, minmax(130px, 1fr));
      gap: 10px;
      margin-bottom: 22px;
    }

    .summary-grid,
    .momentum-map-section,
    .watchlist-alert-panel,
    .dashboard-section {
      scroll-margin-top: 170px;
    }

    .momentum-map-section {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      box-shadow: var(--shadow);
      margin: 0 0 18px;
      padding: 16px;
    }

    .momentum-map-summary {
      display: grid;
      grid-template-columns: repeat(4, minmax(130px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }

    .momentum-map-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(340px, 0.85fr);
      gap: 12px;
      align-items: stretch;
    }

    .momentum-map-panel {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfb;
      padding: 12px;
    }

    .momentum-map-panel-header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }

    .momentum-bars {
      display: grid;
      gap: 3px;
    }

    .momentum-bar-row {
      display: grid;
      grid-template-columns: minmax(118px, 168px) minmax(0, 1fr);
      gap: 8px;
      align-items: center;
      min-height: 24px;
      border-bottom: 1px solid rgba(217, 224, 220, 0.72);
      padding: 2px 0;
    }

    .momentum-bar-row.has-risk {
      border-bottom-color: rgba(180, 83, 9, 0.5);
    }

    .momentum-industry-label {
      min-width: 0;
      font-size: 12px;
      font-weight: 760;
      overflow-wrap: anywhere;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .momentum-industry-meta {
      display: none;
    }

    .momentum-inline-tickers {
      margin-left: 4px;
      color: var(--blue);
      font-size: 10px;
      font-weight: 820;
    }

    .momentum-bar-track {
      position: relative;
      min-height: 18px;
      border-radius: 5px;
      background: #eef2ef;
      overflow: hidden;
    }

    .momentum-zero-line {
      position: absolute;
      top: 0;
      bottom: 0;
      width: 1px;
      background: rgba(23, 33, 29, 0.28);
    }

    .momentum-bar-fill {
      position: absolute;
      top: 0;
      bottom: 0;
      min-width: 2px;
      border-radius: 5px;
      background: var(--green);
      opacity: 0.88;
    }

    .momentum-bar-fill.early_recovery {
      background: var(--teal);
    }

    .momentum-bar-fill.neutral {
      background: #8a9993;
    }

    .momentum-bar-fill.weak {
      background: #b98989;
    }

    .momentum-bar-value {
      position: absolute;
      right: 8px;
      top: 50%;
      transform: translateY(-50%);
      font-size: 11px;
      font-weight: 780;
      color: var(--ink);
      text-shadow: 0 1px 0 rgba(255, 255, 255, 0.85);
    }

    .momentum-breadth-marker {
      position: absolute;
      top: 3px;
      bottom: 3px;
      width: 2px;
      border-radius: 999px;
      background: #17211d;
      opacity: 0.46;
    }

    .momentum-chip-row {
      display: flex;
      flex-wrap: nowrap;
      gap: 3px;
      margin-top: 3px;
      overflow: hidden;
    }

    .momentum-ticker-chip {
      display: inline-flex;
      align-items: center;
      border: 1px solid rgba(29, 78, 216, 0.24);
      border-radius: 999px;
      background: rgba(239, 246, 255, 0.9);
      color: var(--blue);
      font-size: 9px;
      font-weight: 780;
      line-height: 1;
      padding: 2px 5px;
      white-space: nowrap;
    }

    .momentum-risk-badge {
      display: inline-flex;
      margin-left: 5px;
      border-radius: 999px;
      background: #fef3c7;
      color: #92400e;
      font-size: 9px;
      font-weight: 780;
      padding: 1px 4px;
      vertical-align: middle;
    }

    .momentum-scatter {
      position: relative;
      min-height: 430px;
      border: 1px solid rgba(217, 224, 220, 0.8);
      border-radius: 8px;
      background:
        linear-gradient(90deg, transparent calc(50% - 0.5px), rgba(23, 33, 29, 0.24) calc(50% - 0.5px), rgba(23, 33, 29, 0.24) calc(50% + 0.5px), transparent calc(50% + 0.5px)),
        linear-gradient(0deg, transparent calc(50% - 0.5px), rgba(23, 33, 29, 0.24) calc(50% - 0.5px), rgba(23, 33, 29, 0.24) calc(50% + 0.5px), transparent calc(50% + 0.5px)),
        #ffffff;
      overflow: hidden;
    }

    .momentum-scatter-axis {
      position: absolute;
      color: var(--muted);
      font-size: 11px;
      font-weight: 680;
      pointer-events: none;
    }

    .momentum-scatter-axis.x {
      right: 10px;
      bottom: 8px;
    }

    .momentum-scatter-axis.y {
      left: 9px;
      top: 8px;
    }

    .momentum-point {
      position: absolute;
      transform: translate(-50%, -50%);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 34px;
      height: 24px;
      border: 1px solid rgba(23, 33, 29, 0.22);
      border-radius: 999px;
      background: var(--surface);
      color: var(--ink);
      font-size: 11px;
      font-weight: 790;
      box-shadow: 0 7px 18px rgba(23, 33, 29, 0.12);
      padding: 0 7px;
      white-space: nowrap;
    }

    .momentum-point.red {
      border-color: rgba(185, 28, 28, 0.35);
      background: #fee2e2;
      color: var(--red);
    }

    .momentum-point.orange,
    .momentum-point.yellow {
      border-color: rgba(180, 83, 9, 0.35);
      background: #fef3c7;
      color: #92400e;
    }

    .momentum-point.green {
      border-color: rgba(4, 120, 87, 0.30);
      background: #dcfce7;
      color: var(--green);
    }

    .momentum-point:hover::after,
    .momentum-point:focus::after {
      content: attr(data-caption);
      position: absolute;
      left: 50%;
      bottom: calc(100% + 8px);
      transform: translateX(-50%);
      z-index: 45;
      width: min(280px, 72vw);
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      background: var(--surface);
      box-shadow: 0 16px 38px rgba(23, 33, 29, 0.18);
      color: var(--ink);
      font-size: 12px;
      font-weight: 620;
      line-height: 1.45;
      padding: 9px 10px;
      white-space: normal;
    }

    .momentum-gap-list {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 10px;
    }

    .momentum-gap-chip {
      border: 1px solid rgba(15, 118, 110, 0.24);
      border-radius: 999px;
      background: rgba(240, 253, 250, 0.9);
      color: var(--teal);
      font-size: 12px;
      font-weight: 760;
      padding: 5px 8px;
    }

    .watchlist-alert-panel {
      border: 1px solid rgba(194, 111, 0, 0.34);
      border-radius: 8px;
      background: #fffaf0;
      box-shadow: var(--shadow);
      margin: 0 0 18px;
      padding: 16px;
    }

    .watchlist-alert-panel.healthy {
      border-color: rgba(0, 122, 85, 0.28);
      background: #f0fbf6;
    }

    .watchlist-alert-panel.unknown {
      border-color: rgba(96, 111, 108, 0.28);
      background: #f7f9f8;
    }

    .watchlist-alert-header {
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 12px;
    }

    .watchlist-alert-kicker {
      color: var(--muted);
      font-size: 12px;
      font-weight: 780;
    }

    .watchlist-alert-title {
      margin-top: 4px;
      font-size: 22px;
      font-weight: 780;
      line-height: 1.25;
    }

    .watchlist-alert-subtitle {
      margin-top: 5px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }

    .watchlist-alert-counts {
      display: flex;
      flex-wrap: wrap;
      justify-content: end;
      gap: 6px;
    }

    .alert-pill,
    .alert-level {
      display: inline-flex;
      align-items: center;
      width: fit-content;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 780;
      line-height: 1;
      padding: 5px 8px;
      white-space: nowrap;
    }

    .alert-pill.red,
    .alert-level.red {
      background: #fee2e2;
      color: var(--red);
    }

    .alert-pill.orange,
    .alert-level.orange {
      background: #ffedd5;
      color: var(--amber);
    }

    .alert-pill.yellow,
    .alert-level.yellow {
      background: #fef3c7;
      color: #92400e;
    }

    .alert-pill.green,
    .alert-level.green {
      background: #dcfce7;
      color: var(--green);
    }

    .alert-pill.unknown,
    .alert-level.unknown {
      background: #e5e7eb;
      color: var(--muted);
    }

    .watchlist-alert-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .watchlist-alert-card {
      min-width: 0;
      border: 1px solid rgba(217, 224, 220, 0.9);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.78);
      padding: 13px;
    }

    .watchlist-alert-card.red {
      border-color: rgba(185, 28, 28, 0.28);
      background: rgba(254, 242, 242, 0.86);
    }

    .watchlist-alert-card.orange {
      border-color: rgba(180, 83, 9, 0.30);
      background: rgba(255, 247, 237, 0.9);
    }

    .watchlist-alert-card.yellow {
      border-color: rgba(146, 64, 14, 0.26);
      background: rgba(255, 251, 235, 0.88);
    }

    .watchlist-alert-card.green {
      border-color: rgba(4, 120, 87, 0.24);
      background: rgba(240, 253, 244, 0.86);
    }

    .watchlist-alert-card-header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 9px;
    }

    .watchlist-alert-ticker {
      min-width: 0;
      font-size: 20px;
      font-weight: 790;
      overflow-wrap: anywhere;
    }

    .watchlist-alert-meta,
    .watchlist-alert-reason {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
      overflow-wrap: anywhere;
    }

    .watchlist-alert-reason {
      margin-top: 8px;
      color: var(--ink);
      font-weight: 650;
    }

    .watchlist-alert-route {
      border: 1px solid rgba(217, 224, 220, 0.9);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.78);
      color: var(--ink);
      font-size: 13px;
      font-weight: 650;
      line-height: 1.45;
      margin: 0 0 10px;
      overflow-wrap: anywhere;
      padding: 11px 12px;
    }

    .candidate-chip {
      position: relative;
      display: inline-flex;
      align-items: center;
      border: 1px solid rgba(77, 116, 96, 0.28);
      border-radius: 999px;
      background: rgba(237, 247, 241, 0.92);
      color: #183a2b;
      font-size: 12px;
      font-weight: 780;
      line-height: 1;
      margin: 0 2px;
      padding: 4px 7px;
      white-space: nowrap;
    }

    .candidate-chip:focus {
      outline: 2px solid rgba(42, 103, 72, 0.34);
      outline-offset: 2px;
    }

    .candidate-chip:hover::after,
    .candidate-chip:focus::after {
      content: attr(data-caption);
      position: absolute;
      left: 0;
      top: calc(100% + 8px);
      z-index: 40;
      width: min(360px, 78vw);
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      background: var(--surface);
      box-shadow: 0 16px 38px rgba(23, 33, 29, 0.18);
      color: var(--ink);
      font-size: 12px;
      font-weight: 620;
      line-height: 1.45;
      padding: 10px 11px;
      white-space: normal;
    }

    .watchlist-alert-metrics {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 9px;
    }

    .watchlist-alert-metric {
      border: 1px solid rgba(217, 224, 220, 0.86);
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.72);
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      padding: 5px 7px;
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

    .summary-description {
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }

    .daily-brief-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
    }

    .daily-brief-card {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      padding: 14px;
      box-shadow: var(--shadow);
    }

    .daily-brief-card.healthy {
      border-color: rgba(0, 122, 85, 0.28);
      background: rgba(235, 249, 243, 0.72);
    }

    .daily-brief-card.warning {
      border-color: rgba(194, 111, 0, 0.35);
      background: rgba(255, 248, 229, 0.76);
    }

    .daily-brief-card.unknown {
      border-color: rgba(96, 111, 108, 0.32);
      background: rgba(242, 245, 244, 0.88);
    }

    .daily-brief-title {
      color: var(--muted);
      font-size: 12px;
      font-weight: 780;
    }

    .daily-brief-headline {
      margin-top: 8px;
      color: var(--ink);
      font-size: 15px;
      font-weight: 760;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }

    .daily-brief-list {
      display: grid;
      gap: 6px;
      margin: 10px 0 0;
      padding: 0;
      list-style: none;
    }

    .daily-brief-list li {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
      overflow-wrap: anywhere;
    }

    .quality-grid {
      display: grid;
      grid-template-columns: repeat(6, minmax(130px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }

    .quality-tile {
      min-height: 70px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      padding: 12px 14px;
      box-shadow: var(--shadow);
    }

    .quality-tile.warning {
      border-color: rgba(194, 111, 0, 0.35);
      background: rgba(255, 248, 229, 0.86);
    }

    .quality-tile.healthy {
      border-color: rgba(0, 122, 85, 0.28);
      background: rgba(235, 249, 243, 0.82);
    }

    .quality-tile.unknown {
      border-color: rgba(96, 111, 108, 0.32);
      background: rgba(242, 245, 244, 0.9);
    }

    .quality-label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 760;
    }

    .quality-value {
      margin-top: 6px;
      font-size: 18px;
      font-weight: 760;
    }

    .quality-description {
      margin-top: 5px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }

    .quality-value a {
      color: var(--green);
      text-decoration: none;
    }

    .quality-value a:hover {
      text-decoration: underline;
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

    .section-note {
      margin: -4px 0 10px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }

    .field-help {
      display: flex;
      flex-wrap: wrap;
      align-items: start;
      gap: 6px 8px;
      margin: -2px 0 10px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }

    .field-help[hidden] {
      display: none;
    }

    .field-help-intro {
      padding: 5px 0;
      color: var(--ink);
      font-weight: 760;
      white-space: nowrap;
    }

    .field-help-item {
      display: inline-flex;
      gap: 5px;
      max-width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.72);
      padding: 4px 7px;
    }

    .field-help-term {
      color: var(--ink);
      font-weight: 760;
      white-space: nowrap;
    }

    .field-help-description {
      min-width: 0;
      overflow-wrap: anywhere;
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

    .mobile-card-list {
      display: none;
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

    .industry-link {
      appearance: none;
      border: 0;
      background: transparent;
      color: var(--teal);
      cursor: pointer;
      font: inherit;
      font-weight: 720;
      padding: 0;
      text-align: left;
      text-decoration: underline;
      text-decoration-color: rgba(15, 118, 110, 0.35);
      text-underline-offset: 3px;
    }

    .industry-link:hover,
    .industry-link:focus-visible {
      color: var(--blue);
      text-decoration-color: currentColor;
    }

    .mixed-signal-row td {
      background: #fff7ed;
    }

    .mixed-signal-row:hover td {
      background: #ffedd5;
    }

    .mixed-signal-badge {
      display: inline-flex;
      align-items: center;
      width: fit-content;
      border: 1px solid rgba(180, 83, 9, 0.35);
      border-radius: 999px;
      background: #ffedd5;
      color: var(--amber);
      font-size: 11px;
      font-weight: 780;
      line-height: 1;
      margin-left: 7px;
      padding: 3px 6px;
      vertical-align: middle;
      white-space: nowrap;
    }

    .mixed-signal-card {
      border-color: rgba(180, 83, 9, 0.42);
      background: linear-gradient(0deg, rgba(255, 237, 213, 0.58), rgba(255, 255, 255, 0.96));
    }

    .rotation-block .table-wrap,
    .intelligence-block .table-wrap {
      overflow-x: visible;
    }

    .rotation-block table,
    .intelligence-block table {
      width: 100%;
      min-width: 0;
      table-layout: fixed;
    }

    .rotation-block th,
    .rotation-block td,
    .intelligence-block th,
    .intelligence-block td {
      padding: 8px 7px;
      font-size: 12px;
      line-height: 1.3;
      white-space: normal;
      overflow-wrap: anywhere;
    }

    .rotation-block th,
    .intelligence-block th {
      font-size: 11px;
    }

    .rotation-block .rank,
    .intelligence-block .rank {
      width: auto;
    }

    .rotation-block .field-help,
    .intelligence-block .field-help {
      margin: -2px 0 8px;
      font-size: 11px;
      gap: 5px;
    }

    .rotation-block .field-help-item,
    .intelligence-block .field-help-item {
      flex: 1 1 150px;
      padding: 4px 6px;
    }

    .mobile-row-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      padding: 13px 14px;
      box-shadow: var(--shadow);
    }

    .mobile-card-header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 9px;
    }

    .mobile-card-title {
      min-width: 0;
      font-size: 15px;
      font-weight: 760;
      overflow-wrap: anywhere;
    }

    .mobile-card-rank {
      color: var(--muted);
      font-size: 12px;
      font-weight: 760;
      white-space: nowrap;
    }

    .mobile-card-subtitle {
      margin: -3px 0 11px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }

    .mobile-metrics {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px 12px;
    }

    .mobile-metric {
      min-width: 0;
      border-top: 1px solid var(--line);
      padding-top: 7px;
    }

    .mobile-metric-label {
      color: var(--muted);
      font-size: 11px;
      font-weight: 740;
      text-transform: uppercase;
    }

    .mobile-metric-value {
      margin-top: 2px;
      font-size: 14px;
      font-weight: 680;
      overflow-wrap: anywhere;
    }

    .mobile-metric-hint {
      margin-top: 3px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.3;
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

    .industry-modal {
      position: fixed;
      inset: 0;
      z-index: 20;
      display: grid;
      place-items: center;
      padding: 22px;
    }

    .industry-modal[hidden] {
      display: none;
    }

    .industry-modal-backdrop {
      position: absolute;
      inset: 0;
      background: rgba(23, 33, 29, 0.42);
    }

    .industry-modal-panel {
      position: relative;
      z-index: 1;
      display: grid;
      grid-template-rows: auto 1fr;
      width: min(1040px, 100%);
      max-height: min(86vh, 780px);
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      background: var(--surface);
      box-shadow: 0 24px 80px rgba(23, 33, 29, 0.26);
      overflow: hidden;
    }

    .industry-modal-header {
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 18px;
      border-bottom: 1px solid var(--line);
      background: var(--surface-soft);
      padding: 16px 18px;
    }

    .industry-modal-title {
      font-size: 20px;
      font-weight: 780;
    }

    .industry-modal-subtitle {
      margin-top: 4px;
      color: var(--muted);
      font-size: 13px;
    }

    .industry-modal-close {
      flex: 0 0 auto;
      width: 32px;
      height: 32px;
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      background: var(--surface);
      color: var(--ink);
      cursor: pointer;
      font-size: 20px;
      line-height: 1;
    }

    .industry-modal-body {
      min-height: 0;
      overflow: auto;
      padding: 14px 18px 18px;
    }

    .industry-modal-summary {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
      color: var(--muted);
      font-size: 12px;
    }

    .industry-modal-pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(238, 244, 242, 0.7);
      padding: 5px 8px;
    }

    .industry-modal-table-wrap {
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }

    .industry-modal-table {
      min-width: 760px;
      box-shadow: none;
    }

    .industry-modal-table th,
    .industry-modal-table td {
      padding: 8px 9px;
      font-size: 12px;
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

      .quality-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      .momentum-map-summary {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      .momentum-map-grid {
        grid-template-columns: 1fr;
      }

      .daily-brief-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      .watchlist-alert-grid {
        grid-template-columns: 1fr;
      }

      .summary-grid,
      .watchlist-alert-panel,
      .dashboard-section {
        scroll-margin-top: 220px;
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

      .momentum-map-section {
        padding: 12px;
      }

      .momentum-map-summary {
        grid-template-columns: 1fr;
      }

      .momentum-bar-row {
        grid-template-columns: 1fr;
      }

      .momentum-scatter {
        min-height: 360px;
      }

      .watchlist-alert-header {
        flex-direction: column;
      }

      .watchlist-alert-counts {
        justify-content: start;
      }

      .industry-modal {
        padding: 10px;
      }

      .industry-modal-panel {
        max-height: 92vh;
      }

      .industry-modal-header {
        padding: 13px 14px;
      }

      .industry-modal-body {
        padding: 12px 14px 14px;
      }
    }

    @media (max-width: 640px) {
      .dashboard-nav {
        margin-top: -10px;
        position: sticky;
        top: 8px;
        z-index: 40;
      }
      .daily-brief-grid {
        grid-template-columns: 1fr;
      }

      .summary-grid,
      .watchlist-alert-panel,
      .dashboard-section {
        scroll-margin-top: 120px;
      }

      .field-help {
        display: none;
      }

      .table-wrap {
        display: none;
      }

      .mobile-card-list:not([hidden]) {
        display: grid;
        gap: 10px;
      }

      .mobile-metrics {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }

    @media (max-width: 360px) {
      .mobile-metrics {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <header>
    <div class="header-inner">
      <div>
        <h1>股票動能追蹤</h1>
        <p class="subhead">由最新 CSV 輸出產生的靜態動能訊號儀表板。</p>
      </div>
      <p class="timestamp" id="generated-date"></p>
    </div>
  </header>

  <main>
    <nav class="dashboard-nav" aria-label="Dashboard 區塊導覽">
      <div class="nav-panel">
        <select class="section-nav-select" id="section-jump" aria-label="跳到 dashboard 區塊">
          <option value="">跳到區塊</option>
          <option value="#momentum-map">動能地圖</option>
          <option value="#watchlist-alert">追蹤名單轉換提醒</option>
          <option value="#overview">總覽</option>
          <option value="#daily-brief">每日重點</option>
          <option value="#update-health">更新健康</option>
          <option value="#data-quality">資料品質</option>
          <option value="#industry-momentum">產業動能</option>
          <option value="#industry-confirmed">確認比例</option>
          <option value="#industry-breadth">產業廣度</option>
          <option value="#rotation-trend">輪動趨勢</option>
          <option value="#trend-intelligence">趨勢判讀</option>
          <option value="#leader-filter">領導股篩選</option>
          <option value="#portfolio-simulator">追蹤名單檢查</option>
          <option value="#relative-strength">相對強度</option>
          <option value="#early-momentum">早期動能</option>
          <option value="#strong-momentum">強勢動能</option>
          <option value="#risk-warning">風險提醒</option>
        </select>
      </div>
    </nav>

    <section class="momentum-map-section" id="momentum-map" aria-label="動能地圖">
      <div class="section-heading">
        <h2>動能地圖</h2>
      </div>
      <div class="momentum-map-summary" id="momentum-map-summary"></div>
      <div class="momentum-map-grid">
        <section class="momentum-map-panel" aria-label="產業動能長條圖">
          <div class="momentum-map-panel-header">
            <h3>產業動能</h3>
            <span class="row-count" id="momentum-map-industry-count"></span>
          </div>
          <div class="momentum-bars" id="momentum-industry-bars"></div>
        </section>
        <section class="momentum-map-panel" aria-label="持股相對產業位置">
          <div class="momentum-map-panel-header">
            <h3>持股相對產業</h3>
            <span class="row-count" id="momentum-map-holding-count"></span>
          </div>
          <div class="momentum-scatter" id="momentum-alignment-scatter"></div>
        </section>
      </div>
      <div class="momentum-gap-list" id="momentum-gap-list"></div>
    </section>

    <section class="watchlist-alert-panel" id="watchlist-alert" aria-label="追蹤名單轉換提醒">
      <div class="watchlist-alert-header">
        <div>
          <div class="watchlist-alert-kicker">開盤前優先資訊</div>
          <h2 class="watchlist-alert-title" id="watchlist-alert-title">追蹤名單轉換提醒</h2>
          <p class="watchlist-alert-subtitle" id="watchlist-alert-subtitle"></p>
        </div>
        <div class="watchlist-alert-counts" id="watchlist-alert-counts"></div>
      </div>
      <div class="watchlist-alert-route" id="watchlist-alert-route"></div>
      <div class="watchlist-alert-grid" id="watchlist-alert-grid"></div>
    </section>

    <section class="summary-grid" id="overview" data-summary-grid aria-label="動能摘要"></section>

    <section class="dashboard-section daily-brief-section" id="daily-brief">
      <div class="section-heading">
        <h2>每日重點摘要</h2>
      </div>
      <p class="section-note">把資料健康、產業主線、輪動變化、研究候選與風險焦點濃縮成第一眼判讀；這是既有輸出的確定性彙整，不是新指標，也不是投資建議。</p>
      <div class="daily-brief-grid" id="daily-brief-grid"></div>
    </section>

    <section class="dashboard-section update-health-section" id="update-health">
      <div class="section-heading">
        <h2>更新健康狀態</h2>
      </div>
      <p class="section-note">這張表回答：排程是否有跑、資料是否新鮮、這次輸出是否可追溯？這不是投資訊號。</p>
      <div class="quality-grid" id="update-health-grid"></div>
    </section>

    <section class="dashboard-section data-quality-section" id="data-quality">
      <div class="section-heading">
        <h2>資料來源與品質</h2>
        <span class="row-count" data-count-for="data-quality-issues"></span>
      </div>
      <p class="section-note">這張表回答：資料從哪裡來、這次抓得完整嗎、哪些 ticker 需要注意？資料來自 Yahoo Finance via yfinance，適合研究與觀察，不是機構級資料源，也不是正式交易或投資建議。</p>
      <div class="quality-grid" id="data-quality-grid"></div>
      <div class="table-wrap">
        <table data-table="data-quality-issues"></table>
      </div>
      <p class="empty-state" data-empty-for="data-quality-issues" hidden>目前沒有缺資料或資料落後的 ticker。</p>
    </section>

    <section class="dashboard-section" id="industry-momentum">
      <div class="section-heading">
        <h2>產業動能排名：依平均 10 日報酬排序</h2>
        <span class="row-count" data-count-for="industry-momentum"></span>
      </div>
      <p class="section-note">這張表回答：現在價格動能集中在哪些產業？排序仍依平均 10 日報酬；欄位按 5 日、10 日、20 日排列，方便比較短中期動能。</p>
      <div class="table-wrap">
        <table data-table="industry-momentum"></table>
      </div>
      <p class="empty-state" data-empty-for="industry-momentum" hidden>目前沒有產業資料。</p>
    </section>

    <section class="dashboard-section" id="industry-confirmed">
      <div class="section-heading">
        <h2>產業確認動能比例排名</h2>
        <span class="row-count" data-count-for="industry-confirmed"></span>
      </div>
      <p class="section-note">這張表回答：產業內部有多少股票已經符合確認動能？比例越高代表產業內部動能越一致。</p>
      <div class="table-wrap">
        <table data-table="industry-confirmed"></table>
      </div>
      <p class="empty-state" data-empty-for="industry-confirmed" hidden>目前沒有產業訊號資料。</p>
    </section>

    <section class="dashboard-section" id="industry-breadth">
      <div class="section-heading">
        <h2>產業廣度</h2>
        <span class="row-count" id="breadth-status"></span>
      </div>
      <p class="section-note">這張表回答：產業是多數股票一起轉強，還是少數股票拉高平均？廣度越高，產業動能越不依賴單一領頭股。</p>
      <div class="intelligence-grid">
        <div class="intelligence-block">
          <h3>依廣度分數排名的產業</h3>
          <div class="table-wrap">
            <table data-table="breadth-ranked"></table>
          </div>
          <p class="empty-state" data-empty-for="breadth-ranked" hidden>目前沒有產業廣度資料。</p>
        </div>
        <div class="intelligence-block">
          <h3>高報酬但廣度偏弱</h3>
          <div class="table-wrap">
            <table data-table="breadth-high-return-weak"></table>
          </div>
          <p class="empty-state" data-empty-for="breadth-high-return-weak" hidden>目前沒有高報酬但廣度偏弱的產業。</p>
        </div>
        <div class="intelligence-block">
          <h3>報酬中等但廣度改善</h3>
          <div class="table-wrap">
            <table data-table="breadth-moderate-improving"></table>
          </div>
          <p class="empty-state" data-empty-for="breadth-moderate-improving" hidden>目前沒有報酬中等但廣度改善的產業。</p>
        </div>
      </div>
    </section>

    <section class="dashboard-section" id="rotation-trend">
      <div class="section-heading">
        <h2>產業輪動趨勢</h2>
        <span class="row-count" id="rotation-history-status"></span>
      </div>
      <p class="section-note">這張表回答：哪些產業正在上升或退場？比較歷史快照中的排名與平均報酬變化；至少需要兩個不同日期的快照才會出現完整趨勢。</p>
      <div class="rotation-grid">
        <div class="rotation-block">
          <h3>排名上升的產業</h3>
          <div class="table-wrap">
            <table data-table="rotation-gaining"></table>
          </div>
          <p class="empty-state" data-empty-for="rotation-gaining" hidden>目前沒有排名上升的產業。</p>
        </div>
        <div class="rotation-block">
          <h3>排名下滑的產業</h3>
          <div class="table-wrap">
            <table data-table="rotation-losing"></table>
          </div>
          <p class="empty-state" data-empty-for="rotation-losing" hidden>目前沒有排名下滑的產業。</p>
        </div>
        <div class="rotation-block">
          <h3>近 5 個交易日改善最強</h3>
          <div class="table-wrap">
            <table data-table="rotation-improving-5d"></table>
          </div>
          <p class="empty-state" data-empty-for="rotation-improving-5d" hidden>目前沒有 5 日改善資料。</p>
        </div>
      </div>
    </section>

    <section class="dashboard-section" id="trend-intelligence">
      <div class="section-heading">
        <h2>產業趨勢判讀</h2>
        <span class="row-count" id="trend-intelligence-status"></span>
      </div>
      <p class="section-note">這張表回答：動能是否持續、加速、或出現衰竭？用排名變化、持續性、加速與衰竭警示輔助判斷產業動能品質。</p>
      <div class="intelligence-grid">
        <div class="intelligence-block">
          <h3>改善最強的產業</h3>
          <div class="table-wrap">
            <table data-table="trend-strongest-improving"></table>
          </div>
          <p class="empty-state" data-empty-for="trend-strongest-improving" hidden>目前沒有改善中的產業。</p>
        </div>
        <div class="intelligence-block">
          <h3>持續領先的產業</h3>
          <div class="table-wrap">
            <table data-table="trend-strongest-persistent"></table>
          </div>
          <p class="empty-state" data-empty-for="trend-strongest-persistent" hidden>目前沒有持續維持前三名的產業。</p>
        </div>
        <div class="intelligence-block">
          <h3>可能動能衰竭</h3>
          <div class="table-wrap">
            <table data-table="trend-momentum-exhaustion"></table>
          </div>
          <p class="empty-state" data-empty-for="trend-momentum-exhaustion" hidden>目前沒有衰竭警示。</p>
        </div>
        <div class="intelligence-block">
          <h3>最新動能修復</h3>
          <div class="table-wrap">
            <table data-table="trend-momentum-recovery"></table>
          </div>
          <p class="empty-state" data-empty-for="trend-momentum-recovery" hidden>目前沒有動能修復產業。</p>
        </div>
      </div>
    </section>

    <section class="dashboard-section" id="leader-filter">
      <div class="section-heading">
        <h2>領導股累積篩選</h2>
        <span class="row-count" id="leader-accumulation-status"></span>
      </div>
      <p class="section-note">這張表回答：在領先或修復產業中，哪些標的適合進一步研究？這是確定性研究篩選器；研究候選需要先補上 curated leader_type 與 industry_quality_score。</p>
      <div class="intelligence-grid">
        <div class="intelligence-block">
          <h3>研究候選</h3>
          <div class="table-wrap">
            <table data-table="leader-research-candidates"></table>
          </div>
          <p class="empty-state" data-empty-for="leader-research-candidates" hidden>目前沒有研究候選。這是預期情況：需要先在 tickers.csv 補上 curated leader_type 與 industry_quality_score 後，研究候選輸出才有意義。</p>
        </div>
        <div class="intelligence-block">
          <h3>等待穩定</h3>
          <div class="table-wrap">
            <table data-table="leader-wait-for-stabilization"></table>
          </div>
          <p class="empty-state" data-empty-for="leader-wait-for-stabilization" hidden>目前沒有等待穩定名單。</p>
        </div>
        <div class="intelligence-block">
          <h3>延伸偏高領導股</h3>
          <div class="table-wrap">
            <table data-table="leader-too-extended"></table>
          </div>
          <p class="empty-state" data-empty-for="leader-too-extended" hidden>目前沒有延伸偏高的領導股。</p>
        </div>
        <div class="intelligence-block">
          <h3>產業動能未確認，暫列觀察</h3>
          <div class="table-wrap">
            <table data-table="leader-not-eligible-industries"></table>
          </div>
          <p class="empty-state" data-empty-for="leader-not-eligible-industries" hidden>目前沒有產業動能未確認的觀察資料。</p>
        </div>
      </div>
    </section>

    <section class="dashboard-section" id="portfolio-simulator">
      <div class="section-heading">
        <h2>追蹤名單快速檢查</h2>
      </div>
      <p class="section-note">輸入你關注的 ticker，系統會用本頁最新動能與風險訊號做快速體檢，並給出下一步研究方向。此工具僅做研究排序，不構成投資建議。</p>
      <div class="table-wrap" style="padding: 14px; border: 1px solid var(--line); border-radius: 10px; background: var(--surface);">
        <label for="portfolio-input" style="display:block; font-weight:650; margin-bottom:8px;">追蹤名單（每行一個 ticker）</label>
        <textarea id="portfolio-input" rows="7" style="width:100%; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; border:1px solid var(--line); border-radius:8px; padding:10px;" placeholder="VRT&#10;DELL&#10;SPOT&#10;SOFI"></textarea>
        <div style="display:flex; gap:8px; margin-top:10px; flex-wrap:wrap;">
          <button id="portfolio-run" type="button" style="border:1px solid var(--line-strong); border-radius:8px; padding:8px 12px; background:var(--surface-soft); cursor:pointer;">執行模擬</button>
          <button id="portfolio-sample" type="button" style="border:1px solid var(--line); border-radius:8px; padding:8px 12px; background:var(--surface); cursor:pointer;">填入範例</button>
        </div>
      </div>
      <div id="portfolio-result" class="quality-grid" style="margin-top:12px;"></div>
      <ul id="portfolio-actions" class="daily-brief-list" style="margin-top:10px;"></ul>
      <div id="portfolio-alignment" class="table-wrap" style="margin-top:12px;" hidden>
        <table id="portfolio-alignment-table"></table>
      </div>
    </section>

    <section class="dashboard-section" id="relative-strength">
      <div class="section-heading">
        <h2>相對產業強度前 10 名</h2>
        <span class="row-count" data-count-for="top-relative-strength"></span>
      </div>
      <p class="section-note">這張表回答：哪些個股跑贏同產業？相對強度 = 個股 10 日報酬減去所屬產業平均 10 日報酬；正值代表跑贏同產業。</p>
      <div class="table-wrap">
        <table data-table="top-relative-strength"></table>
      </div>
      <p class="empty-state" data-empty-for="top-relative-strength" hidden>目前沒有個股資料。</p>
    </section>

    <section class="dashboard-section" id="early-momentum">
      <div class="section-heading">
        <h2>早期動能候選</h2>
        <span class="row-count" data-count-for="early-candidates"></span>
      </div>
      <p class="section-note">這張表回答：哪些股票剛開始轉強、但仍需確認？3 日與 5 日報酬轉強，且 5 日報酬已高於 10 日報酬的一半。標示「動能+風險」代表同時出現在風險提醒名單。</p>
      <div class="table-wrap">
        <table data-table="early-candidates"></table>
      </div>
      <p class="empty-state" data-empty-for="early-candidates" hidden>目前沒有早期動能候選。</p>
    </section>

    <section class="dashboard-section" id="strong-momentum">
      <div class="section-heading">
        <h2>強勢動能候選</h2>
        <span class="row-count" data-count-for="strong-candidates"></span>
      </div>
      <p class="section-note">這張表回答：哪些股票同時有確認動能、相對強度與量能？同時符合確認動能、跑贏同產業，且相對量大於 1.2。</p>
      <div class="table-wrap">
        <table data-table="strong-candidates"></table>
      </div>
      <p class="empty-state" data-empty-for="strong-candidates" hidden>目前沒有強勢動能候選。</p>
    </section>

    <section class="dashboard-section" id="risk-warning">
      <div class="section-heading">
        <h2>風險提醒名單</h2>
        <span class="row-count" data-count-for="risk-warnings"></span>
      </div>
      <p class="section-note">這張表回答：哪些股票有回撤或延伸風險，不能只看動能？風險提醒代表近期最大回撤較深，或價格已高出 20 日均線 15% 以上。</p>
      <div class="table-wrap">
        <table data-table="risk-warnings"></table>
      </div>
      <p class="empty-state" data-empty-for="risk-warnings" hidden>目前沒有風險提醒。</p>
    </section>
  </main>

  <div class="industry-modal" id="industry-modal" hidden>
    <div class="industry-modal-backdrop" data-close-industry-modal></div>
    <section class="industry-modal-panel" role="dialog" aria-modal="true" aria-labelledby="industry-modal-title">
      <div class="industry-modal-header">
        <div>
          <h2 class="industry-modal-title" id="industry-modal-title">產業成分</h2>
          <p class="industry-modal-subtitle" id="industry-modal-subtitle"></p>
        </div>
        <button class="industry-modal-close" type="button" data-close-industry-modal aria-label="關閉產業成分">×</button>
      </div>
      <div class="industry-modal-body">
        <div class="industry-modal-summary" id="industry-modal-summary"></div>
        <div class="industry-modal-table-wrap">
          <table class="industry-modal-table" id="industry-modal-table"></table>
        </div>
        <p class="empty-state" id="industry-modal-empty" hidden>目前沒有這個產業的成分股資料。</p>
      </div>
    </section>
  </div>

  <script id="dashboard-data" type="application/json">__DASHBOARD_DATA__</script>
  <script>
    const dashboardData = JSON.parse(document.getElementById("dashboard-data").textContent);

    const industryLabels = {
      "AI Infrastructure": "AI 基礎設施",
      "Semiconductors": "半導體",
      "Defense": "國防",
      "Defense Drones": "國防無人機",
      "Space": "太空與衛星",
      "Cybersecurity": "資安",
      "Cloud Software": "雲端軟體",
      "Adtech": "廣告科技",
      "Quantum Computing": "量子運算",
      "Nuclear": "核能",
      "Renewables": "再生能源",
      "Energy Storage": "儲能",
      "Robotics Automation": "機器人與自動化",
      "Fintech": "金融科技",
      "Consumer Platforms": "消費平台",
      "Defensive Healthcare": "防禦型醫療",
      "Defensive Staples": "防禦型必需消費",
      "Market ETFs": "市場 ETF"
    };

    const regimeLabels = {
      momentum_leader: "動能領先",
      early_recovery: "早期修復",
      neutral: "中性",
      weak: "偏弱"
    };

    const industryRiskFlagLabels = {
      none: "無",
      momentum_exhaustion: "動能衰竭",
      narrow_leadership: "領漲偏窄",
      late_cycle_momentum: "後段動能",
      data_limited: "資料有限"
    };

    const rotationTypeLabels = {
      risk_on_growth: "成長風險偏好",
      defensive_rotation: "防禦輪動",
      commodity_inflation: "商品通膨",
      policy_driven: "政策驅動",
      panic_rebound: "急跌反彈",
      liquidity_rebound: "流動性反彈",
      unclear: "不明"
    };

    const causalHypothesisLabels = {
      industry_flow_leads_leaders: "產業流向帶動",
      leader_strength_leads_industry: "領導股帶動",
      macro_liquidity_rebound: "流動性修復",
      policy_or_thematic_support: "政策或題材支撐",
      defensive_rotation: "防禦輪動",
      unclear: "不明"
    };

    const evidenceStatusLabels = {
      observed: "已觀察",
      inferred: "推論",
      needs_review: "待檢查",
      unsupported: "未支持"
    };

    const leaderTypeLabels = {
      core_leader: "核心領導",
      challenger: "挑戰者",
      infrastructure_leader: "基礎設施領導",
      emerging_leader: "新興領導",
      specialist: "專門型",
      non_leader: "未標註領導"
    };

    const priceZoneLabels = {
      deep_pullback: "深度回落",
      reasonable_pullback: "合理回落",
      neutral: "中性",
      extended: "偏高",
      very_extended: "明顯偏高"
    };

    const currentStateLabels = {
      strong_uptrend: "強勢上行",
      early_recovery: "早期修復",
      pullback_in_uptrend: "趨勢內回落",
      sideways_base: "橫向整理",
      falling_knife: "急跌中",
      overextended: "延伸偏高"
    };

    const watchStatusLabels = {
      research_candidate: "研究候選",
      wait_for_stabilization: "等待穩定",
      too_extended: "延伸偏高",
      avoid_for_now: "暫不列入",
      not_eligible_industry: "產業動能未確認，暫列觀察"
    };

    const dataStatusLabels = {
      ok: "正常",
      missing: "缺資料",
      stale: "資料落後",
      limited_history: "歷史不足"
    };

    const pegRatingLabels = {
      undervalued: "低於成長",
      fair: "合理",
      expensive: "偏貴",
      unavailable: "無法判讀"
    };

    const pegStatusLabels = {
      ok: "正常",
      missing_pe: "缺 Forward P/E",
      missing_growth: "缺成長率",
      invalid_growth: "成長率無效",
      fetch_error: "抓取失敗"
    };

    const updateHealthLabels = {
      healthy: "正常",
      warning: "注意",
      unknown: "未知"
    };

    const explanations = {
      return3d: "最近 3 個交易日的報酬率。",
      return5d: "最近 5 個交易日的報酬率，用來觀察短線動能。",
      return10d: "最近 10 個交易日的報酬率，這是主要排名依據。",
      return20d: "最近 20 個交易日的報酬率，用來對照較長週期趨勢。",
      return1m: "約 1 個月的報酬率，用來看短中期趨勢背景。",
      return3m: "約 1 季的報酬率，用來看季度趨勢背景。",
      return6m: "約半年的報酬率，用來看中期趨勢背景。",
      return1y: "約 1 年的報酬率；若下載資料不足完整一年，使用目前可用的最長區間。",
      latestVolume: "最新交易日成交量；用來觀察這個訊號背後的流動性基礎。",
      avgVolume20d: "最近 20 個交易日的平均成交量；可用來對照最新量是否異常放大。",
      relativeVolume: "最新成交量 / 20 日平均成交量；大於 1 代表量能高於近期平均。相對量看放大倍數，原始量看流動性基礎。",
      relativeStrength: "個股 10 日報酬減去所屬產業平均 10 日報酬；正值代表跑贏同產業。",
      maxDrawdown: "最近 10 個交易日從高點回落的最大幅度；數值越負代表回撤越深。",
      upDays: "最近 10 個交易日中，上漲日的天數。",
      rotation: "目前產業排名相對近 5 個歷史交易日的變化；正值代表排名上升。",
      acceleration: "目前平均 5 日報酬減去前一個快照的平均 5 日報酬。",
      persistence: "產業連續維持在前三名的歷史交易日數。",
      confirmedPct: "產業內符合確認動能訊號的股票比例。",
      breadthScore: "廣度分數 = 5 日正報酬比例 20% + 10 日正報酬比例 25% + 確認比例 25% + 強勢比例 20% + 高相對量比例 10%。",
      positive5dPct: "產業內 5 日報酬大於 0 的股票比例。",
      positive10dPct: "產業內 10 日報酬大於 0 的股票比例。",
      strongPct: "產業內符合強勢動能訊號的股票比例。",
      highRelativeVolumePct: "產業內相對量大於 1.2 的股票比例。",
      industryRegime: "產業趨勢狀態由 10 日報酬排名、廣度分數與持續性決定；風險另列。",
      industryRiskFlag: "產業風險旗標保留衰竭、領漲偏窄、後段動能或資料有限等狀況。",
      rotationType: "手動或設定檔分類的輪動類型；不從價格與量能自動推論複雜原因。",
      causalHypothesis: "用於研究檢查的因果假說，不代表已證實原因。",
      evidenceStatus: "目前系統內可觀察資料對假說的支持程度。",
      leaderType: "手動維護的領導股類型；目前預設為未標註領導。",
      industryQuality: "手動維護的產業品質分數，1 到 5 分；研究候選需要至少 4 分。",
      currentState: "依近期報酬、回撤、均線與 price position 判斷的目前價格狀態。",
      priceZone: "綜合短期與長期技術價格位置後的區間，用於 watch status。",
      shortTermPriceZone: "由價格相對 20 日均線的距離判斷。",
      longTermPriceZone: "由價格在 52 週區間中的位置判斷。",
      watchStatus: "風險優先的確定性研究狀態，不代表投資建議。",
      distance20d: "最新價格相對 20 日均線的距離。",
      distance52wHigh: "最新價格相對 52 週高點的距離。",
      position52wRange: "最新價格在 52 週高低區間中的位置，越接近 1 代表越靠近區間上緣。",
      forwardPe: "Yahoo Finance forwardPE；資料缺漏時不補值。",
      earningsGrowth: "Yahoo Finance earningsGrowth，0.25 代表預期 EPS 成長 25%。",
      pegRatio: "Forward PEG = Forward P/E / 預期 EPS 成長百分比；只在 P/E 與成長率皆為正值時產生。",
      pegRating: "PEG < 1 為低於成長，1 到 2 為合理，大於 2 為偏貴；缺資料時不判讀。",
      pegStatus: "PEG 資料狀態，用來辨識缺 P/E、缺成長率、成長率無效或抓取失敗。"
    };

    const tableConfigs = {
      "data-quality-issues": {
        rows: dashboardData.data_quality.issue_tickers,
        columns: [
          { key: "ticker", label: "代號", type: "ticker" },
          { key: "company_name", label: "公司", type: "company" },
          { key: "industry_group", label: "產業" },
          { key: "latest_date", label: "最新日期" },
          { key: "data_points", label: "資料筆數", type: "integer" },
          { key: "data_status", label: "資料狀態" },
          { key: "data_quality_note", label: "資料註記" }
        ]
      },
      "industry-momentum": {
        rows: dashboardData.industry_momentum,
        columns: [
          { key: "__rank", label: "排名", type: "rank" },
          { key: "industry_group", label: "產業" },
          { key: "return_5d", label: "平均 5日", type: "percent", description: explanations.return5d },
          { key: "return_10d", label: "平均 10日", type: "percent", description: explanations.return10d },
          { key: "return_20d", label: "平均 20日", type: "percent", description: explanations.return20d },
          { key: "return_1m", label: "平均 1月", type: "percent", description: explanations.return1m },
          { key: "return_3m", label: "平均 1季", type: "percent", description: explanations.return3m },
          { key: "return_6m", label: "平均 半年", type: "percent", description: explanations.return6m },
          { key: "return_1y", label: "平均 1年", type: "percent", description: explanations.return1y },
          { key: "relative_volume", label: "相對量", type: "number", digits: 2, description: explanations.relativeVolume },
          { key: "industry_regime", label: "狀態", description: explanations.industryRegime },
          { key: "industry_risk_flag", label: "風險", description: explanations.industryRiskFlag },
          { key: "rotation_type", label: "輪動", description: explanations.rotationType },
          { key: "tickers_with_data", label: "有資料", type: "integer" },
          { key: "ticker_count", label: "檔數", type: "integer" }
        ]
      },
      "industry-confirmed": {
        rows: dashboardData.industry_confirmed,
        columns: [
          { key: "__rank", label: "排名", type: "rank" },
          { key: "industry_group", label: "產業" },
          { key: "confirmed_signal_pct", label: "確認比例", type: "percent", description: explanations.confirmedPct },
          { key: "confirmed_count", label: "確認檔數", type: "integer" },
          { key: "tickers_with_data", label: "有資料", type: "integer" },
          { key: "ticker_count", label: "檔數", type: "integer" }
        ]
      },
      "breadth-ranked": {
        rows: dashboardData.industry_breadth.ranked,
        columns: [
          { key: "__rank", label: "排名", type: "rank" },
          { key: "industry_group", label: "產業" },
          { key: "breadth_score", label: "廣度分數", type: "percent", description: explanations.breadthScore },
          { key: "positive_5d_pct", label: "5日正報酬", type: "percent", description: explanations.positive5dPct },
          { key: "positive_10d_pct", label: "10日正報酬", type: "percent", description: explanations.positive10dPct },
          { key: "confirmed_signal_pct", label: "確認比例", type: "percent", description: explanations.confirmedPct },
          { key: "strong_signal_pct", label: "強勢比例", type: "percent", description: explanations.strongPct },
          { key: "high_relative_volume_pct", label: "高相對量", type: "percent", description: explanations.highRelativeVolumePct },
          { key: "return_10d", label: "平均 10日", type: "percent", description: explanations.return10d },
          { key: "industry_regime", label: "狀態", description: explanations.industryRegime },
          { key: "industry_risk_flag", label: "風險", description: explanations.industryRiskFlag },
          { key: "rotation_type", label: "輪動", description: explanations.rotationType }
        ]
      },
      "breadth-high-return-weak": {
        rows: dashboardData.industry_breadth.high_return_weak_breadth,
        columns: [
          { key: "industry_group", label: "產業" },
          { key: "return_10d", label: "平均 10日", type: "percent", description: explanations.return10d },
          { key: "breadth_score", label: "廣度分數", type: "percent", description: explanations.breadthScore },
          { key: "positive_10d_pct", label: "10日正報酬", type: "percent", description: explanations.positive10dPct },
          { key: "confirmed_signal_pct", label: "確認比例", type: "percent", description: explanations.confirmedPct },
          { key: "strong_signal_pct", label: "強勢比例", type: "percent", description: explanations.strongPct },
          { key: "high_relative_volume_pct", label: "高相對量", type: "percent", description: explanations.highRelativeVolumePct },
          { key: "industry_regime", label: "狀態", description: explanations.industryRegime },
          { key: "industry_risk_flag", label: "風險", description: explanations.industryRiskFlag },
          { key: "rotation_type", label: "輪動", description: explanations.rotationType }
        ]
      },
      "breadth-moderate-improving": {
        rows: dashboardData.industry_breadth.moderate_return_improving_breadth,
        columns: [
          { key: "industry_group", label: "產業" },
          { key: "return_5d", label: "平均 5日", type: "percent", description: explanations.return5d },
          { key: "return_10d", label: "平均 10日", type: "percent", description: explanations.return10d },
          { key: "positive_5d_pct", label: "5日正報酬", type: "percent", description: explanations.positive5dPct },
          { key: "positive_10d_pct", label: "10日正報酬", type: "percent", description: explanations.positive10dPct },
          { key: "breadth_score", label: "廣度分數", type: "percent", description: explanations.breadthScore },
          { key: "high_relative_volume_pct", label: "高相對量", type: "percent", description: explanations.highRelativeVolumePct },
          { key: "industry_regime", label: "狀態", description: explanations.industryRegime },
          { key: "industry_risk_flag", label: "風險", description: explanations.industryRiskFlag },
          { key: "rotation_type", label: "輪動", description: explanations.rotationType }
        ]
      },
      "rotation-gaining": {
        rows: dashboardData.rotation_trend.gaining_rank,
        columns: [
          { key: "industry_group", label: "產業" },
          { key: "start_rank", label: "起始排名", type: "integer" },
          { key: "current_rank", label: "目前排名", type: "integer" },
          { key: "rank_change", label: "排名變化", type: "signedInteger", description: explanations.rotation },
          { key: "current_average_10d_return", label: "目前 10日", type: "percent", description: explanations.return10d },
          { key: "average_10d_return_change", label: "10日變化", type: "signedPercent" }
        ]
      },
      "rotation-losing": {
        rows: dashboardData.rotation_trend.losing_rank,
        columns: [
          { key: "industry_group", label: "產業" },
          { key: "start_rank", label: "起始排名", type: "integer" },
          { key: "current_rank", label: "目前排名", type: "integer" },
          { key: "rank_change", label: "排名變化", type: "signedInteger", description: explanations.rotation },
          { key: "current_average_10d_return", label: "目前 10日", type: "percent", description: explanations.return10d },
          { key: "average_10d_return_change", label: "10日變化", type: "signedPercent" }
        ]
      },
      "rotation-improving-5d": {
        rows: dashboardData.rotation_trend.improving_5d,
        columns: [
          { key: "industry_group", label: "產業" },
          { key: "average_10d_return_change", label: "10日變化", type: "signedPercent" },
          { key: "current_average_10d_return", label: "目前 10日", type: "percent", description: explanations.return10d },
          { key: "rank_change", label: "排名變化", type: "signedInteger", description: explanations.rotation },
          { key: "current_rank", label: "目前排名", type: "integer" },
          { key: "current_confirmed_signal_pct", label: "確認比例", type: "percent", description: explanations.confirmedPct }
        ]
      },
      "trend-strongest-improving": {
        rows: dashboardData.industry_trend_intelligence.strongest_improving,
        columns: [
          { key: "industry_group", label: "產業" },
          { key: "rotation_score", label: "輪動分數", type: "signedInteger", description: explanations.rotation },
          { key: "momentum_acceleration", label: "動能加速", type: "signedPercent", description: explanations.acceleration },
          { key: "return_10d", label: "平均 10日", type: "percent", description: explanations.return10d },
          { key: "return_5d", label: "平均 5日", type: "percent", description: explanations.return5d },
          { key: "relative_volume", label: "相對量", type: "number", digits: 2, description: explanations.relativeVolume },
          { key: "industry_regime", label: "狀態", description: explanations.industryRegime },
          { key: "industry_risk_flag", label: "風險", description: explanations.industryRiskFlag },
          { key: "rotation_type", label: "輪動", description: explanations.rotationType }
        ]
      },
      "trend-strongest-persistent": {
        rows: dashboardData.industry_trend_intelligence.strongest_persistent,
        columns: [
          { key: "industry_group", label: "產業" },
          { key: "momentum_persistence", label: "前三持續天數", type: "integer", description: explanations.persistence },
          { key: "return_10d", label: "平均 10日", type: "percent", description: explanations.return10d },
          { key: "confirmed_signal_pct", label: "確認比例", type: "percent", description: explanations.confirmedPct },
          { key: "rotation_score", label: "輪動分數", type: "signedInteger", description: explanations.rotation },
          { key: "momentum_acceleration", label: "動能加速", type: "signedPercent", description: explanations.acceleration },
          { key: "industry_regime", label: "狀態", description: explanations.industryRegime },
          { key: "industry_risk_flag", label: "風險", description: explanations.industryRiskFlag },
          { key: "rotation_type", label: "輪動", description: explanations.rotationType }
        ]
      },
      "trend-momentum-exhaustion": {
        rows: dashboardData.industry_trend_intelligence.momentum_exhaustion,
        columns: [
          { key: "industry_group", label: "產業" },
          { key: "return_10d", label: "平均 10日", type: "percent", description: explanations.return10d },
          { key: "return_3d", label: "平均 3日", type: "percent", description: explanations.return3d },
          { key: "relative_volume", label: "相對量", type: "number", digits: 2, description: explanations.relativeVolume },
          { key: "momentum_acceleration", label: "動能加速", type: "signedPercent", description: explanations.acceleration },
          { key: "momentum_persistence", label: "前三持續天數", type: "integer", description: explanations.persistence },
          { key: "industry_regime", label: "狀態", description: explanations.industryRegime },
          { key: "industry_risk_flag", label: "風險", description: explanations.industryRiskFlag },
          { key: "rotation_type", label: "輪動", description: explanations.rotationType }
        ]
      },
      "trend-momentum-recovery": {
        rows: dashboardData.industry_trend_intelligence.momentum_recovery,
        columns: [
          { key: "industry_group", label: "產業" },
          { key: "momentum_acceleration", label: "動能加速", type: "signedPercent", description: explanations.acceleration },
          { key: "rotation_score", label: "輪動分數", type: "signedInteger", description: explanations.rotation },
          { key: "return_5d", label: "平均 5日", type: "percent", description: explanations.return5d },
          { key: "return_10d", label: "平均 10日", type: "percent", description: explanations.return10d },
          { key: "relative_volume", label: "相對量", type: "number", digits: 2, description: explanations.relativeVolume },
          { key: "industry_regime", label: "狀態", description: explanations.industryRegime },
          { key: "industry_risk_flag", label: "風險", description: explanations.industryRiskFlag },
          { key: "rotation_type", label: "輪動", description: explanations.rotationType }
        ]
      },
      "leader-research-candidates": {
        rows: dashboardData.leader_accumulation.research_candidates,
        columns: [
          { key: "ticker", label: "代號", type: "ticker" },
          { key: "company_name", label: "公司", type: "company" },
          { key: "industry_group", label: "產業" },
          { key: "industry_regime", label: "產業狀態", description: explanations.industryRegime },
          { key: "industry_risk_flag", label: "風險", description: explanations.industryRiskFlag },
          { key: "rotation_type", label: "輪動", description: explanations.rotationType },
          { key: "causal_hypothesis", label: "假說", description: explanations.causalHypothesis },
          { key: "evidence_status", label: "證據", description: explanations.evidenceStatus },
          { key: "leader_type", label: "領導類型", description: explanations.leaderType },
          { key: "industry_quality_score", label: "品質分數", type: "integer", description: explanations.industryQuality },
          { key: "current_state", label: "目前狀態", description: explanations.currentState },
          { key: "price_zone", label: "價格位置", description: explanations.priceZone },
          { key: "peg_ratio", label: "PEG", type: "number", digits: 2, description: explanations.pegRatio },
          { key: "peg_rating", label: "PEG判讀", description: explanations.pegRating },
          { key: "return_10d", label: "10日", type: "percent", description: explanations.return10d },
          { key: "relative_strength_vs_industry", label: "相對強度", type: "percent", description: explanations.relativeStrength },
          { key: "relative_volume", label: "相對量", type: "number", digits: 2, description: explanations.relativeVolume }
        ]
      },
      "leader-wait-for-stabilization": {
        rows: dashboardData.leader_accumulation.wait_for_stabilization,
        columns: [
          { key: "ticker", label: "代號", type: "ticker" },
          { key: "company_name", label: "公司", type: "company" },
          { key: "industry_group", label: "產業" },
          { key: "industry_regime", label: "產業狀態", description: explanations.industryRegime },
          { key: "industry_risk_flag", label: "風險", description: explanations.industryRiskFlag },
          { key: "rotation_type", label: "輪動", description: explanations.rotationType },
          { key: "causal_hypothesis", label: "假說", description: explanations.causalHypothesis },
          { key: "evidence_status", label: "證據", description: explanations.evidenceStatus },
          { key: "leader_type", label: "領導類型", description: explanations.leaderType },
          { key: "industry_quality_score", label: "品質分數", type: "integer", description: explanations.industryQuality },
          { key: "current_state", label: "目前狀態", description: explanations.currentState },
          { key: "short_term_price_zone", label: "短期位置", description: explanations.shortTermPriceZone },
          { key: "long_term_price_zone", label: "長期位置", description: explanations.longTermPriceZone },
          { key: "peg_ratio", label: "PEG", type: "number", digits: 2, description: explanations.pegRatio },
          { key: "peg_rating", label: "PEG判讀", description: explanations.pegRating },
          { key: "return_5d", label: "5日", type: "percent", description: explanations.return5d },
          { key: "max_drawdown_10d", label: "最大回撤", type: "warningPercent", description: explanations.maxDrawdown }
        ]
      },
      "leader-too-extended": {
        rows: dashboardData.leader_accumulation.too_extended_leaders,
        columns: [
          { key: "ticker", label: "代號", type: "ticker" },
          { key: "company_name", label: "公司", type: "company" },
          { key: "industry_group", label: "產業" },
          { key: "industry_regime", label: "產業狀態", description: explanations.industryRegime },
          { key: "industry_risk_flag", label: "風險", description: explanations.industryRiskFlag },
          { key: "rotation_type", label: "輪動", description: explanations.rotationType },
          { key: "causal_hypothesis", label: "假說", description: explanations.causalHypothesis },
          { key: "evidence_status", label: "證據", description: explanations.evidenceStatus },
          { key: "leader_type", label: "領導類型", description: explanations.leaderType },
          { key: "current_state", label: "目前狀態", description: explanations.currentState },
          { key: "price_zone", label: "價格位置", description: explanations.priceZone },
          { key: "peg_ratio", label: "PEG", type: "number", digits: 2, description: explanations.pegRatio },
          { key: "peg_rating", label: "PEG判讀", description: explanations.pegRating },
          { key: "distance_from_20d_ma", label: "距20日均線", type: "percent", description: explanations.distance20d },
          { key: "position_in_52w_range", label: "52週位置", type: "percent", description: explanations.position52wRange },
          { key: "return_10d", label: "10日", type: "percent", description: explanations.return10d }
        ]
      },
      "leader-not-eligible-industries": {
        rows: dashboardData.leader_accumulation.not_eligible_industries,
        columns: [
          { key: "industry_group", label: "產業" },
          { key: "industry_regime", label: "產業狀態", description: explanations.industryRegime },
          { key: "industry_risk_flag", label: "風險", description: explanations.industryRiskFlag },
          { key: "rotation_type", label: "輪動", description: explanations.rotationType },
          { key: "causal_hypothesis", label: "假說", description: explanations.causalHypothesis },
          { key: "evidence_status", label: "證據", description: explanations.evidenceStatus },
          { key: "return_5d", label: "平均 5日", type: "percent", description: explanations.return5d },
          { key: "return_10d", label: "平均 10日", type: "percent", description: explanations.return10d },
          { key: "breadth_score", label: "廣度分數", type: "percent", description: explanations.breadthScore },
          { key: "confirmed_signal_pct", label: "確認比例", type: "percent", description: explanations.confirmedPct },
          { key: "momentum_persistence", label: "前三持續天數", type: "integer", description: explanations.persistence },
          { key: "momentum_exhaustion_warning", label: "衰竭警示" }
        ]
      },
      "top-relative-strength": {
        rows: dashboardData.top_relative_strength,
        columns: [
          { key: "__rank", label: "排名", type: "rank" },
          { key: "ticker", label: "代號", type: "ticker" },
          { key: "company_name", label: "公司", type: "company" },
          { key: "industry_group", label: "產業" },
          { key: "relative_strength_vs_industry", label: "相對強度", type: "percent", description: explanations.relativeStrength },
          { key: "return_10d", label: "10日", type: "percent", description: explanations.return10d },
          { key: "relative_volume", label: "相對量", type: "number", digits: 2, description: explanations.relativeVolume },
          { key: "up_days_10d", label: "上漲天數", type: "integer", description: explanations.upDays }
        ]
      },
      "early-candidates": {
        rows: dashboardData.early_candidates,
        columns: [
          { key: "ticker", label: "代號", type: "ticker" },
          { key: "company_name", label: "公司", type: "company" },
          { key: "industry_group", label: "產業" },
          { key: "return_3d", label: "3日", type: "percent", description: explanations.return3d },
          { key: "return_5d", label: "5日", type: "percent", description: explanations.return5d },
          { key: "return_10d", label: "10日", type: "percent", description: explanations.return10d },
          { key: "relative_strength_vs_industry", label: "相對強度", type: "percent", description: explanations.relativeStrength },
          { key: "relative_volume", label: "相對量", type: "number", digits: 2, description: explanations.relativeVolume }
        ]
      },
      "strong-candidates": {
        rows: dashboardData.strong_candidates,
        columns: [
          { key: "ticker", label: "代號", type: "ticker" },
          { key: "company_name", label: "公司", type: "company" },
          { key: "industry_group", label: "產業" },
          { key: "relative_strength_vs_industry", label: "相對強度", type: "percent", description: explanations.relativeStrength },
          { key: "return_10d", label: "10日", type: "percent", description: explanations.return10d },
          { key: "relative_volume", label: "相對量", type: "number", digits: 2, description: explanations.relativeVolume },
          { key: "up_days_10d", label: "上漲天數", type: "integer", description: explanations.upDays },
          { key: "max_drawdown_10d", label: "最大回撤", type: "percent", description: explanations.maxDrawdown }
        ]
      },
      "risk-warnings": {
        rows: dashboardData.risk_warnings,
        columns: [
          { key: "ticker", label: "代號", type: "ticker" },
          { key: "company_name", label: "公司", type: "company" },
          { key: "industry_group", label: "產業" },
          { key: "max_drawdown_10d", label: "最大回撤", type: "warningPercent", description: explanations.maxDrawdown },
          { key: "return_10d", label: "10日", type: "percent", description: explanations.return10d },
          { key: "return_20d", label: "20日", type: "percent", description: explanations.return20d },
          { key: "relative_volume", label: "相對量", type: "number", digits: 2, description: explanations.relativeVolume },
          { key: "up_days_10d", label: "上漲天數", type: "integer", description: explanations.upDays }
        ]
      }
    };

    const constituentColumns = [
      { key: "ticker", label: "代號", type: "ticker" },
      { key: "company_name", label: "公司", type: "company" },
      { key: "latest_date", label: "最新日期" },
      { key: "data_points", label: "資料", type: "dataStatus" },
      { key: "data_status", label: "資料狀態" },
      { key: "leader_type", label: "領導類型", description: explanations.leaderType },
      { key: "watch_status", label: "觀察狀態", description: explanations.watchStatus },
      { key: "forward_pe", label: "Forward P/E", type: "number", digits: 2, description: explanations.forwardPe },
      { key: "earnings_growth", label: "成長率", type: "percent", description: explanations.earningsGrowth },
      { key: "peg_ratio", label: "PEG", type: "number", digits: 2, description: explanations.pegRatio },
      { key: "peg_rating", label: "PEG判讀", description: explanations.pegRating },
      { key: "peg_status", label: "PEG資料", description: explanations.pegStatus },
      { key: "return_5d", label: "5日", type: "percent", description: explanations.return5d },
      { key: "return_10d", label: "10日", type: "percent", description: explanations.return10d },
      { key: "return_20d", label: "20日", type: "percent", description: explanations.return20d },
      { key: "return_1m", label: "1月", type: "percent", description: explanations.return1m },
      { key: "return_3m", label: "1季", type: "percent", description: explanations.return3m },
      { key: "return_6m", label: "半年", type: "percent", description: explanations.return6m },
      { key: "return_1y", label: "1年", type: "percent", description: explanations.return1y },
      { key: "relative_strength_vs_industry", label: "相對強度", type: "percent", description: explanations.relativeStrength },
      { key: "latest_volume", label: "最新量", type: "compactVolume", description: explanations.latestVolume },
      { key: "avg_volume_20d", label: "20日平均量", type: "compactVolume", description: explanations.avgVolume20d },
      { key: "relative_volume", label: "相對量", type: "number", digits: 2, description: explanations.relativeVolume },
      { key: "confirmed_momentum_signal", label: "確認" },
      { key: "strong_momentum_signal", label: "強勢" },
      { key: "risk_warning", label: "風險" }
    ];

    function isMissing(value) {
      return value === null || value === undefined || Number.isNaN(value);
    }

    function displayText(value) {
      if (isMissing(value)) return "";
      const text = String(value);
      return industryLabels[text] || text;
    }

    function createIndustryButton(industryGroup) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "industry-link";
      button.textContent = displayText(industryGroup);
      button.title = `查看 ${displayText(industryGroup)} 的 ticker 組合`;
      button.addEventListener("click", () => openIndustryModal(industryGroup, button));
      return button;
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

    const alertLevelLabels = {
      red: "紅色",
      orange: "橘色",
      yellow: "黃色",
      green: "綠色",
      unknown: "未知"
    };

    const alertActionLabels = {
      review_replacement: "優先檢查並評估轉換",
      watch_transition: "觀察是否持續轉弱",
      monitor: "保留觀察",
      watch_ok: "暫無警示",
      add_to_tickers: "補進追蹤資料"
    };

    function alertPriority(row) {
      const levelRank = { red: 0, orange: 1, yellow: 2, green: 3, unknown: 4 };
      return levelRank[row?.alert_level] ?? 9;
    }

    function sortedWatchlistAlerts(rows) {
      return [...rows].sort((a, b) => {
        const priorityDiff = alertPriority(a) - alertPriority(b);
        if (priorityDiff !== 0) return priorityDiff;
        return String(a.ticker || "").localeCompare(String(b.ticker || ""));
      });
    }

    function allTickerRows() {
      return Object.values(dashboardData.industry_constituents || {}).flat();
    }

    const tickerDetailMap = new Map(
      allTickerRows()
        .filter((row) => row?.ticker)
        .map((row) => [String(row.ticker).toUpperCase(), row])
    );

    function parseCandidateText(text) {
      return String(text || "")
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean)
        .map((label) => ({
          label,
          ticker: label.split(/\\s+/)[0].replace(/[()]/g, "").toUpperCase()
        }));
    }

    function candidateCaption(candidate) {
      const row = tickerDetailMap.get(candidate.ticker);
      if (!row) return `${candidate.label}: 目前沒有完整資料。`;
      const riskText = row.risk_warning === true ? "有風險提醒" : "無風險提醒";
      return [
        `${row.ticker} · ${row.company_name || "公司名稱未取得"} · ${displayText(row.industry_group) || "未知產業"}`,
        `量能 ${formatCompactVolume(row.latest_volume) || "n/a"}，相對量 ${formatNumber(row.relative_volume, 2) || "n/a"}`,
        `動能 5日 ${formatSignedPercent(row.return_5d) || "n/a"}，10日 ${formatSignedPercent(row.return_10d) || "n/a"}，20日 ${formatSignedPercent(row.return_20d) || "n/a"}`,
        `趨勢 1月 ${formatSignedPercent(row.return_1m) || "n/a"}，1季 ${formatSignedPercent(row.return_3m) || "n/a"}，半年 ${formatSignedPercent(row.return_6m) || "n/a"}，1年 ${formatSignedPercent(row.return_1y) || "n/a"}`,
        `狀態 ${watchStatusLabels[row.watch_status] || displayText(row.watch_status) || "n/a"}，${riskText}`
      ].join("；");
    }

    function appendReplacementRoute(route, routeSource) {
      const industries = routeSource.replacement_industries || "目前沒有替代產業";
      const candidates = parseCandidateText(routeSource.replacement_candidates);
      route.replaceChildren();
      route.append(document.createTextNode(`整體替代路徑：先看產業動能較強的 ${industries}，再研究候選 `));
      if (!candidates.length) {
        route.append(document.createTextNode("目前沒有替代候選。"));
        return;
      }
      candidates.forEach((candidate, index) => {
        const chip = document.createElement("span");
        chip.className = "candidate-chip";
        chip.tabIndex = 0;
        chip.textContent = candidate.label;
        chip.setAttribute("data-caption", candidateCaption(candidate));
        route.appendChild(chip);
        route.append(document.createTextNode(index === candidates.length - 1 ? "。" : "、"));
      });
    }

    function appendText(parent, className, text) {
      const element = document.createElement("div");
      element.className = className;
      element.textContent = text;
      parent.appendChild(element);
      return element;
    }

    function renderWatchlistAlert() {
      const panel = document.getElementById("watchlist-alert");
      const title = document.getElementById("watchlist-alert-title");
      const subtitle = document.getElementById("watchlist-alert-subtitle");
      const counts = document.getElementById("watchlist-alert-counts");
      const route = document.getElementById("watchlist-alert-route");
      const grid = document.getElementById("watchlist-alert-grid");
      const rows = sortedWatchlistAlerts(dashboardData.watchlist_alerts || []);
      const redRows = rows.filter((row) => row.alert_level === "red");
      const orangeRows = rows.filter((row) => row.alert_level === "orange");
      const yellowRows = rows.filter((row) => row.alert_level === "yellow");
      const greenRows = rows.filter((row) => row.alert_level === "green");
      const unknownRows = rows.filter((row) => row.alert_level === "unknown");
      const reviewRows = redRows.concat(orangeRows);
      const focusRows = (reviewRows.length ? reviewRows : rows).slice(0, 6);
      const panelClass = redRows.length || orangeRows.length ? "" : greenRows.length ? "healthy" : "unknown";

      panel.className = `watchlist-alert-panel ${panelClass}`.trim();
      counts.replaceChildren();
      grid.replaceChildren();
      route.textContent = "";

      const addPill = (level, label, value) => {
        const pill = document.createElement("span");
        pill.className = `alert-pill ${level}`;
        pill.textContent = `${label} ${value}`;
        counts.appendChild(pill);
      };
      addPill("red", "紅色", redRows.length);
      addPill("orange", "橘色", orangeRows.length);
      addPill("yellow", "黃色", yellowRows.length);
      addPill("green", "綠色", greenRows.length);
      if (unknownRows.length) addPill("unknown", "未知", unknownRows.length);

      if (!rows.length) {
        title.textContent = "追蹤名單尚未設定";
        subtitle.textContent = "更新 watchlist.csv 後，這裡會直接顯示需要優先檢查的 ticker、原因與替代方向。";
        route.hidden = true;
        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.textContent = "目前沒有追蹤名單資料。";
        grid.appendChild(empty);
        return;
      }

      const focusTickers = focusRows
        .map((row) => `${row.ticker} ${alertLevelLabels[row.alert_level] || row.alert_level}`)
        .join("、");
      title.textContent = reviewRows.length ? `${reviewRows.length} 檔需開盤前檢查` : "追蹤名單目前沒有紅橘警示";
      subtitle.textContent = reviewRows.length
        ? `優先名單：${focusTickers}。先確認弱動能原因，再往強勢產業與領導候選移動。`
        : "目前追蹤名單沒有明顯轉換警示，仍需用每日資料檢查是否轉弱。";
      const routeSource = reviewRows[0] || rows[0];
      route.hidden = false;
      appendReplacementRoute(route, routeSource);

      for (const row of focusRows) {
        const level = row.alert_level || "unknown";
        const card = document.createElement("article");
        card.className = `watchlist-alert-card ${level}`;

        const header = document.createElement("div");
        header.className = "watchlist-alert-card-header";
        const ticker = document.createElement("div");
        ticker.className = "watchlist-alert-ticker";
        ticker.textContent = row.ticker || "未命名";
        const badge = document.createElement("span");
        badge.className = `alert-level ${level}`;
        badge.textContent = `${alertLevelLabels[level] || level} / ${alertActionLabels[row.action] || displayText(row.action)}`;
        header.append(ticker, badge);
        card.appendChild(header);

        appendText(card, "watchlist-alert-meta", `${displayText(row.industry_group) || "未知產業"} · ${row.company_name || "公司名稱未取得"}`);

        const metrics = document.createElement("div");
        metrics.className = "watchlist-alert-metrics";
        [
          ["5日", formatSignedPercent(row.return_5d)],
          ["10日", formatSignedPercent(row.return_10d)],
          ["相對產業", formatSignedPercent(row.relative_strength_vs_industry)]
        ].forEach(([label, value]) => {
          const metric = document.createElement("span");
          metric.className = "watchlist-alert-metric";
          metric.textContent = `${label} ${value || "n/a"}`;
          metrics.appendChild(metric);
        });
        card.appendChild(metrics);

        appendText(
          card,
          "watchlist-alert-reason",
          `因為 ${row.alert_reason || "目前沒有明確警示"}，所以列為「${alertActionLabels[row.action] || displayText(row.action)}」。`
        );
        grid.appendChild(card);
      }
    }

    function compactWithUnit(value, divisor, suffix, digits) {
      return `${(value / divisor).toFixed(digits).replace(/\\.0$/, "")}${suffix}`;
    }

    function formatCompactVolume(value) {
      if (isMissing(value)) return "";
      const number = Number(value);
      if (!Number.isFinite(number)) return "";
      const abs = Math.abs(number);
      if (abs >= 1_000_000_000) return compactWithUnit(number, 1_000_000_000, "B", abs >= 10_000_000_000 ? 1 : 2);
      if (abs >= 1_000_000) return compactWithUnit(number, 1_000_000, "M", 1);
      if (abs >= 1_000) return compactWithUnit(number, 1_000, "K", abs >= 100_000 ? 0 : 1);
      return number.toLocaleString(undefined, { maximumFractionDigits: 0 });
    }

    function formatCell(value, column, rowIndex) {
      if (column.type === "rank") return String(rowIndex + 1);
      if (column.type === "percent" || column.type === "warningPercent") return formatPercent(value);
      if (column.type === "signedPercent") return formatSignedPercent(value);
      if (column.type === "number") return formatNumber(value, column.digits ?? 2);
      if (column.type === "compactVolume") return formatCompactVolume(value);
      if (column.type === "integer") return formatInteger(value);
      if (column.type === "signedInteger") return formatSignedInteger(value);
      if (column.type === "dataStatus") return Number(value) > 0 ? "有資料" : "無資料";
      if (column.key === "industry_group") return displayText(value);
      if (column.key === "data_status") return dataStatusLabels[value] || displayText(value);
      if (column.key === "peg_rating") return pegRatingLabels[value] || displayText(value);
      if (column.key === "peg_status") return pegStatusLabels[value] || displayText(value);
      if (column.key === "industry_regime") return regimeLabels[value] || displayText(value);
      if (column.key === "industry_risk_flag") return industryRiskFlagLabels[value] || displayText(value);
      if (column.key === "rotation_type") return rotationTypeLabels[value] || displayText(value);
      if (column.key === "causal_hypothesis") return causalHypothesisLabels[value] || displayText(value);
      if (column.key === "evidence_status") return evidenceStatusLabels[value] || displayText(value);
      if (column.key === "leader_type") return leaderTypeLabels[value] || displayText(value);
      if (["short_term_price_zone", "long_term_price_zone", "price_zone"].includes(column.key)) {
        return priceZoneLabels[value] || displayText(value);
      }
      if (column.key === "current_state") return currentStateLabels[value] || displayText(value);
      if (column.key === "watch_status") return watchStatusLabels[value] || displayText(value);
      if (typeof value === "boolean") return value ? "是" : "否";
      return isMissing(value) ? "" : String(value);
    }

    function classForCell(value, column) {
      const classes = [];
      if (["rank", "number", "compactVolume", "integer", "signedInteger", "percent", "signedPercent", "warningPercent"].includes(column.type)) {
        classes.push("numeric");
      }
      if (column.type === "rank") classes.push("rank");
      if (column.type === "ticker") classes.push("ticker");
      if (column.type === "company") classes.push("company");
      if (column.type === "dataStatus" && Number(value) <= 0) classes.push("warning");
      if (column.key === "data_status" && ["missing", "stale"].includes(String(value))) classes.push("warning");
      if (column.key === "data_status" && String(value) === "limited_history") classes.push("negative");
      if (column.key === "peg_rating" && String(value) === "undervalued") classes.push("positive");
      if (column.key === "peg_rating" && String(value) === "expensive") classes.push("warning");
      if (column.key === "peg_status" && String(value) !== "ok") classes.push("negative");
      if ((column.type === "percent" || column.type === "signedPercent" || column.type === "warningPercent" || column.type === "signedInteger") && !isMissing(value)) {
        if (value > 0) classes.push("positive");
        if (value < 0) classes.push(column.type === "warningPercent" ? "warning" : "negative");
      }
      return classes.join(" ");
    }

    function valueForColumn(row, column, rowIndex) {
      return column.key === "__rank" ? rowIndex : row[column.key];
    }

    function hasMixedSignal(row) {
      return row.early_momentum_signal === true && row.risk_warning === true;
    }

    function createMixedSignalBadge() {
      const badge = document.createElement("span");
      badge.className = "mixed-signal-badge";
      badge.textContent = "動能+風險";
      badge.title = "同時符合早期動能與風險提醒；判斷時不要只依賴單一候選名單。";
      return badge;
    }

    function isMobileIdentityColumn(column) {
      return ["ticker", "company_name", "industry_group"].includes(column.key);
    }

    function mobileTitleForRow(row, config, rowIndex) {
      if (!isMissing(row.ticker)) return String(row.ticker);
      if (!isMissing(row.industry_group)) return displayText(row.industry_group);

      const titleColumn = config.columns.find((column) => column.key !== "__rank" && !isMobileIdentityColumn(column));
      if (titleColumn) return formatCell(valueForColumn(row, titleColumn, rowIndex), titleColumn, rowIndex);
      return `第 ${rowIndex + 1} 筆`;
    }

    function mobileSubtitleForRow(row) {
      const parts = [];
      if (!isMissing(row.company_name)) parts.push(String(row.company_name));
      if (!isMissing(row.ticker) && !isMissing(row.industry_group)) parts.push(displayText(row.industry_group));
      return parts.join(" / ");
    }

    function ensureMobileCardList(id, tableWrap) {
      const existing = tableWrap.parentElement.querySelector(`[data-mobile-card-list="${id}"]`);
      if (existing) return existing;

      const list = document.createElement("div");
      list.className = "mobile-card-list";
      list.dataset.mobileCardList = id;
      tableWrap.insertAdjacentElement("afterend", list);
      return list;
    }

    function ensureFieldHelp(id, tableWrap) {
      const existing = tableWrap.parentElement.querySelector(`[data-field-help="${id}"]`);
      if (existing) return existing;

      const help = document.createElement("div");
      help.className = "field-help";
      help.dataset.fieldHelp = id;
      tableWrap.insertAdjacentElement("beforebegin", help);
      return help;
    }

    function renderFieldHelp(id, config, tableWrap) {
      const help = ensureFieldHelp(id, tableWrap);
      const describedColumns = config.columns.filter((column) => column.description);
      help.replaceChildren();

      if (!describedColumns.length) {
        help.hidden = true;
        return;
      }

      help.hidden = false;
      const intro = document.createElement("span");
      intro.className = "field-help-intro";
      intro.textContent = "欄位說明";
      help.appendChild(intro);

      for (const column of describedColumns) {
        const item = document.createElement("span");
        item.className = "field-help-item";

        const term = document.createElement("span");
        term.className = "field-help-term";
        term.textContent = column.label;

        const description = document.createElement("span");
        description.className = "field-help-description";
        description.textContent = column.description;

        item.append(term, description);
        help.appendChild(item);
      }
    }

    function renderMobileCards(id, config, rows, tableWrap) {
      const list = ensureMobileCardList(id, tableWrap);
      list.replaceChildren();

      if (!rows.length) {
        list.hidden = true;
        return;
      }

      list.hidden = false;
      rows.forEach((row, rowIndex) => {
        const card = document.createElement("article");
        card.className = "mobile-row-card";
        if (hasMixedSignal(row)) {
          card.classList.add("mixed-signal-card");
        }

        const header = document.createElement("div");
        header.className = "mobile-card-header";

        const title = document.createElement("div");
        title.className = "mobile-card-title";
        if (!isMissing(row.industry_group) && isMissing(row.ticker)) {
          title.appendChild(createIndustryButton(row.industry_group));
        } else {
          title.appendChild(document.createTextNode(mobileTitleForRow(row, config, rowIndex)));
        }
        if (hasMixedSignal(row)) {
          title.appendChild(createMixedSignalBadge());
        }
        header.appendChild(title);

        if (config.columns.some((column) => column.key === "__rank")) {
          const rank = document.createElement("div");
          rank.className = "mobile-card-rank";
          rank.textContent = `第 ${rowIndex + 1} 名`;
          header.appendChild(rank);
        }

        card.appendChild(header);

        const subtitleText = mobileSubtitleForRow(row);
        if (subtitleText) {
          const subtitle = document.createElement("div");
          subtitle.className = "mobile-card-subtitle";
          if (!isMissing(row.ticker) && !isMissing(row.industry_group)) {
            if (!isMissing(row.company_name)) {
              subtitle.appendChild(document.createTextNode(`${row.company_name} / `));
            }
            subtitle.appendChild(createIndustryButton(row.industry_group));
          } else {
            subtitle.textContent = subtitleText;
          }
          card.appendChild(subtitle);
        }

        const metrics = document.createElement("div");
        metrics.className = "mobile-metrics";
        for (const column of config.columns) {
          if (isMobileIdentityColumn(column)) continue;

          const value = valueForColumn(row, column, rowIndex);
          const metric = document.createElement("div");
          metric.className = "mobile-metric";

          const label = document.createElement("div");
          label.className = "mobile-metric-label";
          label.textContent = column.label;

          const metricValue = document.createElement("div");
          metricValue.className = `mobile-metric-value ${classForCell(value, column)}`.trim();
          metricValue.textContent = formatCell(value, column, rowIndex);

          if (column.description) {
            metric.title = column.description;
          }

          metric.append(label, metricValue);
          if (column.description) {
            const hint = document.createElement("div");
            hint.className = "mobile-metric-hint";
            hint.textContent = column.description;
            metric.appendChild(hint);
          }
          metrics.appendChild(metric);
        }

        card.appendChild(metrics);
        list.appendChild(card);
      });
    }

    function finiteNumber(value) {
      const number = Number(value);
      return Number.isFinite(number) ? number : null;
    }

    function renderMomentumMap() {
      const momentumMap = dashboardData.momentum_map || {};
      renderMomentumSummary(momentumMap.summary || {});
      renderMomentumBars(momentumMap.industry_bars || []);
      renderMomentumScatter(momentumMap.holding_alignment || []);
      renderMomentumGaps(momentumMap.momentum_exposure_gaps || []);
    }

    function renderMomentumSummary(summary) {
      const grid = document.getElementById("momentum-map-summary");
      grid.replaceChildren();
      const tiles = [
        ["強勢產業曝險", summary.strong_industry_holding_count, "持股落在動能領先或早期修復產業的檔數。", ""],
        ["落後同產業", summary.lagging_holding_count, "相對產業低於 -10% 的持股檔數。", Number(summary.lagging_holding_count || 0) > 0 ? "warning" : ""],
        ["優先檢查", summary.priority_review_count, "alert level 為紅色或橘色的持股檔數。", Number(summary.priority_review_count || 0) > 0 ? "warning" : "healthy"],
        ["曝險缺口", summary.momentum_exposure_gap_count, "目前強勢或修復產業中，watchlist 尚未映射到持股的產業數。", ""]
      ];
      for (const [label, value, description, stateClass] of tiles) {
        appendQualityTile(grid, label, formatInteger(value), description, stateClass);
      }
    }

    function renderMomentumBars(rows) {
      const container = document.getElementById("momentum-industry-bars");
      const count = document.getElementById("momentum-map-industry-count");
      container.replaceChildren();
      count.textContent = `${rows.length} 個產業`;
      if (!rows.length) {
        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.textContent = "目前沒有產業動能資料。";
        container.appendChild(empty);
        return;
      }

      const values = rows.map((row) => finiteNumber(row.return_10d)).filter((value) => value !== null);
      const minValue = Math.min(0, ...values);
      const maxValue = Math.max(0, ...values);
      const range = maxValue - minValue || 1;
      const zeroPct = ((0 - minValue) / range) * 100;
      const visibleRows = window.innerWidth <= 520 ? rows.slice(0, 10) : rows;

      visibleRows.forEach((row) => {
        const value = finiteNumber(row.return_10d) ?? 0;
        const startPct = ((Math.min(0, value) - minValue) / range) * 100;
        const endPct = ((Math.max(0, value) - minValue) / range) * 100;
        const widthPct = Math.max(1, endPct - startPct);
        const breadth = finiteNumber(row.breadth_score);
        const riskFlag = String(row.industry_risk_flag || "none");
        const regime = String(row.industry_regime || "neutral");
        const tickersForIndustry = Array.isArray(row.holding_tickers) ? row.holding_tickers : [];
        const rowEl = document.createElement("div");
        rowEl.className = `momentum-bar-row ${riskFlag !== "none" ? "has-risk" : ""}`.trim();
        rowEl.title = `${displayText(row.industry_group)} · 10日 ${formatSignedPercent(row.return_10d) || "n/a"} · 5日 ${formatSignedPercent(row.return_5d) || "n/a"} · 廣度 ${formatPercent(row.breadth_score) || "n/a"} · ${regimeLabels[regime] || displayText(regime) || "n/a"}${tickersForIndustry.length ? ` · 持股 ${tickersForIndustry.join(", ")}` : ""}`;

        const label = document.createElement("div");
        label.className = "momentum-industry-label";
        label.textContent = displayText(row.industry_group);
        if (riskFlag !== "none") {
          const badge = document.createElement("span");
          badge.className = "momentum-risk-badge";
          badge.textContent = industryRiskFlagLabels[riskFlag] || displayText(riskFlag);
          label.appendChild(badge);
        }
        if (tickersForIndustry.length) {
          const tickerText = document.createElement("span");
          tickerText.className = "momentum-inline-tickers";
          tickerText.textContent = tickersForIndustry.join(",");
          label.appendChild(tickerText);
        }
        const meta = document.createElement("div");
        meta.className = "momentum-industry-meta";
        const tickerSummary = tickersForIndustry.length ? ` · 持股 ${tickersForIndustry.join(", ")}` : "";
        meta.textContent = `5日 ${formatSignedPercent(row.return_5d) || "n/a"} · 廣度 ${formatPercent(row.breadth_score) || "n/a"} · ${regimeLabels[regime] || displayText(regime) || "n/a"}${tickerSummary}`;
        label.appendChild(meta);

        const track = document.createElement("div");
        track.className = "momentum-bar-track";
        track.title = `${displayText(row.industry_group)} · 10日 ${formatSignedPercent(row.return_10d) || "n/a"} · 廣度 ${formatPercent(row.breadth_score) || "n/a"}`;
        const zero = document.createElement("div");
        zero.className = "momentum-zero-line";
        zero.style.left = `${zeroPct}%`;
        const fill = document.createElement("div");
        fill.className = `momentum-bar-fill ${regime}`.trim();
        fill.style.left = `${startPct}%`;
        fill.style.width = `${widthPct}%`;
        const valueLabel = document.createElement("div");
        valueLabel.className = "momentum-bar-value";
        valueLabel.textContent = formatSignedPercent(row.return_10d) || "n/a";
        track.append(zero, fill, valueLabel);
        if (breadth !== null) {
          const marker = document.createElement("div");
          marker.className = "momentum-breadth-marker";
          marker.style.left = `${Math.max(0, Math.min(100, breadth * 100))}%`;
          marker.title = `廣度 ${formatPercent(row.breadth_score)}`;
          track.appendChild(marker);
        }

        rowEl.append(label, track);
        container.appendChild(rowEl);
      });
    }

    function renderMomentumScatter(rows) {
      const scatter = document.getElementById("momentum-alignment-scatter");
      const count = document.getElementById("momentum-map-holding-count");
      scatter.replaceChildren();
      count.textContent = `${rows.length} 檔持股`;
      const xAxis = document.createElement("div");
      xAxis.className = "momentum-scatter-axis x";
      xAxis.textContent = "相對產業 →";
      const yAxis = document.createElement("div");
      yAxis.className = "momentum-scatter-axis y";
      yAxis.textContent = "10日報酬 ↑";
      scatter.append(xAxis, yAxis);

      const plottedRows = rows.filter((row) => finiteNumber(row.relative_strength_vs_industry) !== null && finiteNumber(row.return_10d) !== null);
      if (!plottedRows.length) {
        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.textContent = "目前沒有可視覺化的持股相對產業資料。";
        scatter.appendChild(empty);
        return;
      }

      const xs = plottedRows.map((row) => finiteNumber(row.relative_strength_vs_industry));
      const ys = plottedRows.map((row) => finiteNumber(row.return_10d));
      const xPad = 0.04;
      const yPad = 0.04;
      const minX = Math.min(-0.10, 0, ...xs) - xPad;
      const maxX = Math.max(0.10, 0, ...xs) + xPad;
      const minY = Math.min(-0.05, 0, ...ys) - yPad;
      const maxY = Math.max(0.05, 0, ...ys) + yPad;
      const xRange = maxX - minX || 1;
      const yRange = maxY - minY || 1;

      plottedRows.forEach((row) => {
        const x = finiteNumber(row.relative_strength_vs_industry);
        const y = finiteNumber(row.return_10d);
        const left = 8 + (((x - minX) / xRange) * 84);
        const top = 92 - (((y - minY) / yRange) * 84);
        const point = document.createElement("button");
        point.type = "button";
        point.className = `momentum-point ${row.alert_level || ""}`.trim();
        point.style.left = `${Math.max(8, Math.min(92, left))}%`;
        point.style.top = `${Math.max(8, Math.min(92, top))}%`;
        point.textContent = row.ticker || "";
        const alertLabel = alertLevelLabels[row.alert_level] || displayText(row.alert_level) || "n/a";
        const watchLabel = watchStatusLabels[row.watch_status] || displayText(row.watch_status) || "n/a";
        point.setAttribute(
          "data-caption",
          `${row.ticker || ""} · ${displayText(row.industry_group) || "n/a"} · 10日 ${formatSignedPercent(row.return_10d) || "n/a"} · 相對產業 ${formatSignedPercent(row.relative_strength_vs_industry) || "n/a"} · 相對量 ${formatNumber(row.relative_volume, 2) || "n/a"} · ${alertLabel} · ${watchLabel}`
        );
        scatter.appendChild(point);
      });
    }

    function renderMomentumGaps(rows) {
      const container = document.getElementById("momentum-gap-list");
      container.replaceChildren();
      if (!rows.length) return;
      rows.slice(0, 6).forEach((row) => {
        const chip = document.createElement("span");
        chip.className = "momentum-gap-chip";
        chip.textContent = `${displayText(row.industry_group)} exposure gap · 10日 ${formatSignedPercent(row.return_10d) || "n/a"}`;
        chip.title = `${regimeLabels[row.industry_regime] || displayText(row.industry_regime)} · ${industryRiskFlagLabels[row.industry_risk_flag] || displayText(row.industry_risk_flag) || "無風險旗標"}`;
        container.appendChild(chip);
      });
    }

    function renderSummary() {
      const summary = dashboardData.summary;
      const grid = document.querySelector("[data-summary-grid]");
      const tiles = [
        ["追蹤檔數", summary.total_tickers, "目前觀察清單內的全部標的。"],
        ["有資料", summary.tickers_with_data, "成功取得近 1 年日線資料的標的。"],
        ["早期", summary.early_count, "3 日與 5 日轉強，偏早期觀察名單。"],
        ["確認", summary.confirmed_count, "5 日/10 日轉正，且均線與上漲天數支持。"],
        ["強勢", summary.strong_count, "確認動能、跑贏同產業，且量能放大。"],
        ["風險", summary.risk_count, "近期回撤過深，或價格偏離 20 日均線過多。"]
      ];

      document.getElementById("generated-date").textContent = summary.latest_date
        ? `最新市場日期：${summary.latest_date}`
        : "最新市場日期無法取得";

      for (const [label, value, description] of tiles) {
        const tile = document.createElement("div");
        tile.className = "summary-tile";

        const labelEl = document.createElement("div");
        labelEl.className = "summary-label";
        labelEl.textContent = label;

        const valueEl = document.createElement("div");
        valueEl.className = "summary-value";
        valueEl.textContent = formatInteger(value);

        const descriptionEl = document.createElement("div");
        descriptionEl.className = "summary-description";
        descriptionEl.textContent = description;

        tile.append(labelEl, valueEl, descriptionEl);
        grid.appendChild(tile);
      }
    }

    function renderDailyBrief() {
      const grid = document.getElementById("daily-brief-grid");
      const cards = dashboardData.daily_brief?.cards || [];
      grid.replaceChildren();

      for (const card of cards) {
        const cardEl = document.createElement("article");
        cardEl.className = `daily-brief-card ${card.status || ""}`.trim();

        const title = document.createElement("div");
        title.className = "daily-brief-title";
        title.textContent = card.title || "摘要";

        const headline = document.createElement("div");
        headline.className = "daily-brief-headline";
        headline.textContent = card.headline || "目前沒有摘要資料";

        const list = document.createElement("ul");
        list.className = "daily-brief-list";
        const details = Array.isArray(card.details) ? card.details : [];
        for (const detail of details) {
          const item = document.createElement("li");
          item.textContent = detail || "";
          list.appendChild(item);
        }

        cardEl.append(title, headline, list);
        grid.appendChild(cardEl);
      }
    }

    function appendQualityTile(grid, label, value, description, stateClass = "", linkHref = "") {
      const tile = document.createElement("div");
      tile.className = `quality-tile ${stateClass}`.trim();

      const labelEl = document.createElement("div");
      labelEl.className = "quality-label";
      labelEl.textContent = label;

      const valueEl = document.createElement("div");
      valueEl.className = "quality-value";
      if (linkHref) {
        const link = document.createElement("a");
        link.href = linkHref;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = value;
        valueEl.appendChild(link);
      } else {
        valueEl.textContent = value;
      }

      const descriptionEl = document.createElement("div");
      descriptionEl.className = "quality-description";
      descriptionEl.textContent = description;

      tile.append(labelEl, valueEl, descriptionEl);
      grid.appendChild(tile);
    }

    function renderUpdateHealth() {
      const health = dashboardData.update_health || {};
      const grid = document.getElementById("update-health-grid");
      const status = health.update_health_status || "unknown";
      const stateClass = status === "healthy" ? "healthy" : status === "warning" ? "warning" : "unknown";
      const runLabel = health.github_run_url
        ? `Run ${health.github_run_id || ""}`.trim()
        : health.run_context === "github_actions"
          ? "GitHub Actions"
          : "本機執行";
      const sha = health.git_sha ? String(health.git_sha).slice(0, 7) : "";
      const cards = [
        ["健康狀態", updateHealthLabels[status] || displayText(status), health.update_health_note || "尚無健康狀態說明。", stateClass, ""],
        ["最新市場日期", health.latest_market_date || "無法取得", "本次輸出使用的最新市場資料日期。", "", ""],
        ["資料年齡", isMissing(health.market_data_age_days) ? "無法取得" : `${formatInteger(health.market_data_age_days)} 天`, "以紐約日期計算；大於 3 天會提示注意。", "", ""],
        ["產生時間", health.generated_at_new_york || "無法取得", "America/New_York 時區的產生時間。", "", ""],
        ["資料成功率", formatPercent(health.success_rate), "成功取得資料的 ticker 比例。", "", ""],
        ["執行來源", runLabel, sha ? `git ${sha}` : "本機或 Actions 執行資訊。", "", health.github_run_url || ""]
      ];

      for (const [label, value, description, cardClass, href] of cards) {
        appendQualityTile(grid, label, value, description, cardClass, href);
      }
    }

    function renderDataQuality() {
      const quality = dashboardData.data_quality.summary;
      const grid = document.getElementById("data-quality-grid");
      const missingCount = Number(quality.missing_count || 0);
      const staleCount = Number(quality.stale_count || 0);
      const limitedCount = Number(quality.limited_history_count || 0);
      const cards = [
        ["資料來源", quality.data_source || "Yahoo Finance via yfinance", "透過 yfinance 抓取 Yahoo Finance 日線資料。", ""],
        ["最新市場日期", quality.latest_market_date || "無法取得", "本次有效資料中最新的交易日期。", ""],
        ["成功取得資料", `${formatInteger(quality.tickers_with_data)}/${formatInteger(quality.total_tickers)}`, "有至少一筆可用日線資料的 ticker。", ""],
        ["缺資料", formatInteger(missingCount), "完全沒有可用日線資料的 ticker。", missingCount > 0 ? "warning" : ""],
        ["資料落後", formatInteger(staleCount), "最新日期早於本次市場日期的 ticker。", staleCount > 0 ? "warning" : ""],
        ["歷史不足", formatInteger(limitedCount), "少於 60 筆日線資料，長週期位置需保守解讀。", limitedCount > 0 ? "warning" : ""]
      ];

      for (const [label, value, description, stateClass] of cards) {
        appendQualityTile(grid, label, value, description, stateClass);
      }
    }

    function parsePortfolioInput(raw) {
      const entries = [];
      const lines = String(raw || "").split(/\\n+/).map((line) => line.trim()).filter(Boolean);
      for (const line of lines) {
        const tokens = line.split(/[,\\s]+/).filter(Boolean);
        const ticker = (tokens[0] || "").toUpperCase();
        if (!ticker) continue;
        entries.push({ ticker, weight: 1 });
      }
      return entries;
    }

    function renderPortfolioSimulator() {
      const runBtn = document.getElementById("portfolio-run");
      const sampleBtn = document.getElementById("portfolio-sample");
      const input = document.getElementById("portfolio-input");
      const resultGrid = document.getElementById("portfolio-result");
      const actionsList = document.getElementById("portfolio-actions");
      const alignmentWrap = document.getElementById("portfolio-alignment");
      const alignmentTable = document.getElementById("portfolio-alignment-table");
      const tickerRows = (dashboardData.top_relative_strength || [])
        .concat(dashboardData.strong_candidates || [])
        .concat(dashboardData.early_candidates || [])
        .concat(dashboardData.risk_warnings || []);
      const byTicker = new Map();
      tickerRows.forEach((row) => {
        if (row?.ticker && !byTicker.has(row.ticker)) byTicker.set(row.ticker, row);
      });

      sampleBtn.addEventListener("click", () => {
        input.value = "VRT\\nDELL\\nSPOT\\nSOFI";
      });

      runBtn.addEventListener("click", () => {
        const holdings = parsePortfolioInput(input.value);
        resultGrid.replaceChildren();
        actionsList.replaceChildren();
        alignmentWrap.hidden = true;
        alignmentTable.replaceChildren();
        if (!holdings.length) return;

        let modeledWeight = 0;
        let riskWeight = 0;
        let strongWeight = 0;
        let earlyWeight = 0;
        let alignedWeight = 0;
        const missing = [];
        const alignmentRows = [];
        for (const h of holdings) {
          const row = byTicker.get(h.ticker);
          if (!row) {
            missing.push(h.ticker);
            alignmentRows.push({ ticker: h.ticker, weight: h.weight, status: "無資料", note: "未納入追蹤清單" });
            continue;
          }
          modeledWeight += h.weight;
          if (row.risk_warning === true) riskWeight += h.weight;
          if (row.strong_momentum_signal === true) strongWeight += h.weight;
          if (row.early_momentum_signal === true) earlyWeight += h.weight;
          const aligned = (row.strong_momentum_signal === true || row.early_momentum_signal === true) && row.risk_warning !== true;
          if (aligned) alignedWeight += h.weight;
          alignmentRows.push({
            ticker: h.ticker,
            weight: h.weight,
            status: aligned ? "一致" : "不一致",
            note: row.risk_warning === true
              ? "有風險警示"
              : row.strong_momentum_signal === true
                ? "強勢動能"
                : row.early_momentum_signal === true
                  ? "早期動能"
                  : "無動能訊號"
          });
        }

        const addCard = (label, value, desc, state = "") => appendQualityTile(resultGrid, label, value, desc, state);
        addCard("可判讀檔數", formatInteger(modeledWeight), "有出現在本頁資料中的追蹤 ticker。");
        addCard("風險檔數", formatInteger(riskWeight), "被標記為 risk warning 的追蹤 ticker。", riskWeight >= 1 ? "warning" : "");
        addCard("強勢動能檔數", formatInteger(strongWeight), "符合 strong momentum 的追蹤 ticker。", strongWeight >= 1 ? "healthy" : "");
        addCard("早期動能檔數", formatInteger(earlyWeight), "符合 early momentum 的追蹤 ticker。");
        addCard("動能一致檔數", formatInteger(alignedWeight), "符合早期/強勢動能且無風險警示的追蹤 ticker。", alignedWeight >= 1 ? "healthy" : "");

        const actions = [];
        if (riskWeight >= 1) actions.push("優先檢視 risk warning 的追蹤 ticker，確認是否需要從弱動能名單移到替代候選研究。");
        if (strongWeight < 1) actions.push("追蹤名單缺少強勢動能，可從「強勢動能候選」挑 1-2 檔加入觀察。");
        if (earlyWeight > strongWeight) actions.push("追蹤名單偏早期訊號，建議等待確認訊號再提高優先級。");
        if (!actions.length) actions.push("追蹤名單結構相對平衡，持續用每日訊號檢查是否轉弱。");
        if (missing.length) actions.push(`以下 ticker 不在目前追蹤資料：${missing.join(", ")}。建議加入 tickers.csv 以納入完整檢查。`);
        actions.forEach((text) => {
          const li = document.createElement("li");
          li.textContent = text;
          actionsList.appendChild(li);
        });

        if (alignmentRows.length) {
          alignmentWrap.hidden = false;
          const thead = document.createElement("thead");
          const headRow = document.createElement("tr");
          ["Ticker", "計數", "與動能是否一致", "判讀"].forEach((label) => {
            const th = document.createElement("th");
            th.textContent = label;
            headRow.appendChild(th);
          });
          thead.appendChild(headRow);

          const tbody = document.createElement("tbody");
          alignmentRows.forEach((item) => {
            const tr = document.createElement("tr");
            const cells = [
              item.ticker,
              formatInteger(item.weight),
              item.status === "一致" ? "一致" : item.status === "不一致" ? "不一致" : "-",
              item.note
            ];
            cells.forEach((text, idx) => {
              const td = document.createElement("td");
              if (idx === 1) td.className = "numeric";
              td.textContent = text;
              tr.appendChild(td);
            });
            tbody.appendChild(tr);
          });
          alignmentTable.replaceChildren(thead, tbody);
        }
      });
    }

    function renderTable(id, config) {
      const rows = config.rows || [];
      const table = document.querySelector(`[data-table="${id}"]`);
      const tableWrap = table.parentElement;
      const empty = document.querySelector(`[data-empty-for="${id}"]`);
      const count = document.querySelector(`[data-count-for="${id}"]`);
      renderFieldHelp(id, config, tableWrap);
      if (count) {
        count.textContent = `${rows.length} 筆`;
      }

      if (!rows.length) {
        tableWrap.hidden = true;
        renderMobileCards(id, config, rows, tableWrap);
        empty.hidden = false;
        return;
      }

      tableWrap.hidden = false;
      empty.hidden = true;
      const thead = document.createElement("thead");
      const headerRow = document.createElement("tr");
      for (const column of config.columns) {
        const th = document.createElement("th");
        th.textContent = column.label;
        if (column.description) {
          th.title = column.description;
          th.setAttribute("aria-label", `${column.label}：${column.description}`);
        }
        if (["rank", "number", "integer", "signedInteger", "percent", "signedPercent", "warningPercent"].includes(column.type)) {
          th.className = "numeric";
        }
        headerRow.appendChild(th);
      }
      thead.appendChild(headerRow);

      const tbody = document.createElement("tbody");
      rows.forEach((row, rowIndex) => {
        const tr = document.createElement("tr");
        if (hasMixedSignal(row)) {
          tr.classList.add("mixed-signal-row");
        }
        for (const column of config.columns) {
          const value = valueForColumn(row, column, rowIndex);
          const td = document.createElement("td");
          td.className = classForCell(value, column);
          if (column.key === "industry_group" && !isMissing(value)) {
            td.appendChild(createIndustryButton(value));
          } else {
            td.textContent = formatCell(value, column, rowIndex);
          }
          if (column.type === "ticker" && hasMixedSignal(row)) {
            td.appendChild(createMixedSignalBadge());
          }
          tr.appendChild(td);
        }
        tbody.appendChild(tr);
      });

      table.replaceChildren(thead, tbody);
      renderMobileCards(id, config, rows, tableWrap);
    }

    let lastIndustryTrigger = null;

    function renderIndustryModalTable(rows) {
      const table = document.getElementById("industry-modal-table");
      const thead = document.createElement("thead");
      const headerRow = document.createElement("tr");
      for (const column of constituentColumns) {
        const th = document.createElement("th");
        th.textContent = column.label;
        if (column.description) {
          th.title = column.description;
        }
        if (["number", "compactVolume", "integer", "percent", "dataStatus"].includes(column.type)) {
          th.className = "numeric";
        }
        headerRow.appendChild(th);
      }
      thead.appendChild(headerRow);

      const tbody = document.createElement("tbody");
      for (const row of rows) {
        const tr = document.createElement("tr");
        if (hasMixedSignal(row)) {
          tr.classList.add("mixed-signal-row");
        }
        for (const column of constituentColumns) {
          const value = row[column.key];
          const td = document.createElement("td");
          td.className = classForCell(value, column);
          td.textContent = formatCell(value, column, 0);
          if (column.type === "ticker" && hasMixedSignal(row)) {
            td.appendChild(createMixedSignalBadge());
          }
          tr.appendChild(td);
        }
        tbody.appendChild(tr);
      }

      table.replaceChildren(thead, tbody);
    }

    function renderIndustrySummary(rows) {
      const summary = document.getElementById("industry-modal-summary");
      summary.replaceChildren();
      const withData = rows.filter((row) => Number(row.data_points) > 0).length;
      const confirmed = rows.filter((row) => row.confirmed_momentum_signal === true).length;
      const strong = rows.filter((row) => row.strong_momentum_signal === true).length;
      const risk = rows.filter((row) => row.risk_warning === true).length;
      const trackedLeaders = rows.filter((row) => row.leader_type && row.leader_type !== "non_leader").length;
      const pills = [
        `${rows.length} 檔 ticker`,
        `${withData} 檔有資料`,
        `${confirmed} 檔確認動能`,
        `${strong} 檔強勢動能`,
        `${risk} 檔風險提醒`,
        `${trackedLeaders} 檔已標註 leader metadata`
      ];

      for (const text of pills) {
        const pill = document.createElement("span");
        pill.className = "industry-modal-pill";
        pill.textContent = text;
        summary.appendChild(pill);
      }
    }

    function openIndustryModal(industryGroup, trigger) {
      const modal = document.getElementById("industry-modal");
      const rows = dashboardData.industry_constituents[String(industryGroup)] || [];
      lastIndustryTrigger = trigger || null;
      document.getElementById("industry-modal-title").textContent = `${displayText(industryGroup)} ticker 組合`;
      document.getElementById("industry-modal-subtitle").textContent =
        "此清單就是產業平均、廣度、輪動與趨勢判讀所依據的 watchlist 成分。";
      document.getElementById("industry-modal-empty").hidden = rows.length > 0;
      document.querySelector(".industry-modal-table-wrap").hidden = rows.length === 0;
      renderIndustrySummary(rows);
      renderIndustryModalTable(rows);
      modal.hidden = false;
      document.querySelector(".industry-modal-close").focus();
    }

    function closeIndustryModal() {
      const modal = document.getElementById("industry-modal");
      modal.hidden = true;
      if (lastIndustryTrigger) {
        lastIndustryTrigger.focus();
      }
    }

    document.querySelectorAll("[data-close-industry-modal]").forEach((element) => {
      element.addEventListener("click", closeIndustryModal);
    });

    const navContainer = document.querySelector(".dashboard-nav");
    function jumpToSection(target) {
      if (!target) return;
      const section = document.querySelector(target);
      if (!section) return;
      const navHeight = navContainer ? Math.ceil(navContainer.getBoundingClientRect().height) : 0;
      const y = section.getBoundingClientRect().top + window.scrollY - navHeight - 12;
      window.scrollTo({ top: Math.max(0, y), behavior: "smooth" });
      if (window.location.hash !== target) {
        window.history.replaceState(null, "", target);
      }
    }

    const sectionJump = document.getElementById("section-jump");
    if (sectionJump) {
      sectionJump.addEventListener("change", (event) => {
        const target = event.target.value;
        if (!target) return;
        jumpToSection(target);
        event.target.value = "";
      });
    }

    if (navContainer) {
      let lastScrollY = window.scrollY;
      const collapseOnMobile = () => {
        if (window.innerWidth > 640) {
          navContainer.classList.remove("is-collapsed");
          lastScrollY = window.scrollY;
          return;
        }
        const currentY = window.scrollY;
        const scrollingDown = currentY > lastScrollY + 6;
        navContainer.classList.toggle("is-collapsed", scrollingDown && currentY > 80);
        lastScrollY = currentY;
      };
      collapseOnMobile();
      window.addEventListener("scroll", collapseOnMobile, { passive: true });
      window.addEventListener("resize", collapseOnMobile);
    }

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !document.getElementById("industry-modal").hidden) {
        closeIndustryModal();
      }
    });

    renderMomentumMap();
    renderWatchlistAlert();
    renderSummary();
    renderDailyBrief();
    renderUpdateHealth();
    renderDataQuality();
    renderPortfolioSimulator();
    const breadthRows = Object.values(dashboardData.industry_breadth)
      .reduce((total, rows) => total + rows.length, 0);
    document.getElementById("breadth-status").textContent =
      `${breadthRows} 筆觀察`;
    document.getElementById("rotation-history-status").textContent =
      `${dashboardData.rotation_trend.date_count} 個歷史日期`;
    const intelligenceRows = Object.values(dashboardData.industry_trend_intelligence)
      .reduce((total, rows) => total + rows.length, 0);
    document.getElementById("trend-intelligence-status").textContent =
      `${intelligenceRows} 筆訊號`;
    const leaderRows = Object.values(dashboardData.leader_accumulation)
      .reduce((total, rows) => total + rows.length, 0);
    document.getElementById("leader-accumulation-status").textContent =
      `${leaderRows} 筆觀察`;
    for (const [id, config] of Object.entries(tableConfigs)) {
      renderTable(id, config);
    }
  </script>
</body>
</html>
"""
    return html.replace("__DASHBOARD_DATA__", data_json)


def write_dashboard(
    ticker_output: pd.DataFrame,
    industry_output: pd.DataFrame,
    rotation_history: pd.DataFrame,
    path: Path,
    update_health_output: pd.DataFrame | None = None,
    watchlist_alerts: pd.DataFrame | None = None,
) -> None:
    dashboard_data = build_dashboard_data(
        ticker_output,
        industry_output,
        rotation_history,
        update_health_output,
        watchlist_alerts,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_dashboard_html(dashboard_data), encoding="utf-8")
