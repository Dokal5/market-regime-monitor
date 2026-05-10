from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import BREADTH_COLUMNS, INDUSTRY_REGIME_COLUMN, INDUSTRY_TREND_COLUMNS, METRIC_COLUMNS
from src.data_quality import build_data_quality_summary
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
) -> dict[str, Any]:
    tickers = ticker_output.copy()
    industries = industry_output.copy()

    ticker_numeric_columns = [
        "data_points",
        "return_3d",
        "return_5d",
        "return_10d",
        "return_20d",
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
    ]
    for column in ticker_numeric_columns:
        if column in tickers.columns:
            tickers[column] = pd.to_numeric(tickers[column], errors="coerce")

    for column in ["early_momentum_signal", "confirmed_momentum_signal", "strong_momentum_signal", "risk_warning"]:
        if column not in tickers.columns:
            tickers[column] = False
        tickers[column] = tickers[column].fillna(False).astype(bool)
    for column, default in [
        ("leader_type", "non_leader"),
        (INDUSTRY_REGIME_COLUMN, "neutral"),
        ("short_term_price_zone", "neutral"),
        ("long_term_price_zone", "neutral"),
        ("price_zone", "neutral"),
        ("current_state", "sideways_base"),
        ("watch_status", "avoid_for_now"),
        ("data_status", "missing"),
        ("data_quality_note", ""),
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
        "relative_volume",
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
        "early_momentum_signal",
        "risk_warning",
    ]
    leader_stock_columns = [
        "ticker",
        "company_name",
        "industry_group",
        INDUSTRY_REGIME_COLUMN,
        "leader_type",
        "industry_quality_score",
        "watch_status",
        "current_state",
        "short_term_price_zone",
        "long_term_price_zone",
        "price_zone",
        "return_5d",
        "return_10d",
        "relative_strength_vs_industry",
        "relative_volume",
        "max_drawdown_10d",
        "distance_from_20d_ma",
        "distance_from_52w_high",
        "position_in_52w_range",
        "risk_warning",
    ]
    leader_industry_columns = [
        "industry_group",
        INDUSTRY_REGIME_COLUMN,
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
        "relative_strength_vs_industry",
        "latest_volume",
        "avg_volume_20d",
        "relative_volume",
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
    not_eligible_industries = industries[
        industries[INDUSTRY_REGIME_COLUMN].isin(["neutral", "weak", "exhaustion"])
    ].sort_values(["return_10d", "breadth_score"], ascending=[False, False], na_position="last")
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

    .summary-description {
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
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
    <section class="summary-grid" id="summary-grid" aria-label="動能摘要"></section>

    <section class="dashboard-section update-health-section">
      <div class="section-heading">
        <h2>更新健康狀態</h2>
      </div>
      <p class="section-note">檢查本次輸出是否由排程正常產生、資料是否新鮮，以及 GitHub Actions run 是否可追溯；這不是投資訊號。</p>
      <div class="quality-grid" id="update-health-grid"></div>
    </section>

    <section class="dashboard-section data-quality-section">
      <div class="section-heading">
        <h2>資料來源與品質</h2>
        <span class="row-count" data-count-for="data-quality-issues"></span>
      </div>
      <p class="section-note">資料來自 Yahoo Finance via yfinance，適合研究與觀察，不是機構級資料源，也不是正式交易或投資建議。</p>
      <div class="quality-grid" id="data-quality-grid"></div>
      <div class="table-wrap">
        <table data-table="data-quality-issues"></table>
      </div>
      <p class="empty-state" data-empty-for="data-quality-issues" hidden>目前沒有缺資料或資料落後的 ticker。</p>
    </section>

    <section class="dashboard-section">
      <div class="section-heading">
        <h2>產業動能排名：依平均 10 日報酬排序</h2>
        <span class="row-count" data-count-for="industry-momentum"></span>
      </div>
      <p class="section-note">排序仍依平均 10 日報酬；欄位按 5 日、10 日、20 日排列，方便比較短中期動能。</p>
      <div class="table-wrap">
        <table data-table="industry-momentum"></table>
      </div>
      <p class="empty-state" data-empty-for="industry-momentum" hidden>目前沒有產業資料。</p>
    </section>

    <section class="dashboard-section">
      <div class="section-heading">
        <h2>產業確認動能比例排名</h2>
        <span class="row-count" data-count-for="industry-confirmed"></span>
      </div>
      <p class="section-note">顯示各產業中符合確認動能條件的股票比例，比例越高代表產業內部動能越一致。</p>
      <div class="table-wrap">
        <table data-table="industry-confirmed"></table>
      </div>
      <p class="empty-state" data-empty-for="industry-confirmed" hidden>目前沒有產業訊號資料。</p>
    </section>

    <section class="dashboard-section">
      <div class="section-heading">
        <h2>Industry Breadth</h2>
        <span class="row-count" id="breadth-status"></span>
      </div>
      <p class="section-note">產業廣度用來判斷動能是多數成分股一起轉強，還是少數領頭股拉高平均報酬。</p>
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

    <section class="dashboard-section">
      <div class="section-heading">
        <h2>產業輪動趨勢</h2>
        <span class="row-count" id="rotation-history-status"></span>
      </div>
      <p class="section-note">比較歷史快照中的排名與平均報酬變化；至少需要兩個不同日期的快照才會出現完整趨勢。</p>
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

    <section class="dashboard-section">
      <div class="section-heading">
        <h2>產業趨勢判讀</h2>
        <span class="row-count" id="trend-intelligence-status"></span>
      </div>
      <p class="section-note">用排名變化、持續性、加速與衰竭警示輔助判斷產業動能品質。</p>
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

    <section class="dashboard-section">
      <div class="section-heading">
        <h2>Leader Accumulation Filter</h2>
        <span class="row-count" id="leader-accumulation-status"></span>
      </div>
      <p class="section-note">這是確定性研究篩選器，只在產業處於動能領先或早期修復時評估個股領導品質；研究候選需要先補上 curated leader_type 與 industry_quality_score。</p>
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
          <h3>產業不符合</h3>
          <div class="table-wrap">
            <table data-table="leader-not-eligible-industries"></table>
          </div>
          <p class="empty-state" data-empty-for="leader-not-eligible-industries" hidden>目前沒有產業不符合資料。</p>
        </div>
      </div>
    </section>

    <section class="dashboard-section">
      <div class="section-heading">
        <h2>相對產業強度前 10 名</h2>
        <span class="row-count" data-count-for="top-relative-strength"></span>
      </div>
      <p class="section-note">相對強度 = 個股 10 日報酬減去所屬產業平均 10 日報酬；正值代表跑贏同產業。</p>
      <div class="table-wrap">
        <table data-table="top-relative-strength"></table>
      </div>
      <p class="empty-state" data-empty-for="top-relative-strength" hidden>目前沒有個股資料。</p>
    </section>

    <section class="dashboard-section">
      <div class="section-heading">
        <h2>早期動能候選</h2>
        <span class="row-count" data-count-for="early-candidates"></span>
      </div>
      <p class="section-note">3 日與 5 日報酬轉強，且 5 日報酬已高於 10 日報酬的一半，適合當作早期觀察名單。標示「動能+風險」代表同時出現在風險提醒名單，判斷時不要只依賴單一候選名單。</p>
      <div class="table-wrap">
        <table data-table="early-candidates"></table>
      </div>
      <p class="empty-state" data-empty-for="early-candidates" hidden>目前沒有早期動能候選。</p>
    </section>

    <section class="dashboard-section">
      <div class="section-heading">
        <h2>強勢動能候選</h2>
        <span class="row-count" data-count-for="strong-candidates"></span>
      </div>
      <p class="section-note">同時符合確認動能、跑贏同產業，且相對量大於 1.2，代表價格與量能一起支持。</p>
      <div class="table-wrap">
        <table data-table="strong-candidates"></table>
      </div>
      <p class="empty-state" data-empty-for="strong-candidates" hidden>目前沒有強勢動能候選。</p>
    </section>

    <section class="dashboard-section">
      <div class="section-heading">
        <h2>風險提醒名單</h2>
        <span class="row-count" data-count-for="risk-warnings"></span>
      </div>
      <p class="section-note">風險提醒代表近期最大回撤較深，或價格已高出 20 日均線 15% 以上；可作為追高與波動風險檢查。標示「動能+風險」代表同時符合早期動能與風險條件。</p>
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
      weak: "偏弱",
      exhaustion: "可能衰竭"
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
      not_eligible_industry: "產業不符合"
    };

    const dataStatusLabels = {
      ok: "正常",
      missing: "缺資料",
      stale: "資料落後",
      limited_history: "歷史不足"
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
      industryRegime: "產業狀態由 10 日報酬排名、廣度分數、持續性與衰竭警示決定。",
      leaderType: "手動維護的領導股類型；目前預設為未標註領導。",
      industryQuality: "手動維護的產業品質分數，1 到 5 分；研究候選需要至少 4 分。",
      currentState: "依近期報酬、回撤、均線與 price zone 判斷的目前價格狀態。",
      priceZone: "綜合短期與長期價格位置後的區間，用於 watch status。",
      shortTermPriceZone: "由價格相對 20 日均線的距離判斷。",
      longTermPriceZone: "由價格在 52 週區間中的位置判斷。",
      watchStatus: "風險優先的確定性研究狀態，不代表投資建議。",
      distance20d: "最新價格相對 20 日均線的距離。",
      distance52wHigh: "最新價格相對 52 週高點的距離。",
      position52wRange: "最新價格在 52 週高低區間中的位置，越接近 1 代表越靠近區間上緣。"
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
          { key: "relative_volume", label: "相對量", type: "number", digits: 2, description: explanations.relativeVolume },
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
          { key: "return_10d", label: "平均 10日", type: "percent", description: explanations.return10d }
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
          { key: "high_relative_volume_pct", label: "高相對量", type: "percent", description: explanations.highRelativeVolumePct }
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
          { key: "high_relative_volume_pct", label: "高相對量", type: "percent", description: explanations.highRelativeVolumePct }
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
          { key: "relative_volume", label: "相對量", type: "number", digits: 2, description: explanations.relativeVolume }
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
          { key: "momentum_acceleration", label: "動能加速", type: "signedPercent", description: explanations.acceleration }
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
          { key: "momentum_persistence", label: "前三持續天數", type: "integer", description: explanations.persistence }
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
          { key: "relative_volume", label: "相對量", type: "number", digits: 2, description: explanations.relativeVolume }
        ]
      },
      "leader-research-candidates": {
        rows: dashboardData.leader_accumulation.research_candidates,
        columns: [
          { key: "ticker", label: "代號", type: "ticker" },
          { key: "company_name", label: "公司", type: "company" },
          { key: "industry_group", label: "產業" },
          { key: "industry_regime", label: "產業狀態", description: explanations.industryRegime },
          { key: "leader_type", label: "領導類型", description: explanations.leaderType },
          { key: "industry_quality_score", label: "品質分數", type: "integer", description: explanations.industryQuality },
          { key: "current_state", label: "目前狀態", description: explanations.currentState },
          { key: "price_zone", label: "綜合區間", description: explanations.priceZone },
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
          { key: "leader_type", label: "領導類型", description: explanations.leaderType },
          { key: "industry_quality_score", label: "品質分數", type: "integer", description: explanations.industryQuality },
          { key: "current_state", label: "目前狀態", description: explanations.currentState },
          { key: "short_term_price_zone", label: "短期區間", description: explanations.shortTermPriceZone },
          { key: "long_term_price_zone", label: "長期區間", description: explanations.longTermPriceZone },
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
          { key: "leader_type", label: "領導類型", description: explanations.leaderType },
          { key: "current_state", label: "目前狀態", description: explanations.currentState },
          { key: "price_zone", label: "綜合區間", description: explanations.priceZone },
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
      { key: "return_5d", label: "5日", type: "percent", description: explanations.return5d },
      { key: "return_10d", label: "10日", type: "percent", description: explanations.return10d },
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
      if (column.key === "industry_regime") return regimeLabels[value] || displayText(value);
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

    function renderSummary() {
      const summary = dashboardData.summary;
      const grid = document.getElementById("summary-grid");
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

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !document.getElementById("industry-modal").hidden) {
        closeIndustryModal();
      }
    });

    renderSummary();
    renderUpdateHealth();
    renderDataQuality();
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
) -> None:
    dashboard_data = build_dashboard_data(ticker_output, industry_output, rotation_history, update_health_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_dashboard_html(dashboard_data), encoding="utf-8")
