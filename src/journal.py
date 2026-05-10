from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import JOURNAL_DIR, LATEST_JOURNAL_PATH
from src.data_quality import build_data_quality_summary
from src.update_health import build_update_health_output


def format_percent(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 100:.2f}%"


def format_number(value: Any, digits: int = 2) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def format_bool(value: Any) -> str:
    if pd.isna(value):
        return ""
    return "true" if bool(value) else "false"


def table_from_records(records: list[dict[str, Any]], columns: list[tuple[str, str, str]]) -> str:
    headers = [label for _, label, _ in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    if not records:
        lines.append("| " + " | ".join("None" if index == 0 else "" for index, _ in enumerate(headers)) + " |")
        return "\n".join(lines)

    for record in records:
        cells = []
        for key, _, column_type in columns:
            value = record.get(key)
            if column_type == "percent":
                cells.append(format_percent(value))
            elif column_type == "number":
                cells.append(format_number(value))
            elif column_type == "bool":
                cells.append(format_bool(value))
            else:
                cells.append("" if pd.isna(value) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def prepare_tickers(ticker_output: pd.DataFrame) -> pd.DataFrame:
    tickers = ticker_output.copy()
    numeric_columns = [
        "data_points",
        "return_3d",
        "return_5d",
        "return_10d",
        "return_20d",
        "relative_volume",
        "max_drawdown_10d",
        "up_days_10d",
        "relative_strength_vs_industry",
    ]
    for column in numeric_columns:
        if column in tickers.columns:
            tickers[column] = pd.to_numeric(tickers[column], errors="coerce")

    for column in ["early_momentum_signal", "confirmed_momentum_signal", "strong_momentum_signal", "risk_warning"]:
        if column in tickers.columns:
            tickers[column] = tickers[column].fillna(False).astype(bool)
        else:
            tickers[column] = False
    for column, default in [("data_status", "missing"), ("data_quality_note", "")]:
        if column not in tickers.columns:
            tickers[column] = default
        tickers[column] = tickers[column].fillna(default).astype(str)
    return tickers


def prepare_industries(industry_output: pd.DataFrame) -> pd.DataFrame:
    industries = industry_output.copy()
    numeric_columns = [
        "return_3d",
        "return_5d",
        "return_10d",
        "relative_volume",
        "positive_5d_pct",
        "positive_10d_pct",
        "confirmed_signal_pct",
        "strong_signal_pct",
        "high_relative_volume_pct",
        "breadth_score",
    ]
    for column in numeric_columns:
        if column in industries.columns:
            industries[column] = pd.to_numeric(industries[column], errors="coerce")
    return industries


def top_records(data: pd.DataFrame, sort_columns: list[str], ascending: list[bool], limit: int) -> list[dict[str, Any]]:
    if data.empty:
        return []
    return data.sort_values(sort_columns, ascending=ascending, na_position="last").head(limit).to_dict("records")


def build_market_snapshot(tickers: pd.DataFrame, market_date: str) -> dict[str, int | str]:
    tradable = tickers[tickers["data_points"] > 0] if "data_points" in tickers.columns else tickers
    return {
        "latest_market_date": market_date,
        "total_tickers": int(len(tickers)),
        "tickers_with_data": int(len(tradable)),
        "early_momentum_count": int(tickers["early_momentum_signal"].sum()),
        "confirmed_momentum_count": int(tickers["confirmed_momentum_signal"].sum()),
        "strong_momentum_count": int(tickers["strong_momentum_signal"].sum()),
        "risk_warning_count": int(tickers["risk_warning"].sum()),
    }


def build_interpretation(
    tickers: pd.DataFrame,
    industries: pd.DataFrame,
    leading_industries: list[dict[str, Any]],
    broad_strength_industries: list[dict[str, Any]],
    market_date: str,
) -> list[str]:
    summary = build_market_snapshot(tickers, market_date)
    lines = []

    leading = leading_industries[0] if leading_industries else None
    strongest_breadth = broad_strength_industries[0] if broad_strength_industries else None
    if leading:
        lines.append(
            f"- The leading industry by 10 day return is {leading['industry_group']} "
            f"at {format_percent(leading.get('return_10d'))}."
        )
    else:
        lines.append("- No leading industry is available from the current generated data.")

    if strongest_breadth:
        lines.append(
            f"- The strongest industry by breadth score is {strongest_breadth['industry_group']} "
            f"at {format_percent(strongest_breadth.get('breadth_score'))}."
        )
    else:
        lines.append("- No breadth leader is available from the current generated data.")

    risk_count = int(summary["risk_warning_count"])
    strong_count = int(summary["strong_momentum_count"])
    if risk_count == 0:
        lines.append("- Risk warnings are not present in the current generated output.")
    elif risk_count > strong_count:
        lines.append(
            f"- Risk warnings are elevated relative to strong momentum signals "
            f"({risk_count} risk warnings versus {strong_count} strong signals)."
        )
    else:
        lines.append(
            f"- Risk warnings are present but do not exceed strong momentum signals "
            f"({risk_count} risk warnings versus {strong_count} strong signals)."
        )

    if leading and strongest_breadth and leading["industry_group"] == strongest_breadth["industry_group"]:
        lines.append("- Leadership appears broad because the 10 day return leader also has the strongest breadth score.")
    elif leading and not industries.empty:
        median_breadth = industries["breadth_score"].median()
        leading_breadth = leading.get("breadth_score")
        if pd.notna(leading_breadth) and pd.notna(median_breadth) and leading_breadth >= median_breadth:
            lines.append("- Leadership appears broad because the return leader has above-median industry breadth.")
        else:
            lines.append("- Leadership appears narrow because the return leader does not rank as a breadth leader.")
    else:
        lines.append("- Leadership breadth cannot be classified from the current generated data.")

    recovery = industries[
        (industries["return_5d"] > 0)
        & (industries["return_5d"] > industries["return_10d"])
        & (industries["positive_5d_pct"] > industries["positive_10d_pct"])
    ].sort_values(["positive_5d_pct", "return_5d"], ascending=[False, False], na_position="last")
    if recovery.empty:
        lines.append("- No notable early recovery industries are present under the current rule set.")
    else:
        names = ", ".join(recovery["industry_group"].head(3).astype(str).tolist())
        lines.append(f"- Notable early recovery industries by the current rule set: {names}.")

    return lines


def build_journal_markdown(
    ticker_output: pd.DataFrame,
    industry_output: pd.DataFrame,
    market_date: str,
    update_health_output: pd.DataFrame | None = None,
) -> str:
    tickers = prepare_tickers(ticker_output)
    industries = prepare_industries(industry_output)
    tradable_tickers = tickers[tickers["data_points"] > 0] if "data_points" in tickers.columns else tickers
    snapshot = build_market_snapshot(tickers, market_date)

    leading_industries = top_records(industries, ["return_10d"], [False], 5)
    broad_strength_industries = top_records(industries, ["breadth_score"], [False], 5)
    strongest_relative_strength = top_records(
        tradable_tickers, ["relative_strength_vs_industry"], [False], 10
    )
    early_candidates = top_records(
        tradable_tickers[tradable_tickers["early_momentum_signal"]],
        ["return_5d"],
        [False],
        10,
    )
    risk_warnings = top_records(
        tradable_tickers[tradable_tickers["risk_warning"]],
        ["max_drawdown_10d"],
        [True],
        10,
    )
    data_quality_summary = build_data_quality_summary(tickers)
    data_quality_issues = tickers[tickers["data_status"] != "ok"].sort_values(
        ["data_status", "ticker"], ascending=[True, True], na_position="last"
    )
    if update_health_output is None:
        update_health_output = build_update_health_output(tickers)
    update_health_records = update_health_output.to_dict("records")
    interpretation = build_interpretation(tickers, industries, leading_industries, broad_strength_industries, market_date)

    snapshot_table = table_from_records(
        [snapshot],
        [
            ("latest_market_date", "latest market date", "text"),
            ("total_tickers", "total tickers", "text"),
            ("tickers_with_data", "tickers with data", "text"),
            ("early_momentum_count", "early momentum count", "text"),
            ("confirmed_momentum_count", "confirmed momentum count", "text"),
            ("strong_momentum_count", "strong momentum count", "text"),
            ("risk_warning_count", "risk warning count", "text"),
        ],
    )
    data_quality_table = table_from_records(
        [data_quality_summary],
        [
            ("data_source", "data source", "text"),
            ("latest_market_date", "latest market date", "text"),
            ("tickers_with_data", "tickers with data", "text"),
            ("total_tickers", "total tickers", "text"),
            ("success_rate", "success rate", "percent"),
            ("missing_count", "missing", "text"),
            ("stale_count", "stale", "text"),
            ("limited_history_count", "limited history", "text"),
        ],
    )
    data_quality_issue_table = table_from_records(
        data_quality_issues[
            [
                "ticker",
                "company_name",
                "industry_group",
                "latest_date",
                "data_points",
                "data_status",
                "data_quality_note",
            ]
        ].to_dict("records"),
        [
            ("ticker", "ticker", "text"),
            ("company_name", "company_name", "text"),
            ("industry_group", "industry_group", "text"),
            ("latest_date", "latest_date", "text"),
            ("data_points", "data_points", "text"),
            ("data_status", "data_status", "text"),
            ("data_quality_note", "data_quality_note", "text"),
        ],
    )
    update_health_table = table_from_records(
        update_health_records,
        [
            ("update_health_status", "update_health_status", "text"),
            ("update_health_note", "update_health_note", "text"),
            ("generated_at_new_york", "generated_at_new_york", "text"),
            ("run_context", "run_context", "text"),
            ("github_run_url", "github_run_url", "text"),
            ("latest_market_date", "latest_market_date", "text"),
            ("market_data_age_days", "market_data_age_days", "text"),
            ("success_rate", "success_rate", "percent"),
        ],
    )
    leading_table = table_from_records(
        leading_industries,
        [
            ("industry_group", "industry_group", "text"),
            ("return_10d", "return_10d", "percent"),
            ("return_5d", "return_5d", "percent"),
            ("relative_volume", "relative_volume", "number"),
            ("breadth_score", "breadth_score", "percent"),
            ("confirmed_signal_pct", "confirmed_signal_pct", "percent"),
        ],
    )
    breadth_table = table_from_records(
        broad_strength_industries,
        [
            ("industry_group", "industry_group", "text"),
            ("breadth_score", "breadth_score", "percent"),
            ("positive_5d_pct", "positive_5d_pct", "percent"),
            ("positive_10d_pct", "positive_10d_pct", "percent"),
            ("confirmed_signal_pct", "confirmed_signal_pct", "percent"),
            ("strong_signal_pct", "strong_signal_pct", "percent"),
            ("high_relative_volume_pct", "high_relative_volume_pct", "percent"),
        ],
    )
    relative_strength_table = table_from_records(
        strongest_relative_strength,
        [
            ("ticker", "ticker", "text"),
            ("company_name", "company_name", "text"),
            ("industry_group", "industry_group", "text"),
            ("return_10d", "return_10d", "percent"),
            ("relative_strength_vs_industry", "relative_strength_vs_industry", "percent"),
            ("relative_volume", "relative_volume", "number"),
            ("risk_warning", "risk_warning", "bool"),
        ],
    )
    early_table = table_from_records(
        early_candidates,
        [
            ("ticker", "ticker", "text"),
            ("company_name", "company_name", "text"),
            ("industry_group", "industry_group", "text"),
            ("return_3d", "return_3d", "percent"),
            ("return_5d", "return_5d", "percent"),
            ("return_10d", "return_10d", "percent"),
            ("relative_volume", "relative_volume", "number"),
        ],
    )
    risk_table = table_from_records(
        risk_warnings,
        [
            ("ticker", "ticker", "text"),
            ("company_name", "company_name", "text"),
            ("industry_group", "industry_group", "text"),
            ("return_10d", "return_10d", "percent"),
            ("max_drawdown_10d", "max_drawdown_10d", "percent"),
            ("relative_volume", "relative_volume", "number"),
        ],
    )

    return "\n\n".join(
        [
            f"# Market Regime Monitor Journal: {market_date}",
            "## Market Snapshot\n" + snapshot_table,
            "## Update Health\n" + update_health_table,
            "## Data Quality\n" + data_quality_table + "\n\n" + data_quality_issue_table,
            "## Leading Industries\n" + leading_table,
            "## Broad Strength Industries\n" + breadth_table,
            "## Strongest Relative Strength Stocks\n" + relative_strength_table,
            "## Early Momentum Candidates\n" + early_table,
            "## Risk Warnings\n" + risk_table,
            "## Deterministic System Interpretation\n" + "\n".join(interpretation),
            "## My Interpretation\n",
            "## Possible Action\n",
            "## What Would Invalidate This View\n",
            "",
        ]
    )


def write_journal(
    ticker_output: pd.DataFrame,
    industry_output: pd.DataFrame,
    market_date: str,
    update_health_output: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    markdown = build_journal_markdown(ticker_output, industry_output, market_date, update_health_output)
    dated_path = JOURNAL_DIR / f"{market_date}.md"

    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    dated_path.write_text(markdown, encoding="utf-8")
    LATEST_JOURNAL_PATH.write_text(markdown, encoding="utf-8")
    return dated_path, LATEST_JOURNAL_PATH
