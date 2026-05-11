from __future__ import annotations

from typing import Any

import pandas as pd

from src.data_quality import build_data_quality_summary
from src.update_health import build_update_health_output


INDUSTRY_LABELS = {
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
    "Market ETFs": "市場 ETF",
}

HEALTH_LABELS = {
    "healthy": "正常",
    "warning": "注意",
    "unknown": "未知",
}


def is_missing(value: Any) -> bool:
    return value is None or bool(pd.isna(value))


def format_percent(value: Any, default: str = "無法取得") -> str:
    if is_missing(value):
        return default
    return f"{float(value) * 100:.2f}%"


def format_signed_percent(value: Any, default: str = "無法取得") -> str:
    if is_missing(value):
        return default
    return f"{float(value) * 100:+.2f}%"


def format_integer(value: Any, default: str = "0") -> str:
    if is_missing(value):
        return default
    return f"{int(value):,}"


def display_industry(value: Any) -> str:
    if is_missing(value):
        return "無法取得"
    text = str(value)
    return INDUSTRY_LABELS.get(text, text)


def ticker_list(records: list[dict[str, Any]], limit: int = 5) -> str:
    tickers = [str(record.get("ticker", "")).strip() for record in records[:limit]]
    tickers = [ticker for ticker in tickers if ticker]
    return "、".join(tickers) if tickers else "目前沒有符合條件的標的"


def prepare_tickers(ticker_output: pd.DataFrame) -> pd.DataFrame:
    tickers = ticker_output.copy()
    numeric_columns = [
        "data_points",
        "return_5d",
        "return_10d",
        "relative_strength_vs_industry",
        "max_drawdown_10d",
        "relative_volume",
    ]
    for column in numeric_columns:
        if column in tickers.columns:
            tickers[column] = pd.to_numeric(tickers[column], errors="coerce")

    for column in ["early_momentum_signal", "strong_momentum_signal", "risk_warning"]:
        if column not in tickers.columns:
            tickers[column] = False
        tickers[column] = tickers[column].fillna(False).astype(bool)

    for column, default in [("watch_status", ""), ("data_status", "missing")]:
        if column not in tickers.columns:
            tickers[column] = default
        tickers[column] = tickers[column].fillna(default).astype(str)

    return tickers


def prepare_industries(industry_output: pd.DataFrame) -> pd.DataFrame:
    industries = industry_output.copy()
    numeric_columns = [
        "return_10d",
        "breadth_score",
        "rotation_score",
        "momentum_acceleration",
        "max_drawdown_10d",
    ]
    for column in numeric_columns:
        if column not in industries.columns:
            industries[column] = pd.NA
        industries[column] = pd.to_numeric(industries[column], errors="coerce")

    if "momentum_exhaustion_warning" not in industries.columns:
        industries["momentum_exhaustion_warning"] = False
    industries["momentum_exhaustion_warning"] = (
        industries["momentum_exhaustion_warning"].fillna(False).astype(str).str.lower().isin(["true", "1"])
    )

    return industries


def first_record(data: pd.DataFrame | None) -> dict[str, Any]:
    if data is None or data.empty:
        return {}
    return data.iloc[0].to_dict()


def top_records(data: pd.DataFrame, sort_columns: list[str], ascending: list[bool], limit: int) -> list[dict[str, Any]]:
    if data.empty:
        return []
    return data.sort_values(sort_columns, ascending=ascending, na_position="last").head(limit).to_dict("records")


def build_data_status_card(
    tickers: pd.DataFrame,
    update_health_output: pd.DataFrame | None,
) -> dict[str, Any]:
    summary = build_data_quality_summary(tickers)
    health = first_record(update_health_output)
    if not health:
        health = first_record(build_update_health_output(tickers))

    status = str(health.get("update_health_status") or "unknown")
    status_label = HEALTH_LABELS.get(status, status)
    latest_date = health.get("latest_market_date") or summary.get("latest_market_date") or "無法取得"
    success_rate = health.get("success_rate", summary.get("success_rate"))
    missing_count = health.get("missing_count", summary.get("missing_count", 0))
    stale_count = health.get("stale_count", summary.get("stale_count", 0))
    limited_count = health.get("limited_history_count", summary.get("limited_history_count", 0))

    if status == "healthy":
        headline = f"資料狀態正常，最新市場日期 {latest_date}"
        details = [
            f"成功率 {format_percent(success_rate)}；缺資料 {format_integer(missing_count)}，資料落後 {format_integer(stale_count)}。",
            health.get("update_health_note") or "排程輸出與資料新鮮度目前正常。",
        ]
    else:
        headline = f"資料狀態為「{status_label}」，本日訊號需先保守解讀"
        details = [
            f"最新市場日期 {latest_date}；成功率 {format_percent(success_rate)}。",
            f"缺資料 {format_integer(missing_count)}，資料落後 {format_integer(stale_count)}，歷史不足 {format_integer(limited_count)}。",
            health.get("update_health_note") or "尚無健康狀態說明。",
        ]

    return {
        "key": "data_status",
        "title": "資料狀態",
        "status": status if status in ["healthy", "warning", "unknown"] else "unknown",
        "headline": headline,
        "details": details,
    }


def build_market_theme_card(industries: pd.DataFrame) -> dict[str, Any]:
    if industries.empty:
        return {
            "key": "market_theme",
            "title": "市場主線",
            "status": "unknown",
            "headline": "目前沒有足夠產業資料",
            "details": ["需要有效的產業輸出後，才會顯示 10 日報酬與廣度主線。"],
        }

    leading = top_records(industries, ["return_10d"], [False], 1)
    breadth = top_records(industries, ["breadth_score"], [False], 1)
    leading_row = leading[0] if leading else {}
    breadth_row = breadth[0] if breadth else {}
    leading_name = display_industry(leading_row.get("industry_group"))
    breadth_name = display_industry(breadth_row.get("industry_group"))
    median_breadth = industries["breadth_score"].median() if "breadth_score" in industries.columns else pd.NA
    leading_breadth = leading_row.get("breadth_score")

    if leading_row.get("industry_group") and leading_row.get("industry_group") == breadth_row.get("industry_group"):
        breadth_note = "主線偏廣：10 日報酬領先產業同時也是廣度分數領先產業。"
    elif not is_missing(leading_breadth) and not is_missing(median_breadth) and float(leading_breadth) >= float(median_breadth):
        breadth_note = "主線偏廣：10 日報酬領先產業的廣度高於產業中位數。"
    else:
        breadth_note = "主線偏窄：10 日報酬領先產業尚未同步成為廣度領先產業。"

    return {
        "key": "market_theme",
        "title": "市場主線",
        "status": "neutral",
        "headline": f"10 日主線：{leading_name}；廣度主線：{breadth_name}",
        "details": [
            f"{leading_name} 10 日報酬 {format_percent(leading_row.get('return_10d'))}。",
            f"{breadth_name} 廣度分數 {format_percent(breadth_row.get('breadth_score'))}。",
            breadth_note,
        ],
    }


def build_rotation_card(industries: pd.DataFrame) -> dict[str, Any]:
    improving = industries[
        (industries.get("rotation_score", pd.Series(dtype=float)) > 0)
        | (industries.get("momentum_acceleration", pd.Series(dtype=float)) > 0)
    ]
    improving_records = top_records(improving, ["rotation_score", "momentum_acceleration", "return_10d"], [False, False, False], 3)
    weakening = industries[industries.get("rotation_score", pd.Series(dtype=float)) < 0]
    weakening_records = top_records(weakening, ["rotation_score", "return_10d"], [True, True], 3)
    exhaustion_count = int(industries["momentum_exhaustion_warning"].sum()) if "momentum_exhaustion_warning" in industries else 0

    improving_names = "、".join(display_industry(row.get("industry_group")) for row in improving_records)
    weakening_names = "、".join(display_industry(row.get("industry_group")) for row in weakening_records)
    improving_text = improving_names or "目前沒有明顯改善產業"
    weakening_text = weakening_names or "目前沒有明顯轉弱產業"

    return {
        "key": "rotation_change",
        "title": "輪動變化",
        "status": "warning" if exhaustion_count > 0 else "neutral",
        "headline": f"改善最強：{improving_text}",
        "details": [
            f"排名或加速轉強：{improving_text}。",
            f"排名轉弱：{weakening_text}。",
            f"可能動能衰竭：{exhaustion_count} 個產業。",
        ],
    }


def build_research_card(tickers: pd.DataFrame) -> dict[str, Any]:
    tradable = tickers[tickers["data_points"] > 0] if "data_points" in tickers.columns else tickers
    research_candidates = top_records(
        tradable[tradable["watch_status"] == "research_candidate"],
        ["relative_strength_vs_industry", "return_10d"],
        [False, False],
        5,
    )
    strong_records = top_records(
        tradable[tradable["strong_momentum_signal"]],
        ["relative_strength_vs_industry", "return_10d"],
        [False, False],
        5,
    )
    early_without_risk = top_records(
        tradable[tradable["early_momentum_signal"] & ~tradable["risk_warning"]],
        ["return_5d", "relative_volume"],
        [False, False],
        5,
    )

    if research_candidates:
        headline = f"研究候選：{ticker_list(research_candidates)}"
        first_detail = f"研究候選：{ticker_list(research_candidates)}。"
        status = "neutral"
    else:
        headline = "研究候選需要先維護 leader metadata"
        first_detail = "目前沒有研究候選；需要 curated leader_type 與 industry_quality_score 後，輸出才會有意義。"
        status = "warning"

    return {
        "key": "research_focus",
        "title": "研究候選",
        "status": status,
        "headline": headline,
        "details": [
            first_detail,
            f"強勢動能：{ticker_list(strong_records)}。",
            f"早期動能且無風險提醒：{ticker_list(early_without_risk)}。",
        ],
    }


def build_risk_card(tickers: pd.DataFrame) -> dict[str, Any]:
    tradable = tickers[tickers["data_points"] > 0] if "data_points" in tickers.columns else tickers
    risk_rows = tradable[tradable["risk_warning"]]
    mixed_rows = tradable[tradable["early_momentum_signal"] & tradable["risk_warning"]]
    drawdown_records = top_records(risk_rows, ["max_drawdown_10d", "relative_volume"], [True, False], 5)
    mixed_records = top_records(mixed_rows, ["return_5d", "relative_volume"], [False, False], 5)

    drawdown_text = "、".join(
        f"{record.get('ticker')} {format_signed_percent(record.get('max_drawdown_10d'))}"
        for record in drawdown_records
        if record.get("ticker")
    )
    drawdown_text = drawdown_text or "目前沒有風險提醒標的"

    return {
        "key": "risk_focus",
        "title": "風險焦點",
        "status": "warning" if len(risk_rows) else "healthy",
        "headline": f"{len(risk_rows)} 檔風險提醒；{len(mixed_rows)} 檔同時有早期動能與風險",
        "details": [
            f"最大回撤代表：{drawdown_text}。",
            f"動能與風險重疊：{ticker_list(mixed_records)}。",
            "讀法提醒：不能只依賴單一候選名單，需要同時檢查風險欄位。",
        ],
    }


def build_daily_brief(
    ticker_output: pd.DataFrame,
    industry_output: pd.DataFrame,
    update_health_output: pd.DataFrame | None = None,
) -> dict[str, Any]:
    tickers = prepare_tickers(ticker_output)
    industries = prepare_industries(industry_output)
    return {
        "cards": [
            build_data_status_card(tickers, update_health_output),
            build_market_theme_card(industries),
            build_rotation_card(industries),
            build_research_card(tickers),
            build_risk_card(tickers),
        ]
    }


def daily_brief_to_markdown(brief: dict[str, Any]) -> str:
    lines: list[str] = []
    for card in brief.get("cards", []):
        title = card.get("title", "摘要")
        headline = card.get("headline", "")
        details = card.get("details", [])
        lines.append(f"### {title}")
        if headline:
            lines.append(f"**{headline}**")
        for detail in details:
            if detail:
                lines.append(f"- {detail}")
        lines.append("")
    return "\n".join(lines).strip()
