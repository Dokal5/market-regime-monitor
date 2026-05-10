from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.config import (
    DEFAULT_INDUSTRY_QUALITY_SCORE,
    DEFAULT_LEADER_TYPE,
    HISTORY_DIR,
    INDUSTRY_REGIME_COLUMN,
    INDUSTRY_REGIME_MIN_PERSISTENCE,
    INDUSTRY_REGIME_TOP_BREADTH_RANK,
    INDUSTRY_REGIME_TOP_RETURN_RANK,
    LEADER_FILTER_COLUMNS,
    LEADER_MIN_QUALITY_SCORE,
    PRICE_ZONE_DEEP_PULLBACK_DISTANCE_20D,
    PRICE_ZONE_DEEP_PULLBACK_RANGE_POSITION,
    PRICE_ZONE_EXTENDED_DISTANCE_20D,
    PRICE_ZONE_EXTENDED_RANGE_POSITION,
    PRICE_ZONE_REASONABLE_PULLBACK_DISTANCE_20D,
    PRICE_ZONE_REASONABLE_PULLBACK_RANGE_POSITION,
    PRICE_ZONE_VERY_EXTENDED_DISTANCE_20D,
    PRICE_ZONE_VERY_EXTENDED_RANGE_POSITION,
    RESEARCH_LEADER_TYPES,
    RISK_DRAWDOWN_THRESHOLD,
    TICKER_VOLUME_COLUMNS,
)

ELIGIBLE_INDUSTRY_REGIMES = {"momentum_leader", "early_recovery"}
EXTENDED_PRICE_ZONES = {"extended", "very_extended"}
RESEARCH_PRICE_ZONES = {"reasonable_pullback", "neutral"}
RESEARCH_CURRENT_STATES = {"early_recovery", "pullback_in_uptrend", "sideways_base"}


def as_float(value: Any) -> float:
    if pd.isna(value):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def as_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def normalize_leader_type(value: Any) -> str:
    if pd.isna(value):
        return DEFAULT_LEADER_TYPE
    text = str(value).strip().lower()
    return text or DEFAULT_LEADER_TYPE


def load_previous_breadth_scores(before_date: str) -> dict[str, float]:
    if not HISTORY_DIR.exists():
        return {}

    previous_dates = [path.name for path in HISTORY_DIR.iterdir() if path.is_dir() and path.name < before_date]
    for snapshot_date in sorted(previous_dates, reverse=True):
        industry_path = HISTORY_DIR / snapshot_date / "industry_momentum.csv"
        if not industry_path.exists():
            continue

        try:
            snapshot = pd.read_csv(industry_path)
        except Exception as exc:
            print(f"Warning: could not read previous breadth snapshot {snapshot_date}: {exc}")
            continue

        if "industry_group" not in snapshot.columns or "breadth_score" not in snapshot.columns:
            continue

        snapshot["breadth_score"] = pd.to_numeric(snapshot["breadth_score"], errors="coerce")
        return dict(zip(snapshot["industry_group"].astype(str), snapshot["breadth_score"], strict=False))

    return {}


def classify_industry_regime(row: pd.Series, previous_breadth_scores: dict[str, float]) -> str:
    if as_bool(row.get("momentum_exhaustion_warning")):
        return "exhaustion"

    return_rank = as_float(row.get("_return_10d_rank"))
    breadth_rank = as_float(row.get("_breadth_score_rank"))
    persistence = as_float(row.get("momentum_persistence"))
    if (
        (pd.notna(return_rank) and return_rank <= INDUSTRY_REGIME_TOP_RETURN_RANK)
        or (pd.notna(breadth_rank) and breadth_rank <= INDUSTRY_REGIME_TOP_BREADTH_RANK)
        or (pd.notna(persistence) and persistence >= INDUSTRY_REGIME_MIN_PERSISTENCE)
    ):
        return "momentum_leader"

    return_5d = as_float(row.get("return_5d"))
    return_10d = as_float(row.get("return_10d"))
    positive_5d_pct = as_float(row.get("positive_5d_pct"))
    positive_10d_pct = as_float(row.get("positive_10d_pct"))
    current_breadth = as_float(row.get("breadth_score"))
    previous_breadth = previous_breadth_scores.get(str(row.get("industry_group")))
    breadth_improving = pd.notna(current_breadth) and pd.notna(previous_breadth) and current_breadth > previous_breadth
    short_breadth_stronger = (
        pd.notna(positive_5d_pct) and pd.notna(positive_10d_pct) and positive_5d_pct > positive_10d_pct
    )
    if return_5d > 0 and return_10d <= 0 and (breadth_improving or short_breadth_stronger):
        return "early_recovery"

    if return_5d < 0 and return_10d < 0:
        return "weak"

    return "neutral"


def add_industry_regime_column(industry_output: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    output_columns = list(industry_output.columns) + [
        INDUSTRY_REGIME_COLUMN for _ in [0] if INDUSTRY_REGIME_COLUMN not in industry_output.columns
    ]
    if industry_output.empty:
        return pd.DataFrame(columns=output_columns)

    industry = industry_output.copy()
    numeric_columns = [
        "return_5d",
        "return_10d",
        "breadth_score",
        "positive_5d_pct",
        "positive_10d_pct",
        "momentum_persistence",
    ]
    for column in numeric_columns:
        if column not in industry.columns:
            industry[column] = math.nan
        industry[column] = pd.to_numeric(industry[column], errors="coerce")

    if "momentum_exhaustion_warning" not in industry.columns:
        industry["momentum_exhaustion_warning"] = False

    industry["_return_10d_rank"] = industry["return_10d"].rank(method="min", ascending=False, na_option="bottom")
    industry["_breadth_score_rank"] = industry["breadth_score"].rank(method="min", ascending=False, na_option="bottom")
    previous_breadth_scores = load_previous_breadth_scores(snapshot_date)
    industry[INDUSTRY_REGIME_COLUMN] = industry.apply(
        lambda row: classify_industry_regime(row, previous_breadth_scores), axis=1
    )
    industry = industry.drop(columns=["_return_10d_rank", "_breadth_score_rank"], errors="ignore")
    return industry[output_columns]


def classify_short_term_price_zone(distance_from_20d_ma: Any) -> str:
    distance = as_float(distance_from_20d_ma)
    if pd.isna(distance):
        return "neutral"
    if distance > PRICE_ZONE_VERY_EXTENDED_DISTANCE_20D:
        return "very_extended"
    if distance > PRICE_ZONE_EXTENDED_DISTANCE_20D:
        return "extended"
    if distance <= PRICE_ZONE_DEEP_PULLBACK_DISTANCE_20D:
        return "deep_pullback"
    if distance <= PRICE_ZONE_REASONABLE_PULLBACK_DISTANCE_20D:
        return "reasonable_pullback"
    return "neutral"


def classify_long_term_price_zone(position_in_52w_range: Any) -> str:
    position = as_float(position_in_52w_range)
    if pd.isna(position):
        return "neutral"
    if position >= PRICE_ZONE_VERY_EXTENDED_RANGE_POSITION:
        return "very_extended"
    if position >= PRICE_ZONE_EXTENDED_RANGE_POSITION:
        return "extended"
    if position <= PRICE_ZONE_DEEP_PULLBACK_RANGE_POSITION:
        return "deep_pullback"
    if position <= PRICE_ZONE_REASONABLE_PULLBACK_RANGE_POSITION:
        return "reasonable_pullback"
    return "neutral"


def combine_price_zones(short_term_price_zone: str, long_term_price_zone: str) -> str:
    zones = {short_term_price_zone, long_term_price_zone}
    for zone in ["very_extended", "extended", "deep_pullback", "reasonable_pullback"]:
        if zone in zones:
            return zone
    return "neutral"


def classify_current_state(row: pd.Series) -> str:
    return_3d = as_float(row.get("return_3d"))
    return_5d = as_float(row.get("return_5d"))
    return_10d = as_float(row.get("return_10d"))
    return_20d = as_float(row.get("return_20d"))
    ma_5d = as_float(row.get("ma_5d"))
    ma_10d = as_float(row.get("ma_10d"))
    max_drawdown_10d = as_float(row.get("max_drawdown_10d"))
    price_zone = str(row.get("price_zone") or "neutral")
    risk_warning = as_bool(row.get("risk_warning"))

    if return_5d < 0 and return_10d < 0 and max_drawdown_10d <= RISK_DRAWDOWN_THRESHOLD:
        return "falling_knife"

    if price_zone == "very_extended" or (price_zone == "extended" and risk_warning):
        return "overextended"

    if price_zone in {"deep_pullback", "reasonable_pullback"} and return_20d > 0 and return_10d >= 0:
        return "pullback_in_uptrend"

    if return_3d > 0 and return_5d > 0 and (return_10d <= 0 or return_5d > return_10d):
        return "early_recovery"

    if return_5d > 0 and return_10d > 0 and ma_5d > ma_10d:
        return "strong_uptrend"

    return "sideways_base"


def classify_watch_status(row: pd.Series) -> str:
    risk_warning = as_bool(row.get("risk_warning"))
    return_10d = as_float(row.get("return_10d"))
    current_state = str(row.get("current_state") or "sideways_base")
    industry_regime = str(row.get(INDUSTRY_REGIME_COLUMN) or "neutral")
    price_zone = str(row.get("price_zone") or "neutral")
    leader_type = normalize_leader_type(row.get("leader_type"))
    industry_quality_score = as_float(row.get("industry_quality_score"))
    eligible_industry = industry_regime in ELIGIBLE_INDUSTRY_REGIMES
    tracked_leader_type = leader_type != DEFAULT_LEADER_TYPE
    quality_ready = pd.notna(industry_quality_score) and industry_quality_score >= LEADER_MIN_QUALITY_SCORE

    if current_state == "falling_knife" or (risk_warning and return_10d < 0):
        return "avoid_for_now"

    if not eligible_industry:
        return "not_eligible_industry"

    if tracked_leader_type and price_zone in EXTENDED_PRICE_ZONES:
        return "too_extended"

    if (
        quality_ready
        and leader_type in RESEARCH_LEADER_TYPES
        and price_zone in RESEARCH_PRICE_ZONES
        and current_state in RESEARCH_CURRENT_STATES
        and not risk_warning
    ):
        return "research_candidate"

    if quality_ready and tracked_leader_type and price_zone == "deep_pullback" and current_state != "falling_knife":
        return "wait_for_stabilization"

    return "avoid_for_now"


def add_leader_filter_columns(
    ticker_output: pd.DataFrame, industry_output: pd.DataFrame, snapshot_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    industry = add_industry_regime_column(industry_output, snapshot_date)
    base_columns = [column for column in ticker_output.columns if column not in TICKER_VOLUME_COLUMNS]
    ticker_output_columns = (
        base_columns
        + [column for column in LEADER_FILTER_COLUMNS if column not in base_columns]
        + [column for column in TICKER_VOLUME_COLUMNS if column in ticker_output.columns or ticker_output.empty]
    )
    if ticker_output.empty:
        return pd.DataFrame(columns=ticker_output_columns), industry

    tickers = ticker_output.copy()
    tickers = tickers.drop(columns=[INDUSTRY_REGIME_COLUMN], errors="ignore")
    if not industry.empty:
        regimes = industry[["industry_group", INDUSTRY_REGIME_COLUMN]]
        tickers = tickers.merge(regimes, on="industry_group", how="left")
        tickers[INDUSTRY_REGIME_COLUMN] = tickers[INDUSTRY_REGIME_COLUMN].fillna("neutral")
    else:
        tickers[INDUSTRY_REGIME_COLUMN] = "neutral"

    if "leader_type" not in tickers.columns:
        tickers["leader_type"] = DEFAULT_LEADER_TYPE
    tickers["leader_type"] = tickers["leader_type"].apply(normalize_leader_type)
    if "industry_quality_score" not in tickers.columns:
        tickers["industry_quality_score"] = DEFAULT_INDUSTRY_QUALITY_SCORE
    tickers["industry_quality_score"] = (
        pd.to_numeric(tickers["industry_quality_score"], errors="coerce")
        .fillna(DEFAULT_INDUSTRY_QUALITY_SCORE)
        .clip(lower=1, upper=5)
        .round()
        .astype(int)
    )

    tickers["short_term_price_zone"] = tickers["distance_from_20d_ma"].apply(classify_short_term_price_zone)
    tickers["long_term_price_zone"] = tickers["position_in_52w_range"].apply(classify_long_term_price_zone)
    tickers["price_zone"] = tickers.apply(
        lambda row: combine_price_zones(row["short_term_price_zone"], row["long_term_price_zone"]), axis=1
    )
    tickers["current_state"] = tickers.apply(classify_current_state, axis=1)
    tickers["watch_status"] = tickers.apply(classify_watch_status, axis=1)
    return tickers[ticker_output_columns], industry
