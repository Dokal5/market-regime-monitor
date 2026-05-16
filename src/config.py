from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
TICKERS_PATH = BASE_DIR / "tickers.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
TICKER_OUTPUT_PATH = OUTPUT_DIR / "ticker_momentum.csv"
INDUSTRY_OUTPUT_PATH = OUTPUT_DIR / "industry_momentum.csv"
DATA_QUALITY_OUTPUT_PATH = OUTPUT_DIR / "data_quality.csv"
UPDATE_HEALTH_OUTPUT_PATH = OUTPUT_DIR / "update_health.csv"
DASHBOARD_OUTPUT_PATH = OUTPUT_DIR / "index.html"
HISTORY_DIR = OUTPUT_DIR / "history"
INDUSTRY_ROTATION_HISTORY_PATH = HISTORY_DIR / "industry_rotation_history.csv"
JOURNAL_DIR = OUTPUT_DIR / "journal"
LATEST_JOURNAL_PATH = JOURNAL_DIR / "latest.md"

LOOKBACK_PERIOD = "1y"
MARKET_DATA_INTERVAL = "1d"

INPUT_COLUMNS = ["ticker", "company_name", "industry_group"]
OPTIONAL_TICKER_COLUMNS = ["leader_type", "industry_quality_score"]
DEFAULT_LEADER_TYPE = "non_leader"
DEFAULT_INDUSTRY_QUALITY_SCORE = 3
ALLOWED_LEADER_TYPES = [
    "core_leader",
    "challenger",
    "infrastructure_leader",
    "emerging_leader",
    "specialist",
    "non_leader",
]
RESEARCH_LEADER_TYPES = [
    "core_leader",
    "challenger",
    "infrastructure_leader",
    "emerging_leader",
]
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
TICKER_VOLUME_COLUMNS = ["latest_volume"]
DATA_QUALITY_COLUMNS = ["data_status", "data_quality_note"]
DATA_QUALITY_EXPORT_COLUMNS = [
    "ticker",
    "company_name",
    "industry_group",
    "latest_date",
    "data_points",
    "data_status",
    "data_quality_note",
]
LIMITED_HISTORY_MIN_DATA_POINTS = 60
UPDATE_HEALTH_COLUMNS = [
    "generated_at_utc",
    "generated_at_new_york",
    "run_context",
    "github_workflow",
    "github_run_id",
    "github_run_url",
    "git_sha",
    "latest_market_date",
    "market_data_age_days",
    "success_rate",
    "missing_count",
    "stale_count",
    "limited_history_count",
    "update_health_status",
    "update_health_note",
]
UPDATE_HEALTH_MAX_DATA_AGE_DAYS = 3
UPDATE_HEALTH_MIN_SUCCESS_RATE = 0.98
PRICE_POSITION_COLUMNS = [
    "distance_from_20d_ma",
    "distance_from_52w_high",
    "position_in_52w_range",
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
BREADTH_COLUMNS = [
    "positive_5d_pct",
    "positive_10d_pct",
    "confirmed_signal_pct",
    "strong_signal_pct",
    "high_relative_volume_pct",
    "breadth_score",
]

EARLY_MOMENTUM_MIN_RETURN_3D = 0.0
EARLY_MOMENTUM_MIN_RETURN_5D = 0.0
EARLY_MOMENTUM_5D_TO_10D_RATIO = 0.5
EARLY_MOMENTUM_10D_RETURN_DIVISOR = 2

CONFIRMED_MOMENTUM_MIN_RETURN_5D = 0.0
CONFIRMED_MOMENTUM_MIN_RETURN_10D = 0.0
CONFIRMED_MOMENTUM_MIN_UP_DAYS_10D = 6

STRONG_MOMENTUM_RELATIVE_VOLUME_THRESHOLD = 1.2
HIGH_RELATIVE_VOLUME_THRESHOLD = STRONG_MOMENTUM_RELATIVE_VOLUME_THRESHOLD

RISK_DRAWDOWN_THRESHOLD = -0.08
RISK_EXTENSION_THRESHOLD = 0.15
RISK_EXTENSION_MULTIPLE = 1.15

TOP_INDUSTRY_RANK_THRESHOLD = 3
RELATIVE_VOLUME_STRONG_THRESHOLD = STRONG_MOMENTUM_RELATIVE_VOLUME_THRESHOLD
RELATIVE_VOLUME_LOW_THRESHOLD = 0.8

MOMENTUM_EXHAUSTION_STRONG_10D_RETURN_THRESHOLD = 0.05
MOMENTUM_EXHAUSTION_WEAK_3D_RETURN_RATIO = 1 / 3
MOMENTUM_EXHAUSTION_WEAK_3D_RETURN_DIVISOR = 3

BREADTH_POSITIVE_5D_WEIGHT = 0.20
BREADTH_POSITIVE_10D_WEIGHT = 0.25
BREADTH_CONFIRMED_SIGNAL_WEIGHT = 0.25
BREADTH_STRONG_SIGNAL_WEIGHT = 0.20
BREADTH_HIGH_RELATIVE_VOLUME_WEIGHT = 0.10

INDUSTRY_REGIME_COLUMN = "industry_regime"
INDUSTRY_RISK_FLAG_COLUMN = "industry_risk_flag"
ROTATION_TYPE_COLUMN = "rotation_type"
CAUSAL_HYPOTHESIS_COLUMN = "causal_hypothesis"
EVIDENCE_STATUS_COLUMN = "evidence_status"
INDUSTRY_REGIME_TOP_RETURN_RANK = 5
INDUSTRY_REGIME_TOP_BREADTH_RANK = 5
INDUSTRY_REGIME_MIN_PERSISTENCE = 2
INDUSTRY_NARROW_LEADERSHIP_BREADTH_THRESHOLD = 0.40
INDUSTRY_OBSERVED_EVIDENCE_BREADTH_THRESHOLD = 0.60

ALLOWED_INDUSTRY_RISK_FLAGS = [
    "none",
    "momentum_exhaustion",
    "narrow_leadership",
    "late_cycle_momentum",
    "data_limited",
]
ALLOWED_ROTATION_TYPES = [
    "risk_on_growth",
    "defensive_rotation",
    "commodity_inflation",
    "policy_driven",
    "panic_rebound",
    "liquidity_rebound",
    "unclear",
]
ALLOWED_CAUSAL_HYPOTHESES = [
    "industry_flow_leads_leaders",
    "leader_strength_leads_industry",
    "macro_liquidity_rebound",
    "policy_or_thematic_support",
    "defensive_rotation",
    "unclear",
]
ALLOWED_EVIDENCE_STATUSES = [
    "observed",
    "inferred",
    "needs_review",
    "unsupported",
]
ALLOWED_EVIDENCE_STATUS = ALLOWED_EVIDENCE_STATUSES
INDUSTRY_ROTATION_TYPE_MAP = {
    "Defensive Healthcare": "defensive_rotation",
    "Defensive Staples": "defensive_rotation",
    "Nuclear": "policy_driven",
    "Energy Nuclear": "policy_driven",
    "Semiconductors": "risk_on_growth",
    "AI Infrastructure": "risk_on_growth",
}
ROTATION_TYPE_CAUSAL_HYPOTHESIS_MAP = {
    "risk_on_growth": "industry_flow_leads_leaders",
    "defensive_rotation": "defensive_rotation",
    "policy_driven": "policy_or_thematic_support",
    "liquidity_rebound": "macro_liquidity_rebound",
    "panic_rebound": "macro_liquidity_rebound",
    "commodity_inflation": "policy_or_thematic_support",
    "unclear": "unclear",
}

LEADER_FILTER_COLUMNS = [
    "industry_regime",
    "industry_risk_flag",
    "rotation_type",
    "causal_hypothesis",
    "evidence_status",
    "short_term_price_zone",
    "long_term_price_zone",
    "price_zone",
    "current_state",
    "watch_status",
]
LEADER_MIN_QUALITY_SCORE = 4

PRICE_ZONE_DEEP_PULLBACK_DISTANCE_20D = -0.12
PRICE_ZONE_REASONABLE_PULLBACK_DISTANCE_20D = -0.03
PRICE_ZONE_EXTENDED_DISTANCE_20D = 0.08
PRICE_ZONE_VERY_EXTENDED_DISTANCE_20D = 0.15
PRICE_ZONE_DEEP_PULLBACK_RANGE_POSITION = 0.25
PRICE_ZONE_REASONABLE_PULLBACK_RANGE_POSITION = 0.45
PRICE_ZONE_EXTENDED_RANGE_POSITION = 0.82
PRICE_ZONE_VERY_EXTENDED_RANGE_POSITION = 0.92
