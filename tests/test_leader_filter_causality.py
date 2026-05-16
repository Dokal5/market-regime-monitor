from __future__ import annotations

import pandas as pd

from src.config import CAUSAL_HYPOTHESIS_COLUMN, EVIDENCE_STATUS_COLUMN, INDUSTRY_RISK_FLAG_COLUMN, ROTATION_TYPE_COLUMN
from src.leader_filter import (
    add_industry_regime_column,
    add_leader_filter_columns,
    classify_industry_risk_flag,
    classify_causal_hypothesis,
)


def base_industry_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "industry_group": "Semiconductors",
        "ticker_count": 3,
        "tickers_with_data": 3,
        "return_3d": 0.01,
        "return_5d": 0.04,
        "return_10d": 0.10,
        "return_20d": 0.15,
        "avg_volume_3d": 100.0,
        "avg_volume_5d": 100.0,
        "avg_volume_20d": 100.0,
        "relative_volume": 1.3,
        "ma_5d": 10.0,
        "ma_10d": 9.0,
        "ma_20d": 8.0,
        "max_drawdown_10d": -0.02,
        "up_days_10d": 7,
        "positive_5d_pct": 0.8,
        "positive_10d_pct": 0.8,
        "confirmed_signal_pct": 0.7,
        "strong_signal_pct": 0.5,
        "high_relative_volume_pct": 0.5,
        "breadth_score": 0.70,
        "rotation_score": 2,
        "momentum_persistence": 2,
        "momentum_acceleration": 0.02,
        "momentum_exhaustion_warning": False,
    }
    row.update(overrides)
    return row


def test_industry_regime_and_momentum_exhaustion_risk_can_coexist() -> None:
    industry = pd.DataFrame([base_industry_row(momentum_exhaustion_warning=True)])

    result = add_industry_regime_column(industry, "2099-01-01")

    assert result.loc[0, "industry_regime"] == "momentum_leader"
    assert result.loc[0, INDUSTRY_RISK_FLAG_COLUMN] == "momentum_exhaustion"


def test_momentum_exhaustion_takes_precedence_over_data_limited() -> None:
    row = pd.Series(
        base_industry_row(
            ticker_count=3,
            tickers_with_data=0,
            momentum_exhaustion_warning=True,
        )
    )

    assert classify_industry_risk_flag(row) == "momentum_exhaustion"


def test_rotation_type_defaults_to_unclear() -> None:
    industry = pd.DataFrame([base_industry_row(industry_group="Space")])

    result = add_industry_regime_column(industry, "2099-01-01")

    assert result.loc[0, ROTATION_TYPE_COLUMN] == "unclear"


def test_causal_hypothesis_maps_from_rotation_type() -> None:
    assert classify_causal_hypothesis("risk_on_growth") == "industry_flow_leads_leaders"
    assert classify_causal_hypothesis("policy_driven") == "policy_or_thematic_support"
    assert classify_causal_hypothesis("unclear") == "unclear"


def test_evidence_status_needs_review_when_cause_is_unclear() -> None:
    industry = pd.DataFrame([base_industry_row(industry_group="Space")])

    result = add_industry_regime_column(industry, "2099-01-01")

    assert result.loc[0, EVIDENCE_STATUS_COLUMN] == "needs_review"


def test_ticker_output_receives_industry_causality_fields() -> None:
    industry = pd.DataFrame([base_industry_row()])
    tickers = pd.DataFrame(
        [
            {
                "ticker": "NVDA",
                "company_name": "NVIDIA Corporation",
                "industry_group": "Semiconductors",
                "latest_date": "2099-01-01",
                "data_points": 252,
                "return_3d": 0.01,
                "return_5d": 0.03,
                "return_10d": 0.08,
                "return_20d": 0.12,
                "avg_volume_3d": 100.0,
                "avg_volume_5d": 100.0,
                "avg_volume_20d": 100.0,
                "relative_volume": 1.1,
                "ma_5d": 10.0,
                "ma_10d": 9.0,
                "ma_20d": 8.0,
                "max_drawdown_10d": -0.02,
                "up_days_10d": 7,
                "early_momentum_signal": True,
                "confirmed_momentum_signal": True,
                "strong_momentum_signal": True,
                "risk_warning": False,
                "relative_strength_vs_industry": 0.02,
                "leader_type": "core_leader",
                "industry_quality_score": 5,
                "distance_from_20d_ma": 0.02,
                "distance_from_52w_high": -0.05,
                "position_in_52w_range": 0.75,
            }
        ]
    )

    ticker_result, _ = add_leader_filter_columns(tickers, industry, "2099-01-01")

    for column in [INDUSTRY_RISK_FLAG_COLUMN, ROTATION_TYPE_COLUMN, CAUSAL_HYPOTHESIS_COLUMN, EVIDENCE_STATUS_COLUMN]:
        assert column in ticker_result.columns
    assert ticker_result.loc[0, ROTATION_TYPE_COLUMN] == "risk_on_growth"
    assert ticker_result.loc[0, CAUSAL_HYPOTHESIS_COLUMN] == "industry_flow_leads_leaders"
