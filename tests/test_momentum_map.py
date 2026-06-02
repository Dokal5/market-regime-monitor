from __future__ import annotations

import pandas as pd

from src.dashboard import build_dashboard_html, build_momentum_map


def sample_tickers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "SMR",
                "company_name": "NuScale Power Corporation",
                "industry_group": "Nuclear",
                "return_10d": 0.15,
                "relative_strength_vs_industry": 0.07,
                "relative_volume": 1.26,
                "watch_status": "not_eligible_industry",
                "industry_regime": "neutral",
                "industry_risk_flag": "momentum_exhaustion",
            },
            {
                "ticker": "TSM",
                "company_name": "Taiwan Semiconductor Manufacturing Co. Ltd.",
                "industry_group": "Semiconductors",
                "return_10d": 0.07,
                "relative_strength_vs_industry": -0.12,
                "relative_volume": 1.35,
                "watch_status": "not_eligible_industry",
                "industry_regime": "neutral",
                "industry_risk_flag": "momentum_exhaustion",
            },
        ]
    )


def sample_industries() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "industry_group": "Nuclear",
                "return_10d": 0.08,
                "return_5d": 0.04,
                "breadth_score": 0.60,
                "industry_regime": "neutral",
                "industry_risk_flag": "momentum_exhaustion",
                "rotation_type": "policy_driven",
            },
            {
                "industry_group": "Cloud Software",
                "return_10d": 0.20,
                "return_5d": 0.10,
                "breadth_score": 0.70,
                "industry_regime": "momentum_leader",
                "industry_risk_flag": "none",
                "rotation_type": "unclear",
            },
            {
                "industry_group": "Semiconductors",
                "return_10d": 0.19,
                "return_5d": 0.09,
                "breadth_score": 0.55,
                "industry_regime": "early_recovery",
                "industry_risk_flag": "none",
                "rotation_type": "risk_on_growth",
            },
        ]
    )


def sample_alerts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker": "SMR", "alert_level": "yellow"},
            {"ticker": "TSM", "alert_level": "red"},
        ]
    )


def test_momentum_map_maps_watchlist_tickers_to_industries() -> None:
    momentum_map = build_momentum_map(sample_tickers(), sample_industries(), sample_alerts())
    bars = {row["industry_group"]: row for row in momentum_map["industry_bars"]}

    assert bars["Nuclear"]["holding_tickers"] == ["SMR"]
    assert bars["Semiconductors"]["holding_tickers"] == ["TSM"]


def test_momentum_map_detects_lagging_holdings() -> None:
    momentum_map = build_momentum_map(sample_tickers(), sample_industries(), sample_alerts())

    assert momentum_map["summary"]["lagging_holding_count"] == 1


def test_momentum_map_counts_red_or_orange_priority_review() -> None:
    momentum_map = build_momentum_map(sample_tickers(), sample_industries(), sample_alerts())

    assert momentum_map["summary"]["priority_review_count"] == 1


def test_momentum_map_detects_strong_industry_exposure_gap() -> None:
    momentum_map = build_momentum_map(sample_tickers(), sample_industries(), sample_alerts())
    gaps = {row["industry_group"] for row in momentum_map["momentum_exposure_gaps"]}

    assert "Cloud Software" in gaps
    assert "Semiconductors" not in gaps


def test_dashboard_html_contains_momentum_map_section() -> None:
    html = build_dashboard_html(
        {
            "summary": {},
            "momentum_map": {
                "industry_bars": [],
                "holding_alignment": [],
                "summary": {},
                "momentum_exposure_gaps": [],
            },
        }
    )

    assert "動能地圖" in html
    assert 'id="momentum-map"' in html
