from __future__ import annotations

import pandas as pd

from src.dashboard import build_dashboard_html, build_research_workbench


def base_tickers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "MU",
                "company_name": "Micron",
                "industry_group": "Semiconductors",
                "data_status": "ok",
                "watch_status": "hold_or_add_on_pullback",
                "current_state": "strong_uptrend",
                "leader_type": "core_leader",
                "relative_strength_vs_industry": 0.04,
                "return_10d": 0.08,
                "relative_volume": 1.4,
                "strong_momentum_signal": True,
                "risk_warning": False,
            },
            {
                "ticker": "VRT",
                "company_name": "Vertiv",
                "industry_group": "AI Infrastructure",
                "data_status": "ok",
                "watch_status": "research_candidate",
                "current_state": "early_uptrend",
                "leader_type": "infrastructure_leader",
                "relative_strength_vs_industry": 0.06,
                "return_10d": 0.10,
                "relative_volume": 1.1,
                "strong_momentum_signal": False,
                "risk_warning": False,
            },
        ]
    )


def base_industries() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "industry_group": "Semiconductors",
                "industry_regime": "momentum_leader",
                "industry_risk_flag": "none",
                "return_10d": 0.07,
                "breadth_score": 0.82,
                "momentum_acceleration": 0.02,
            },
            {
                "industry_group": "Cloud Software",
                "industry_regime": "early_recovery",
                "industry_risk_flag": "none",
                "return_10d": 0.05,
                "breadth_score": 0.74,
                "momentum_acceleration": 0.03,
            },
        ]
    )


def test_research_workbench_maps_priority_holding_alert_to_thesis() -> None:
    alerts = pd.DataFrame(
        [
            {
                "ticker": "MU",
                "company_name": "Micron",
                "industry_group": "Semiconductors",
                "alert_level": "red",
                "action": "review before open",
                "alert_reason": "holding is lagging its industry",
                "holding_status": "holding",
            }
        ]
    )

    workbench = build_research_workbench(base_tickers(), base_industries(), alerts)

    thesis_tasks = [task for task in workbench["tasks"] if task["target"] == "MU" and task["command"] == "/thesis"]
    assert thesis_tasks
    assert thesis_tasks[0]["priority"] == "high"
    assert workbench["summary"]["high_priority_count"] >= 1


def test_research_workbench_maps_research_candidate_to_initiate() -> None:
    workbench = build_research_workbench(base_tickers(), base_industries(), pd.DataFrame())

    initiate_tasks = [task for task in workbench["tasks"] if task["target"] == "VRT" and task["command"] == "/initiate"]
    assert initiate_tasks
    assert initiate_tasks[0]["workflow"] == "深度研究"


def test_research_workbench_maps_unheld_strong_industry_to_sector() -> None:
    alerts = pd.DataFrame(
        [
            {
                "ticker": "MU",
                "company_name": "Micron",
                "industry_group": "Semiconductors",
                "alert_level": "green",
                "holding_status": "holding",
            }
        ]
    )

    workbench = build_research_workbench(base_tickers(), base_industries(), alerts)

    sector_tasks = [
        task for task in workbench["tasks"] if task["target"] == "Cloud Software" and task["command"] == "/sector"
    ]
    assert sector_tasks
    assert sector_tasks[0]["target_type"] == "industry"


def test_dashboard_html_contains_research_workbench_section() -> None:
    html = build_dashboard_html(
        {
            "summary": {},
            "momentum_map": {
                "industry_bars": [],
                "holding_alignment": [],
                "summary": {},
                "momentum_exposure_gaps": [],
            },
            "research_workbench": {
                "summary": {},
                "command_cards": [],
                "tasks": [],
            },
        }
    )

    assert "AI 研究工作台" in html
    assert 'id="research-workbench"' in html
    assert 'data-table="research-tasks"' in html
