from __future__ import annotations

import pandas as pd

from src.watchlist import build_watchlist_alerts, clean_watchlist


def test_clean_watchlist_normalizes_tickers_and_optional_columns() -> None:
    watchlist = clean_watchlist(pd.DataFrame([{"ticker": " vrt "}]))

    assert watchlist.loc[0, "ticker"] == "VRT"
    assert "theme" in watchlist.columns
    assert "notes" in watchlist.columns


def test_build_watchlist_alerts_flags_weak_ticker_for_replacement() -> None:
    watchlist = pd.DataFrame([{"ticker": "VRT", "theme": "AI Infrastructure"}])
    tickers = pd.DataFrame(
        [
            {
                "ticker": "VRT",
                "company_name": "Vertiv Holdings Co.",
                "industry_group": "AI Infrastructure",
                "latest_date": "2099-01-01",
                "data_points": 251,
                "return_5d": -0.02,
                "return_10d": -0.16,
                "relative_strength_vs_industry": -0.37,
                "risk_warning": True,
                "watch_status": "avoid_for_now",
                "current_state": "falling_knife",
                "industry_regime": "momentum_leader",
                "industry_risk_flag": "none",
                "data_status": "ok",
                "early_momentum_signal": False,
                "strong_momentum_signal": False,
            },
            {
                "ticker": "SPOT",
                "company_name": "Spotify Technology S.A.",
                "industry_group": "Consumer Platforms",
                "latest_date": "2099-01-01",
                "data_points": 251,
                "return_5d": 0.03,
                "return_10d": 0.15,
                "relative_strength_vs_industry": 0.12,
                "risk_warning": False,
                "watch_status": "not_eligible_industry",
                "current_state": "strong_uptrend",
                "industry_regime": "momentum_leader",
                "industry_risk_flag": "none",
                "data_status": "ok",
                "early_momentum_signal": True,
                "strong_momentum_signal": True,
            },
        ]
    )
    industries = pd.DataFrame(
        [
            {
                "industry_group": "Consumer Platforms",
                "return_5d": 0.03,
                "return_10d": 0.10,
                "breadth_score": 0.70,
                "rotation_score": 2,
                "momentum_acceleration": 0.01,
                "momentum_exhaustion_warning": False,
                "industry_regime": "momentum_leader",
                "industry_risk_flag": "none",
            },
            {
                "industry_group": "AI Infrastructure",
                "return_5d": 0.20,
                "return_10d": 0.25,
                "breadth_score": 0.60,
                "rotation_score": 0,
                "momentum_acceleration": 0.02,
                "momentum_exhaustion_warning": False,
                "industry_regime": "momentum_leader",
                "industry_risk_flag": "none",
            },
        ]
    )

    alerts = build_watchlist_alerts(watchlist, tickers, industries)

    assert alerts.loc[0, "alert_level"] == "red"
    assert alerts.loc[0, "action"] == "review_replacement"
    assert "risk_warning" in alerts.loc[0, "alert_reason"]
    assert "SPOT" in alerts.loc[0, "replacement_candidates"]


def test_build_watchlist_alerts_marks_untracked_ticker_unknown() -> None:
    watchlist = pd.DataFrame([{"ticker": "NEW"}])

    alerts = build_watchlist_alerts(watchlist, pd.DataFrame(), pd.DataFrame())

    assert alerts.loc[0, "alert_level"] == "unknown"
    assert alerts.loc[0, "action"] == "add_to_tickers"
