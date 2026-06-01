from __future__ import annotations

import pandas as pd

from src.dashboard import build_dashboard_data
from src.metrics import build_ticker_output
from src.signals import add_signal_columns
from src.valuation import calculate_peg_metrics, download_fundamentals


def test_calculate_peg_metrics_rates_valid_forward_peg() -> None:
    metrics = calculate_peg_metrics({"forwardPE": 24, "earningsGrowth": 0.30})

    assert metrics["forward_pe"] == 24
    assert metrics["earnings_growth"] == 0.30
    assert metrics["peg_ratio"] == 0.8
    assert metrics["peg_rating"] == "undervalued"
    assert metrics["peg_status"] == "ok"


def test_calculate_peg_metrics_handles_missing_forward_pe() -> None:
    metrics = calculate_peg_metrics({"earningsGrowth": 0.30})

    assert metrics["peg_ratio"] is None
    assert metrics["peg_rating"] == "unavailable"
    assert metrics["peg_status"] == "missing_pe"


def test_calculate_peg_metrics_handles_missing_growth() -> None:
    metrics = calculate_peg_metrics({"forwardPE": 24})

    assert metrics["peg_ratio"] is None
    assert metrics["peg_rating"] == "unavailable"
    assert metrics["peg_status"] == "missing_growth"


def test_calculate_peg_metrics_handles_invalid_growth() -> None:
    metrics = calculate_peg_metrics({"forwardPE": 24, "earningsGrowth": 0})

    assert metrics["peg_ratio"] is None
    assert metrics["peg_rating"] == "unavailable"
    assert metrics["peg_status"] == "invalid_growth"


def test_download_fundamentals_continues_after_fetch_error(monkeypatch) -> None:
    class FakeTicker:
        def __init__(self, ticker: str) -> None:
            self.ticker = ticker

        @property
        def info(self) -> dict[str, float]:
            if self.ticker == "BAD":
                raise RuntimeError("boom")
            return {"forwardPE": 20, "earningsGrowth": 0.25}

    monkeypatch.setattr("src.valuation.yf.Ticker", FakeTicker)

    fundamentals = download_fundamentals(["GOOD", "BAD"])

    assert fundamentals["GOOD"] == {"forwardPE": 20, "earningsGrowth": 0.25}
    assert fundamentals["BAD"] is None


def test_build_ticker_output_adds_peg_columns_for_every_row() -> None:
    tickers = pd.DataFrame(
        [
            {
                "ticker": "GOOD",
                "company_name": "Good Co.",
                "industry_group": "Software",
                "leader_type": "core_leader",
                "industry_quality_score": 5,
            },
            {
                "ticker": "MISS",
                "company_name": "Missing Co.",
                "industry_group": "Software",
                "leader_type": "non_leader",
                "industry_quality_score": 3,
            },
        ]
    )
    dates = pd.date_range("2099-01-01", periods=30)
    downloaded_data = {
        "GOOD": pd.DataFrame(
            {
                "Adj Close": range(100, 130),
                "Volume": [1000] * 30,
            },
            index=dates,
        )
    }
    fundamentals = {
        "GOOD": {"forwardPE": 30, "earningsGrowth": 0.15},
        "MISS": None,
    }

    output = build_ticker_output(tickers, downloaded_data, fundamentals)

    assert ["forward_pe", "earnings_growth", "peg_ratio", "peg_rating", "peg_status", "peg_note"] == [
        column for column in output.columns if column.startswith("peg") or column in ["forward_pe", "earnings_growth"]
    ]
    assert output.loc[output["ticker"] == "GOOD", "peg_ratio"].iloc[0] == 2
    assert output.loc[output["ticker"] == "MISS", "peg_status"].iloc[0] == "fetch_error"


def test_signal_step_preserves_peg_columns() -> None:
    ticker_output = pd.DataFrame(
        [
            {
                "ticker": "GOOD",
                "company_name": "Good Co.",
                "industry_group": "Software",
                "latest_date": "2099-01-30",
                "data_points": 30,
                "return_3d": 0.03,
                "return_5d": 0.05,
                "return_10d": 0.10,
                "return_20d": 0.12,
                "return_1m": 0.12,
                "return_3m": 0.12,
                "return_6m": 0.12,
                "return_1y": 0.12,
                "avg_volume_3d": 1000,
                "avg_volume_5d": 1000,
                "avg_volume_20d": 1000,
                "relative_volume": 1.1,
                "ma_5d": 110,
                "ma_10d": 105,
                "ma_20d": 100,
                "max_drawdown_10d": -0.02,
                "up_days_10d": 7,
                "latest_price": 110,
                "leader_type": "core_leader",
                "industry_quality_score": 5,
                "distance_from_20d_ma": 0.10,
                "distance_from_52w_high": -0.05,
                "position_in_52w_range": 0.90,
                "latest_volume": 1100,
                "forward_pe": 30,
                "earnings_growth": 0.15,
                "peg_ratio": 2,
                "peg_rating": "fair",
                "peg_status": "ok",
                "peg_note": "test",
            }
        ]
    )
    industry_output = pd.DataFrame([{"industry_group": "Software", "return_10d": 0.06}])

    signaled = add_signal_columns(ticker_output, industry_output)

    assert signaled.loc[0, "peg_ratio"] == 2
    assert signaled.loc[0, "peg_status"] == "ok"


def test_dashboard_data_keeps_peg_columns_in_industry_constituents() -> None:
    ticker_output = pd.DataFrame(
        [
            {
                "ticker": "GOOD",
                "company_name": "Good Co.",
                "industry_group": "Software",
                "latest_date": "2099-01-30",
                "data_points": 30,
                "data_status": "ok",
                "leader_type": "core_leader",
                "industry_quality_score": 5,
                "watch_status": "research_candidate",
                "current_state": "strong_uptrend",
                "return_5d": 0.05,
                "return_10d": 0.10,
                "return_20d": 0.12,
                "return_1m": 0.12,
                "return_3m": 0.12,
                "return_6m": 0.12,
                "return_1y": 0.12,
                "relative_strength_vs_industry": 0.04,
                "latest_volume": 1000,
                "avg_volume_20d": 900,
                "relative_volume": 1.11,
                "early_momentum_signal": True,
                "confirmed_momentum_signal": True,
                "strong_momentum_signal": False,
                "risk_warning": False,
                "forward_pe": 30,
                "earnings_growth": 0.15,
                "peg_ratio": 2,
                "peg_rating": "fair",
                "peg_status": "ok",
                "peg_note": "test",
            }
        ]
    )
    for column in ["return_3d", "max_drawdown_10d", "up_days_10d", "distance_from_20d_ma", "distance_from_52w_high", "position_in_52w_range"]:
        ticker_output[column] = 0
    for column in ["industry_regime", "industry_risk_flag", "rotation_type", "causal_hypothesis", "evidence_status", "short_term_price_zone", "long_term_price_zone", "price_zone"]:
        ticker_output[column] = "neutral" if column.endswith("zone") or column == "industry_regime" else "none"
    industry_output = pd.DataFrame(
        [
            {
                "industry_group": "Software",
                "ticker_count": 1,
                "tickers_with_data": 1,
                "return_3d": 0.03,
                "return_5d": 0.05,
                "return_10d": 0.10,
                "return_20d": 0.12,
                "return_1m": 0.12,
                "return_3m": 0.12,
                "return_6m": 0.12,
                "return_1y": 0.12,
                "relative_volume": 1.11,
                "breadth_score": 1,
                "positive_5d_pct": 1,
                "positive_10d_pct": 1,
                "confirmed_signal_pct": 1,
                "strong_signal_pct": 0,
                "high_relative_volume_pct": 0,
                "rotation_score": 0,
                "momentum_persistence": 1,
                "momentum_acceleration": 0,
                "momentum_exhaustion_warning": False,
                "industry_regime": "momentum_leader",
                "industry_risk_flag": "none",
                "rotation_type": "unclear",
                "causal_hypothesis": "unclear",
                "evidence_status": "observed",
            }
        ]
    )

    dashboard = build_dashboard_data(ticker_output, industry_output, pd.DataFrame())
    row = dashboard["industry_constituents"]["Software"][0]

    assert row["peg_ratio"] == 2
    assert row["peg_rating"] == "fair"
    assert row["peg_status"] == "ok"
