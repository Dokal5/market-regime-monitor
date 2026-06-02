from __future__ import annotations

import pandas as pd

from src.data_quality import add_data_quality_columns, build_data_quality_summary, latest_market_date


def test_latest_market_date_uses_primary_market_tickers_as_benchmark() -> None:
    tickers = pd.DataFrame(
        [
            {"ticker": "MSFT", "latest_date": "2026-06-01", "data_points": 251},
            {"ticker": "VUAA.L", "latest_date": "2026-06-02", "data_points": 253},
        ]
    )

    assert latest_market_date(tickers) == "2026-06-01"


def test_cross_market_later_date_does_not_mark_primary_market_stale() -> None:
    tickers = pd.DataFrame(
        [
            {"ticker": "MSFT", "latest_date": "2026-06-01", "data_points": 251},
            {"ticker": "EWT", "latest_date": "2026-06-01", "data_points": 251},
            {"ticker": "VUAA.L", "latest_date": "2026-06-02", "data_points": 253},
        ]
    )

    quality = add_data_quality_columns(tickers)
    summary = build_data_quality_summary(quality)

    assert summary["latest_market_date"] == "2026-06-01"
    assert summary["stale_count"] == 0
    assert quality.set_index("ticker").loc["MSFT", "data_status"] == "ok"
    assert quality.set_index("ticker").loc["VUAA.L", "data_status"] == "ok"
    assert "跨市場交易日差異" in quality.set_index("ticker").loc["VUAA.L", "data_quality_note"]
