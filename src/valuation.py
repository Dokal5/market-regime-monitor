from __future__ import annotations

import math
from typing import Any

import pandas as pd
import yfinance as yf


def numeric_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def calculate_peg_metrics(fundamentals: dict[str, Any] | None) -> dict[str, Any]:
    if fundamentals is None:
        return {
            "forward_pe": None,
            "earnings_growth": None,
            "peg_ratio": None,
            "peg_rating": "unavailable",
            "peg_status": "fetch_error",
            "peg_note": "Could not fetch Yahoo Finance fundamentals.",
        }

    forward_pe = numeric_or_none(fundamentals.get("forwardPE"))
    earnings_growth = numeric_or_none(fundamentals.get("earningsGrowth"))

    if forward_pe is None or forward_pe <= 0:
        return {
            "forward_pe": forward_pe,
            "earnings_growth": earnings_growth,
            "peg_ratio": None,
            "peg_rating": "unavailable",
            "peg_status": "missing_pe",
            "peg_note": "Yahoo Finance forwardPE is missing or not positive.",
        }
    if earnings_growth is None:
        return {
            "forward_pe": forward_pe,
            "earnings_growth": None,
            "peg_ratio": None,
            "peg_rating": "unavailable",
            "peg_status": "missing_growth",
            "peg_note": "Yahoo Finance earningsGrowth is missing.",
        }
    if earnings_growth <= 0:
        return {
            "forward_pe": forward_pe,
            "earnings_growth": earnings_growth,
            "peg_ratio": None,
            "peg_rating": "unavailable",
            "peg_status": "invalid_growth",
            "peg_note": "Yahoo Finance earningsGrowth is zero or negative.",
        }

    peg_ratio = forward_pe / (earnings_growth * 100)
    if peg_ratio < 1:
        peg_rating = "undervalued"
    elif peg_ratio <= 2:
        peg_rating = "fair"
    else:
        peg_rating = "expensive"

    return {
        "forward_pe": forward_pe,
        "earnings_growth": earnings_growth,
        "peg_ratio": peg_ratio,
        "peg_rating": peg_rating,
        "peg_status": "ok",
        "peg_note": "Forward PEG uses Yahoo forwardPE divided by earningsGrowth percent.",
    }


def download_fundamentals(ticker_symbols: list[str]) -> dict[str, dict[str, Any] | None]:
    fundamentals = {}
    for ticker in ticker_symbols:
        try:
            fundamentals[ticker] = yf.Ticker(ticker).info
        except Exception as exc:
            print(f"Warning: could not download fundamentals for {ticker}: {exc}")
            fundamentals[ticker] = None
    return fundamentals
