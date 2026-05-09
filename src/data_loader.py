from __future__ import annotations

import pandas as pd
import yfinance as yf

from src.config import INPUT_COLUMNS, LOOKBACK_PERIOD, MARKET_DATA_INTERVAL


def clean_tickers(raw_tickers: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in INPUT_COLUMNS if column not in raw_tickers.columns]
    if missing_columns:
        raise ValueError(f"tickers.csv is missing required columns: {', '.join(missing_columns)}")

    tickers = raw_tickers[INPUT_COLUMNS].copy()
    tickers["ticker"] = tickers["ticker"].astype(str).str.strip().str.upper()
    tickers["company_name"] = tickers["company_name"].fillna("").astype(str).str.strip()
    tickers["industry_group"] = tickers["industry_group"].fillna("Unknown").astype(str).str.strip()
    tickers = tickers[tickers["ticker"] != ""].drop_duplicates(subset=["ticker"])
    return tickers.reset_index(drop=True)


def download_market_data(ticker_symbols: list[str]) -> dict[str, pd.DataFrame]:
    market_data = {}
    if not ticker_symbols:
        return market_data

    for ticker in ticker_symbols:
        try:
            market_data[ticker] = yf.download(
                tickers=ticker,
                period=LOOKBACK_PERIOD,
                interval=MARKET_DATA_INTERVAL,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            print(f"Warning: could not download data for {ticker}: {exc}")
            market_data[ticker] = pd.DataFrame()

    return market_data


def get_ticker_frame(downloaded_data: dict[str, pd.DataFrame], ticker: str) -> pd.DataFrame:
    ticker_data = downloaded_data.get(ticker, pd.DataFrame()).copy()
    if ticker_data.empty:
        return pd.DataFrame()

    if isinstance(ticker_data.columns, pd.MultiIndex):
        if ticker in ticker_data.columns.get_level_values(0):
            ticker_data = ticker_data[ticker].copy()
        elif ticker in ticker_data.columns.get_level_values(1):
            ticker_data = ticker_data.xs(ticker, axis=1, level=1, drop_level=True).copy()
        else:
            ticker_data.columns = ticker_data.columns.get_level_values(-1)

    close_column = "Adj Close" if "Adj Close" in ticker_data.columns else "Close"
    required_columns = [close_column, "Volume"]
    if any(column not in ticker_data.columns for column in required_columns):
        return pd.DataFrame()

    prices = ticker_data[required_columns].rename(columns={close_column: "adjusted_close"})
    prices = prices.dropna(subset=["adjusted_close", "Volume"])
    prices = prices[prices["Volume"].notna()]
    return prices.sort_index()
