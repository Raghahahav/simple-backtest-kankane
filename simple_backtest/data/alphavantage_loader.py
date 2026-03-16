"""Alpha Vantage data loader."""

from __future__ import annotations

import importlib
from datetime import datetime

import pandas as pd

from simple_backtest.data.base import DataLoader


class AlphaVantageLoader(DataLoader):
    """Load OHLCV data from Alpha Vantage daily endpoint."""

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str):
        """Initialize loader with API key."""
        if not api_key:
            raise ValueError("api_key is required for AlphaVantageLoader")
        self.api_key = api_key

    def load(
        self,
        symbol: str,
        start: datetime | str,
        end: datetime | str,
        outputsize: str = "full",
    ) -> pd.DataFrame:
        """Fetch daily OHLCV data from Alpha Vantage."""
        try:
            requests = importlib.import_module("requests")
        except ModuleNotFoundError as exc:
            raise ImportError(
                "requests is not installed. Install it with: pip install requests"
            ) from exc

        response = requests.get(
            self.BASE_URL,
            params={
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": outputsize,
                "apikey": self.api_key,
            },
            timeout=30,
        )
        payload = response.json()

        for error_key in ("Error Message", "Information", "Note"):
            if error_key in payload:
                raise ValueError(payload[error_key])

        time_series_key = "Time Series (Daily)"
        if time_series_key not in payload:
            raise ValueError(
                "Unexpected Alpha Vantage response format: missing daily time series"
            )

        rows = []
        for date_str, values in payload[time_series_key].items():
            rows.append(
                {
                    "Date": pd.to_datetime(date_str),
                    "Open": float(values["1. open"]),
                    "High": float(values["2. high"]),
                    "Low": float(values["3. low"]),
                    "Close": float(values["4. close"]),
                    "Volume": float(values["5. volume"]),
                }
            )

        data = pd.DataFrame(rows).set_index("Date").sort_index()

        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        data = data[(data.index >= start_ts) & (data.index <= end_ts)]

        if data.empty:
            raise ValueError(
                f"No Alpha Vantage data for symbol '{symbol}' between {start} and {end}"
            )

        return self._finalize_dataframe(data)
