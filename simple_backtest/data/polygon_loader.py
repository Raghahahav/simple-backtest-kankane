"""Polygon.io data loader."""

from __future__ import annotations

import importlib
from datetime import datetime

import pandas as pd

from simple_backtest.data.base import DataLoader


class PolygonLoader(DataLoader):
    """Load OHLCV aggregates from Polygon.io."""

    BASE_URL = "https://api.polygon.io/v2/aggs/ticker"

    def __init__(self, api_key: str):
        """Initialize loader with API key."""
        if not api_key:
            raise ValueError("api_key is required for PolygonLoader")
        self.api_key = api_key

    def load(
        self,
        symbol: str,
        start: datetime | str,
        end: datetime | str,
        timespan: str = "day",
        multiplier: int = 1,
    ) -> pd.DataFrame:
        """Fetch aggregate bars from Polygon.io with pagination."""
        try:
            requests = importlib.import_module("requests")
        except ModuleNotFoundError as exc:
            raise ImportError(
                "requests is not installed. Install it with: pip install requests"
            ) from exc

        url = f"{self.BASE_URL}/{symbol}/range/{multiplier}/{timespan}/{start}/{end}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key,
        }

        results: list[dict] = []

        while url:
            response = requests.get(url, params=params, timeout=30)
            payload = response.json()

            status = payload.get("status")
            if status not in {"OK", "DELAYED"}:
                message = payload.get("error") or payload.get("message") or str(payload)
                raise ValueError(
                    f"Polygon API error for symbol '{symbol}' between {start} and {end}: {message}"
                )

            batch_results = payload.get("results", [])
            if batch_results:
                results.extend(batch_results)

            next_url = payload.get("next_url")
            if next_url:
                url = next_url
                params = {"apiKey": self.api_key}
            else:
                url = ""

        if not results:
            raise ValueError(
                f"No Polygon data returned for symbol '{symbol}' between {start} and {end}"
            )

        data = pd.DataFrame(results)
        data = data.rename(
            columns={
                "o": "Open",
                "h": "High",
                "l": "Low",
                "c": "Close",
                "v": "Volume",
                "t": "timestamp",
            }
        )

        required = {"timestamp", "Open", "High", "Low", "Close", "Volume"}
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(
                f"Polygon response missing expected fields {missing} for symbol '{symbol}'"
            )

        data["timestamp"] = pd.to_datetime(
            data["timestamp"], unit="ms", utc=True
        ).dt.tz_localize(None)
        data = data.set_index("timestamp").sort_index()

        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        data = data[(data.index >= start_ts) & (data.index <= end_ts)]

        if data.empty:
            raise ValueError(
                f"No Polygon data returned for symbol '{symbol}' between {start} and {end}"
            )

        return self._finalize_dataframe(data)
