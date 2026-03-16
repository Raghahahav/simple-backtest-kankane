"""CCXT data loader."""

from __future__ import annotations

import importlib
from datetime import datetime
from typing import Any

import pandas as pd

from simple_backtest.data.base import DataLoader


class CCXTLoader(DataLoader):
    """Load OHLCV data from cryptocurrency exchanges via ccxt."""

    def __init__(
        self,
        exchange_name: str,
        api_key: str | None = None,
        api_secret: str | None = None,
    ):
        """Initialize ccxt exchange client."""
        try:
            self._ccxt = importlib.import_module("ccxt")
        except ModuleNotFoundError as exc:
            raise ImportError(
                "ccxt is not installed. Install it with: pip install ccxt"
            ) from exc

        self.exchange_name = exchange_name

        if exchange_name not in self._ccxt.exchanges:
            available = sorted(self._ccxt.exchanges)[:10]
            raise ValueError(
                f"Exchange '{exchange_name}' is not supported. Available exchanges: {available}"
            )

        exchange_cls = getattr(self._ccxt, exchange_name)
        exchange_kwargs: dict[str, Any] = {}
        if api_key:
            exchange_kwargs["apiKey"] = api_key
        if api_secret:
            exchange_kwargs["secret"] = api_secret

        self.exchange = exchange_cls(exchange_kwargs)

    def load(
        self,
        symbol: str,
        start: datetime | str,
        end: datetime | str,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """Load OHLCV candles from ccxt with automatic pagination."""
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)

        since_ms = int(start_ts.timestamp() * 1000)
        end_ms = int(end_ts.timestamp() * 1000)

        all_rows: list[list[float]] = []
        limit = 1000
        current_since = since_ms

        while True:
            batch = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=current_since,
                limit=limit,
            )

            if not batch:
                break

            all_rows.extend(batch)

            last_timestamp = int(batch[-1][0])
            if last_timestamp >= end_ms:
                break

            if len(batch) < limit:
                break

            next_since = last_timestamp + 1
            if next_since <= current_since:
                break
            current_since = next_since

        if not all_rows:
            raise ValueError(
                f"No data returned for symbol '{symbol}' on exchange '{self.exchange_name}' "
                f"between {start} and {end}"
            )

        data = pd.DataFrame(
            all_rows,
            columns=["timestamp", "Open", "High", "Low", "Close", "Volume"],
        )

        data["timestamp"] = pd.to_datetime(
            data["timestamp"], unit="ms", utc=True
        ).dt.tz_localize(None)
        data = data.set_index("timestamp").sort_index()

        data = data[(data.index >= start_ts) & (data.index <= end_ts)]

        if data.empty:
            raise ValueError(
                f"No data returned for symbol '{symbol}' on exchange '{self.exchange_name}' "
                f"between {start} and {end}"
            )

        return self._finalize_dataframe(data)
