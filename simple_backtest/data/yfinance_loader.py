"""YFinance data loader."""

from __future__ import annotations

import importlib
from datetime import datetime

import pandas as pd

from simple_backtest.data.base import DataLoader


class YFinanceLoader(DataLoader):
    """Load OHLCV data from Yahoo Finance via yfinance."""

    def load(
        self, symbol: str, start: datetime | str, end: datetime | str
    ) -> pd.DataFrame:
        """Load OHLCV data for a symbol and date range."""
        try:
            yf = importlib.import_module("yfinance")
        except ModuleNotFoundError as exc:
            raise ImportError(
                "yfinance is not installed. Install it with: pip install yfinance"
            ) from exc

        data = yf.download(
            symbol, start=start, end=end, progress=False, auto_adjust=False
        )

        if data.empty:
            raise ValueError(
                f"No data returned for symbol '{symbol}' between {start} and {end}"
            )

        if isinstance(data.columns, pd.MultiIndex):
            if data.columns.nlevels == 2:
                tickers = list(data.columns.get_level_values(1).unique())
                if len(tickers) == 1:
                    data.columns = data.columns.droplevel(1)
                elif symbol in tickers:
                    data = data.xs(symbol, level=1, axis=1)
                else:
                    data = data.xs(tickers[0], level=1, axis=1)
            else:
                # Best-effort flatten for unexpected MultiIndex structures.
                data.columns = data.columns.get_level_values(0)

        return self._finalize_dataframe(data)
