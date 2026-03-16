"""Base interfaces and helpers for data source loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd

from simple_backtest.utils.validation import DataValidationError, validate_dataframe


class DataLoader(ABC):
    """Abstract base class for all data loaders."""

    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

    @abstractmethod
    def load(
        self, symbol: str, start: datetime | str | None, end: datetime | str | None
    ) -> pd.DataFrame:
        """Load market data and return a validated OHLCV DataFrame."""

    def get_name(self) -> str:
        """Return loader name."""
        return self.__class__.__name__

    @staticmethod
    def _normalize_ohlcv_columns(data: pd.DataFrame) -> pd.DataFrame:
        """Normalize common OHLCV column variants to standard names."""
        if data.empty:
            return data

        def normalize_key(value: Any) -> str:
            return str(value).strip().lower().replace("_", "").replace(" ", "")

        variants = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adjclose": "Close",
            "adjustedclose": "Close",
            "volume": "Volume",
        }

        rename_map: dict[str, str] = {}
        normalized_existing = {normalize_key(col) for col in data.columns}

        for col in data.columns:
            normalized_col = normalize_key(col)

            # Avoid duplicate 'Close' when both Close and Adj Close exist
            if (
                normalized_col in {"adjclose", "adjustedclose"}
                and "close" in normalized_existing
            ):
                continue

            canonical = variants.get(normalized_col)
            if canonical is not None:
                rename_map[str(col)] = canonical

        if not rename_map:
            return data

        return data.rename(columns=rename_map)

    def _finalize_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and return a standardized OHLCV DataFrame."""
        if not isinstance(data, pd.DataFrame):
            raise DataValidationError(
                f"Data loader '{self.get_name()}' must return a pandas DataFrame, "
                f"got {type(data).__name__}"
            )

        data = self._normalize_ohlcv_columns(data)

        missing_columns = [
            col for col in self.REQUIRED_COLUMNS if col not in data.columns
        ]
        if missing_columns:
            raise DataValidationError(
                f"Missing required columns: {missing_columns}. "
                f"Your DataFrame has: {list(data.columns)}"
            )

        finalized = data[self.REQUIRED_COLUMNS].copy()
        validate_dataframe(finalized, require_volume=True)
        return finalized
