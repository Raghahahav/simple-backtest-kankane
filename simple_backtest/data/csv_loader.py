"""CSV data loader."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from simple_backtest.data.base import DataLoader
from simple_backtest.utils.validation import DataValidationError


class CSVLoader(DataLoader):
    """Load OHLCV data from CSV files."""

    DATE_CANDIDATES = ["Date", "date", "Datetime", "datetime"]

    def load(
        self,
        filepath: str,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
    ) -> pd.DataFrame:
        """Load OHLCV data from CSV and optionally filter by date range."""
        file_path = Path(filepath)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        data = pd.read_csv(file_path)

        date_column = next(
            (col for col in self.DATE_CANDIDATES if col in data.columns), None
        )

        if date_column is not None:
            data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
            if data[date_column].isna().all():
                raise DataValidationError(
                    f"Date column '{date_column}' could not be parsed as datetime values"
                )
            data = data.set_index(date_column)
        else:
            # Try parsing first column as datetime index.
            first_column = data.columns[0]
            parsed_first_column = pd.to_datetime(data[first_column], errors="coerce")
            if parsed_first_column.notna().all():
                data[first_column] = parsed_first_column
                data = data.set_index(first_column)
            else:
                # Fallback: read first column as index and parse index.
                indexed_data = pd.read_csv(file_path, index_col=0)
                parsed_index = pd.to_datetime(indexed_data.index, errors="coerce")
                if parsed_index.notna().all():
                    indexed_data.index = parsed_index
                    data = indexed_data
                else:
                    raise DataValidationError(
                        "Could not detect a datetime column/index. "
                        "Expected one of ['Date', 'date', 'Datetime', 'datetime'] "
                        "or a datetime index."
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

        data = data.sort_index()

        if start is not None:
            start_ts = pd.to_datetime(start)
            data = data[data.index >= start_ts]

        if end is not None:
            end_ts = pd.to_datetime(end)
            data = data[data.index <= end_ts]

        return self._finalize_dataframe(data)
