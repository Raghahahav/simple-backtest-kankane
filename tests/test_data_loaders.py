"""Tests for optional data loader integrations."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from simple_backtest.data import (
    AlphaVantageLoader,
    CCXTLoader,
    CSVLoader,
    DataLoader,
    PolygonLoader,
    YFinanceLoader,
)
from simple_backtest.utils.validation import DataValidationError


def _make_mock_response(payload: dict) -> Mock:
    response = Mock()
    response.json.return_value = payload
    return response


class TestCSVLoader:
    """Tests for CSVLoader."""

    def test_valid_temp_csv(self, tmp_path):
        """Loads valid OHLCV CSV."""
        csv_path = tmp_path / "sample.csv"
        pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-01", periods=3, freq="D"),
                "Open": [1.0, 2.0, 3.0],
                "High": [2.0, 3.0, 4.0],
                "Low": [0.5, 1.5, 2.5],
                "Close": [1.5, 2.5, 3.5],
                "Volume": [100, 200, 300],
            }
        ).to_csv(csv_path, index=False)

        loader = CSVLoader()
        data = loader.load(str(csv_path))

        assert list(data.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert len(data) == 3
        assert isinstance(data.index, pd.DatetimeIndex)

    def test_missing_file_raises_file_not_found(self):
        """Raises clear file not found error."""
        loader = CSVLoader()

        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            loader.load("/path/does/not/exist.csv")

    def test_missing_required_columns_raises_validation_error(self, tmp_path):
        """Raises DataValidationError when required columns are missing."""
        csv_path = tmp_path / "missing_cols.csv"
        pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-01", periods=2, freq="D"),
                "Open": [1.0, 2.0],
                "Close": [1.1, 2.1],
            }
        ).to_csv(csv_path, index=False)

        loader = CSVLoader()

        with pytest.raises(DataValidationError, match="Missing required columns"):
            loader.load(str(csv_path))

    def test_date_range_filtering(self, tmp_path):
        """Filters loaded data by start/end date."""
        csv_path = tmp_path / "range.csv"
        pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-01", periods=5, freq="D"),
                "Open": [1, 2, 3, 4, 5],
                "High": [2, 3, 4, 5, 6],
                "Low": [0, 1, 2, 3, 4],
                "Close": [1.5, 2.5, 3.5, 4.5, 5.5],
                "Volume": [10, 20, 30, 40, 50],
            }
        ).to_csv(csv_path, index=False)

        loader = CSVLoader()
        data = loader.load(str(csv_path), start="2020-01-02", end="2020-01-04")

        assert len(data) == 3
        assert data.index.min() == pd.Timestamp("2020-01-02")
        assert data.index.max() == pd.Timestamp("2020-01-04")

    def test_lowercase_columns_are_normalized(self, tmp_path):
        """Normalizes lowercase OHLCV column names."""
        csv_path = tmp_path / "lowercase.csv"
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=2, freq="D"),
                "open": [1.0, 2.0],
                "high": [2.0, 3.0],
                "low": [0.5, 1.5],
                "close": [1.5, 2.5],
                "volume": [100, 200],
            }
        ).to_csv(csv_path, index=False)

        loader = CSVLoader()
        data = loader.load(str(csv_path))

        assert list(data.columns) == ["Open", "High", "Low", "Close", "Volume"]


class TestYFinanceLoader:
    """Tests for YFinanceLoader."""

    def test_import_error_when_yfinance_missing(self):
        """Raises import error when yfinance is unavailable."""
        loader = YFinanceLoader()

        with patch(
            "simple_backtest.data.yfinance_loader.importlib.import_module",
            side_effect=ModuleNotFoundError,
        ):
            with pytest.raises(
                ImportError,
                match="yfinance is not installed. Install it with: pip install yfinance",
            ):
                loader.load("AAPL", "2020-01-01", "2020-01-10")

    def test_empty_download_raises_value_error(self):
        """Raises clear error when yfinance returns empty dataframe."""
        mock_yf = SimpleNamespace(download=Mock(return_value=pd.DataFrame()))
        loader = YFinanceLoader()

        with patch(
            "simple_backtest.data.yfinance_loader.importlib.import_module",
            return_value=mock_yf,
        ):
            with pytest.raises(ValueError, match="No data returned for symbol 'AAPL'"):
                loader.load("AAPL", "2020-01-01", "2023-12-31")

    def test_multiindex_columns_are_flattened(self):
        """Flattens yfinance MultiIndex columns to standard OHLCV columns."""
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        columns = pd.MultiIndex.from_tuples(
            [
                ("Open", "AAPL"),
                ("High", "AAPL"),
                ("Low", "AAPL"),
                ("Close", "AAPL"),
                ("Volume", "AAPL"),
            ]
        )
        yfinance_df = pd.DataFrame(
            [[1, 2, 0.5, 1.5, 100], [2, 3, 1.5, 2.5, 200], [3, 4, 2.5, 3.5, 300]],
            index=dates,
            columns=columns,
        )

        mock_yf = SimpleNamespace(download=Mock(return_value=yfinance_df))
        loader = YFinanceLoader()

        with patch(
            "simple_backtest.data.yfinance_loader.importlib.import_module",
            return_value=mock_yf,
        ):
            data = loader.load("AAPL", "2020-01-01", "2020-01-03")

        assert list(data.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert len(data) == 3


class TestCCXTLoader:
    """Tests for CCXTLoader."""

    def test_import_error_when_ccxt_missing(self):
        """Raises import error when ccxt is unavailable."""
        with patch(
            "simple_backtest.data.ccxt_loader.importlib.import_module",
            side_effect=ModuleNotFoundError,
        ):
            with pytest.raises(ImportError, match="ccxt is not installed"):
                CCXTLoader("binance")

    def test_unsupported_exchange_raises_value_error(self):
        """Raises clear error for unsupported exchange."""
        mock_ccxt = SimpleNamespace(exchanges=["binance", "coinbase"])

        with patch(
            "simple_backtest.data.ccxt_loader.importlib.import_module",
            return_value=mock_ccxt,
        ):
            with pytest.raises(ValueError, match="Exchange 'xyz' is not supported"):
                CCXTLoader("xyz")

    def test_parses_ohlcv_data_from_ccxt_format(self):
        """Parses ccxt OHLCV list format to OHLCV DataFrame."""

        class FakeExchange:
            def __init__(self, _kwargs):
                self.fetch_ohlcv = Mock(
                    return_value=[
                        [1577836800000, 1.0, 2.0, 0.5, 1.5, 100.0],
                        [1577923200000, 2.0, 3.0, 1.5, 2.5, 200.0],
                    ]
                )

        mock_ccxt = SimpleNamespace(exchanges=["binance"], binance=FakeExchange)

        with patch(
            "simple_backtest.data.ccxt_loader.importlib.import_module",
            return_value=mock_ccxt,
        ):
            loader = CCXTLoader("binance")
            data = loader.load("BTC/USDT", "2020-01-01", "2020-01-02")

        assert list(data.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert len(data) >= 1

    def test_empty_response_raises_value_error(self):
        """Raises clear error when ccxt returns no candles."""

        class FakeExchange:
            def __init__(self, _kwargs):
                self.fetch_ohlcv = Mock(return_value=[])

        mock_ccxt = SimpleNamespace(exchanges=["binance"], binance=FakeExchange)

        with patch(
            "simple_backtest.data.ccxt_loader.importlib.import_module",
            return_value=mock_ccxt,
        ):
            loader = CCXTLoader("binance")
            with pytest.raises(
                ValueError, match="No data returned for symbol 'BTC/USDT'"
            ):
                loader.load("BTC/USDT", "2020-01-01", "2020-01-10")


class TestAlphaVantageLoader:
    """Tests for AlphaVantageLoader."""

    def test_import_error_when_requests_missing(self):
        """Raises import error when requests is unavailable."""
        loader = AlphaVantageLoader(api_key="test-key")

        with patch(
            "simple_backtest.data.alphavantage_loader.importlib.import_module",
            side_effect=ModuleNotFoundError,
        ):
            with pytest.raises(ImportError, match="requests is not installed"):
                loader.load("AAPL", "2020-01-01", "2020-01-10")

    def test_api_error_in_json_raises_value_error(self):
        """Raises API error message from JSON response."""
        mock_requests = SimpleNamespace(
            get=Mock(
                return_value=_make_mock_response({"Error Message": "Invalid API call."})
            )
        )
        loader = AlphaVantageLoader(api_key="bad-key")

        with patch(
            "simple_backtest.data.alphavantage_loader.importlib.import_module",
            return_value=mock_requests,
        ):
            with pytest.raises(ValueError, match="Invalid API call"):
                loader.load("AAPL", "2020-01-01", "2020-01-10")

    def test_parses_alpha_vantage_json_format(self):
        """Parses valid Alpha Vantage JSON into OHLCV DataFrame."""
        payload = {
            "Time Series (Daily)": {
                "2020-01-02": {
                    "1. open": "10.0",
                    "2. high": "11.0",
                    "3. low": "9.5",
                    "4. close": "10.5",
                    "5. volume": "1000",
                },
                "2020-01-03": {
                    "1. open": "10.5",
                    "2. high": "11.5",
                    "3. low": "10.0",
                    "4. close": "11.0",
                    "5. volume": "1500",
                },
            }
        }

        mock_requests = SimpleNamespace(
            get=Mock(return_value=_make_mock_response(payload))
        )
        loader = AlphaVantageLoader(api_key="test-key")

        with patch(
            "simple_backtest.data.alphavantage_loader.importlib.import_module",
            return_value=mock_requests,
        ):
            data = loader.load("AAPL", "2020-01-01", "2020-01-05")

        assert list(data.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert len(data) == 2

    def test_empty_date_range_after_filtering_raises_value_error(self):
        """Raises clear error when filtered range has no rows."""
        payload = {
            "Time Series (Daily)": {
                "2020-01-02": {
                    "1. open": "10.0",
                    "2. high": "11.0",
                    "3. low": "9.5",
                    "4. close": "10.5",
                    "5. volume": "1000",
                }
            }
        }

        mock_requests = SimpleNamespace(
            get=Mock(return_value=_make_mock_response(payload))
        )
        loader = AlphaVantageLoader(api_key="test-key")

        with patch(
            "simple_backtest.data.alphavantage_loader.importlib.import_module",
            return_value=mock_requests,
        ):
            with pytest.raises(ValueError, match="No Alpha Vantage data"):
                loader.load("AAPL", "2021-01-01", "2021-01-10")


class TestPolygonLoader:
    """Tests for PolygonLoader."""

    def test_import_error_when_requests_missing(self):
        """Raises import error when requests is unavailable."""
        loader = PolygonLoader(api_key="test-key")

        with patch(
            "simple_backtest.data.polygon_loader.importlib.import_module",
            side_effect=ModuleNotFoundError,
        ):
            with pytest.raises(ImportError, match="requests is not installed"):
                loader.load("AAPL", "2020-01-01", "2020-01-10")

    def test_api_error_response_raises_value_error(self):
        """Raises clear error for Polygon API errors."""
        payload = {"status": "ERROR", "error": "Invalid API Key"}
        mock_requests = SimpleNamespace(
            get=Mock(return_value=_make_mock_response(payload))
        )
        loader = PolygonLoader(api_key="bad-key")

        with patch(
            "simple_backtest.data.polygon_loader.importlib.import_module",
            return_value=mock_requests,
        ):
            with pytest.raises(ValueError, match="Polygon API error"):
                loader.load("AAPL", "2020-01-01", "2020-01-10")

    def test_parses_polygon_json_format(self):
        """Parses valid Polygon aggregate response."""
        payload = {
            "status": "OK",
            "results": [
                {
                    "o": 10.0,
                    "h": 11.0,
                    "l": 9.5,
                    "c": 10.5,
                    "v": 1000.0,
                    "t": 1577836800000,
                }
            ],
        }
        mock_requests = SimpleNamespace(
            get=Mock(return_value=_make_mock_response(payload))
        )
        loader = PolygonLoader(api_key="test-key")

        with patch(
            "simple_backtest.data.polygon_loader.importlib.import_module",
            return_value=mock_requests,
        ):
            data = loader.load("AAPL", "2020-01-01", "2020-01-10")

        assert list(data.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert len(data) == 1

    def test_pagination_handling(self):
        """Follows next_url pagination and merges all rows."""
        first_payload = {
            "status": "OK",
            "results": [
                {
                    "o": 10.0,
                    "h": 11.0,
                    "l": 9.5,
                    "c": 10.5,
                    "v": 1000.0,
                    "t": 1577836800000,
                }
            ],
            "next_url": "https://api.polygon.io/v2/aggs/ticker/AAPL/next",
        }
        second_payload = {
            "status": "OK",
            "results": [
                {
                    "o": 10.5,
                    "h": 11.5,
                    "l": 10.0,
                    "c": 11.0,
                    "v": 1500.0,
                    "t": 1577923200000,
                }
            ],
        }

        mock_requests = SimpleNamespace(
            get=Mock(
                side_effect=[
                    _make_mock_response(first_payload),
                    _make_mock_response(second_payload),
                ]
            )
        )
        loader = PolygonLoader(api_key="test-key")

        with patch(
            "simple_backtest.data.polygon_loader.importlib.import_module",
            return_value=mock_requests,
        ):
            data = loader.load("AAPL", "2020-01-01", "2020-01-10")

        assert len(data) == 2
        assert mock_requests.get.call_count == 2


class TestDataLoaderBase:
    """Tests for DataLoader abstract class."""

    def test_cannot_instantiate_abstract_base(self):
        """DataLoader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DataLoader()
