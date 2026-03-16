"""Data source loaders."""

from simple_backtest.data.alphavantage_loader import AlphaVantageLoader
from simple_backtest.data.base import DataLoader
from simple_backtest.data.ccxt_loader import CCXTLoader
from simple_backtest.data.csv_loader import CSVLoader
from simple_backtest.data.polygon_loader import PolygonLoader
from simple_backtest.data.yfinance_loader import YFinanceLoader

__all__ = [
    "DataLoader",
    "YFinanceLoader",
    "CSVLoader",
    "CCXTLoader",
    "AlphaVantageLoader",
    "PolygonLoader",
]
