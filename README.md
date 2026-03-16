<div align="center">

# Simple Backtest

I have removed my githistory because i want to treat this as my base with imporvement in data loader and custom error message
**A high-performance, asset-agnostic backtesting framework for Python**

[![PyPI version](https://img.shields.io/pypi/v/simple-backtest-kankane)](https://pypi.org/project/simple-backtest-kankane/)
[![Python](https://img.shields.io/pypi/pyversions/simple-backtest-kankane)](https://pypi.org/project/simple-backtest-kankane/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-268%20passed-success.svg)](#)

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Data Loaders](#-data-loaders) • [Documentation](#-documentation)

</div>

---

## 📖 About

Simple Backtest provides a clean framework for running strategy backtests with strong validation, robust metrics, and extensible architecture.

You can still bring your own pandas DataFrame, but the project now also includes **optional data source integrations** so users can load and normalize OHLCV data faster.

## ✨ Features

- **Backtesting Engine**: Fast, deterministic strategy execution
- **Validation First**: Actionable errors for data, config, and strategy inputs
- **Asset-Agnostic Design**: Works with stocks, forex, crypto, ETFs, and more
- **20+ Metrics**: Return, drawdown, Sharpe, Sortino, Calmar, alpha/beta, etc.
- **Optimization**: Grid search, random search, walk-forward
- **Optional Data Integrations**:
  - `CSVLoader`
  - `YFinanceLoader`
  - `CCXTLoader`
  - `AlphaVantageLoader`
  - `PolygonLoader`

## 📦 Installation

### Core package

```bash
pip install simple-backtest
```

### Optional loader dependencies

Install only what you use:

```bash
# Yahoo Finance
pip install yfinance

# Crypto exchange data
pip install ccxt

# REST API loaders (Alpha Vantage, Polygon)
pip install requests
```

### Development setup

```bash
git clone <your-repository-url>
cd simple-backtest
pip install -e ".[dev]"
```

**Requirements**: Python 3.10+

## 🚀 Quick Start

### Backtest from any OHLCV DataFrame

```python
from simple_backtest import Backtest, BacktestConfig, MovingAverageStrategy

# data must contain: Open, High, Low, Close (Volume optional unless configured)
strategy = MovingAverageStrategy(short_window=10, long_window=30, shares=10)
config = BacktestConfig.default(initial_capital=10000)

backtest = Backtest(data, config)
results = backtest.run([strategy])

print(results.get_strategy(strategy.get_name()).summary())
```

### Backtest using built-in CSV loader

```python
from simple_backtest import Backtest, BacktestConfig, CSVLoader, MovingAverageStrategy

loader = CSVLoader()
data = loader.load("data/aapl.csv", start="2020-01-01", end="2023-12-31")

backtest = Backtest(data, BacktestConfig.default(initial_capital=10000))
results = backtest.run([MovingAverageStrategy(short_window=10, long_window=30, shares=10)])
```

## 🔌 Data Loaders

All loaders inherit from `DataLoader` and return a validated DataFrame with standardized columns:

`Open`, `High`, `Low`, `Close`, `Volume`

Validation is always run internally before the DataFrame is returned.

### `CSVLoader`

- Reads local CSV files
- Auto-detects date column (`Date`, `date`, `Datetime`, `datetime`, or datetime index)
- Normalizes common column variants (`open` → `Open`, etc.)
- Supports optional date filtering via `start` and `end`

```python
from simple_backtest import CSVLoader

data = CSVLoader().load("prices.csv", start="2021-01-01", end="2021-12-31")
```

### `YFinanceLoader`

- Uses `yfinance.download(...)`
- Handles yfinance MultiIndex column outputs
- Raises clear import/data errors

```python
from simple_backtest import YFinanceLoader

data = YFinanceLoader().load("AAPL", "2020-01-01", "2023-12-31")
```

### `CCXTLoader`

- Uses `ccxt` exchange clients
- Supports constructor args: `exchange_name`, optional `api_key`, `api_secret`
- Converts millisecond timestamps to `DatetimeIndex`
- Auto-paginates OHLCV fetches for larger ranges

```python
from simple_backtest import CCXTLoader

loader = CCXTLoader(exchange_name="binance")
data = loader.load("BTC/USDT", "2021-01-01", "2021-12-31", timeframe="1d")
```

### `AlphaVantageLoader`

- Uses Alpha Vantage daily REST endpoint
- Constructor requires `api_key`
- Parses API JSON into standardized OHLCV DataFrame
- Applies `start` / `end` filtering post-load

```python
from simple_backtest import AlphaVantageLoader

loader = AlphaVantageLoader(api_key="YOUR_KEY")
data = loader.load("AAPL", "2020-01-01", "2023-12-31")
```

### `PolygonLoader`

- Uses Polygon aggregates REST endpoint
- Constructor requires `api_key`
- Handles `next_url` pagination
- Parses `o/h/l/c/v/t` fields into standardized OHLCV DataFrame

```python
from simple_backtest import PolygonLoader

loader = PolygonLoader(api_key="YOUR_KEY")
data = loader.load("AAPL", "2020-01-01", "2023-12-31", timespan="day", multiplier=1)
```

### Create your own loader

```python
import pandas as pd
from simple_backtest import DataLoader


class MyCustomLoader(DataLoader):
    def load(self, symbol, start, end) -> pd.DataFrame:
        # fetch/construct your data
        data = pd.DataFrame(...)
        return self._finalize_dataframe(data)
```

## 📚 Documentation

### Built-in strategy helpers

When writing a custom strategy (subclass of `Strategy`), you can use:

- `self.has_position()`
- `self.get_position()`
- `self.get_cash()`
- `self.get_portfolio_value()`
- `self.buy(shares)`
- `self.sell(shares)`
- `self.sell_all()`
- `self.hold()`
- `self.buy_percent(percent)`
- `self.buy_cash(amount)`

### Config presets

```python
from simple_backtest import BacktestConfig

config = BacktestConfig.default(initial_capital=10000)
config_zero_fees = BacktestConfig.zero_commission(initial_capital=10000)
config_hft = BacktestConfig.high_frequency(initial_capital=100000)
config_swing = BacktestConfig.swing_trading(initial_capital=10000)
```

### Optimizers

- `GridSearchOptimizer`
- `RandomSearchOptimizer`
- `WalkForwardOptimizer`

## 🧩 API Reference (What to Import)

This section explains:

- what can be imported directly from `simple_backtest`
- what should be imported from submodules
- what each import is typically used for

### ✅ Import from top-level package

These are re-exported in `simple_backtest/__init__.py` and are stable entry points for most users.

```python
from simple_backtest import (
  # Core
  Backtest,
  BacktestConfig,
  Portfolio,
  Strategy,
  BacktestResults,
  StrategyResult,

  # Built-in strategies
  BuyAndHoldStrategy,
  DCAStrategy,
  MovingAverageStrategy,

  # Optimizers
  Optimizer,
  GridSearchOptimizer,
  RandomSearchOptimizer,
  WalkForwardOptimizer,

  # Commission models
  Commission,
  PercentageCommission,
  FlatCommission,
  TieredCommission,

  # Data loaders
  DataLoader,
  CSVLoader,
  YFinanceLoader,
  CCXTLoader,
  AlphaVantageLoader,
  PolygonLoader,
)
```

### ✅ Import from `simple_backtest.utils`

Use these for execution helpers, validation, logging, and custom commission wiring.

```python
from simple_backtest.utils import (
  # Execution price helpers
  get_execution_price,
  create_execution_price_extractor,

  # Data/strategy validation
  validate_dataframe,
  validate_date_range,
  validate_strategies,

  # Validation exceptions
  BacktestError,
  DataValidationError,
  DateRangeError,
  StrategyError,

  # Commission helper factory
  get_commission_calculator,
  create_custom_commission,

  # Logging
  get_logger,
  setup_logging,
  disable_logging,
  enable_debug_logging,
)
```

### ✅ Import from `simple_backtest.metrics`

```python
from simple_backtest.metrics import calculate_metrics, format_metrics
```

- `calculate_metrics`: compute full metric dictionary from returns/trades/portfolio data
- `format_metrics`: convert metric dictionary into readable report text

### ✅ Import from `simple_backtest.visualization`

```python
from simple_backtest.visualization import (
  plot_equity_curve,
  plot_drawdowns,
  plot_returns_distribution,
  plot_monthly_returns,
  plot_trades,
  plot_strategy_trades,
  plot_rolling_metrics,
  create_comparison_table,
  plot_all,
)
```

### ⚠️ What is importable but not recommended as public API

You can technically import internals like:

```python
from simple_backtest.metrics.definitions import calculate_sharpe_ratio
from simple_backtest.utils.execution import get_vwap
```

But these are lower-level internals and may change more often.
Prefer top-level package imports and subpackage `__init__` exports shown above.

### Quick guide: “which import does what?”

- `Backtest`: runs strategies on price data
- `BacktestConfig`: execution settings (capital, commission, execution price, etc.)
- `Strategy`: base class for custom strategy logic
- `Portfolio`: tracks cash, positions, and trades
- `...Loader` classes: fetch/normalize OHLCV data from source
- `...Optimizer` classes: parameter search and evaluation
- `Commission` classes: trading-cost models
- `utils` functions: validation, execution-price extraction, logging helpers
- `metrics` functions: calculate/format performance metrics
- `visualization` functions: charts and comparison views

### 📌 One-page Import Cheat Sheet

| You want to...                 | Import from                     | Import this                                                                                                                                                                                      |
| ------------------------------ | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Run a backtest                 | `simple_backtest`               | `Backtest`, `BacktestConfig`                                                                                                                                                                     |
| Build custom strategy          | `simple_backtest`               | `Strategy`                                                                                                                                                                                       |
| Use built-in strategies        | `simple_backtest`               | `MovingAverageStrategy`, `BuyAndHoldStrategy`, `DCAStrategy`                                                                                                                                     |
| Load CSV data                  | `simple_backtest`               | `CSVLoader`                                                                                                                                                                                      |
| Load Yahoo Finance data        | `simple_backtest`               | `YFinanceLoader`                                                                                                                                                                                 |
| Load crypto exchange data      | `simple_backtest`               | `CCXTLoader`                                                                                                                                                                                     |
| Load Alpha Vantage data        | `simple_backtest`               | `AlphaVantageLoader`                                                                                                                                                                             |
| Load Polygon data              | `simple_backtest`               | `PolygonLoader`                                                                                                                                                                                  |
| Create your own loader base    | `simple_backtest`               | `DataLoader`                                                                                                                                                                                     |
| Use commission models          | `simple_backtest`               | `Commission`, `PercentageCommission`, `FlatCommission`, `TieredCommission`                                                                                                                       |
| Optimize parameters            | `simple_backtest`               | `GridSearchOptimizer`, `RandomSearchOptimizer`, `WalkForwardOptimizer`                                                                                                                           |
| Validate input data/strategies | `simple_backtest.utils`         | `validate_dataframe`, `validate_date_range`, `validate_strategies`                                                                                                                               |
| Build execution price logic    | `simple_backtest.utils`         | `get_execution_price`, `create_execution_price_extractor`                                                                                                                                        |
| Configure logging              | `simple_backtest.utils`         | `setup_logging`, `enable_debug_logging`, `disable_logging`                                                                                                                                       |
| Compute metrics                | `simple_backtest.metrics`       | `calculate_metrics`, `format_metrics`                                                                                                                                                            |
| Plot charts                    | `simple_backtest.visualization` | `plot_equity_curve`, `plot_drawdowns`, `plot_returns_distribution`, `plot_monthly_returns`, `plot_trades`, `plot_strategy_trades`, `plot_rolling_metrics`, `create_comparison_table`, `plot_all` |

**Rule of thumb**: Prefer `simple_backtest` and package `__init__` exports first. Use deep/internal module imports only when you intentionally need low-level internals.

## 📓 Notebooks

Jupyter examples are available in the [notebooks](notebooks) folder:

- `01_basic_usage.ipynb`
- `02_candle_strategies.ipynb`
- `03_ta_strategies.ipynb`
- `04_ml_strategies.ipynb`
- `05_commission_usage.ipynb`
- `06_advanced_optimization.ipynb`

## 🛠️ Development

### Run tests

```bash
pytest
```

### Run linting

```bash
ruff check simple_backtest tests
```

### Format

```bash
ruff format simple_backtest tests
```

## 🤝 Contributing

Contributions are welcome.

1. Fork repository
2. Create branch
3. Add tests for changes
4. Run `pytest` and `ruff check`
5. Open pull request

## 📄 License

MIT. See [LICENSE](LICENSE).

## 📬 Support

- Issues: Use your repository issue tracker
- Discussions: Use your repository discussions page
