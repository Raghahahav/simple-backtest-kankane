from simple_backtest import (
    Backtest,
    BacktestConfig,
    MovingAverageStrategy,
    YFinanceLoader,
)

# ---- user inputs ----
symbol = "AAPL"
start_year = 2018
end_year = 2023

start = f"{start_year}-01-01"
end = f"{end_year}-12-31"

# 1) Load data from yfinance
loader = YFinanceLoader()
data = loader.load(symbol=symbol, start=start, end=end)

# 2) Define strategy
strategy = MovingAverageStrategy(short_window=20, long_window=50, shares=10)

# 3) Configure backtest
config = BacktestConfig.default(initial_capital=10_000)

# 4) Run
backtest = Backtest(data, config)
results = backtest.run([strategy])

# 5) Print summary
result = results.get_strategy(strategy.get_name())
print(f"Symbol: {symbol}")
print(f"Range: {start} to {end}")
print(result.summary())
