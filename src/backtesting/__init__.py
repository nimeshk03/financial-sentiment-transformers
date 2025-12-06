"""Backtesting module for strategy evaluation."""

from src.backtesting.strategy import (
    StrategyConfig,
    generate_signals,
    calculate_strategy_returns,
    long_only_strategy,
    momentum_strategy,
)
from src.backtesting.metrics import (
    BacktestResults,
    calculate_metrics,
    print_results,
)

__all__ = [
    "StrategyConfig",
    "generate_signals",
    "calculate_strategy_returns",
    "long_only_strategy",
    "momentum_strategy",
    "BacktestResults",
    "calculate_metrics",
    "print_results",
]
