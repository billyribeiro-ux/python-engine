"""Backtesting engines - vectorised and event-driven. Built in Module 9."""

from .cv import PurgedKFold, WalkForward, deflated_sharpe
from .vectorized import BacktestResult, vectorized_backtest

__all__ = [
    "BacktestResult",
    "PurgedKFold",
    "WalkForward",
    "deflated_sharpe",
    "vectorized_backtest",
]
