"""
Performance metrics for backtesting.

Calculates standard trading performance metrics.
"""
from typing import Dict, Any
from dataclasses import dataclass

import pandas as pd
import numpy as np


@dataclass
class BacktestResults:
    """Container for backtest results."""
    
    # Returns
    total_return: float
    market_return: float
    excess_return: float
    
    # Risk metrics
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Trade statistics
    num_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Additional
    num_days: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'market_return': self.market_return,
            'excess_return': self.excess_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'num_trades': self.num_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'num_days': self.num_days,
        }


def calculate_metrics(
    df: pd.DataFrame,
    strategy_return_col: str = 'strategy_return',
    market_return_col: str = 'return_1d',
    signal_col: str = 'signal',
    risk_free_rate: float = 0.0,
) -> BacktestResults:
    """Calculate comprehensive backtest metrics.
    
    Args:
        df: DataFrame with strategy returns
        strategy_return_col: Column with strategy returns
        market_return_col: Column with market returns
        signal_col: Column with trading signals
        risk_free_rate: Annual risk-free rate (default 0)
    
    Returns:
        BacktestResults object with all metrics
    """
    # Filter valid data
    valid = df.dropna(subset=[strategy_return_col, market_return_col])
    
    if len(valid) == 0:
        return BacktestResults(
            total_return=0, market_return=0, excess_return=0,
            volatility=0, sharpe_ratio=0, max_drawdown=0,
            num_trades=0, win_rate=0, avg_win=0, avg_loss=0,
            profit_factor=0, num_days=0,
        )
    
    strategy_returns = valid[strategy_return_col]
    market_returns = valid[market_return_col]
    signals = valid[signal_col]
    
    # Total returns
    total_return = (1 + strategy_returns).prod() - 1
    market_return = (1 + market_returns).prod() - 1
    excess_return = total_return - market_return
    
    # Volatility (annualized, assuming daily returns)
    volatility = strategy_returns.std() * np.sqrt(252)
    
    # Sharpe ratio (annualized)
    if volatility > 0:
        mean_return = strategy_returns.mean() * 252
        sharpe_ratio = (mean_return - risk_free_rate) / volatility
    else:
        sharpe_ratio = 0
    
    # Maximum drawdown
    cumulative = (1 + strategy_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Trade statistics
    trades = valid[signals != 0]
    num_trades = len(trades)
    
    if num_trades > 0:
        trade_returns = trades[strategy_return_col]
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]
        
        win_rate = len(wins) / num_trades
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        # Profit factor = gross profits / gross losses
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
    
    return BacktestResults(
        total_return=total_return,
        market_return=market_return,
        excess_return=excess_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        num_trades=num_trades,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        num_days=len(valid),
    )


def print_results(results: BacktestResults, strategy_name: str = "Strategy") -> None:
    """Print formatted backtest results.
    
    Args:
        results: BacktestResults object
        strategy_name: Name of the strategy
    """
    print(f"\n{'='*50}")
    print(f"{strategy_name} Results")
    print(f"{'='*50}")
    
    print(f"\nReturns:")
    print(f"  Total Return:    {results.total_return:+.2%}")
    print(f"  Market Return:   {results.market_return:+.2%}")
    print(f"  Excess Return:   {results.excess_return:+.2%}")
    
    print(f"\nRisk Metrics:")
    print(f"  Volatility:      {results.volatility:.2%}")
    print(f"  Sharpe Ratio:    {results.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:    {results.max_drawdown:.2%}")
    
    print(f"\nTrade Statistics:")
    print(f"  Number of Trades: {results.num_trades}")
    print(f"  Win Rate:         {results.win_rate:.1%}")
    print(f"  Avg Win:          {results.avg_win:+.2%}")
    print(f"  Avg Loss:         {results.avg_loss:+.2%}")
    print(f"  Profit Factor:    {results.profit_factor:.2f}")
    
    print(f"\n  Days Tested:      {results.num_days}")
