"""
Trading strategy definitions for backtesting.

Defines simple sentiment-based trading rules.
"""
from typing import Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np


@dataclass
class StrategyConfig:
    """Configuration for a sentiment trading strategy."""
    
    # Signal thresholds
    long_threshold: float = 0.2      # Go long when sentiment > this
    short_threshold: float = -0.2    # Go short when sentiment < this
    
    # Position sizing
    position_size: float = 1.0       # Fraction of capital per trade
    
    # Which sentiment column to use
    signal_column: str = 'sentiment_mean'
    
    # Return column for evaluation
    return_column: str = 'return_1d'


def generate_signals(
    df: pd.DataFrame,
    config: Optional[StrategyConfig] = None,
) -> pd.DataFrame:
    """Generate trading signals from sentiment data.
    
    Signal logic:
    - sentiment > long_threshold  -> signal = 1 (long)
    - sentiment < short_threshold -> signal = -1 (short)
    - otherwise                   -> signal = 0 (no position)
    
    Args:
        df: DataFrame with sentiment and price data
        config: Strategy configuration
    
    Returns:
        DataFrame with added 'signal' column
    """
    if config is None:
        config = StrategyConfig()
    
    df = df.copy()
    
    sentiment = df[config.signal_column]
    
    # Generate signals
    df['signal'] = 0
    df.loc[sentiment > config.long_threshold, 'signal'] = 1
    df.loc[sentiment < config.short_threshold, 'signal'] = -1
    
    return df


def calculate_strategy_returns(
    df: pd.DataFrame,
    config: Optional[StrategyConfig] = None,
) -> pd.DataFrame:
    """Calculate returns from trading signals.
    
    Strategy return = signal * forward_return
    - If signal=1 (long) and stock goes up 2%, we make 2%
    - If signal=-1 (short) and stock goes down 2%, we make 2%
    - If signal=0, we make 0%
    
    Args:
        df: DataFrame with signals and returns
        config: Strategy configuration
    
    Returns:
        DataFrame with strategy returns
    """
    if config is None:
        config = StrategyConfig()
    
    df = df.copy()
    
    # Strategy return = signal * market return
    df['strategy_return'] = df['signal'] * df[config.return_column]
    
    # Cumulative returns
    df['cumulative_market'] = (1 + df[config.return_column]).cumprod() - 1
    df['cumulative_strategy'] = (1 + df['strategy_return']).cumprod() - 1
    
    return df


def long_only_strategy(
    df: pd.DataFrame,
    threshold: float = 0.1,
    signal_column: str = 'sentiment_mean',
    return_column: str = 'return_1d',
) -> pd.DataFrame:
    """Simple long-only strategy.
    
    Only takes long positions when sentiment is positive.
    Safer than long/short as it doesn't require shorting.
    
    Args:
        df: DataFrame with sentiment and returns
        threshold: Minimum sentiment to go long
        signal_column: Column with sentiment scores
        return_column: Column with forward returns
    
    Returns:
        DataFrame with strategy returns
    """
    df = df.copy()
    
    # Long when sentiment > threshold, else flat
    df['signal'] = (df[signal_column] > threshold).astype(int)
    
    # Strategy return
    df['strategy_return'] = df['signal'] * df[return_column]
    
    # Cumulative
    df['cumulative_market'] = (1 + df[return_column]).cumprod() - 1
    df['cumulative_strategy'] = (1 + df['strategy_return']).cumprod() - 1
    
    return df


def momentum_strategy(
    df: pd.DataFrame,
    momentum_threshold: float = 0.05,
    return_column: str = 'return_1d',
) -> pd.DataFrame:
    """Strategy based on sentiment momentum.
    
    Goes long when sentiment is improving (momentum > 0).
    Goes short when sentiment is deteriorating (momentum < 0).
    
    Args:
        df: DataFrame with sentiment_momentum column
        momentum_threshold: Minimum momentum to trade
        return_column: Column with forward returns
    
    Returns:
        DataFrame with strategy returns
    """
    df = df.copy()
    
    if 'sentiment_momentum' not in df.columns:
        raise ValueError("DataFrame must have 'sentiment_momentum' column")
    
    # Signal based on momentum
    df['signal'] = 0
    df.loc[df['sentiment_momentum'] > momentum_threshold, 'signal'] = 1
    df.loc[df['sentiment_momentum'] < -momentum_threshold, 'signal'] = -1
    
    # Strategy return
    df['strategy_return'] = df['signal'] * df[return_column]
    
    # Cumulative
    df['cumulative_market'] = (1 + df[return_column]).cumprod() - 1
    df['cumulative_strategy'] = (1 + df['strategy_return']).cumprod() - 1
    
    return df
