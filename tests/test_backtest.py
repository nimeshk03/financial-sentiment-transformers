"""
Tests for Milestone 3.5: Backtesting Framework.

Tests trading strategies and performance metrics.
"""
import pytest
import pandas as pd
import numpy as np


class TestStrategy:
    """Test trading strategy functions."""
    
    def test_generate_signals_long(self):
        """Test long signal generation."""
        from src.backtesting.strategy import generate_signals, StrategyConfig
        
        df = pd.DataFrame({
            'sentiment_mean': [0.5, 0.1, -0.3],
            'return_1d': [0.01, 0.02, -0.01],
        })
        
        config = StrategyConfig(long_threshold=0.2, short_threshold=-0.2)
        result = generate_signals(df, config)
        
        assert result.iloc[0]['signal'] == 1   # 0.5 > 0.2 -> long
        assert result.iloc[1]['signal'] == 0   # 0.1 between thresholds
        assert result.iloc[2]['signal'] == -1  # -0.3 < -0.2 -> short
    
    def test_calculate_strategy_returns(self):
        """Test strategy return calculation."""
        from src.backtesting.strategy import (
            generate_signals, 
            calculate_strategy_returns,
            StrategyConfig,
        )
        
        df = pd.DataFrame({
            'sentiment_mean': [0.5, -0.5],
            'return_1d': [0.02, -0.01],
        })
        
        config = StrategyConfig(long_threshold=0.2, short_threshold=-0.2)
        df = generate_signals(df, config)
        df = calculate_strategy_returns(df, config)
        
        # Long signal (1) * positive return (0.02) = 0.02
        assert abs(df.iloc[0]['strategy_return'] - 0.02) < 0.001
        
        # Short signal (-1) * negative return (-0.01) = 0.01 (profit)
        assert abs(df.iloc[1]['strategy_return'] - 0.01) < 0.001
    
    def test_long_only_strategy(self):
        """Test long-only strategy."""
        from src.backtesting.strategy import long_only_strategy
        
        df = pd.DataFrame({
            'sentiment_mean': [0.5, -0.5, 0.2],
            'return_1d': [0.02, -0.01, 0.01],
        })
        
        result = long_only_strategy(df, threshold=0.1)
        
        # Only positive sentiment gets signal
        assert result.iloc[0]['signal'] == 1
        assert result.iloc[1]['signal'] == 0
        assert result.iloc[2]['signal'] == 1
    
    def test_momentum_strategy(self):
        """Test momentum-based strategy."""
        from src.backtesting.strategy import momentum_strategy
        
        df = pd.DataFrame({
            'sentiment_momentum': [0.1, -0.1, 0.02],
            'return_1d': [0.02, -0.01, 0.01],
        })
        
        result = momentum_strategy(df, momentum_threshold=0.05)
        
        assert result.iloc[0]['signal'] == 1   # Positive momentum
        assert result.iloc[1]['signal'] == -1  # Negative momentum
        assert result.iloc[2]['signal'] == 0   # Below threshold


class TestMetrics:
    """Test performance metrics."""
    
    def test_calculate_metrics_basic(self):
        """Test basic metrics calculation."""
        from src.backtesting.metrics import calculate_metrics
        
        df = pd.DataFrame({
            'strategy_return': [0.01, 0.02, -0.01, 0.015],
            'return_1d': [0.01, 0.01, 0.01, 0.01],
            'signal': [1, 1, 1, 1],
        })
        
        results = calculate_metrics(df)
        
        assert results.num_trades == 4
        assert results.total_return > 0
        assert results.num_days == 4
    
    def test_win_rate_calculation(self):
        """Test win rate is calculated correctly."""
        from src.backtesting.metrics import calculate_metrics
        
        df = pd.DataFrame({
            'strategy_return': [0.01, 0.02, -0.01, -0.02],
            'return_1d': [0.01, 0.02, -0.01, -0.02],
            'signal': [1, 1, 1, 1],
        })
        
        results = calculate_metrics(df)
        
        # 2 wins out of 4 trades = 50%
        assert abs(results.win_rate - 0.5) < 0.01
    
    def test_empty_dataframe(self):
        """Test handling of empty data."""
        from src.backtesting.metrics import calculate_metrics
        
        df = pd.DataFrame({
            'strategy_return': [],
            'return_1d': [],
            'signal': [],
        })
        
        results = calculate_metrics(df)
        
        assert results.num_trades == 0
        assert results.total_return == 0
    
    def test_max_drawdown(self):
        """Test max drawdown calculation."""
        from src.backtesting.metrics import calculate_metrics
        
        # Simulate a drawdown scenario
        df = pd.DataFrame({
            'strategy_return': [0.10, -0.15, -0.05, 0.05],
            'return_1d': [0.10, -0.15, -0.05, 0.05],
            'signal': [1, 1, 1, 1],
        })
        
        results = calculate_metrics(df)
        
        # Should have negative max drawdown
        assert results.max_drawdown < 0


class TestIntegration:
    """Integration tests."""
    
    def test_full_backtest_flow(self):
        """Test complete backtest workflow."""
        from src.backtesting.strategy import (
            generate_signals,
            calculate_strategy_returns,
            StrategyConfig,
        )
        from src.backtesting.metrics import calculate_metrics
        
        # Simulate realistic data
        np.random.seed(42)
        n = 20
        
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=n),
            'ticker': ['AAPL'] * n,
            'sentiment_mean': np.random.uniform(-0.5, 0.5, n),
            'return_1d': np.random.normal(0.001, 0.02, n),
        })
        
        # Run strategy
        config = StrategyConfig()
        df = generate_signals(df, config)
        df = calculate_strategy_returns(df, config)
        
        # Calculate metrics
        results = calculate_metrics(df)
        
        # Verify we got results
        assert results.num_days == n
        assert isinstance(results.sharpe_ratio, float)
        assert isinstance(results.total_return, float)
