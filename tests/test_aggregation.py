"""
Tests for Milestone 3.4: Sentiment Aggregation.

Tests the aggregation of news sentiment into daily signals
and the merging with price data for backtesting.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestAggregator:
    """Test aggregation functions."""
    
    def test_aggregate_daily_sentiment(self):
        """Test daily aggregation combines multiple articles per day."""
        from src.signals.aggregator import aggregate_daily_sentiment
        
        # Two articles on day 1, one on day 2
        df = pd.DataFrame({
            'published': [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 14, 0),
                datetime(2024, 1, 2, 9, 0),
            ],
            'ticker': ['AAPL', 'AAPL', 'AAPL'],
            'sentiment_score': [1.0, 0.0, -1.0],
            'sentiment_label': [2, 1, 0],
        })
        
        result = aggregate_daily_sentiment(df)
        
        assert len(result) == 2  # 2 unique days
        assert 'sentiment_mean' in result.columns
        assert 'article_count' in result.columns
        
        # Day 1 should have mean of (1.0 + 0.0) / 2 = 0.5
        day1 = result[result['date'] == pd.Timestamp('2024-01-01')]
        assert len(day1) == 1
        assert day1.iloc[0]['article_count'] == 2
    
    def test_aggregate_multiple_tickers(self):
        """Test aggregation handles multiple tickers."""
        from src.signals.aggregator import aggregate_daily_sentiment
        
        df = pd.DataFrame({
            'published': [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 11, 0),
            ],
            'ticker': ['AAPL', 'MSFT'],
            'sentiment_score': [1.0, -1.0],
            'sentiment_label': [2, 0],
        })
        
        result = aggregate_daily_sentiment(df)
        
        assert len(result) == 2  # 2 tickers
        assert set(result['ticker']) == {'AAPL', 'MSFT'}
    
    def test_add_rolling_sentiment(self):
        """Test rolling averages are computed correctly."""
        from src.signals.aggregator import add_rolling_sentiment
        
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'ticker': ['AAPL'] * 5,
            'sentiment_mean': [0.2, 0.4, 0.6, 0.8, 1.0],
        })
        
        result = add_rolling_sentiment(df, windows=[3])
        
        assert 'sentiment_rolling_3d' in result.columns
        
        # 3-day rolling mean at position 2 should be (0.2+0.4+0.6)/3 = 0.4
        assert abs(result.iloc[2]['sentiment_rolling_3d'] - 0.4) < 0.01
    
    def test_compute_sentiment_momentum(self):
        """Test momentum is short MA minus long MA."""
        from src.signals.aggregator import compute_sentiment_momentum
        
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'ticker': ['AAPL'] * 3,
            'sentiment_mean': [0.5, 0.5, 0.5],
            'sentiment_rolling_3d': [0.6, 0.6, 0.6],
            'sentiment_rolling_7d': [0.4, 0.4, 0.4],
        })
        
        result = compute_sentiment_momentum(df)
        
        assert 'sentiment_momentum' in result.columns
        # Momentum should be 0.6 - 0.4 = 0.2
        assert abs(result.iloc[0]['sentiment_momentum'] - 0.2) < 0.01


class TestPrices:
    """Test price fetching functions."""
    
    def test_fetch_prices_returns_dataframe(self):
        """Test price fetching returns proper DataFrame."""
        from src.signals.prices import fetch_prices
        
        df = fetch_prices(['AAPL'], period='5d')
        
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert 'date' in df.columns
            assert 'ticker' in df.columns
            assert 'close' in df.columns
    
    def test_compute_returns_forward(self):
        """Test forward returns are computed correctly."""
        from src.signals.prices import compute_returns
        
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'ticker': ['AAPL'] * 5,
            'close': [100, 102, 101, 105, 103],
        })
        
        result = compute_returns(df, periods=[1])
        
        assert 'return_1d' in result.columns
        
        # Return from day 0 to day 1: (102-100)/100 = 0.02
        assert abs(result.iloc[0]['return_1d'] - 0.02) < 0.001
        
        # Last day should have NaN (no future data)
        assert pd.isna(result.iloc[-1]['return_1d'])
    
    def test_merge_sentiment_prices(self):
        """Test merging aligns on date and ticker."""
        from src.signals.prices import merge_sentiment_prices
        
        sentiment_df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-02', '2024-01-03']),
            'ticker': ['AAPL', 'AAPL'],
            'sentiment_mean': [0.5, -0.3],
        })
        
        price_df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04']),
            'ticker': ['AAPL', 'AAPL', 'AAPL'],
            'close': [150, 152, 151],
        })
        
        result = merge_sentiment_prices(sentiment_df, price_df)
        
        # Should only have 2 rows (inner join)
        assert len(result) == 2
        assert 'sentiment_mean' in result.columns
        assert 'close' in result.columns


class TestIntegration:
    """Integration tests."""
    
    def test_full_aggregation_pipeline(self):
        """Test complete aggregation flow."""
        from src.signals.aggregator import (
            aggregate_daily_sentiment,
            add_rolling_sentiment,
            compute_sentiment_momentum,
        )
        
        # Simulate sentiment data
        df = pd.DataFrame({
            'published': pd.date_range('2024-01-01', periods=10, freq='D'),
            'ticker': ['AAPL'] * 10,
            'sentiment_score': [0.5, -0.3, 0.8, 0.2, -0.5, 0.1, 0.6, -0.2, 0.4, 0.3],
            'sentiment_label': [2, 0, 2, 1, 0, 1, 2, 0, 2, 1],
        })
        
        # Run pipeline
        daily = aggregate_daily_sentiment(df)
        daily = add_rolling_sentiment(daily, windows=[3, 7])
        daily = compute_sentiment_momentum(daily)
        
        # Verify output
        assert len(daily) == 10
        assert 'sentiment_mean' in daily.columns
        assert 'sentiment_rolling_3d' in daily.columns
        assert 'sentiment_rolling_7d' in daily.columns
        assert 'sentiment_momentum' in daily.columns
