"""
Sentiment aggregation utilities.

Aggregates news sentiment into daily scores and rolling averages.
"""
from typing import List

import pandas as pd


def aggregate_daily_sentiment(
    df: pd.DataFrame,
    date_col: str = 'published',
    ticker_col: str = 'ticker',
    score_col: str = 'sentiment_score',
) -> pd.DataFrame:
    """Aggregate sentiment scores to daily level per ticker.
    
    Takes individual article sentiments and computes daily averages.
    This is essential because we may have multiple articles per day
    and need a single daily signal for trading decisions.
    
    Args:
        df: DataFrame with sentiment predictions
        date_col: Column containing datetime
        ticker_col: Column containing ticker symbol
        score_col: Column containing sentiment score (-1, 0, 1)
    
    Returns:
        DataFrame with columns: [date, ticker, sentiment_mean, sentiment_sum, 
                                 article_count, positive_pct, negative_pct]
    """
    df = df.copy()
    
    # Extract date from datetime
    df['date'] = pd.to_datetime(df[date_col]).dt.date
    
    # Group by date and ticker, compute aggregates
    agg = df.groupby(['date', ticker_col]).agg({
        score_col: ['mean', 'sum', 'count'],
    }).reset_index()
    
    # Flatten column names
    agg.columns = ['date', 'ticker', 'sentiment_mean', 'sentiment_sum', 'article_count']
    
    # Calculate positive/negative percentages
    if 'sentiment_label' in df.columns:
        pos_counts = df.groupby(['date', ticker_col])['sentiment_label'].apply(
            lambda x: (x == 2).sum()
        ).reset_index(name='positive_count')
        
        neg_counts = df.groupby(['date', ticker_col])['sentiment_label'].apply(
            lambda x: (x == 0).sum()
        ).reset_index(name='negative_count')
    else:
        agg['positive_count'] = 0
        agg['negative_count'] = 0
        pos_counts = None
        neg_counts = None
    
    if pos_counts is not None:
        agg = agg.merge(pos_counts, on=['date', 'ticker'], how='left')
    if neg_counts is not None:
        agg = agg.merge(neg_counts, on=['date', 'ticker'], how='left')
    
    agg['positive_pct'] = agg['positive_count'] / agg['article_count']
    agg['negative_pct'] = agg['negative_count'] / agg['article_count']
    
    # Sort by date
    agg['date'] = pd.to_datetime(agg['date'])
    agg = agg.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    return agg


def add_rolling_sentiment(
    df: pd.DataFrame,
    windows: List[int] = [3, 7],
    score_col: str = 'sentiment_mean',
) -> pd.DataFrame:
    """Add rolling average sentiment scores.
    
    Rolling averages smooth out daily noise and capture trends.
    - 3-day: Short-term sentiment momentum
    - 7-day: Medium-term sentiment trend
    
    Args:
        df: DataFrame with daily sentiment (from aggregate_daily_sentiment)
        windows: List of rolling window sizes in days
        score_col: Column to compute rolling average on
    
    Returns:
        DataFrame with added rolling columns (e.g., sentiment_rolling_3d)
    """
    df = df.copy()
    df = df.sort_values(['ticker', 'date'])
    
    for window in windows:
        col_name = f'sentiment_rolling_{window}d'
        df[col_name] = df.groupby('ticker')[score_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    return df


def compute_sentiment_momentum(
    df: pd.DataFrame,
    short_window: int = 3,
    long_window: int = 7,
) -> pd.DataFrame:
    """Compute sentiment momentum (short MA - long MA).
    
    Momentum captures whether sentiment is improving or deteriorating.
    - Positive momentum: Recent sentiment better than average (bullish)
    - Negative momentum: Recent sentiment worse than average (bearish)
    
    This is similar to MACD but for sentiment instead of price.
    
    Args:
        df: DataFrame with rolling sentiment columns
        short_window: Short-term window (default 3 days)
        long_window: Long-term window (default 7 days)
    
    Returns:
        DataFrame with sentiment_momentum column
    """
    df = df.copy()
    
    short_col = f'sentiment_rolling_{short_window}d'
    long_col = f'sentiment_rolling_{long_window}d'
    
    if short_col in df.columns and long_col in df.columns:
        df['sentiment_momentum'] = df[short_col] - df[long_col]
    
    return df
