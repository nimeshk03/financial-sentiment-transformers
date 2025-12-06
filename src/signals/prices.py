"""
Price data fetching utilities.

Fetches historical prices and computes returns for backtesting.
"""
from typing import List, Optional

import pandas as pd
import yfinance as yf


def fetch_prices(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "1mo",
) -> pd.DataFrame:
    """Fetch historical prices for tickers.
    
    Uses Yahoo Finance to get OHLCV data. This is needed to:
    1. Align sentiment signals with actual trading days
    2. Compute returns for backtesting
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD). If None, uses period.
        end_date: End date (YYYY-MM-DD). If None, uses today.
        period: Period to fetch if dates not specified (1d, 5d, 1mo, 3mo, 1y)
    
    Returns:
        DataFrame with columns: [date, ticker, open, high, low, close, volume]
    """
    all_prices = []
    
    for ticker in tickers:
        try:
            print(f"  Fetching prices for {ticker}...")
            stock = yf.Ticker(ticker)
            
            if start_date and end_date:
                hist = stock.history(start=start_date, end=end_date)
            else:
                hist = stock.history(period=period)
            
            if hist.empty:
                continue
            
            hist = hist.reset_index()
            hist['ticker'] = ticker
            hist = hist.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
            })
            
            # Keep only needed columns
            cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
            hist = hist[[c for c in cols if c in hist.columns]]
            all_prices.append(hist)
            
        except Exception as e:
            print(f"  Error fetching prices for {ticker}: {e}")
    
    if not all_prices:
        return pd.DataFrame(columns=['date', 'ticker', 'open', 'high', 'low', 'close', 'volume'])
    
    df = pd.concat(all_prices, ignore_index=True)
    
    # Remove timezone info for easier merging
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    
    return df


def compute_returns(
    df: pd.DataFrame,
    periods: List[int] = [1, 5],
) -> pd.DataFrame:
    """Compute forward returns for backtesting.
    
    Forward returns tell us what happened AFTER we would have traded.
    - 1-day return: Next day's performance
    - 5-day return: Week-ahead performance
    
    These are used to evaluate if sentiment signals predict future returns.
    
    Args:
        df: Price DataFrame
        periods: List of forward periods (in days)
    
    Returns:
        DataFrame with return columns (e.g., return_1d, return_5d)
    """
    df = df.copy()
    df = df.sort_values(['ticker', 'date'])
    
    for period in periods:
        col_name = f'return_{period}d'
        # Shift negative to get FUTURE returns
        df[col_name] = df.groupby('ticker')['close'].transform(
            lambda x: x.shift(-period) / x - 1
        )
    
    return df


def merge_sentiment_prices(
    sentiment_df: pd.DataFrame,
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge sentiment and price data.
    
    Aligns sentiment signals with price data on trading days.
    Only keeps dates where we have BOTH sentiment and price data.
    
    Args:
        sentiment_df: Daily sentiment DataFrame
        price_df: Price DataFrame
    
    Returns:
        Merged DataFrame ready for backtesting
    """
    sentiment_df = sentiment_df.copy()
    price_df = price_df.copy()
    
    # Normalize dates
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.normalize()
    price_df['date'] = pd.to_datetime(price_df['date']).dt.normalize()
    
    # Merge on date and ticker
    merged = pd.merge(
        sentiment_df,
        price_df,
        on=['date', 'ticker'],
        how='inner',
    )
    
    return merged
