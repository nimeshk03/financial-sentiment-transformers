"""
News processing and cleaning utilities.
"""
import re

import pandas as pd


def clean_headline(text: str) -> str:
    """Clean a news headline for sentiment analysis.
    
    Args:
        text: Raw headline text
    
    Returns:
        Cleaned headline
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def deduplicate_headlines(df: pd.DataFrame, title_col: str = 'title') -> pd.DataFrame:
    """Remove duplicate headlines.
    
    Args:
        df: DataFrame with headlines
        title_col: Column name containing headlines
    
    Returns:
        DataFrame with duplicates removed
    """
    # Normalize for comparison
    df = df.copy()
    df['_normalized'] = df[title_col].str.lower().str.strip()
    
    # Keep first occurrence
    df = df.drop_duplicates(subset=['_normalized'], keep='first')
    
    # Remove temp column
    df = df.drop(columns=['_normalized'])
    
    return df


def filter_financial_headlines(
    df: pd.DataFrame,
    title_col: str = 'title',
    min_length: int = 20,
) -> pd.DataFrame:
    """Filter to keep only relevant financial headlines.
    
    Args:
        df: DataFrame with headlines
        title_col: Column name containing headlines
        min_length: Minimum headline length
    
    Returns:
        Filtered DataFrame
    """
    df = df.copy()
    
    # Remove very short headlines
    df = df[df[title_col].str.len() >= min_length]
    
    # Remove headlines that are mostly non-ASCII (likely garbage)
    # Allow any printable characters - be permissive
    df = df[df[title_col].str.contains(r'[A-Za-z]{3,}', na=False)]
    
    return df


def process_news_dataframe(
    df: pd.DataFrame,
    title_col: str = 'title',
) -> pd.DataFrame:
    """Full processing pipeline for news DataFrame.
    
    Args:
        df: Raw news DataFrame
        title_col: Column name containing headlines
    
    Returns:
        Processed DataFrame
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Clean headlines
    df[title_col] = df[title_col].apply(clean_headline)
    
    # Remove empty headlines
    df = df[df[title_col].str.len() > 0]
    
    # Deduplicate
    df = deduplicate_headlines(df, title_col)
    
    # Filter
    df = filter_financial_headlines(df, title_col)
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df
