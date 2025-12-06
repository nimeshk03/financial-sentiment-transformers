"""
News fetching utilities.

Fetches financial news from multiple sources:
- yfinance (Yahoo Finance news)
"""
from typing import List, Optional
from datetime import datetime
from dataclasses import dataclass
import time

import pandas as pd
import yfinance as yf

from src.news import STOCK_TICKERS


@dataclass
class NewsArticle:
    """Represents a news article."""
    title: str
    source: str
    published: datetime
    ticker: str
    url: Optional[str] = None


def fetch_yahoo_news(ticker: str, max_articles: int = 50) -> List[NewsArticle]:
    """Fetch news for a ticker from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        max_articles: Maximum number of articles to fetch
    
    Returns:
        List of NewsArticle objects
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if not news:
            return []
        
        articles = []
        for item in news[:max_articles]:
            # Handle new yfinance API structure (nested under 'content')
            content = item.get('content', item)
            
            # Parse publish time
            pub_date = content.get('pubDate', '')
            if pub_date:
                # Parse ISO format: '2025-12-06T11:00:17Z'
                try:
                    pub_time = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                except ValueError:
                    pub_time = datetime.now()
            else:
                # Fallback to old format
                pub_time = datetime.fromtimestamp(item.get('providerPublishTime', 0))
            
            # Get title from content or directly
            title = content.get('title', item.get('title', ''))
            
            # Get provider/source
            provider = content.get('provider', {})
            source = provider.get('displayName', item.get('publisher', 'Yahoo Finance'))
            
            # Get URL
            canonical = content.get('canonicalUrl', {})
            url = canonical.get('url', item.get('link', ''))
            
            if title:  # Only add if we have a title
                article = NewsArticle(
                    title=title,
                    source=source,
                    published=pub_time,
                    ticker=ticker,
                    url=url,
                )
                articles.append(article)
        
        return articles
    
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []


def fetch_news_for_tickers(
    tickers: Optional[List[str]] = None,
    max_per_ticker: int = 20,
    delay: float = 0.5,
) -> pd.DataFrame:
    """Fetch news for multiple tickers.
    
    Args:
        tickers: List of ticker symbols. If None, uses default STOCK_TICKERS.
        max_per_ticker: Max articles per ticker
        delay: Delay between requests (seconds)
    
    Returns:
        DataFrame with columns: [ticker, title, source, published, url]
    """
    if tickers is None:
        tickers = STOCK_TICKERS
    
    all_articles = []
    
    for ticker in tickers:
        print(f"Fetching news for {ticker}...")
        articles = fetch_yahoo_news(ticker, max_per_ticker)
        all_articles.extend(articles)
        
        if delay > 0 and ticker != tickers[-1]:
            time.sleep(delay)
    
    if not all_articles:
        return pd.DataFrame(columns=['ticker', 'title', 'source', 'published', 'url'])
    
    df = pd.DataFrame([
        {
            'ticker': a.ticker,
            'title': a.title,
            'source': a.source,
            'published': a.published,
            'url': a.url,
        }
        for a in all_articles
    ])
    
    # Sort by date
    df = df.sort_values('published', ascending=False)
    
    return df
