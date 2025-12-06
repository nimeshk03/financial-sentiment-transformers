"""
Tests for Milestone 3.3: News Ingestion Pipeline.

Verifies:
- News fetching works
- Headline processing works
- Sentiment prediction works
"""
import pytest
import pandas as pd
from pathlib import Path


class TestNewsFetcher:
    """Test news fetching functions."""
    
    def test_stock_tickers_defined(self):
        """Test that default tickers are defined."""
        from src.news import STOCK_TICKERS
        
        assert isinstance(STOCK_TICKERS, list)
        assert len(STOCK_TICKERS) == 10
        assert 'AAPL' in STOCK_TICKERS
        assert 'NVDA' in STOCK_TICKERS
    
    def test_fetch_yahoo_news(self):
        """Test fetching news from Yahoo Finance."""
        from src.news.fetcher import fetch_yahoo_news
        
        # Fetch news for a major ticker
        articles = fetch_yahoo_news("AAPL", max_articles=5)
        
        # Should return a list
        assert isinstance(articles, list)
        
        # If articles found, check structure
        if len(articles) > 0:
            article = articles[0]
            assert hasattr(article, 'title')
            assert hasattr(article, 'ticker')
            assert hasattr(article, 'published')
            assert article.ticker == "AAPL"
    
    def test_fetch_news_for_tickers(self):
        """Test fetching news for multiple tickers."""
        from src.news.fetcher import fetch_news_for_tickers
        
        df = fetch_news_for_tickers(["AAPL", "MSFT"], max_per_ticker=3, delay=0.1)
        
        assert isinstance(df, pd.DataFrame)
        assert 'ticker' in df.columns
        assert 'title' in df.columns
        assert 'published' in df.columns
    
    def test_fetch_news_default_tickers(self):
        """Test fetching with default tickers."""
        from src.news.fetcher import fetch_news_for_tickers
        from src.news import STOCK_TICKERS
        
        # Just verify it doesn't crash with defaults
        df = fetch_news_for_tickers(tickers=STOCK_TICKERS[:2], max_per_ticker=2, delay=0.1)
        assert isinstance(df, pd.DataFrame)


class TestNewsProcessor:
    """Test news processing functions."""
    
    def test_clean_headline(self):
        """Test headline cleaning."""
        from src.news.processor import clean_headline
        
        # Test URL removal
        assert "Check this" == clean_headline("Check this https://example.com")
        
        # Test HTML removal
        assert "Hello world" == clean_headline("<b>Hello</b> world")
        
        # Test whitespace normalization
        assert "Hello world" == clean_headline("Hello    world")
        
        # Test empty input
        assert "" == clean_headline("")
        assert "" == clean_headline(None)
    
    def test_deduplicate_headlines(self):
        """Test deduplication."""
        from src.news.processor import deduplicate_headlines
        
        df = pd.DataFrame({
            'title': ['Hello World', 'hello world', 'Different'],
            'other': [1, 2, 3],
        })
        
        result = deduplicate_headlines(df)
        
        # Should remove the duplicate (case-insensitive)
        assert len(result) == 2
    
    def test_filter_financial_headlines(self):
        """Test headline filtering."""
        from src.news.processor import filter_financial_headlines
        
        df = pd.DataFrame({
            'title': [
                'Apple reports record earnings for Q4',  # Valid
                'Short',  # Too short
                'Valid headline with numbers 123',  # Valid
            ],
        })
        
        result = filter_financial_headlines(df, min_length=10)
        assert len(result) == 2
    
    def test_process_news_dataframe(self):
        """Test full processing pipeline."""
        from src.news.processor import process_news_dataframe
        
        df = pd.DataFrame({
            'title': [
                'Apple reports record earnings for Q4 2024',
                'apple reports record earnings for q4 2024',  # Duplicate
                'Short',  # Too short
                'Microsoft announces new AI features today',
            ],
            'ticker': ['AAPL', 'AAPL', 'MSFT', 'MSFT'],
        })
        
        result = process_news_dataframe(df)
        
        # Should have 2 articles (removed duplicate and short)
        assert len(result) == 2
    
    def test_process_empty_dataframe(self):
        """Test processing empty DataFrame."""
        from src.news.processor import process_news_dataframe
        
        df = pd.DataFrame(columns=['title', 'ticker'])
        result = process_news_dataframe(df)
        
        assert len(result) == 0


class TestSentimentPredictor:
    """Test sentiment prediction."""
    
    @pytest.mark.skipif(
        not Path("outputs/training/finbert_finetuned.pt").exists(),
        reason="Model checkpoint not available"
    )
    def test_predictor_initialization(self):
        """Test predictor can be initialized."""
        from src.inference.predictor import SentimentPredictor
        
        predictor = SentimentPredictor(model_name="finbert")
        assert predictor.model is not None
        assert predictor.device in ["cpu", "cuda"]
    
    @pytest.mark.skipif(
        not Path("outputs/training/finbert_finetuned.pt").exists(),
        reason="Model checkpoint not available"
    )
    def test_predict_single(self):
        """Test prediction on single text."""
        from src.inference.predictor import SentimentPredictor
        
        predictor = SentimentPredictor(model_name="finbert")
        
        results = predictor.predict(["Apple reports record profits"])
        
        assert len(results) == 1
        assert 'label' in results.columns
        assert 'score' in results.columns
        assert results.iloc[0]['label'] in [0, 1, 2]
    
    @pytest.mark.skipif(
        not Path("outputs/training/finbert_finetuned.pt").exists(),
        reason="Model checkpoint not available"
    )
    def test_predict_batch(self):
        """Test batch prediction."""
        from src.inference.predictor import SentimentPredictor
        
        predictor = SentimentPredictor(model_name="finbert")
        
        texts = [
            "Company reports massive losses",
            "Earnings meet expectations",
            "Stock surges on positive news",
        ]
        
        results = predictor.predict(texts)
        
        assert len(results) == 3
        assert all(col in results.columns for col in ['label', 'label_name', 'confidence', 'score'])
    
    @pytest.mark.skipif(
        not Path("outputs/training/finbert_finetuned.pt").exists(),
        reason="Model checkpoint not available"
    )
    def test_predict_dataframe(self):
        """Test DataFrame prediction."""
        from src.inference.predictor import SentimentPredictor
        
        predictor = SentimentPredictor(model_name="finbert")
        
        df = pd.DataFrame({
            'title': ['Good earnings report', 'Stock crashes'],
            'ticker': ['AAPL', 'MSFT'],
        })
        
        result = predictor.predict_dataframe(df)
        
        assert 'sentiment' in result.columns
        assert 'sentiment_score' in result.columns
        assert len(result) == 2


class TestIntegration:
    """Integration tests for full pipeline."""
    
    @pytest.mark.skipif(
        not Path("outputs/training/finbert_finetuned.pt").exists(),
        reason="Model checkpoint not available"
    )
    def test_full_pipeline(self):
        """Test full news pipeline."""
        from src.news.fetcher import fetch_news_for_tickers
        from src.news.processor import process_news_dataframe
        from src.inference.predictor import SentimentPredictor
        
        # Fetch
        raw_df = fetch_news_for_tickers(["AAPL"], max_per_ticker=5, delay=0)
        
        if raw_df.empty:
            pytest.skip("No news available")
        
        # Process
        processed_df = process_news_dataframe(raw_df)
        
        if processed_df.empty:
            pytest.skip("No articles after processing")
        
        # Predict
        predictor = SentimentPredictor(model_name="finbert")
        results_df = predictor.predict_dataframe(processed_df)
        
        # Verify output structure
        assert 'ticker' in results_df.columns
        assert 'title' in results_df.columns
        assert 'sentiment' in results_df.columns
        assert 'sentiment_score' in results_df.columns
        
        # Verify sentiment values are valid
        assert all(results_df['sentiment'].isin(['negative', 'neutral', 'positive']))
        assert all(results_df['sentiment_score'].isin([-1.0, 0.0, 1.0]))
