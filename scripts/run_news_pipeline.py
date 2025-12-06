#!/usr/bin/env python3
"""
Run the news ingestion and sentiment pipeline.

Fetches news for specified tickers, processes headlines,
and generates sentiment predictions.

Usage:
    python scripts/run_news_pipeline.py
    python scripts/run_news_pipeline.py --tickers AAPL MSFT GOOGL
    python scripts/run_news_pipeline.py --output outputs/news/custom.csv
"""
import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.news import STOCK_TICKERS
from src.news.fetcher import fetch_news_for_tickers
from src.news.processor import process_news_dataframe
from src.inference.predictor import SentimentPredictor


def main():
    parser = argparse.ArgumentParser(description="News sentiment pipeline")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=STOCK_TICKERS,
        help="Ticker symbols to fetch news for",
    )
    parser.add_argument(
        "--max-per-ticker",
        type=int,
        default=20,
        help="Maximum articles per ticker",
    )
    parser.add_argument(
        "--model",
        default="finbert",
        choices=["bert", "roberta", "distilbert", "finbert"],
        help="Model to use for predictions",
    )
    parser.add_argument(
        "--output",
        default="outputs/news/sentiment_results.csv",
        help="Output CSV path",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NEWS SENTIMENT PIPELINE")
    print("=" * 60)
    print(f"Tickers: {', '.join(args.tickers)}")
    
    # Step 1: Fetch news
    print(f"\n1. Fetching news...")
    raw_df = fetch_news_for_tickers(
        tickers=args.tickers,
        max_per_ticker=args.max_per_ticker,
    )
    print(f"   Fetched {len(raw_df)} articles")
    
    if raw_df.empty:
        print("No news articles found. Exiting.")
        return
    
    # Step 2: Process headlines
    print("\n2. Processing headlines...")
    processed_df = process_news_dataframe(raw_df)
    print(f"   After processing: {len(processed_df)} articles")
    
    if processed_df.empty:
        print("No articles after processing. Exiting.")
        return
    
    # Step 3: Run sentiment predictions
    print(f"\n3. Running sentiment predictions with {args.model}...")
    predictor = SentimentPredictor(model_name=args.model)
    results_df = predictor.predict_dataframe(processed_df)
    
    # Step 4: Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n4. Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal articles: {len(results_df)}")
    print(f"\nSentiment distribution:")
    sentiment_counts = results_df['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        pct = count / len(results_df) * 100
        print(f"  {sentiment}: {count} ({pct:.1f}%)")
    
    print(f"\nPer-ticker breakdown:")
    for ticker in args.tickers:
        ticker_df = results_df[results_df['ticker'] == ticker]
        if len(ticker_df) > 0:
            avg_score = ticker_df['sentiment_score'].mean()
            if avg_score > 0.1:
                sentiment = "bullish"
            elif avg_score < -0.1:
                sentiment = "bearish"
            else:
                sentiment = "neutral"
            print(f"  {ticker}: {len(ticker_df)} articles, avg score: {avg_score:+.2f} ({sentiment})")
    
    print(f"\nSample predictions:")
    for _, row in results_df.head(5).iterrows():
        title = row['title'][:55] + "..." if len(row['title']) > 55 else row['title']
        print(f"  [{row['ticker']:5}] [{row['sentiment']:8}] {title}")
    
    return results_df


if __name__ == "__main__":
    main()
