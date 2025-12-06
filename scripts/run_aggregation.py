#!/usr/bin/env python3
"""
Run sentiment aggregation pipeline.

Aggregates news sentiment, fetches prices, and prepares data for backtesting.

Usage:
    python scripts/run_aggregation.py
    python scripts/run_aggregation.py --input outputs/news/sentiment_results.csv
"""
import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.signals.aggregator import (
    aggregate_daily_sentiment,
    add_rolling_sentiment,
    compute_sentiment_momentum,
)
from src.signals.prices import fetch_prices, compute_returns, merge_sentiment_prices


def main():
    parser = argparse.ArgumentParser(description="Sentiment aggregation pipeline")
    parser.add_argument(
        "--input",
        default="outputs/news/sentiment_results.csv",
        help="Input sentiment CSV",
    )
    parser.add_argument(
        "--output",
        default="outputs/signals/aggregated_signals.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--price-period",
        default="1mo",
        help="Price history period (1d, 5d, 1mo, 3mo, 1y)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SENTIMENT AGGREGATION PIPELINE")
    print("=" * 60)
    
    # Step 1: Load sentiment data
    print(f"\n1. Loading sentiment data from {args.input}...")
    sentiment_df = pd.read_csv(args.input)
    print(f"   Loaded {len(sentiment_df)} articles")
    
    tickers = sentiment_df['ticker'].unique().tolist()
    print(f"   Tickers: {', '.join(tickers)}")
    
    # Step 2: Aggregate to daily
    print("\n2. Aggregating to daily sentiment...")
    daily_df = aggregate_daily_sentiment(sentiment_df)
    print(f"   Generated {len(daily_df)} daily records")
    
    # Step 3: Add rolling averages
    print("\n3. Computing rolling averages (3d, 7d)...")
    daily_df = add_rolling_sentiment(daily_df, windows=[3, 7])
    daily_df = compute_sentiment_momentum(daily_df)
    
    # Save daily sentiment separately
    daily_output = Path(args.output).parent / "daily_sentiment.csv"
    daily_df.to_csv(daily_output, index=False)
    print(f"   Daily sentiment saved to: {daily_output}")
    
    # Step 4: Fetch prices
    print(f"\n4. Fetching price data (period: {args.price_period})...")
    price_df = fetch_prices(tickers, period=args.price_period)
    print(f"   Fetched {len(price_df)} price records")
    
    if price_df.empty:
        print("   No price data available. Saving daily sentiment only.")
        return daily_df
    
    # Step 5: Compute returns
    print("\n5. Computing forward returns (1d, 5d)...")
    price_df = compute_returns(price_df, periods=[1, 5])
    
    # Step 6: Merge
    print("\n6. Merging sentiment and price data...")
    merged_df = merge_sentiment_prices(daily_df, price_df)
    print(f"   Merged dataset: {len(merged_df)} records")
    
    # Step 7: Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    print(f"\n7. Results saved to: {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\nOutput columns:")
    for col in merged_df.columns:
        print(f"  - {col}")
    
    if len(merged_df) > 0:
        print(f"\nSample data (first 5 rows):")
        display_cols = ['date', 'ticker', 'sentiment_mean', 'sentiment_rolling_3d', 
                        'close', 'return_1d']
        display_cols = [c for c in display_cols if c in merged_df.columns]
        print(merged_df[display_cols].head().to_string())
        
        # Correlation preview
        if 'return_1d' in merged_df.columns and 'sentiment_mean' in merged_df.columns:
            valid_data = merged_df.dropna(subset=['sentiment_mean', 'return_1d'])
            if len(valid_data) > 2:
                corr = valid_data['sentiment_mean'].corr(valid_data['return_1d'])
                print(f"\nSentiment-Return correlation (1d): {corr:.4f}")
    
    return merged_df


if __name__ == "__main__":
    main()
