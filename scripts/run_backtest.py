#!/usr/bin/env python3
"""
Run backtesting on sentiment signals.

Tests if sentiment-based trading strategies outperform the market.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --input outputs/signals/aggregated_signals.csv
"""
import sys
from pathlib import Path
import argparse
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.backtesting.strategy import (
    StrategyConfig,
    generate_signals,
    calculate_strategy_returns,
    long_only_strategy,
    momentum_strategy,
)
from src.backtesting.metrics import calculate_metrics, print_results


def run_all_strategies(df: pd.DataFrame) -> dict:
    """Run all strategies and compare results.
    
    Args:
        df: DataFrame with sentiment and price data
    
    Returns:
        Dictionary of strategy results
    """
    results = {}
    
    # Strategy 1: Basic sentiment threshold
    print("\n" + "="*60)
    print("STRATEGY 1: Sentiment Threshold (Long/Short)")
    print("="*60)
    print("Rule: Long if sentiment > 0.2, Short if sentiment < -0.2")
    
    config = StrategyConfig(
        long_threshold=0.2,
        short_threshold=-0.2,
        signal_column='sentiment_mean',
        return_column='return_1d',
    )
    
    df1 = generate_signals(df, config)
    df1 = calculate_strategy_returns(df1, config)
    metrics1 = calculate_metrics(df1)
    print_results(metrics1, "Sentiment Threshold")
    results['sentiment_threshold'] = metrics1.to_dict()
    
    # Strategy 2: Long-only
    print("\n" + "="*60)
    print("STRATEGY 2: Long Only")
    print("="*60)
    print("Rule: Long if sentiment > 0.1, else flat (no shorting)")
    
    df2 = long_only_strategy(df, threshold=0.1)
    metrics2 = calculate_metrics(df2)
    print_results(metrics2, "Long Only")
    results['long_only'] = metrics2.to_dict()
    
    # Strategy 3: Momentum-based
    if 'sentiment_momentum' in df.columns:
        print("\n" + "="*60)
        print("STRATEGY 3: Sentiment Momentum")
        print("="*60)
        print("Rule: Long if momentum > 0.05, Short if momentum < -0.05")
        
        df3 = momentum_strategy(df, momentum_threshold=0.05)
        metrics3 = calculate_metrics(df3)
        print_results(metrics3, "Sentiment Momentum")
        results['momentum'] = metrics3.to_dict()
    
    # Strategy 4: Rolling sentiment
    if 'sentiment_rolling_3d' in df.columns:
        print("\n" + "="*60)
        print("STRATEGY 4: Rolling Sentiment (3-day)")
        print("="*60)
        print("Rule: Long if 3d rolling sentiment > 0.15, else flat")
        
        df4 = long_only_strategy(
            df, 
            threshold=0.15, 
            signal_column='sentiment_rolling_3d'
        )
        metrics4 = calculate_metrics(df4)
        print_results(metrics4, "Rolling Sentiment 3d")
        results['rolling_3d'] = metrics4.to_dict()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Backtest sentiment strategies")
    parser.add_argument(
        "--input",
        default="outputs/signals/aggregated_signals.csv",
        help="Input signals CSV",
    )
    parser.add_argument(
        "--output",
        default="outputs/backtest/backtest_results.json",
        help="Output JSON path",
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("SENTIMENT TRADING BACKTEST")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from {args.input}...")
    df = pd.read_csv(args.input)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Loaded {len(df)} records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Tickers: {', '.join(df['ticker'].unique())}")
    
    # Check for required columns
    required = ['sentiment_mean', 'return_1d']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        return
    
    # Drop rows with NaN returns (last day has no forward return)
    df_valid = df.dropna(subset=['return_1d'])
    print(f"Valid records (with returns): {len(df_valid)}")
    
    if len(df_valid) < 2:
        print("Not enough data for backtesting. Need more historical data.")
        print("\nTip: Run with more price history:")
        print("  python scripts/run_aggregation.py --price-period 3mo")
        return
    
    # Run strategies
    results = run_all_strategies(df_valid)
    
    # Summary comparison
    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)
    
    print(f"\n{'Strategy':<25} {'Return':>10} {'Sharpe':>10} {'Win Rate':>10}")
    print("-" * 55)
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['total_return']:>+10.2%} "
              f"{metrics['sharpe_ratio']:>10.2f} {metrics['win_rate']:>10.1%}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    best_strategy = max(results.items(), key=lambda x: x[1]['total_return'])
    print(f"\nBest performing strategy: {best_strategy[0]}")
    print(f"  Return: {best_strategy[1]['total_return']:+.2%}")
    
    if best_strategy[1]['excess_return'] > 0:
        print(f"  Beat market by: {best_strategy[1]['excess_return']:+.2%}")
    else:
        print(f"  Underperformed market by: {best_strategy[1]['excess_return']:+.2%}")
    
    print("\nNote: Results are based on limited data. For robust conclusions,")
    print("run with longer history: --price-period 1y")


if __name__ == "__main__":
    main()
