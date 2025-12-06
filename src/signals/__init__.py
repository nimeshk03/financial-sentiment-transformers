"""Trading signals and sentiment aggregation."""

from src.signals.aggregator import (
    aggregate_daily_sentiment,
    add_rolling_sentiment,
    compute_sentiment_momentum,
)
from src.signals.prices import (
    fetch_prices,
    compute_returns,
    merge_sentiment_prices,
)

__all__ = [
    "aggregate_daily_sentiment",
    "add_rolling_sentiment",
    "compute_sentiment_momentum",
    "fetch_prices",
    "compute_returns",
    "merge_sentiment_prices",
]
