"""Model definitions and training utilities."""
from src.models.classifier import (
    SentimentClassifier,
    create_classifier,
    MODEL_NAMES,
)

from src.models.trainer import (
    Trainer,
    TrainingConfig,
    TrainingResult,
    quick_evaluate,
)

__all__ = [
    "SentimentClassifier",
    "create_classifier",
    "MODEL_NAMES",
    "Trainer",
    "TrainingConfig",
    "TrainingResult",
    "quick_evaluate",
]