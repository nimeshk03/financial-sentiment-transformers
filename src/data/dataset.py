"""
Dataset loading and preprocessing for Financial PhraseBank.

This module handles:
- Loading the raw CSV data
- Label encoding (sentiment -> integer)
- Train/validation/test splits with stratification
"""
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd
from sklearn.model_selection import train_test_split


# Label mapping: sentiment string -> integer
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_NAMES = ["negative", "neutral", "positive"]


def load_financial_phrasebank(
    data_path: str = "data/raw/all-data.csv"
) -> pd.DataFrame:
    """Load Financial PhraseBank dataset from CSV.
    
    The CSV has no header and uses latin-1 encoding.
    Columns: [sentiment, text]
    
    Args:
        data_path: Path to the all-data.csv file
        
    Returns:
        DataFrame with columns: sentiment, text, label
    """
    # Read CSV - no header, latin-1 encoding (handles special characters)
    df = pd.read_csv(
        data_path,
        encoding="latin-1",
        header=None,
        names=["sentiment", "text"]
    )
    
    # Map sentiment strings to integer labels
    df["label"] = df["sentiment"].map(LABEL_MAP)
    
    # Validate no missing labels
    if df["label"].isna().any():
        missing = df[df["label"].isna()]["sentiment"].unique()
        raise ValueError(f"Unknown sentiment values: {missing}")
    
    # Convert label to int
    df["label"] = df["label"].astype(int)
    
    return df


def get_class_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """Get count of samples per class.
    
    Args:
        df: DataFrame with 'sentiment' column
        
    Returns:
        Dictionary mapping sentiment -> count
    """
    return df["sentiment"].value_counts().to_dict()


def get_class_weights(df: pd.DataFrame) -> Dict[int, float]:
    """Calculate class weights inversely proportional to frequency.
    
    Used for weighted loss to handle class imbalance.
    Formula: weight = total_samples / (num_classes * class_count)
    
    Args:
        df: DataFrame with 'label' column
        
    Returns:
        Dictionary mapping label -> weight
    """
    counts = df["label"].value_counts()
    total = len(df)
    num_classes = len(counts)
    
    weights = {}
    for label, count in counts.items():
        weights[label] = total / (num_classes * count)
    
    return weights


def create_splits(
    df: pd.DataFrame,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train/validation/test sets.
    
    Uses stratified splitting to maintain class distribution.
    
    Args:
        df: Full dataset DataFrame
        train_size: Fraction for training (default 0.8)
        val_size: Fraction for validation (default 0.1)
        test_size: Fraction for testing (default 0.1)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "Split sizes must sum to 1.0"
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        stratify=df["label"],
        random_state=random_state
    )
    
    # Second split: val vs test (from the temp set)
    # Adjust ratio: if val=0.1 and test=0.1, then val is 0.5 of temp
    val_ratio = val_size / (val_size + test_size)
    
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio,
        stratify=temp_df["label"],
        random_state=random_state
    )
    
    return train_df, val_df, test_df


def get_text_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate text length statistics.
    
    Args:
        df: DataFrame with 'text' column
        
    Returns:
        Dictionary with min, max, mean, median lengths
    """
    lengths = df["text"].str.len()
    word_counts = df["text"].str.split().str.len()
    
    return {
        "char_min": lengths.min(),
        "char_max": lengths.max(),
        "char_mean": lengths.mean(),
        "char_median": lengths.median(),
        "word_min": word_counts.min(),
        "word_max": word_counts.max(),
        "word_mean": word_counts.mean(),
        "word_median": word_counts.median(),
    }