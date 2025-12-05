"""
Tests for data loading module (Milestone 1.2)
Run with: pytest tests/test_data_loading.py -v
"""
import pytest
import pandas as pd
from pathlib import Path


# Data path relative to project root (where pytest runs from)
DATA_PATH = Path("data/raw/all-data.csv")


@pytest.fixture
def data_path():
    """Return data path, skip if file doesn't exist."""
    if not DATA_PATH.exists():
        pytest.skip(f"Dataset not found at {DATA_PATH}. Download from Kaggle first.")
    return DATA_PATH


def test_load_financial_phrasebank(data_path):
    """Test that dataset loads correctly."""
    from src.data.dataset import load_financial_phrasebank
    
    df = load_financial_phrasebank(str(data_path))
    
    # Check shape
    assert len(df) > 4000, f"Expected >4000 samples, got {len(df)}"
    assert len(df.columns) == 3, f"Expected 3 columns, got {df.columns.tolist()}"
    
    # Check columns
    assert "sentiment" in df.columns
    assert "text" in df.columns
    assert "label" in df.columns
    
    # Check label values
    assert set(df["label"].unique()) == {0, 1, 2}
    
    print(f"\nLoaded {len(df):,} samples successfully")


def test_class_distribution(data_path):
    """Test class distribution calculation."""
    from src.data.dataset import load_financial_phrasebank, get_class_distribution
    
    df = load_financial_phrasebank(str(data_path))
    dist = get_class_distribution(df)
    
    # Check all classes present
    assert "negative" in dist
    assert "neutral" in dist
    assert "positive" in dist
    
    # Neutral should be majority class
    assert dist["neutral"] > dist["positive"]
    assert dist["neutral"] > dist["negative"]
    
    print(f"\nClass distribution: {dist}")


def test_class_weights(data_path):
    """Test class weight calculation."""
    from src.data.dataset import load_financial_phrasebank, get_class_weights
    
    df = load_financial_phrasebank(str(data_path))
    weights = get_class_weights(df)
    
    # Check all labels have weights
    assert 0 in weights  # negative
    assert 1 in weights  # neutral
    assert 2 in weights  # positive
    
    # Minority class (negative) should have highest weight
    assert weights[0] > weights[1], "Negative should have higher weight than neutral"
    
    print(f"\nClass weights: {weights}")


def test_create_splits(data_path):
    """Test train/val/test split creation."""
    from src.data.dataset import load_financial_phrasebank, create_splits
    
    df = load_financial_phrasebank(str(data_path))
    train_df, val_df, test_df = create_splits(df)
    
    # Check sizes (80/10/10 split)
    total = len(df)
    assert abs(len(train_df) / total - 0.8) < 0.02, "Train should be ~80%"
    assert abs(len(val_df) / total - 0.1) < 0.02, "Val should be ~10%"
    assert abs(len(test_df) / total - 0.1) < 0.02, "Test should be ~10%"
    
    # Check stratification preserved
    train_ratio = train_df["label"].value_counts(normalize=True)
    val_ratio = val_df["label"].value_counts(normalize=True)
    
    for label in [0, 1, 2]:
        assert abs(train_ratio[label] - val_ratio[label]) < 0.05
    
    print(f"\nSplit sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")


def test_text_statistics(data_path):
    """Test text statistics calculation."""
    from src.data.dataset import load_financial_phrasebank, get_text_statistics
    
    df = load_financial_phrasebank(str(data_path))
    stats = get_text_statistics(df)
    
    # Check all stats present
    required_keys = ["char_min", "char_max", "char_mean", "word_mean"]
    for key in required_keys:
        assert key in stats, f"Missing stat: {key}"
    
    # Sanity checks
    assert stats["word_mean"] > 5
    
    print(f"\nText stats: {stats}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])