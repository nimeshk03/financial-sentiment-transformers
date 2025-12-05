"""
Tests for tokenization pipeline (Milestone 1.3)
Run with: pytest tests/test_tokenization.py -v -s
"""
import pytest
import torch
from pathlib import Path


DATA_PATH = Path("data/raw/all-data.csv")


@pytest.fixture
def sample_texts():
    """Sample financial texts for testing."""
    return [
        "The company reported strong quarterly earnings.",
        "Stock prices fell sharply after the announcement.",
        "Market conditions remain stable.",
    ]


@pytest.fixture
def sample_labels():
    """Labels corresponding to sample texts."""
    return [2, 0, 1]  # positive, negative, neutral


@pytest.fixture
def data_splits():
    """Load and split the actual dataset."""
    if not DATA_PATH.exists():
        pytest.skip(f"Dataset not found at {DATA_PATH}")
    
    from src.data.dataset import load_financial_phrasebank, create_splits
    df = load_financial_phrasebank(str(DATA_PATH))
    return create_splits(df)


class TestGetTokenizer:
    """Tests for tokenizer loading."""
    
    @pytest.mark.parametrize("model_name", ["bert", "roberta", "distilbert", "finbert"])
    def test_load_tokenizer(self, model_name):
        """Test that all four tokenizers load correctly."""
        from src.data.tokenizer import get_tokenizer
        
        tokenizer = get_tokenizer(model_name)
        assert tokenizer is not None
        print(f"\n{model_name} tokenizer loaded: {type(tokenizer).__name__}")
    
    def test_tokenizer_with_full_path(self):
        """Test loading tokenizer with full HuggingFace path."""
        from src.data.tokenizer import get_tokenizer
        
        tokenizer = get_tokenizer("bert-base-uncased")
        assert tokenizer is not None


class TestTokenizeTexts:
    """Tests for text tokenization."""
    
    def test_tokenize_basic(self, sample_texts):
        """Test basic tokenization."""
        from src.data.tokenizer import get_tokenizer, tokenize_texts
        
        tokenizer = get_tokenizer("bert")
        encoded = tokenize_texts(sample_texts, tokenizer, max_length=128)
        
        assert "input_ids" in encoded
        assert "attention_mask" in encoded
        assert encoded["input_ids"].shape == (3, 128)
        assert encoded["attention_mask"].shape == (3, 128)
        
        print(f"\nTokenized shape: {encoded['input_ids'].shape}")
    
    def test_tokenize_truncation(self, sample_texts):
        """Test that long texts are truncated."""
        from src.data.tokenizer import get_tokenizer, tokenize_texts
        
        tokenizer = get_tokenizer("bert")
        # Use very short max_length to force truncation
        encoded = tokenize_texts(sample_texts, tokenizer, max_length=16)
        
        assert encoded["input_ids"].shape == (3, 16)
        print(f"\nTruncated to max_length=16: {encoded['input_ids'].shape}")
    
    def test_tokenize_padding(self):
        """Test that short texts are padded."""
        from src.data.tokenizer import get_tokenizer, tokenize_texts
        
        tokenizer = get_tokenizer("bert")
        short_text = ["Hi"]
        encoded = tokenize_texts(short_text, tokenizer, max_length=128)
        
        # Should be padded to 128
        assert encoded["input_ids"].shape == (1, 128)
        # Most tokens should be padding (0)
        assert (encoded["input_ids"][0] == tokenizer.pad_token_id).sum() > 100
        
        print(f"\nPadded 'Hi' to length 128")
    
    @pytest.mark.parametrize("model_name", ["bert", "roberta", "distilbert", "finbert"])
    def test_tokenize_all_models(self, sample_texts, model_name):
        """Test tokenization works for all four models."""
        from src.data.tokenizer import get_tokenizer, tokenize_texts
        
        tokenizer = get_tokenizer(model_name)
        encoded = tokenize_texts(sample_texts, tokenizer, max_length=128)
        
        assert encoded["input_ids"].shape == (3, 128)
        print(f"\n{model_name}: {encoded['input_ids'].shape}")


class TestFinancialSentimentDataset:
    """Tests for PyTorch Dataset."""
    
    def test_dataset_creation(self, sample_texts, sample_labels):
        """Test dataset creation."""
        from src.data.tokenizer import get_tokenizer, FinancialSentimentDataset
        
        tokenizer = get_tokenizer("bert")
        dataset = FinancialSentimentDataset(
            texts=sample_texts,
            labels=sample_labels,
            tokenizer=tokenizer,
            max_length=128,
        )
        
        assert len(dataset) == 3
        print(f"\nDataset size: {len(dataset)}")
    
    def test_dataset_getitem(self, sample_texts, sample_labels):
        """Test getting items from dataset."""
        from src.data.tokenizer import get_tokenizer, FinancialSentimentDataset
        
        tokenizer = get_tokenizer("bert")
        dataset = FinancialSentimentDataset(
            texts=sample_texts,
            labels=sample_labels,
            tokenizer=tokenizer,
            max_length=128,
        )
        
        item = dataset[0]
        
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "label" in item
        assert item["input_ids"].shape == (128,)
        assert item["label"].item() == sample_labels[0]
        
        print(f"\nItem keys: {list(item.keys())}")
        print(f"input_ids shape: {item['input_ids'].shape}")
        print(f"label: {item['label'].item()}")


class TestDataLoader:
    """Tests for DataLoader creation."""
    
    def test_create_dataloader(self, sample_texts, sample_labels):
        """Test DataLoader creation."""
        from src.data.tokenizer import (
            get_tokenizer, FinancialSentimentDataset, create_dataloader
        )
        
        tokenizer = get_tokenizer("bert")
        dataset = FinancialSentimentDataset(
            texts=sample_texts,
            labels=sample_labels,
            tokenizer=tokenizer,
        )
        
        loader = create_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        
        assert batch["input_ids"].shape == (2, 128)
        assert batch["attention_mask"].shape == (2, 128)
        assert batch["label"].shape == (2,)
        
        print(f"\nBatch shapes:")
        print(f"  input_ids: {batch['input_ids'].shape}")
        print(f"  attention_mask: {batch['attention_mask'].shape}")
        print(f"  labels: {batch['label'].shape}")
    
    @pytest.mark.parametrize("model_name", ["bert", "roberta", "distilbert", "finbert"])
    def test_batch_shape_all_models(self, sample_texts, sample_labels, model_name):
        """Test batch shape is (batch_size, max_length) for all models."""
        from src.data.tokenizer import (
            get_tokenizer, FinancialSentimentDataset, create_dataloader
        )
        
        batch_size = 2
        max_length = 128
        
        tokenizer = get_tokenizer(model_name)
        dataset = FinancialSentimentDataset(
            texts=sample_texts,
            labels=sample_labels,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        
        loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)
        batch = next(iter(loader))
        
        # THE KEY TEST: batch shape must be (batch_size, max_length)
        assert batch["input_ids"].shape == (batch_size, max_length), \
            f"{model_name}: Expected ({batch_size}, {max_length}), got {batch['input_ids'].shape}"
        
        print(f"\n{model_name}: batch shape = {batch['input_ids'].shape}")


class TestCreateDataloadersFromDF:
    """Tests for the convenience function."""
    
    def test_create_dataloaders(self, data_splits):
        """Test creating all dataloaders from DataFrames."""
        from src.data.tokenizer import create_dataloaders_from_df
        
        train_df, val_df, test_df = data_splits
        
        train_loader, val_loader, test_loader = create_dataloaders_from_df(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            model_name="bert",
            max_length=128,
            batch_size=32,
        )
        
        # Check train loader
        train_batch = next(iter(train_loader))
        assert train_batch["input_ids"].shape == (32, 128)
        
        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        print(f"Batch shape: {train_batch['input_ids'].shape}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])