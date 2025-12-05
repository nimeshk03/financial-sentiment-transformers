"""
Test script to verify environment setup for Milestone 1.1
Run with: pytest tests/test_environment.py -v
"""
import pytest


def test_torch_import():
    """Verify PyTorch is installed and reports device."""
    import torch
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Basic tensor operation should work
    x = torch.tensor([1.0, 2.0, 3.0])
    assert x.sum().item() == 6.0


def test_transformers_import():
    """Verify transformers library is installed."""
    import transformers
    
    print(f"\nTransformers version: {transformers.__version__}")
    assert hasattr(transformers, 'AutoTokenizer')
    assert hasattr(transformers, 'AutoModelForSequenceClassification')


def test_bert_model_loads():
    """Verify bert-base-uncased can be loaded."""
    from transformers import AutoTokenizer, AutoModel
    
    model_name = "bert-base-uncased"
    print(f"\nLoading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"{model_name} parameters: {param_count:,}")
    
    # BERT has ~110M parameters
    assert 100_000_000 < param_count < 120_000_000


def test_roberta_model_loads():
    """Verify roberta-base can be loaded."""
    from transformers import AutoTokenizer, AutoModel
    
    model_name = "roberta-base"
    print(f"\nLoading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"{model_name} parameters: {param_count:,}")
    
    # RoBERTa has ~125M parameters
    assert 120_000_000 < param_count < 130_000_000


def test_distilbert_model_loads():
    """Verify distilbert-base-uncased can be loaded."""
    from transformers import AutoTokenizer, AutoModel
    
    model_name = "distilbert-base-uncased"
    print(f"\nLoading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"{model_name} parameters: {param_count:,}")
    
    # DistilBERT has ~66M parameters
    assert 60_000_000 < param_count < 70_000_000


def test_finbert_model_loads():
    """Verify ProsusAI/finbert can be loaded."""
    from transformers import AutoTokenizer, AutoModel
    
    model_name = "ProsusAI/finbert"
    print(f"\nLoading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"{model_name} parameters: {param_count:,}")
    
    # FinBERT has ~110M parameters (same as BERT)
    assert 100_000_000 < param_count < 120_000_000


def test_dataset_loads():
    """Verify we can load a financial sentiment dataset.
    
    Uses 'zeroshot/twitter-financial-news-sentiment' which is in modern format
    and works with current datasets library. We'll use this or Financial PhraseBank
    (via manual download) for actual training.
    """
    from datasets import load_dataset
    
    print("\nLoading Twitter Financial News Sentiment dataset...")
    
    # This dataset is in modern format (no loading scripts)
    # It has similar structure: text + sentiment labels
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
    
    print(f"Dataset splits: {list(dataset.keys())}")
    
    # Check validation split (this dataset has train/validation)
    split_name = "train" if "train" in dataset else "validation"
    print(f"Number of samples in {split_name}: {len(dataset[split_name])}")
    
    # Verify dataset structure
    sample = dataset[split_name][0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Sample text: {sample['text'][:50]}...")
    print(f"Sample label: {sample['label']}")
    
    # Should have reasonable number of samples
    assert len(dataset[split_name]) > 1000
    assert 'text' in sample
    assert 'label' in sample
    
    print("\nNote: For training, we'll download Financial PhraseBank manually.")
    print("This test verifies the datasets library works correctly.")


def test_other_dependencies():
    """Verify other key dependencies are installed."""
    import pandas as pd
    import numpy as np
    import sklearn
    import yfinance
    import pytorch_lightning as pl
    
    print(f"\nPandas: {pd.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"Scikit-learn: {sklearn.__version__}")
    print(f"PyTorch Lightning: {pl.__version__}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])