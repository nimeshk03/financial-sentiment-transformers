"""
Unified tokenization pipeline for all four transformer models.

This module provides:
- A unified tokenization function that works with BERT, RoBERTa, DistilBERT, and FinBERT
- PyTorch Dataset class for tokenized data
- DataLoader factory with consistent batch sizes
"""
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer


# Model name mapping for HuggingFace
MODEL_NAMES = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "distilbert": "distilbert-base-uncased",
    "finbert": "ProsusAI/finbert",
}

# Default tokenization parameters
DEFAULT_MAX_LENGTH = 128
DEFAULT_BATCH_SIZE = 32


def get_tokenizer(model_name: str) -> AutoTokenizer:
    """Load tokenizer for a given model.
    
    Args:
        model_name: One of 'bert', 'roberta', 'distilbert', 'finbert'
                   or a HuggingFace model path
    
    Returns:
        AutoTokenizer instance
    """
    # Map short names to full HuggingFace paths
    hf_name = MODEL_NAMES.get(model_name.lower(), model_name)
    return AutoTokenizer.from_pretrained(hf_name)


def tokenize_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Dict[str, torch.Tensor]:
    """Tokenize a list of texts using the given tokenizer.
    
    Applies padding and truncation to ensure uniform length.
    
    Args:
        texts: List of text strings to tokenize
        tokenizer: HuggingFace tokenizer instance
        max_length: Maximum sequence length (default 128)
    
    Returns:
        Dictionary with 'input_ids' and 'attention_mask' tensors
        Shape: (num_texts, max_length)
    """
    # Tokenize with padding and truncation
    # - padding="max_length": pad all sequences to max_length
    # - truncation=True: truncate sequences longer than max_length
    # - return_tensors="pt": return PyTorch tensors
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


class FinancialSentimentDataset(Dataset):
    """PyTorch Dataset for financial sentiment classification.
    
    Holds tokenized texts and their labels for efficient batching.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        """Initialize dataset with texts and labels.
        
        Args:
            texts: List of text strings
            labels: List of integer labels (0, 1, 2)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-tokenize all texts for efficiency
        # This uses more memory but speeds up training
        self.encodings = tokenize_texts(texts, tokenizer, max_length)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single tokenized sample.
        
        Returns:
            Dictionary with input_ids, attention_mask, and label
        """
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def create_dataloader(
    dataset: FinancialSentimentDataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader from a dataset.
    
    Args:
        dataset: FinancialSentimentDataset instance
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data (True for train, False for val/test)
        num_workers: Number of worker processes for data loading
    
    Returns:
        PyTorch DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
    )


def create_dataloaders_from_df(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from DataFrames.
    
    Convenience function that handles tokenizer loading and dataset creation.
    
    Args:
        train_df: Training DataFrame with 'text' and 'label' columns
        val_df: Validation DataFrame
        test_df: Test DataFrame
        model_name: Model name for tokenizer ('bert', 'roberta', etc.)
        max_length: Maximum sequence length
        batch_size: Batch size for all loaders
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load tokenizer
    tokenizer = get_tokenizer(model_name)
    
    # Create datasets
    train_dataset = FinancialSentimentDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    
    val_dataset = FinancialSentimentDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    
    test_dataset = FinancialSentimentDataset(
        texts=test_df["text"].tolist(),
        labels=test_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    
    # Create dataloaders
    train_loader = create_dataloader(train_dataset, batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size, shuffle=False)
    test_loader = create_dataloader(test_dataset, batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def get_sample_batch(
    dataloader: DataLoader,
) -> Dict[str, torch.Tensor]:
    """Get a single batch from a dataloader for inspection.
    
    Args:
        dataloader: PyTorch DataLoader
    
    Returns:
        Dictionary with input_ids, attention_mask, and labels
    """
    return next(iter(dataloader))