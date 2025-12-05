"""
Transformer-based sentiment classifier.

This module provides a unified classifier that works with:
- BERT, RoBERTa, DistilBERT, FinBERT

Supports both frozen (feature extraction) and unfrozen (fine-tuning) modes.
"""
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


# Model name mapping
MODEL_NAMES = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "distilbert": "distilbert-base-uncased",
    "finbert": "ProsusAI/finbert",
}


class SentimentClassifier(nn.Module):
    """Transformer-based sentiment classifier.
    
    Architecture:
        [Transformer Encoder] -> [Pooled Output] -> [Dropout] -> [Linear] -> [3 classes]
    
    The transformer encoder can be frozen (for baseline) or unfrozen (for fine-tuning).
    """
    
    def __init__(
        self,
        model_name: str = "bert",
        num_classes: int = 3,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ):
        """Initialize the classifier.
        
        Args:
            model_name: One of 'bert', 'roberta', 'distilbert', 'finbert'
            num_classes: Number of output classes (default 3: neg/neu/pos)
            dropout: Dropout probability before classification head
            freeze_encoder: If True, freeze transformer weights (baseline mode)
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder
        
        # Load pre-trained transformer
        hf_name = MODEL_NAMES.get(model_name.lower(), model_name)
        self.config = AutoConfig.from_pretrained(hf_name)
        self.encoder = AutoModel.from_pretrained(hf_name)
        
        # Get hidden size from config
        # Different models use different attribute names
        if hasattr(self.config, "hidden_size"):
            hidden_size = self.config.hidden_size
        elif hasattr(self.config, "dim"):
            hidden_size = self.config.dim
        else:
            hidden_size = 768  # Default BERT size
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # Freeze encoder if requested
        if freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def _unfreeze_encoder(self):
        """Unfreeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input_ids: Token IDs, shape (batch_size, seq_length)
            attention_mask: Attention mask, shape (batch_size, seq_length)
        
        Returns:
            Logits, shape (batch_size, num_classes)
        """
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Get pooled output
        # For BERT/FinBERT: use pooler_output (CLS token representation)
        # For RoBERTa: no pooler, use last_hidden_state[:, 0, :]
        # For DistilBERT: no pooler, use last_hidden_state[:, 0, :]
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            # Use CLS token (first token) from last hidden state
            pooled = outputs.last_hidden_state[:, 0, :]
        
        # Classification head
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits
    
    def get_num_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


def create_classifier(
    model_name: str = "bert",
    num_classes: int = 3,
    freeze_encoder: bool = False,
    device: Optional[str] = None,
) -> SentimentClassifier:
    """Factory function to create a classifier.
    
    Args:
        model_name: Model name ('bert', 'roberta', 'distilbert', 'finbert')
        num_classes: Number of classes
        freeze_encoder: Whether to freeze encoder weights
        device: Device to move model to (None for auto-detect)
    
    Returns:
        SentimentClassifier instance on the specified device
    """
    model = SentimentClassifier(
        model_name=model_name,
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
    )
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return model.to(device)