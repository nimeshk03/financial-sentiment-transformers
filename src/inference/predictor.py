"""
Sentiment prediction for news headlines.
"""
from typing import List, Optional
from pathlib import Path

import pandas as pd
import torch

from src.models.classifier import SentimentClassifier
from src.data.tokenizer import get_tokenizer


LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}
SENTIMENT_SCORES = {0: -1.0, 1: 0.0, 2: 1.0}


class SentimentPredictor:
    """Sentiment predictor for financial headlines."""
    
    def __init__(
        self,
        model_name: str = "finbert",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize predictor.
        
        Args:
            model_name: Model to use ('bert', 'roberta', 'distilbert', 'finbert')
            checkpoint_path: Path to model checkpoint. If None, uses default.
            device: Device to use. If None, auto-detects.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Default checkpoint path
        if checkpoint_path is None:
            checkpoint_path = f"outputs/training/{model_name}_finetuned.pt"
        
        self.checkpoint_path = checkpoint_path
        
        # Load model
        self._load_model()
        
        # Load tokenizer
        self.tokenizer = get_tokenizer(model_name)
    
    def _load_model(self) -> None:
        """Load model from checkpoint."""
        print(f"Loading {self.model_name} from {self.checkpoint_path}...")
        
        self.model = SentimentClassifier(
            model_name=self.model_name,
            num_classes=3,
            freeze_encoder=False,
        )
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
    
    def predict(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 128,
    ) -> pd.DataFrame:
        """Predict sentiment for a list of texts.
        
        Args:
            texts: List of headlines/texts
            batch_size: Batch size for inference
            max_length: Max sequence length
        
        Returns:
            DataFrame with columns: [text, label, label_name, confidence, score]
        """
        all_preds = []
        all_probs = []
        
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Build results DataFrame
        results = pd.DataFrame({
            'text': texts,
            'label': all_preds,
            'label_name': [LABEL_NAMES[p] for p in all_preds],
            'confidence': [probs.max() for probs in all_probs],
            'score': [SENTIMENT_SCORES[p] for p in all_preds],
            'prob_negative': [probs[0] for probs in all_probs],
            'prob_neutral': [probs[1] for probs in all_probs],
            'prob_positive': [probs[2] for probs in all_probs],
        })
        
        return results
    
    def predict_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = 'title',
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """Add sentiment predictions to a DataFrame.
        
        Args:
            df: DataFrame with text column
            text_col: Column containing text to analyze
            batch_size: Batch size for inference
        
        Returns:
            DataFrame with added sentiment columns
        """
        if df.empty:
            return df
        
        texts = df[text_col].tolist()
        predictions = self.predict(texts, batch_size)
        
        # Add prediction columns to original DataFrame
        df = df.copy()
        df['sentiment_label'] = predictions['label'].values
        df['sentiment'] = predictions['label_name'].values
        df['sentiment_confidence'] = predictions['confidence'].values
        df['sentiment_score'] = predictions['score'].values
        
        return df
