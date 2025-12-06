"""
Tests for Milestone 3.1: Error Analysis.

Verifies:
- Model predictions can be loaded
- Disagreement analysis works
- Visualizations are generated
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Skip if data not available
DATA_PATH = Path("data/raw/all-data.csv")
MODEL_DIR = Path("outputs/training")


class TestErrorAnalysisModule:
    """Test error analysis functions."""
    
    def test_label_mappings(self):
        """Test label mappings are correct."""
        from src.analysis.error_analysis import LABEL_NAMES, LABEL_IDS
        
        assert LABEL_NAMES[0] == "negative"
        assert LABEL_NAMES[1] == "neutral"
        assert LABEL_NAMES[2] == "positive"
        
        assert LABEL_IDS["negative"] == 0
        assert LABEL_IDS["neutral"] == 1
        assert LABEL_IDS["positive"] == 2
    
    def test_find_disagreements(self):
        """Test disagreement finding logic."""
        from src.analysis.error_analysis import find_disagreements
        
        # Create mock predictions
        df = pd.DataFrame({
            "text": ["text1", "text2", "text3", "text4"],
            "true_label": [0, 1, 2, 0],
            "bert_pred": [0, 1, 2, 0],
            "roberta_pred": [0, 1, 1, 0],  # Disagrees on text3
            "finbert_pred": [1, 1, 2, 0],  # Disagrees on text1
        })
        
        disagreements = find_disagreements(df)
        
        # Should find 2 disagreements (text1 and text3)
        assert len(disagreements) == 2
        assert "num_unique_preds" in disagreements.columns
    
    def test_find_hard_samples(self):
        """Test hard sample detection."""
        from src.analysis.error_analysis import find_hard_samples
        
        df = pd.DataFrame({
            "text": ["easy", "hard"],
            "bert_confidence": [0.95, 0.45],
            "roberta_confidence": [0.90, 0.50],
        })
        
        hard = find_hard_samples(df, threshold=0.6)
        
        assert len(hard) == 1
        assert hard.iloc[0]["text"] == "hard"
    
    def test_analyze_misclassifications(self):
        """Test misclassification analysis."""
        from src.analysis.error_analysis import analyze_misclassifications
        
        df = pd.DataFrame({
            "text": ["t1", "t2", "t3", "t4"],
            "true_label": [0, 1, 2, 0],
            "bert_pred": [0, 1, 1, 1],  # Wrong on t3 and t4
        })
        
        analysis = analyze_misclassifications(df, "bert")
        
        assert analysis["total_samples"] == 4
        assert analysis["correct"] == 2
        assert analysis["incorrect"] == 2
        assert analysis["accuracy"] == 0.5


def _has_model_checkpoints():
    """Check if at least one model checkpoint exists."""
    for model in ["bert", "roberta", "distilbert", "finbert"]:
        if (MODEL_DIR / f"{model}_finetuned.pt").exists():
            return True
    return False


@pytest.mark.skipif(
    not DATA_PATH.exists() or not _has_model_checkpoints(),
    reason="Data or model checkpoints not available"
)
class TestErrorAnalysisIntegration:
    """Integration tests requiring data and models."""
    
    def test_predictions_generation(self):
        """Test that predictions can be generated."""
        from src.data.dataset import load_financial_phrasebank, create_splits
        from src.analysis.error_analysis import get_all_model_predictions
        
        df = load_financial_phrasebank(str(DATA_PATH))
        _, _, test_df = create_splits(df)
        
        # Use small subset
        texts = test_df["text"].head(10).tolist()
        labels = test_df["label"].head(10).values
        
        predictions_df = get_all_model_predictions(
            texts=texts,
            labels=labels,
            model_dir=str(MODEL_DIR),
            device="cpu",
        )
        
        assert len(predictions_df) == 10
        assert "true_label" in predictions_df.columns
        # At least one model should have predictions
        pred_cols = [c for c in predictions_df.columns if c.endswith("_pred")]
        assert len(pred_cols) > 0


@pytest.mark.skipif(
    not DATA_PATH.exists() or not MODEL_DIR.exists(),
    reason="Data or models not available"
)
class TestVisualizationGeneration:
    """Test visualization generation."""
    
    def test_confusion_matrix_plot(self, tmp_path):
        """Test confusion matrix plotting."""
        from src.analysis.error_analysis import plot_confusion_matrices
        
        # Mock predictions
        df = pd.DataFrame({
            "true_label": [0, 1, 2, 0, 1, 2],
            "bert_pred": [0, 1, 2, 1, 1, 2],
            "finbert_pred": [0, 1, 2, 0, 1, 1],
        })
        
        output_path = tmp_path / "cm.png"
        plot_confusion_matrices(df, str(output_path))
        
        assert output_path.exists()