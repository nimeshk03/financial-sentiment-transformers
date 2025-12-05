"""
Tests for BERT baseline model (Milestone 1.4)
Run with: pytest tests/test_baseline.py -v -s
"""
import pytest
import torch
from pathlib import Path


DATA_PATH = Path("data/raw/all-data.csv")


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class TestSentimentClassifier:
    """Tests for the classifier model."""
    
    def test_create_classifier_bert(self, device):
        """Test BERT classifier creation."""
        from src.models.classifier import create_classifier
        
        model = create_classifier("bert", freeze_encoder=False, device=device)
        
        assert model is not None
        assert model.num_classes == 3
        print(f"\nBERT total params: {model.get_num_total_params():,}")
    
    def test_create_classifier_frozen(self, device):
        """Test frozen encoder has fewer trainable params."""
        from src.models.classifier import create_classifier
        
        frozen = create_classifier("bert", freeze_encoder=True, device=device)
        unfrozen = create_classifier("bert", freeze_encoder=False, device=device)
        
        frozen_trainable = frozen.get_num_trainable_params()
        unfrozen_trainable = unfrozen.get_num_trainable_params()
        
        # Frozen should have way fewer trainable params
        assert frozen_trainable < unfrozen_trainable
        assert frozen_trainable < 10000  # Only classifier head
        
        print(f"\nFrozen trainable: {frozen_trainable:,}")
        print(f"Unfrozen trainable: {unfrozen_trainable:,}")
    
    def test_forward_pass(self, device):
        """Test forward pass produces correct output shape."""
        from src.models.classifier import create_classifier
        
        model = create_classifier("bert", freeze_encoder=True, device=device)
        
        # Create dummy input
        batch_size = 4
        seq_length = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
        attention_mask = torch.ones(batch_size, seq_length).to(device)
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        
        assert logits.shape == (batch_size, 3)
        print(f"\nOutput shape: {logits.shape}")
    
    @pytest.mark.parametrize("model_name", ["bert", "roberta", "distilbert", "finbert"])
    def test_all_models_forward(self, model_name, device):
        """Test forward pass works for all four models."""
        from src.models.classifier import create_classifier
        
        model = create_classifier(model_name, freeze_encoder=True, device=device)
        
        batch_size = 2
        seq_length = 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
        attention_mask = torch.ones(batch_size, seq_length).to(device)
        
        logits = model(input_ids, attention_mask)
        
        assert logits.shape == (batch_size, 3)
        print(f"\n{model_name}: output shape = {logits.shape}")


class TestTrainer:
    """Tests for the trainer."""
    
    @pytest.fixture
    def small_data(self):
        """Create small dataset for testing."""
        if not DATA_PATH.exists():
            pytest.skip("Dataset not found")
        
        from src.data.dataset import load_financial_phrasebank, create_splits
        from src.data.tokenizer import create_dataloaders_from_df
        
        df = load_financial_phrasebank(str(DATA_PATH))
        # Use small subset for fast testing
        df = df.sample(n=100, random_state=42)
        train_df, val_df, test_df = create_splits(df)
        
        train_loader, val_loader, test_loader = create_dataloaders_from_df(
            train_df, val_df, test_df,
            model_name="bert",
            max_length=64,  # Shorter for speed
            batch_size=8,
        )
        
        return train_loader, val_loader, test_loader
    
    def test_trainer_one_epoch(self, small_data, device):
        """Test training for one epoch."""
        from src.models.classifier import create_classifier
        from src.models.trainer import Trainer, TrainingConfig
        
        train_loader, val_loader, _ = small_data
        
        model = create_classifier("bert", freeze_encoder=True, device=device)
        config = TrainingConfig(
            learning_rate=1e-3,
            epochs=1,
            device=device,
        )
        
        trainer = Trainer(model, config)
        result = trainer.train(train_loader, val_loader, epochs=1)
        
        assert len(result.train_losses) == 1
        assert len(result.val_accuracies) == 1
        assert result.best_accuracy > 0
        
        print(f"\nTrain loss: {result.train_losses[0]:.4f}")
        print(f"Val accuracy: {result.val_accuracies[0]*100:.1f}%")
    
    def test_evaluate(self, small_data, device):
        """Test evaluation."""
        from src.models.classifier import create_classifier
        from src.models.trainer import Trainer, TrainingConfig
        
        _, val_loader, _ = small_data
        
        model = create_classifier("bert", freeze_encoder=True, device=device)
        config = TrainingConfig(device=device)
        trainer = Trainer(model, config)
        
        loss, acc, f1, report = trainer.evaluate(val_loader)
        
        assert 0 <= acc <= 1
        assert 0 <= f1 <= 1
        assert "negative" in report
        assert "neutral" in report
        assert "positive" in report
        
        print(f"\nVal accuracy: {acc*100:.1f}%")
        print(f"Val F1: {f1:.4f}")


class TestBaselineAccuracy:
    """Test that baseline achieves >60% accuracy."""
    
    @pytest.mark.slow
    def test_baseline_accuracy_threshold(self, device):
        """Test baseline achieves >60% accuracy (random = 33%)."""
        if not DATA_PATH.exists():
            pytest.skip("Dataset not found")
        
        from src.data.dataset import load_financial_phrasebank, create_splits, get_class_weights
        from src.data.tokenizer import create_dataloaders_from_df
        from src.models.classifier import create_classifier
        from src.models.trainer import Trainer, TrainingConfig
        
        # Load data
        df = load_financial_phrasebank(str(DATA_PATH))
        train_df, val_df, test_df = create_splits(df)
        
        class_weights = get_class_weights(df)
        weights_list = [class_weights[i] for i in range(3)]
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders_from_df(
            train_df, val_df, test_df,
            model_name="bert",
            max_length=128,
            batch_size=32,
        )
        
        # Create frozen model
        model = create_classifier("bert", freeze_encoder=True, device=device)
        
        # Train for 1 epoch
        config = TrainingConfig(
            learning_rate=1e-3,
            epochs=1,
            device=device,
            class_weights=weights_list,
        )
        
        trainer = Trainer(model, config)
        trainer.train(train_loader, val_loader)
        
        # Evaluate on test set
        _, test_acc, test_f1, _ = trainer.evaluate(test_loader)
        
        print(f"\nTest Accuracy: {test_acc*100:.1f}%")
        print(f"Test F1: {test_f1:.4f}")
        
        # THE KEY TEST: >60% accuracy
        assert test_acc > 0.60, f"Baseline accuracy {test_acc*100:.1f}% should be >60%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])