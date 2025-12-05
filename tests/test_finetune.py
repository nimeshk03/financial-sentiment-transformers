"""
Tests for Milestone 2.2: Fine-tuning all models.

Verifies:
- All 4 models can be created and run forward pass
- Training loss decreases over epochs
- No NaN gradients during training
"""
import pytest
import torch
from pathlib import Path

# Skip if data not available
DATA_PATH = Path("data/raw/all-data.csv")
pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason=f"Dataset not found at {DATA_PATH}"
)


class TestAllModelsCreation:
    """Test that all 4 models can be created."""
    
    @pytest.mark.parametrize("model_name", ["bert", "roberta", "distilbert", "finbert"])
    def test_model_creation(self, model_name):
        """Test model can be instantiated."""
        from src.models.classifier import SentimentClassifier, MODEL_NAMES
        
        model = SentimentClassifier(
            model_name=model_name,
            num_classes=3,
            freeze_encoder=False,
        )
        
        assert model is not None
        assert model.model_name == model_name
        print(f"\n{model_name}: {model.get_num_total_params():,} params")
    
    @pytest.mark.parametrize("model_name", ["bert", "roberta", "distilbert", "finbert"])
    def test_forward_pass(self, model_name):
        """Test forward pass works for all models."""
        from src.models.classifier import SentimentClassifier
        
        model = SentimentClassifier(
            model_name=model_name,
            num_classes=3,
            freeze_encoder=False,
        )
        
        # Dummy input
        batch_size = 2
        seq_length = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        
        assert logits.shape == (batch_size, 3)
        assert not torch.isnan(logits).any(), "NaN in output"
        print(f"\n{model_name}: forward pass OK, output shape {logits.shape}")


class TestTrainingBehavior:
    """Test training behavior for fine-tuning."""
    
    @pytest.fixture
    def sample_data(self):
        """Create small sample dataset for testing."""
        from src.data.dataset import load_financial_phrasebank, create_splits
        
        df = load_financial_phrasebank(str(DATA_PATH))
        train_df, val_df, test_df = create_splits(df)
        
        # Use small subset for testing
        return train_df.head(100), val_df.head(50), test_df.head(50)
    
    def test_loss_decreases_bert(self, sample_data):
        """Test that training loss decreases for BERT."""
        self._test_loss_decreases("bert", sample_data)
    
    def test_loss_decreases_distilbert(self, sample_data):
        """Test that training loss decreases for DistilBERT (fastest)."""
        self._test_loss_decreases("distilbert", sample_data)
    
    def _test_loss_decreases(self, model_name, sample_data):
        """Helper to test loss decrease."""
        from src.data.tokenizer import create_dataloaders_from_df
        from src.models.classifier import SentimentClassifier
        from src.models.trainer import Trainer, TrainingConfig
        
        train_df, val_df, test_df = sample_data
        
        # Create small dataloaders
        train_loader, val_loader, _ = create_dataloaders_from_df(
            train_df, val_df, test_df,
            model_name=model_name,
            max_length=64,  # Shorter for speed
            batch_size=8,
        )
        
        # Create model
        model = SentimentClassifier(
            model_name=model_name,
            num_classes=3,
            freeze_encoder=False,
        )
        
        # Train for 2 epochs
        config = TrainingConfig(
            learning_rate=2e-5,
            epochs=2,
            warmup_ratio=0.1,
            device="cpu",
        )
        
        trainer = Trainer(model, config)
        result = trainer.train(train_loader, val_loader)
        
        # Check loss decreased
        assert len(result.train_losses) == 2
        assert result.train_losses[1] < result.train_losses[0], \
            f"Loss should decrease: {result.train_losses}"
        
        print(f"\n{model_name} loss: {result.train_losses[0]:.4f} -> {result.train_losses[1]:.4f}")


class TestNoNaNGradients:
    """Test that gradients don't become NaN."""
    
    @pytest.mark.parametrize("model_name", ["bert", "distilbert"])
    def test_no_nan_gradients(self, model_name):
        """Verify no NaN gradients during training step."""
        from src.models.classifier import SentimentClassifier
        
        model = SentimentClassifier(
            model_name=model_name,
            num_classes=3,
            freeze_encoder=False,
        )
        
        # Dummy batch
        input_ids = torch.randint(0, 1000, (4, 32))
        attention_mask = torch.ones(4, 32)
        labels = torch.tensor([0, 1, 2, 1])
        
        # Forward + backward
        logits = model(input_ids, attention_mask)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        
        # Check gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), \
                    f"NaN gradient in {name}"
        
        print(f"\n{model_name}: No NaN gradients")


@pytest.mark.slow
class TestFullTraining:
    """Full training tests (slow, run with pytest -m slow)."""
    
    def test_all_models_trainable(self):
        """Verify all 4 models can complete training."""
        from src.data.dataset import load_financial_phrasebank, create_splits
        from src.data.tokenizer import create_dataloaders_from_df
        from src.models.classifier import SentimentClassifier
        from src.models.trainer import Trainer, TrainingConfig
        
        df = load_financial_phrasebank(str(DATA_PATH))
        train_df, val_df, test_df = create_splits(df)
        
        # Use subset for faster testing
        train_df = train_df.head(200)
        val_df = val_df.head(100)
        
        results = {}
        
        for model_name in ["bert", "roberta", "distilbert", "finbert"]:
            print(f"\nTesting {model_name}...")
            
            train_loader, val_loader, _ = create_dataloaders_from_df(
                train_df, val_df, test_df.head(50),
                model_name=model_name,
                max_length=64,
                batch_size=8,
            )
            
            model = SentimentClassifier(
                model_name=model_name,
                num_classes=3,
                freeze_encoder=False,
            )
            
            config = TrainingConfig(
                learning_rate=2e-5,
                epochs=1,
                device="cpu",
            )
            
            trainer = Trainer(model, config)
            result = trainer.train(train_loader, val_loader)
            
            results[model_name] = {
                "train_loss": result.train_losses[-1],
                "val_accuracy": result.best_accuracy,
            }
            
            # Basic sanity check
            assert result.train_losses[-1] < 3.0, f"{model_name} loss too high"
            assert result.best_accuracy > 0.3, f"{model_name} accuracy too low"
        
        print("\n" + "="*50)
        print("All models trained successfully!")
        for name, r in results.items():
            print(f"  {name}: loss={r['train_loss']:.4f}, acc={r['val_accuracy']:.2%}")