"""
Train BERT baseline with frozen encoder.

This establishes a baseline by only training the classification head
while keeping BERT's pre-trained weights frozen.

Usage:
    python scripts/train_baseline.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.data.dataset import load_financial_phrasebank, create_splits, get_class_weights
from src.data.tokenizer import create_dataloaders_from_df
from src.models.classifier import create_classifier
from src.models.trainer import Trainer, TrainingConfig


def main():
    print("=" * 60)
    print("BERT BASELINE - Frozen Encoder")
    print("=" * 60)
    
    # Configuration
    MODEL_NAME = "bert"
    MAX_LENGTH = 128
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-4  # Slightly lower for better convergence
    EPOCHS = 3  # More epochs for frozen encoder to learn
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Load data
    print("\n1. Loading data...")
    df = load_financial_phrasebank("data/raw/all-data.csv")
    train_df, val_df, test_df = create_splits(df)
    print(f"   Train: {len(train_df):,} samples")
    print(f"   Val:   {len(val_df):,} samples")
    print(f"   Test:  {len(test_df):,} samples")
    
    # Get class weights for imbalanced data
    class_weights = get_class_weights(df)
    weights_list = [class_weights[i] for i in range(3)]
    print(f"   Class weights: {[f'{w:.2f}' for w in weights_list]}")
    
    # Create dataloaders
    print("\n2. Tokenizing data...")
    train_loader, val_loader, test_loader = create_dataloaders_from_df(
        train_df, val_df, test_df,
        model_name=MODEL_NAME,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    
    # Create model with frozen encoder
    print("\n3. Creating model...")
    model = create_classifier(
        model_name=MODEL_NAME,
        num_classes=3,
        freeze_encoder=True,  # FROZEN for baseline
        device=device,
    )
    print(f"   Total parameters:     {model.get_num_total_params():,}")
    print(f"   Trainable parameters: {model.get_num_trainable_params():,}")
    print(f"   Encoder frozen: {model.freeze_encoder}")
    
    # Setup trainer
    print("\n4. Training...")
    config = TrainingConfig(
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device=device,
        class_weights=weights_list,
    )
    
    trainer = Trainer(model, config)
    result = trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    print("\n5. Evaluating on test set...")
    test_loss, test_acc, test_f1, report = trainer.evaluate(test_loader)
    
    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {test_acc*100:.1f}%")
    print(f"Test F1 Score: {test_f1:.4f}")
    print("\nPer-class metrics:")
    for label in ["negative", "neutral", "positive"]:
        p = report[label]["precision"]
        r = report[label]["recall"]
        f = report[label]["f1-score"]
        print(f"  {label:10s}: P={p:.2f} R={r:.2f} F1={f:.2f}")
    
    # Check if baseline passes (>60% accuracy)
    print("\n" + "=" * 60)
    if test_acc > 0.60:
        print(f"BASELINE PASSED: {test_acc*100:.1f}% > 60%")
    else:
        print(f"BASELINE FAILED: {test_acc*100:.1f}% <= 60%")
    print("=" * 60)
    
    return test_acc, test_f1


if __name__ == "__main__":
    main()