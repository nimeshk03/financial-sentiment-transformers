#!/usr/bin/env python3
"""
Fine-tune all four transformer models for sentiment classification.

Models: BERT, RoBERTa, DistilBERT, FinBERT
Uses standardized hyperparameters for fair comparison.

Usage:
    python scripts/train_all_models.py
    python scripts/train_all_models.py --models bert roberta  # Specific models
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.data.dataset import load_financial_phrasebank, create_splits, get_class_weights
from src.data.tokenizer import create_dataloaders_from_df
from src.models.classifier import SentimentClassifier, MODEL_NAMES
from src.models.trainer import Trainer, TrainingConfig


# Standard hyperparameters (from implementation plan)
STANDARD_CONFIG = {
    "learning_rate": 2e-5,
    "batch_size": 16,
    "epochs": 3,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "max_length": 128,
}


def train_single_model(
    model_name: str,
    train_df,
    val_df,
    test_df,
    class_weights: list,
    device: str,
    output_dir: Path,
) -> dict:
    """Train a single model and return results.
    
    Args:
        model_name: One of 'bert', 'roberta', 'distilbert', 'finbert'
        train_df, val_df, test_df: Data splits
        class_weights: Class weights for imbalanced data
        device: Device to train on
        output_dir: Directory to save results
    
    Returns:
        Dictionary with training results and metrics
    """
    print(f"\n{'='*60}")
    print(f"Training: {model_name.upper()}")
    print(f"HuggingFace model: {MODEL_NAMES[model_name]}")
    print(f"{'='*60}")
    
    # Create dataloaders (model-specific tokenizer)
    print("\n1. Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders_from_df(
        train_df, val_df, test_df,
        model_name=model_name,
        max_length=STANDARD_CONFIG["max_length"],
        batch_size=STANDARD_CONFIG["batch_size"],
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Create model (unfrozen for fine-tuning)
    print("\n2. Creating model...")
    model = SentimentClassifier(
        model_name=model_name,
        num_classes=3,
        freeze_encoder=False,  # Fine-tuning: train everything
    )
    
    total_params = model.get_num_total_params()
    trainable_params = model.get_num_trainable_params()
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Training config
    config = TrainingConfig(
        learning_rate=STANDARD_CONFIG["learning_rate"],
        epochs=STANDARD_CONFIG["epochs"],
        warmup_ratio=STANDARD_CONFIG["warmup_ratio"],
        weight_decay=STANDARD_CONFIG["weight_decay"],
        max_grad_norm=STANDARD_CONFIG["max_grad_norm"],
        batch_size=STANDARD_CONFIG["batch_size"],
        device=device,
        class_weights=class_weights,
        early_stopping_patience=2,
    )
    
    # Train
    print("\n3. Training...")
    trainer = Trainer(model, config)
    train_result = trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    print("\n4. Evaluating on test set...")
    test_loss, test_acc, test_f1, report = trainer.evaluate(test_loader)
    
    # Measure inference speed
    print("\n5. Measuring inference speed...")
    inference_time = measure_inference_speed(model, test_loader, device)
    
    # Compile results
    results = {
        "model_name": model_name,
        "hf_model": MODEL_NAMES[model_name],
        "total_params": total_params,
        "trainable_params": trainable_params,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "test_loss": test_loss,
        "best_val_accuracy": train_result.best_accuracy,
        "best_val_f1": train_result.best_f1,
        "training_time_seconds": train_result.training_time,
        "inference_ms_per_sample": inference_time,
        "classification_report": report,
        "train_losses": train_result.train_losses,
        "val_losses": train_result.val_losses,
        "val_accuracies": train_result.val_accuracies,
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Test Accuracy:     {test_acc*100:.2f}%")
    print(f"Test F1 (weighted): {test_f1:.4f}")
    print(f"Training Time:     {train_result.training_time:.1f}s")
    print(f"Inference Speed:   {inference_time:.2f} ms/sample")
    print(f"Parameters:        {total_params:,}")
    
    # Save model checkpoint
    model_path = output_dir / f"{model_name}_finetuned.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": STANDARD_CONFIG,
        "results": results,
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    return results


def measure_inference_speed(
    model: SentimentClassifier,
    data_loader,
    device: str,
    num_batches: int = 10,
) -> float:
    """Measure inference speed in ms per sample.
    
    Args:
        model: Trained model
        data_loader: Test data loader
        device: Device
        num_batches: Number of batches to measure
    
    Returns:
        Average inference time in milliseconds per sample
    """
    import time
    
    model.eval()
    model.to(device)
    
    total_time = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_size = input_ids.size(0)
            
            # Warm up (first batch)
            if i == 0:
                _ = model(input_ids, attention_mask)
                continue
            
            start = time.perf_counter()
            _ = model(input_ids, attention_mask)
            end = time.perf_counter()
            
            total_time += (end - start)
            total_samples += batch_size
    
    if total_samples == 0:
        return 0.0
    
    # Convert to milliseconds per sample
    return (total_time / total_samples) * 1000


def generate_comparison_table(all_results: list, output_dir: Path):
    """Generate comparison table in markdown and CSV formats.
    
    Args:
        all_results: List of result dictionaries
        output_dir: Directory to save outputs
    """
    # Sort by accuracy (descending)
    sorted_results = sorted(all_results, key=lambda x: x["test_accuracy"], reverse=True)
    
    # Markdown table
    md_lines = [
        "# Model Comparison Results",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary Table",
        "",
        "| Model | Parameters | Accuracy | F1 | Inference (ms) | Training Time |",
        "|-------|------------|----------|-----|----------------|---------------|",
    ]
    
    for r in sorted_results:
        params_m = r["total_params"] / 1e6
        acc = r["test_accuracy"] * 100
        f1 = r["test_f1"]
        inf_ms = r["inference_ms_per_sample"]
        train_min = r["training_time_seconds"] / 60
        
        md_lines.append(
            f"| {r['model_name']} | {params_m:.0f}M | {acc:.1f}% | {f1:.3f} | {inf_ms:.1f}ms | {train_min:.1f}min |"
        )
    
    md_lines.extend([
        "",
        "## Per-Class Metrics",
        "",
    ])
    
    for r in sorted_results:
        md_lines.append(f"### {r['model_name'].upper()}")
        md_lines.append("")
        md_lines.append("| Class | Precision | Recall | F1-Score |")
        md_lines.append("|-------|-----------|--------|----------|")
        
        for label in ["negative", "neutral", "positive"]:
            metrics = r["classification_report"][label]
            md_lines.append(
                f"| {label} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1-score']:.3f} |"
            )
        md_lines.append("")
    
    # Save markdown
    md_path = output_dir / "comparison_results.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"\nMarkdown report saved to: {md_path}")
    
    # Save JSON (full results)
    json_path = output_dir / "comparison_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"JSON results saved to: {json_path}")
    
    # Print summary to console
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"{'Model':<12} {'Params':<10} {'Accuracy':<10} {'F1':<8} {'Inference':<12} {'Training':<10}")
    print("-"*70)
    for r in sorted_results:
        params_m = r["total_params"] / 1e6
        acc = r["test_accuracy"] * 100
        f1 = r["test_f1"]
        inf_ms = r["inference_ms_per_sample"]
        train_min = r["training_time_seconds"] / 60
        print(f"{r['model_name']:<12} {params_m:.0f}M{'':<6} {acc:.1f}%{'':<5} {f1:.3f}{'':<4} {inf_ms:.1f}ms{'':<7} {train_min:.1f}min")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune transformer models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["bert", "roberta", "distilbert", "finbert"],
        choices=["bert", "roberta", "distilbert", "finbert"],
        help="Models to train (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/training",
        help="Output directory for results",
    )
    args = parser.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("MULTI-MODEL FINE-TUNING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Models to train: {args.models}")
    print(f"Output directory: {output_dir}")
    print(f"\nStandard hyperparameters:")
    for k, v in STANDARD_CONFIG.items():
        print(f"  {k}: {v}")
    
    # Load data once
    print("\n" + "="*60)
    print("Loading dataset...")
    print("="*60)
    df = load_financial_phrasebank("data/raw/all-data.csv")
    train_df, val_df, test_df = create_splits(df)
    
    print(f"Train: {len(train_df):,} samples")
    print(f"Val:   {len(val_df):,} samples")
    print(f"Test:  {len(test_df):,} samples")
    
    # Class weights
    class_weights_dict = get_class_weights(df)
    class_weights = [class_weights_dict[i] for i in range(3)]
    print(f"Class weights: {[f'{w:.2f}' for w in class_weights]}")
    
    # Train each model
    all_results = []
    for model_name in args.models:
        try:
            results = train_single_model(
                model_name=model_name,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                class_weights=class_weights,
                device=device,
                output_dir=output_dir,
            )
            all_results.append(results)
        except Exception as e:
            print(f"\nERROR training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate comparison
    if all_results:
        generate_comparison_table(all_results, output_dir)
    
    print("\nTraining complete!")
    return all_results


if __name__ == "__main__":
    main()