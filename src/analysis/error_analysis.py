"""
Error analysis for sentiment classification models.

Provides:
- Model disagreement analysis
- Misclassification pattern identification
- Confusion matrix visualization
- Per-class error analysis
"""
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.classifier import SentimentClassifier, MODEL_NAMES
from src.data.tokenizer import get_tokenizer


# Label mappings
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_IDS = {"negative": 0, "neutral": 1, "positive": 2}


def load_trained_model(
    model_name: str,
    checkpoint_path: str,
    device: str = "cpu",
) -> SentimentClassifier:
    """Load a trained model from checkpoint.
    
    Args:
        model_name: One of 'bert', 'roberta', 'distilbert', 'finbert'
        checkpoint_path: Path to .pt checkpoint file
        device: Device to load model on
    
    Returns:
        Loaded SentimentClassifier
    """
    model = SentimentClassifier(
        model_name=model_name,
        num_classes=3,
        freeze_encoder=False,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model


def predict_batch(
    model: SentimentClassifier,
    texts: List[str],
    model_name: str,
    device: str = "cpu",
    batch_size: int = 32,
    max_length: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get predictions and probabilities for a batch of texts.
    
    Args:
        model: Trained model
        texts: List of text strings
        model_name: Model name for tokenizer
        device: Device
        batch_size: Batch size for inference
        max_length: Max sequence length
    
    Returns:
        Tuple of (predictions, probabilities)
    """
    tokenizer = get_tokenizer(model_name)
    model.eval()
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_probs)


def get_all_model_predictions(
    texts: List[str],
    labels: np.ndarray,
    model_dir: str = "outputs/training",
    device: str = "cpu",
) -> pd.DataFrame:
    """Get predictions from all 4 models for comparison.
    
    Args:
        texts: List of text strings
        labels: Ground truth labels
        model_dir: Directory containing model checkpoints
        device: Device
    
    Returns:
        DataFrame with columns: text, true_label, bert_pred, roberta_pred, etc.
    """
    model_dir = Path(model_dir)
    results = {
        "text": texts,
        "true_label": labels,
        "true_label_name": [LABEL_NAMES[l] for l in labels],
    }
    
    for model_name in ["bert", "roberta", "distilbert", "finbert"]:
        checkpoint_path = model_dir / f"{model_name}_finetuned.pt"
        
        if not checkpoint_path.exists():
            print(f"Warning: {checkpoint_path} not found, skipping {model_name}")
            continue
        
        print(f"Loading {model_name}...")
        model = load_trained_model(model_name, str(checkpoint_path), device)
        
        preds, probs = predict_batch(model, texts, model_name, device)
        
        results[f"{model_name}_pred"] = preds
        results[f"{model_name}_pred_name"] = [LABEL_NAMES[p] for p in preds]
        results[f"{model_name}_confidence"] = probs.max(axis=1)
        
        # Store probabilities for each class
        for i, label_name in LABEL_NAMES.items():
            results[f"{model_name}_prob_{label_name}"] = probs[:, i]
        
        # Clean up
        del model
        torch.cuda.empty_cache() if device == "cuda" else None
    
    return pd.DataFrame(results)


def find_disagreements(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Find samples where models disagree.
    
    Args:
        predictions_df: DataFrame from get_all_model_predictions
    
    Returns:
        DataFrame with only disagreement samples, sorted by disagreement count
    """
    pred_cols = [c for c in predictions_df.columns if c.endswith("_pred") and not c.endswith("_pred_name")]
    
    if len(pred_cols) < 2:
        raise ValueError("Need at least 2 models for disagreement analysis")
    
    # Count unique predictions per sample
    predictions_df["num_unique_preds"] = predictions_df[pred_cols].nunique(axis=1)
    
    # Find where not all models agree
    disagreements = predictions_df[predictions_df["num_unique_preds"] > 1].copy()
    
    # Add disagreement details
    def get_disagreement_type(row):
        preds = [row[c] for c in pred_cols]
        unique_preds = set(preds)
        if len(unique_preds) == 2:
            return "2-way split"
        elif len(unique_preds) == 3:
            return "3-way split"
        else:
            return f"{len(unique_preds)}-way split"
    
    disagreements["disagreement_type"] = disagreements.apply(get_disagreement_type, axis=1)
    
    return disagreements.sort_values("num_unique_preds", ascending=False)


def find_hard_samples(predictions_df: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    """Find samples that are hard for all models (low confidence).
    
    Args:
        predictions_df: DataFrame from get_all_model_predictions
        threshold: Confidence threshold below which sample is "hard"
    
    Returns:
        DataFrame with hard samples
    """
    conf_cols = [c for c in predictions_df.columns if c.endswith("_confidence")]
    
    # Average confidence across models
    predictions_df["avg_confidence"] = predictions_df[conf_cols].mean(axis=1)
    
    # Find low confidence samples
    hard_samples = predictions_df[predictions_df["avg_confidence"] < threshold].copy()
    
    return hard_samples.sort_values("avg_confidence")


def analyze_misclassifications(
    predictions_df: pd.DataFrame,
    model_name: str,
) -> Dict:
    """Analyze misclassification patterns for a specific model.
    
    Args:
        predictions_df: DataFrame from get_all_model_predictions
        model_name: Model to analyze
    
    Returns:
        Dictionary with misclassification analysis
    """
    pred_col = f"{model_name}_pred"
    
    if pred_col not in predictions_df.columns:
        raise ValueError(f"Model {model_name} not found in predictions")
    
    true_labels = predictions_df["true_label"].values
    pred_labels = predictions_df[pred_col].values
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Find misclassified samples
    misclassified_mask = true_labels != pred_labels
    misclassified_df = predictions_df[misclassified_mask].copy()
    
    # Analyze error patterns
    error_patterns = {}
    for true_label in range(3):
        for pred_label in range(3):
            if true_label != pred_label:
                mask = (true_labels == true_label) & (pred_labels == pred_label)
                count = mask.sum()
                if count > 0:
                    key = f"{LABEL_NAMES[true_label]}_as_{LABEL_NAMES[pred_label]}"
                    error_patterns[key] = {
                        "count": int(count),
                        "examples": predictions_df[mask]["text"].head(3).tolist(),
                    }
    
    return {
        "model_name": model_name,
        "total_samples": len(predictions_df),
        "correct": int((true_labels == pred_labels).sum()),
        "incorrect": int(misclassified_mask.sum()),
        "accuracy": float((true_labels == pred_labels).mean()),
        "confusion_matrix": cm.tolist(),
        "error_patterns": error_patterns,
    }


def plot_confusion_matrices(
    predictions_df: pd.DataFrame,
    output_path: str = "outputs/analysis/confusion_matrices.png",
) -> None:
    """Plot confusion matrices for all models side by side.
    
    Args:
        predictions_df: DataFrame from get_all_model_predictions
        output_path: Path to save figure
    """
    pred_cols = [c for c in predictions_df.columns if c.endswith("_pred") and not c.endswith("_pred_name")]
    model_names = [c.replace("_pred", "") for c in pred_cols]
    
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    true_labels = predictions_df["true_label"].values
    labels = ["negative", "neutral", "positive"]
    
    for ax, model_name in zip(axes, model_names):
        pred_labels = predictions_df[f"{model_name}_pred"].values
        cm = confusion_matrix(true_labels, pred_labels)
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized,
            annot=cm,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar=False,
        )
        
        accuracy = (true_labels == pred_labels).mean()
        ax.set_title(f"{model_name.upper()}\nAccuracy: {accuracy:.1%}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    
    plt.tight_layout()
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Confusion matrices saved to: {output_path}")


def plot_model_agreement(
    predictions_df: pd.DataFrame,
    output_path: str = "outputs/analysis/model_agreement.png",
) -> None:
    """Plot model agreement statistics.
    
    Args:
        predictions_df: DataFrame from get_all_model_predictions
        output_path: Path to save figure
    """
    pred_cols = [c for c in predictions_df.columns if c.endswith("_pred") and not c.endswith("_pred_name")]
    
    # Count agreement levels
    predictions_df["num_unique_preds"] = predictions_df[pred_cols].nunique(axis=1)
    
    agreement_counts = predictions_df["num_unique_preds"].value_counts().sort_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart of agreement levels
    ax1 = axes[0]
    bars = ax1.bar(
        [f"{n} predictions" for n in agreement_counts.index],
        agreement_counts.values,
        color=["green", "orange", "red", "darkred"][:len(agreement_counts)],
    )
    ax1.set_xlabel("Number of Unique Predictions")
    ax1.set_ylabel("Number of Samples")
    ax1.set_title("Model Agreement Distribution")
    
    # Add percentage labels
    total = len(predictions_df)
    for bar, count in zip(bars, agreement_counts.values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{count/total:.1%}",
            ha="center",
        )
    
    # Pairwise agreement heatmap
    ax2 = axes[1]
    model_names = [c.replace("_pred", "") for c in pred_cols]
    n_models = len(model_names)
    
    agreement_matrix = np.zeros((n_models, n_models))
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            agreement = (predictions_df[f"{m1}_pred"] == predictions_df[f"{m2}_pred"]).mean()
            agreement_matrix[i, j] = agreement
    
    sns.heatmap(
        agreement_matrix,
        annot=True,
        fmt=".1%",
        cmap="RdYlGn",
        xticklabels=[m.upper() for m in model_names],
        yticklabels=[m.upper() for m in model_names],
        ax=ax2,
        vmin=0.7,
        vmax=1.0,
    )
    ax2.set_title("Pairwise Model Agreement")
    
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Model agreement plot saved to: {output_path}")


def generate_error_report(
    predictions_df: pd.DataFrame,
    output_dir: str = "outputs/analysis",
) -> Dict:
    """Generate comprehensive error analysis report.
    
    Args:
        predictions_df: DataFrame from get_all_model_predictions
        output_dir: Directory to save outputs
    
    Returns:
        Dictionary with full analysis
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pred_cols = [c for c in predictions_df.columns if c.endswith("_pred") and not c.endswith("_pred_name")]
    model_names = [c.replace("_pred", "") for c in pred_cols]
    
    report = {
        "total_samples": len(predictions_df),
        "models_analyzed": model_names,
    }
    
    # Per-model analysis
    report["per_model"] = {}
    for model_name in model_names:
        report["per_model"][model_name] = analyze_misclassifications(predictions_df, model_name)
    
    # Disagreement analysis
    disagreements = find_disagreements(predictions_df)
    report["disagreements"] = {
        "total": len(disagreements),
        "percentage": len(disagreements) / len(predictions_df),
        "by_type": disagreements["disagreement_type"].value_counts().to_dict() if len(disagreements) > 0 else {},
    }
    
    # Hard samples
    hard_samples = find_hard_samples(predictions_df)
    report["hard_samples"] = {
        "total": len(hard_samples),
        "percentage": len(hard_samples) / len(predictions_df),
        "examples": hard_samples.head(5)[["text", "true_label_name", "avg_confidence"]].to_dict("records") if len(hard_samples) > 0 else [],
    }
    
    # Domain-specific insights
    # Check if FinBERT performs better on certain patterns
    if "finbert_pred" in predictions_df.columns and "bert_pred" in predictions_df.columns:
        finbert_correct = predictions_df["finbert_pred"] == predictions_df["true_label"]
        bert_correct = predictions_df["bert_pred"] == predictions_df["true_label"]
        
        # Cases where FinBERT is right but BERT is wrong
        finbert_advantage = finbert_correct & ~bert_correct
        report["domain_insights"] = {
            "finbert_only_correct": int(finbert_advantage.sum()),
            "bert_only_correct": int((~finbert_correct & bert_correct).sum()),
            "both_correct": int((finbert_correct & bert_correct).sum()),
            "both_wrong": int((~finbert_correct & ~bert_correct).sum()),
            "finbert_advantage_examples": predictions_df[finbert_advantage]["text"].head(5).tolist(),
        }
    
    # Save report
    report_path = output_dir / "error_analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Error analysis report saved to: {report_path}")
    
    # Generate visualizations
    plot_confusion_matrices(predictions_df, str(output_dir / "confusion_matrices.png"))
    plot_model_agreement(predictions_df, str(output_dir / "model_agreement.png"))
    
    # Save disagreement examples
    if len(disagreements) > 0:
        disagreements_path = output_dir / "disagreement_examples.csv"
        disagreements.head(100).to_csv(disagreements_path, index=False)
        print(f"Disagreement examples saved to: {disagreements_path}")
    
    return report