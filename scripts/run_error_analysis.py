#!/usr/bin/env python3
"""
Run error analysis on trained models.

Analyzes model disagreements, misclassification patterns,
and generates visualizations.

Usage:
    python scripts/run_error_analysis.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.data.dataset import load_financial_phrasebank, create_splits
from src.analysis.error_analysis import (
    get_all_model_predictions,
    find_disagreements,
    find_hard_samples,
    generate_error_report,
    LABEL_NAMES,
)


def main():
    print("=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load test data
    print("\n1. Loading test data...")
    df = load_financial_phrasebank("data/raw/all-data.csv")
    _, _, test_df = create_splits(df)
    
    texts = test_df["text"].tolist()
    labels = test_df["label"].values
    
    print(f"   Test samples: {len(texts)}")
    
    # Get predictions from all models
    print("\n2. Getting predictions from all models...")
    predictions_df = get_all_model_predictions(
        texts=texts,
        labels=labels,
        model_dir="outputs/training",
        device=device,
    )
    
    # Find disagreements
    print("\n3. Analyzing disagreements...")
    disagreements = find_disagreements(predictions_df)
    print(f"   Total disagreements: {len(disagreements)} ({len(disagreements)/len(predictions_df):.1%})")
    
    if len(disagreements) > 0:
        print("\n   Top 5 disagreement examples:")
        for i, (_, row) in enumerate(disagreements.head(5).iterrows()):
            print(f"\n   [{i+1}] \"{row['text'][:80]}...\"")
            print(f"       True: {row['true_label_name']}")
            for model in ["bert", "roberta", "distilbert", "finbert"]:
                pred_col = f"{model}_pred_name"
                if pred_col in row:
                    print(f"       {model.upper()}: {row[pred_col]}")
    
    # Find hard samples
    print("\n4. Finding hard samples (low confidence)...")
    hard_samples = find_hard_samples(predictions_df, threshold=0.6)
    print(f"   Hard samples: {len(hard_samples)} ({len(hard_samples)/len(predictions_df):.1%})")
    
    # Generate full report
    print("\n5. Generating error analysis report...")
    report = generate_error_report(
        predictions_df=predictions_df,
        output_dir="outputs/analysis",
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\nPer-Model Accuracy:")
    for model_name, analysis in report["per_model"].items():
        print(f"  {model_name.upper()}: {analysis['accuracy']:.1%} ({analysis['correct']}/{analysis['total_samples']})")
    
    print(f"\nDisagreements: {report['disagreements']['total']} samples ({report['disagreements']['percentage']:.1%})")
    
    if "domain_insights" in report:
        insights = report["domain_insights"]
        print("\nDomain-Specific Insights (FinBERT vs BERT):")
        print(f"  FinBERT correct, BERT wrong: {insights['finbert_only_correct']}")
        print(f"  BERT correct, FinBERT wrong: {insights['bert_only_correct']}")
        print(f"  Both correct: {insights['both_correct']}")
        print(f"  Both wrong: {insights['both_wrong']}")
    
    print("\nOutputs saved to: outputs/analysis/")
    print("  - error_analysis_report.json")
    print("  - confusion_matrices.png")
    print("  - model_agreement.png")
    print("  - disagreement_examples.csv")
    
    return report


if __name__ == "__main__":
    main()