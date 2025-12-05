# Financial Sentiment Analysis - Multi-Model Transformer Comparison

A comprehensive pipeline benchmarking **four transformer architectures** (BERT, RoBERTa, DistilBERT, FinBERT) on financial sentiment classification, with trading signal generation and backtesting capabilities.

## Project Overview

This project demonstrates:
- **Architecture Comparison**: Fair benchmarking of multiple transformer models
- **Domain Adaptation**: Comparing general-purpose vs. finance-specific models
- **Production Thinking**: Balancing accuracy, speed, and resource constraints
- **End-to-End Pipeline**: From raw text to trading signals

## Models Under Comparison

| Model | Parameters | HuggingFace ID | Key Insight |
|-------|------------|----------------|-------------|
| BERT | 110M | `bert-base-uncased` | Baseline transformer |
| RoBERTa | 125M | `roberta-base` | Better training procedure |
| DistilBERT | 66M | `distilbert-base-uncased` | 40% smaller, 2x faster |
| FinBERT | 110M | `ProsusAI/finbert` | Domain-specific pre-training |

## Current Progress

### Week 1: Data & Baseline [COMPLETED]

- [x] **Milestone 1.1**: Environment Setup (Docker, dependencies)
- [x] **Milestone 1.2**: Data Loading & Exploration (Financial PhraseBank)
- [x] **Milestone 1.3**: Tokenization Pipeline (unified for all models)
- [x] **Milestone 1.4**: BERT Baseline with frozen encoder (**62.4% accuracy**)

### Week 2: Model Comparison [IN PROGRESS]

- [x] **Milestone 2.1**: Standardized Training Setup
  - Class weights for imbalanced data
  - Learning rate scheduler (linear warmup + decay)
  - Gradient clipping
  - Early stopping
- [ ] **Milestone 2.2**: Fine-tune all four models
- [ ] **Milestone 2.3**: Experiment tracking with W&B
- [ ] **Milestone 2.4**: Build comparison table

## Project Structure

```
financial_sentiment_analysis/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── implementation_plan.md      # Detailed project roadmap
├── src/
│   ├── data/
│   │   ├── dataset.py          # Data loading, splits, class weights
│   │   └── tokenizer.py        # Unified tokenization pipeline
│   ├── models/
│   │   ├── classifier.py       # SentimentClassifier (supports all 4 models)
│   │   └── trainer.py          # Training loop, scheduler, early stopping
│   ├── inference/              # (Week 3)
│   ├── backtesting/            # (Week 3)
│   └── utils/
├── scripts/
│   └── train_baseline.py       # BERT baseline training script
├── tests/
│   ├── test_environment.py     # Environment verification
│   ├── test_data_loading.py    # Data loading tests
│   └── test_baseline.py        # Model and trainer tests
├── data/
│   └── raw/                    # Financial PhraseBank dataset
├── models/                     # Saved model checkpoints
├── outputs/                    # Training outputs, metrics
├── notebooks/                  # EDA and experimentation
└── app/                        # Streamlit dashboard (Week 4)
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- ~4GB disk space for models

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/nimeshk03/financial-sentiment-transformers.git
   cd financial-sentiment-transformers
   ```

2. **Download the dataset**
   
   Download Financial PhraseBank from [Kaggle](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) and place `all-data.csv` in `data/raw/`.

3. **Build and run with Docker**
   ```bash
   docker compose build
   docker compose run --rm app pytest tests/test_environment.py -v
   ```

### Run Baseline Training

```bash
docker compose run --rm app python scripts/train_baseline.py
```

Expected output: ~62% accuracy with frozen BERT encoder.

### Run Tests

```bash
# All tests
docker compose run --rm app pytest -v

# Specific test file
docker compose run --rm app pytest tests/test_baseline.py -v
```

## Technical Details

### Training Configuration

All models use standardized hyperparameters for fair comparison:

```python
TrainingConfig(
    learning_rate=2e-5,
    batch_size=16,
    epochs=3,
    warmup_ratio=0.1,      # 10% of steps for warmup
    weight_decay=0.01,
    max_grad_norm=1.0,     # Gradient clipping
    early_stopping_patience=2,
)
```

### Dataset

- **Financial PhraseBank**: ~4,845 financial news sentences
- **Classes**: Negative (0), Neutral (1), Positive (2)
- **Split**: 80% train, 10% validation, 10% test (stratified)
- **Class weights**: Applied to handle imbalance (neutral is majority)

### Key Components

#### SentimentClassifier (`src/models/classifier.py`)
- Wraps any HuggingFace transformer
- Supports frozen/unfrozen encoder
- Unified interface for all 4 models

#### Trainer (`src/models/trainer.py`)
- Linear warmup + decay learning rate scheduler
- Gradient clipping
- Early stopping
- Weighted cross-entropy loss
- Comprehensive metrics (accuracy, F1, per-class)

#### Tokenizer Pipeline (`src/data/tokenizer.py`)
- Unified tokenization for all models
- PyTorch Dataset and DataLoader creation
- Configurable max length and batch size

## Compute Strategy

| Task | Environment | Reason |
|------|-------------|--------|
| Development | Local (CPU) | Fast iteration |
| Single model training | Local (CPU) | ~30-60 min acceptable |
| All 4 models | Cloud GPU | Parallel, faster |
| Inference | Local (CPU) | Fast enough |

## License

MIT

## Author

Nimesh K
