# Financial Sentiment Analysis - Multi-Model Transformer Comparison

A comprehensive pipeline benchmarking **four transformer architectures** (BERT, RoBERTa, DistilBERT, FinBERT) on financial sentiment classification, with trading signal generation and backtesting capabilities.

## Project Overview

This project demonstrates:
- **Architecture Comparison**: Fair benchmarking of multiple transformer models
- **Domain Adaptation**: Comparing general-purpose vs. finance-specific models
- **Production Thinking**: Balancing accuracy, speed, and resource constraints
- **End-to-End Pipeline**: From raw text to trading signals

## Results

| Model | Parameters | Accuracy | F1 | Inference | Training |
|-------|------------|----------|-----|-----------|----------|
| **FinBERT** | 109M | **87.2%** | 0.873 | 0.7ms | 4.2min |
| RoBERTa | 125M | 84.5% | 0.846 | 0.5ms | 4.3min |
| BERT | 109M | 84.1% | 0.842 | 0.5ms | 4.1min |
| DistilBERT | 66M | 81.4% | 0.816 | 0.2ms | 2.1min |

**Key Findings**:
- Domain-specific **FinBERT outperforms** general models by 3-6%
- **DistilBERT** achieves 93% of FinBERT's accuracy with 40% fewer parameters and 3.5x faster inference
- All models show strong neutral class performance (majority class)

## Models Under Comparison

| Model | Parameters | HuggingFace ID | Key Insight |
|-------|------------|----------------|-------------|
| BERT | 110M | `bert-base-uncased` | Baseline transformer |
| RoBERTa | 125M | `roberta-base` | Better training procedure |
| DistilBERT | 66M | `distilbert-base-uncased` | 40% smaller, 2x faster |
| FinBERT | 110M | `ProsusAI/finbert` | Domain-specific pre-training |

## Current Progress

### Phase 1 : Data & Baseline [COMPLETED]

- [x] **Milestone 1.1**: Environment Setup (Docker, dependencies)
- [x] **Milestone 1.2**: Data Loading & Exploration (Financial PhraseBank)
- [x] **Milestone 1.3**: Tokenization Pipeline (unified for all models)
- [x] **Milestone 1.4**: BERT Baseline with frozen encoder (62.4% accuracy)

### Phase 2 : Model Comparison [COMPLETED]

- [x] **Milestone 2.1**: Standardized Training Setup
  - Class weights for imbalanced data
  - Learning rate scheduler (linear warmup + decay)
  - Gradient clipping & early stopping
- [x] **Milestone 2.2**: Fine-tune all four models (see Results above)
- [x] **Milestone 2.4**: Build comparison table

### Phase 3: Analysis & Trading Signals [COMPLETED]

- [x] **Milestone 3.1**: Error Analysis
  - 87 disagreements (17.9%) between models
  - FinBERT outperforms BERT on 30 samples where BERT fails
  - Confusion matrices and agreement visualizations generated
- [x] **Milestone 3.2**: Model Selection for Deployment
  - FinBERT recommended for accuracy-critical applications
  - DistilBERT for speed/cost-critical applications
- [x] **Milestone 3.3**: News Ingestion Pipeline
  - Yahoo Finance news fetching for 10 tickers
  - Headline cleaning and deduplication
  - FinBERT sentiment predictions
- [x] **Milestone 3.4**: Sentiment Aggregation
  - Daily sentiment scores with rolling averages (3d, 7d)
  - Sentiment momentum calculation
  - Price data alignment for backtesting
- [x] **Milestone 3.5**: Backtesting Framework
  - 4 trading strategies implemented
  - Best strategy: Rolling 3d sentiment (+0.32% return, 9.07 Sharpe)

### Phase 4: Deployment & Documentation [IN PROGRESS]

- [x] **Milestone 4.1**: Streamlit Dashboard
  - Model comparison visualizations
  - Live sentiment by ticker
  - Backtest results display
- [ ] **Milestone 4.2**: Documentation
- [ ] **Milestone 4.3**: Deployment to Hugging Face Spaces
- [ ] **Milestone 4.4**: Demo Video

## Project Structure

```
financial_sentiment_analysis/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── src/
│   ├── data/
│   │   ├── dataset.py          # Data loading, splits, class weights
│   │   └── tokenizer.py        # Unified tokenization pipeline
│   ├── models/
│   │   ├── classifier.py       # SentimentClassifier (supports all 4 models)
│   │   └── trainer.py          # Training loop, scheduler, early stopping
│   ├── analysis/
│   │   └── error_analysis.py   # Model comparison and error analysis
│   ├── news/
│   │   ├── fetcher.py          # Yahoo Finance news fetching
│   │   └── processor.py        # Headline cleaning and processing
│   ├── inference/
│   │   └── predictor.py        # Sentiment prediction pipeline
│   ├── signals/
│   │   ├── aggregator.py       # Daily sentiment aggregation
│   │   └── prices.py           # Price fetching and returns
│   ├── backtesting/
│   │   ├── strategy.py         # Trading strategies
│   │   └── metrics.py          # Performance metrics
│   └── utils/
├── scripts/
│   ├── train_baseline.py       # BERT baseline training script
│   ├── train_all_models.py     # Multi-model fine-tuning script
│   ├── run_error_analysis.py   # Error analysis script
│   ├── run_news_pipeline.py    # News sentiment pipeline
│   ├── run_aggregation.py      # Sentiment aggregation
│   └── run_backtest.py         # Backtesting strategies
├── tests/
│   ├── test_environment.py     # Environment verification
│   ├── test_data_loading.py    # Data loading tests
│   ├── test_baseline.py        # Model and trainer tests
│   ├── test_finetune.py        # Fine-tuning tests
│   ├── test_error_analysis.py  # Error analysis tests
│   ├── test_news_pipeline.py   # News pipeline tests
│   ├── test_aggregation.py     # Aggregation tests
│   └── test_backtest.py        # Backtesting tests
├── data/
│   └── raw/                    # Financial PhraseBank dataset
├── models/                     # Saved model checkpoints
├── outputs/                    # Training outputs, metrics
├── notebooks/                  # EDA and experimentation
├── app/
│   └── dashboard.py            # Streamlit dashboard
├── docs/
│   └── model_selection.md      # Model selection recommendations
└── outputs/
    ├── training/               # Model checkpoints and comparison
    ├── analysis/               # Error analysis outputs
    ├── news/                   # News sentiment results
    ├── signals/                # Aggregated signals
    └── backtest/               # Backtest results
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

### Run Full Model Training

```bash
# Train all 4 models (recommended: use GPU via Google Colab)
docker compose run --rm app python scripts/train_all_models.py

# Train specific models
docker compose run --rm app python scripts/train_all_models.py --models bert distilbert
```

Results saved to `outputs/training/`.

### Run Tests

```bash
# All tests
docker compose run --rm app pytest -v

# Specific test file
docker compose run --rm app pytest tests/test_baseline.py -v
```

### Run News Sentiment Pipeline

```bash
# Fetch news and predict sentiment for default tickers
docker compose run --rm app python scripts/run_news_pipeline.py

# Specific tickers
docker compose run --rm app python scripts/run_news_pipeline.py --tickers AAPL NVDA TSLA
```

### Run Backtesting

```bash
# Aggregate sentiment and run backtest
docker compose run --rm app python scripts/run_aggregation.py
docker compose run --rm app python scripts/run_backtest.py
```

### Run Dashboard

```bash
docker compose run --rm -p 8501:8501 app streamlit run app/dashboard.py --server.address 0.0.0.0
```

Open http://localhost:8501 in your browser.

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

## Trading Strategies

Four sentiment-based strategies are implemented:

| Strategy | Rule | Return | Sharpe |
|----------|------|--------|--------|
| Sentiment Threshold | Long if > 0.2, Short if < -0.2 | +0.03% | 0.59 |
| Long Only | Long if > 0.1, else flat | +0.14% | 2.75 |
| Momentum | Trade on sentiment momentum | +0.18% | 1.87 |
| **Rolling 3d** | Long if 3d rolling > 0.15 | **+0.32%** | **9.07** |

## Default Tickers

```python
STOCK_TICKERS = ['AAPL', 'AMZN', 'BAC', 'GLD', 'GOOGL', 
                 'JPM', 'MSFT', 'NVDA', 'SPY', 'TLT']
```

## License

MIT

## Author

Nimesh Kulatunga

## Acknowledgments

- [Financial PhraseBank](https://www.researchgate.net/publication/251231107_Good_Debt_or_Bad_Debt_Detecting_Semantic_Orientations_in_Economic_Texts) dataset
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) for domain-specific model
