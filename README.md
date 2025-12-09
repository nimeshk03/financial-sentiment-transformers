# Financial Sentiment Analysis - Multi-Model Transformer Comparison

[![Hugging Face Spaces](https://img.shields.io/badge/Demo-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/nimeshk03/financial-sentiment-dashboard)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

A comprehensive pipeline benchmarking **four transformer architectures** (BERT, RoBERTa, DistilBERT, FinBERT) on financial sentiment classification, with trading signal generation and backtesting capabilities.

**[Live Demo](https://huggingface.co/spaces/nimeshk03/financial-sentiment-dashboard)**

---

## Table of Contents

- [Project Overview](#project-overview)
- [Results](#results)
- [Models Under Comparison](#models-under-comparison)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Trading Strategies](#trading-strategies)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

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

## Features

- **Multi-Model Benchmarking**: Fair comparison of BERT, RoBERTa, DistilBERT, and FinBERT
- **End-to-End Pipeline**: From raw financial text to trading signals
- **News Sentiment Analysis**: Real-time sentiment from Yahoo Finance headlines
- **Backtesting Framework**: 4 trading strategies with performance metrics
- **Interactive Dashboard**: Streamlit-based visualization and exploration
- **Production-Ready**: Docker containerization with comprehensive tests


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

### Environment Variables

No API keys are required for basic functionality. The project uses:
- **Yahoo Finance**: Public API for news and price data (no key needed)
- **Hugging Face**: Public model downloads (no key needed for inference)

> **Note**: For production deployments or high-volume usage, consider setting up rate limiting or caching.

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

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce `batch_size` in training config or use CPU |
| `Dataset not found` | Ensure `all-data.csv` is in `data/raw/` directory |
| `Docker build fails` | Check Docker daemon is running; try `docker system prune` |
| `Model download slow` | Models are cached after first download in `~/.cache/huggingface` |
| `Yahoo Finance rate limit` | Add delays between requests or reduce ticker count |

### Performance Tips

- **GPU Training**: Use Google Colab or cloud GPU for training all models
- **Inference**: CPU is sufficient for real-time inference (0.2-0.7ms per sample)
- **Memory**: DistilBERT uses 40% less memory than BERT/FinBERT

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes using [conventional commits](https://www.conventionalcommits.org/)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/financial-sentiment-transformers.git
cd financial-sentiment-transformers

# Build and run tests
docker compose build
docker compose run --rm app pytest -v
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints where applicable
- Write docstrings for public functions
- Add tests for new features

## Citation

If you use this project in your research, please cite:

```bibtex
@software{kulatunga2024financial,
  author = {Kulatunga, Nimesh},
  title = {Financial Sentiment Analysis: Multi-Model Transformer Comparison},
  year = {2024},
  url = {https://github.com/nimeshk03/financial-sentiment-transformers}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Nimesh Kulatunga**

- GitHub: [@nimeshk03](https://github.com/nimeshk03)
- Hugging Face: [nimeshk03](https://huggingface.co/nimeshk03)

## Acknowledgments

- [Financial PhraseBank](https://www.researchgate.net/publication/251231107_Good_Debt_or_Bad_Debt_Detecting_Semantic_Orientations_in_Economic_Texts) dataset by Malo et al.
- [Hugging Face Transformers](https://huggingface.co/transformers/) library
- [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) for domain-specific pre-training
- [Yahoo Finance](https://finance.yahoo.com/) for news and price data

---

<p align="center">
  <i>Built with Transformers and Python</i>
</p>
