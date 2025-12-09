# Benchmarking Transformer Models for Financial Sentiment: A Complete Pipeline from Text to Trading Signals

Machine Learning | NLP | Finance | Transformers

[View Code](https://github.com/nimeshk03/financial-sentiment-transformers) | [Live Demo](https://huggingface.co/spaces/nimeshk03/financial-sentiment-dashboard)

---

## Introduction

Can a transformer model trained on financial text outperform general-purpose language models? And more importantly, can we turn sentiment predictions into actionable trading signals?

This project explores these questions by building a complete pipeline that benchmarks **four transformer architectures** (BERT, RoBERTa, DistilBERT, FinBERT) on financial sentiment classification, then extends the best model into a news-driven trading signal generator with backtesting validation.

The result? **FinBERT achieved 87.2% accuracy**, outperforming general-purpose BERT by 3.9 percentage points. The sentiment-based trading strategy achieved a **9.07 Sharpe ratio**, beating the market benchmark by 0.25%.

---

## The Challenge: Why Financial Sentiment is Hard

Financial text is deceptively difficult for NLP models. Consider this sentence:

> "Only Lannen Tehtaat showed a loss, but it has only recently started streamlining..."

Is this negative (mentions "loss"), neutral (factual reporting), or positive (implies improvement ahead)? Even humans disagree. In my experiments, models disagreed on **17.9% of test samples** - these represent genuinely ambiguous financial statements.

### Why General Models Struggle

Standard language models like BERT are trained on Wikipedia and books. They understand general English but miss financial nuances:

- "Bearish" isn't about animals
- "Correction" isn't fixing a mistake
- "Exposure" isn't about photography

This is where domain-specific models like FinBERT shine - they've seen millions of financial documents during pre-training.

---

## The Models: A Fair Comparison

I compared four transformer architectures, each representing a different approach:

| Model | Parameters | Key Feature | Why Include |
|-------|------------|-------------|-------------|
| **BERT** | 110M | The original transformer | Baseline everyone knows |
| **RoBERTa** | 125M | Better training procedure | Shows training matters |
| **DistilBERT** | 66M | 40% smaller, 2x faster | Production efficiency |
| **FinBERT** | 110M | Pre-trained on financial text | Domain adaptation |

### The Critical Design Decision: Fair Comparison

To ensure a fair comparison, all models used **identical hyperparameters**:

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

This is crucial. Without standardization, you can't tell if Model A beats Model B because it's better, or because you accidentally gave it better hyperparameters.

---

## The Data: Financial PhraseBank

I used the Financial PhraseBank dataset - approximately 4,845 financial news sentences labeled as negative, neutral, or positive.

### The Class Imbalance Problem

| Class | Count | Percentage |
|-------|-------|------------|
| Neutral | 2,879 | 59.4% |
| Positive | 1,363 | 28.1% |
| Negative | 603 | 12.4% |

![Class Distribution](~/Documents/Projects/financial_sentiment_analysis/outputs/training/01_class_distribution.png)

![Class Distribution Pie Chart](~/Documents/Projects/financial_sentiment_analysis/outputs/training/02_class_pie_chart.png)

Neutral dominates. A naive model could achieve 59% accuracy by always predicting "neutral." To handle this, I used:

1. **Stratified splits**: Maintain class ratios in train/val/test
2. **Weighted loss function**: Penalize mistakes on minority classes more heavily
3. **Per-class metrics**: Track precision/recall for each class, not just overall accuracy

![Train/Val/Test Split Distribution](~/Documents/Projects/financial_sentiment_analysis/outputs/training/05_split_distribution.png)

### Dataset Characteristics

Financial sentences tend to be concise - most are under 40 words:

![Text Length Distribution](~/Documents/Projects/financial_sentiment_analysis/outputs/training/03_text_length_distribution.png)

![Word Count by Class](~/Documents/Projects/financial_sentiment_analysis/outputs/training/04_word_count_boxplot.png)

---

## Results: Domain Adaptation Wins

After training all four models on a Google Colab T4 GPU (total time: ~15 minutes), here are the results:

| Model | Accuracy | F1 Score | Inference (ms) | Training Time |
|-------|----------|----------|----------------|---------------|
| **FinBERT** | **87.2%** | **0.873** | 0.7ms | 4.2min |
| RoBERTa | 84.5% | 0.846 | 0.5ms | 4.3min |
| BERT | 84.1% | 0.842 | 0.5ms | 4.1min |
| DistilBERT | 81.4% | 0.816 | 0.2ms | 2.1min |

![Model Comparison - Confusion Matrices](~/Documents/Projects/financial_sentiment_analysis/outputs/analysis/confusion_matrices.png)

### Key Finding 1: Domain Adaptation Provides 3-6% Improvement

FinBERT outperforms all general-purpose models. The financial pre-training gives it an understanding of domain-specific language that BERT simply doesn't have.

### Key Finding 2: DistilBERT Offers Compelling Trade-offs

DistilBERT achieves **93% of FinBERT's accuracy** with:
- **40% fewer parameters** (66M vs 110M)
- **3.5x faster inference** (0.2ms vs 0.7ms)
- **Half the training time** (2.1min vs 4.2min)

For cost-sensitive production deployments, DistilBERT is often the right choice.

### Key Finding 3: Training Procedure Matters

RoBERTa slightly outperforms BERT despite having the same architecture. The difference? RoBERTa was trained with more data and better masking strategies. Architecture isn't everything.

---

## Error Analysis: Where Models Disagree

I analyzed the 87 samples (17.9%) where models disagreed to understand their failure modes.

![Model Agreement Distribution](~/Documents/Projects/financial_sentiment_analysis/outputs/analysis/model_agreement.png)

### FinBERT vs BERT: Head-to-Head

| Scenario | Count |
|----------|-------|
| FinBERT correct, BERT wrong | 30 |
| BERT correct, FinBERT wrong | 11 |
| Both correct | 393 |
| Both wrong | 51 |

FinBERT correctly classified **30 samples that BERT missed**, while only failing on 11 that BERT got right. This is strong evidence that domain-specific pre-training provides real value.

### Example Disagreement

> "Key shareholders of Finnish IT services provider TietoEnator Oyj on Friday rejected..."

- **True label**: Positive
- **BERT**: Negative (triggered by "rejected")
- **FinBERT**: Positive (understands shareholder context)

FinBERT understands that shareholders rejecting something can be positive for the company - a nuance that general-purpose BERT misses.

---

## From Sentiment to Trading Signals

With a working sentiment model, I built a complete pipeline to generate trading signals from real news:

### Step 1: News Ingestion

Fetch news headlines from Yahoo Finance for 10 tickers:

```python
STOCK_TICKERS = ['AAPL', 'AMZN', 'BAC', 'GLD', 'GOOGL', 
                 'JPM', 'MSFT', 'NVDA', 'SPY', 'TLT']
```

### Step 2: Sentiment Prediction

Run each headline through FinBERT to get sentiment scores (-1 to +1).

### Step 3: Signal Aggregation

Aggregate daily sentiment with rolling averages:
- **3-day rolling average**: Short-term sentiment trend
- **7-day rolling average**: Medium-term sentiment trend
- **Sentiment momentum**: Difference between short and long MA

### Step 4: Backtesting

Test four trading strategies:

| Strategy | Rule | Return | Sharpe Ratio |
|----------|------|--------|--------------|
| Sentiment Threshold | Long if > 0.2, Short if < -0.2 | +0.03% | 0.59 |
| Long Only | Long if > 0.1, else flat | +0.14% | 2.75 |
| Momentum | Trade on sentiment momentum | +0.18% | 1.87 |
| **Rolling 3d** | Long if 3d rolling > 0.15 | **+0.32%** | **9.07** |

### Best Strategy: Rolling 3-Day Sentiment

The rolling 3-day strategy achieved the best risk-adjusted returns:
- **Total Return**: +0.32%
- **Sharpe Ratio**: 9.07
- **Beat Market By**: +0.25%

The smoothing effect of the rolling average filters out noise from individual headlines, capturing genuine sentiment shifts.

---

## The Dashboard: Making It Interactive

I built a Streamlit dashboard with three pages:

### 1. Model Comparison
Interactive charts comparing accuracy, speed, and the accuracy-speed trade-off across all four models.

### 2. Live Sentiment
Select a ticker and see:
- Average sentiment score
- Sentiment distribution (pie chart)
- Recent headlines with sentiment labels

### 3. Backtest Results
Strategy comparison with returns, Sharpe ratios, and detailed metrics.

**Try it yourself**: [Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/nimeshk03/financial-sentiment-dashboard)

---

## Technical Implementation

### Project Structure

```
financial_sentiment_analysis/
├── src/
│   ├── data/           # Data loading, tokenization
│   ├── models/         # Classifier, trainer
│   ├── analysis/       # Error analysis
│   ├── news/           # News fetching, processing
│   ├── signals/        # Sentiment aggregation
│   └── backtesting/    # Trading strategies, metrics
├── scripts/            # CLI scripts for each pipeline
├── tests/              # Comprehensive test suite
├── app/                # Streamlit dashboard
└── outputs/            # Results, charts, models
```

### Key Design Decisions

**1. Docker for Reproducibility**

The entire project runs in Docker. Anyone can clone the repo and get identical results:

```bash
docker compose build
docker compose run --rm app pytest -v
```

**2. Modular Architecture**

Each component is independent and testable:
- `SentimentClassifier`: Works with any HuggingFace transformer
- `Trainer`: Handles scheduler, early stopping, metrics
- `SentimentPredictor`: Inference wrapper for production

**3. Hybrid Compute Strategy**

| Task | Environment | Reason |
|------|-------------|--------|
| Development | Local (CPU) | Fast iteration |
| Model training | Google Colab (GPU) | Free T4 GPU |
| Inference | Local (CPU) | Fast enough |
| Dashboard | Hugging Face Spaces | Free hosting |

---

## Lessons Learned

### 1. Domain Adaptation is Worth It

FinBERT's 3-6% improvement over general models is significant. For financial applications, always consider domain-specific models.

### 2. Fair Comparison Requires Discipline

Without identical hyperparameters, you can't draw valid conclusions. This seems obvious but is often overlooked.

### 3. Efficiency Matters in Production

DistilBERT's 3.5x speed advantage makes it viable for real-time applications where FinBERT might be too slow.

### 4. Sentiment Alone Isn't Enough

Raw sentiment scores are noisy. Rolling averages and momentum indicators significantly improve signal quality.

### 5. Start with Docker

Setting up Docker early saved countless hours of "works on my machine" debugging.

---

## Conclusion

This project demonstrates that:

1. **Domain-specific pre-training provides measurable improvements** - FinBERT's 87.2% accuracy vs BERT's 84.1%

2. **Efficiency and accuracy are trade-offs** - DistilBERT offers 93% of the accuracy at 3.5x the speed

3. **Sentiment can generate trading signals** - The rolling 3-day strategy achieved a 9.07 Sharpe ratio

4. **End-to-end pipelines are achievable** - From raw text to trading signals with a live dashboard

The code is open source. The dashboard is live. Try it yourself and let me know what you find.

---

## Resources

- **GitHub**: [github.com/nimeshk03/financial-sentiment-transformers](https://github.com/nimeshk03/financial-sentiment-transformers)
- **Live Demo**: [Hugging Face Spaces](https://huggingface.co/spaces/nimeshk03/financial-sentiment-dashboard)
- **Dataset**: [Financial PhraseBank on Kaggle](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)

---

*Disclaimer: This project is for educational purposes. Past backtest performance does not guarantee future results. Always do your own research before making investment decisions.*
