# Model Comparison Results

Generated: 2025-12-06 14:27:38

## Summary Table

| Model | Parameters | Accuracy | F1 | Inference (ms) | Training Time |
|-------|------------|----------|-----|----------------|---------------|
| finbert | 109M | 87.2% | 0.873 | 0.5ms | 4.3min |
| roberta | 125M | 85.2% | 0.852 | 0.6ms | 4.3min |
| bert | 109M | 83.3% | 0.835 | 0.5ms | 4.2min |
| distilbert | 66M | 81.4% | 0.816 | 0.2ms | 2.2min |

## Per-Class Metrics

### FINBERT

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| negative | 0.821 | 0.902 | 0.859 |
| neutral | 0.929 | 0.865 | 0.896 |
| positive | 0.793 | 0.875 | 0.832 |

### ROBERTA

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| negative | 0.814 | 0.934 | 0.870 |
| neutral | 0.907 | 0.844 | 0.874 |
| positive | 0.769 | 0.831 | 0.799 |

### BERT

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| negative | 0.761 | 0.836 | 0.797 |
| neutral | 0.902 | 0.826 | 0.862 |
| positive | 0.747 | 0.846 | 0.793 |

### DISTILBERT

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| negative | 0.736 | 0.869 | 0.797 |
| neutral | 0.887 | 0.819 | 0.852 |
| positive | 0.721 | 0.779 | 0.749 |
