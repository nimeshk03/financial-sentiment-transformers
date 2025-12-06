# Model Selection for Deployment

## Summary

Based on our comprehensive evaluation of four transformer models on the Financial PhraseBank dataset, we provide deployment recommendations for different use cases.

## Model Comparison

| Model | Accuracy | F1 Score | Inference | Parameters | Training Time |
|-------|----------|----------|-----------|------------|---------------|
| **FinBERT** | 87.2% | 0.873 | 0.7ms | 109M | 4.2min |
| RoBERTa | 85.2% | 0.846 | 0.5ms | 125M | 4.3min |
| BERT | 83.3% | 0.842 | 0.5ms | 109M | 4.1min |
| DistilBERT | 81.4% | 0.816 | 0.2ms | 66M | 2.1min |

## Recommendations

### Primary Recommendation: FinBERT

**Use FinBERT when:**
- Accuracy is the top priority
- Processing financial reports, earnings calls, or research
- Batch processing where latency is not critical
- Domain-specific understanding is important

**Why FinBERT:**
- Highest accuracy (87.2%)
- Pre-trained on financial text - understands domain nuances
- Correctly classifies 30 samples that BERT misses
- Best performance on all three sentiment classes

### Alternative: DistilBERT

**Use DistilBERT when:**
- Real-time or high-volume processing required
- Resource-constrained environments (edge devices, limited GPU)
- Cost optimization is important
- Slight accuracy trade-off is acceptable

**Why DistilBERT:**
- 3.5x faster inference (0.2ms vs 0.7ms)
- 40% fewer parameters (66M vs 109M)
- Achieves 93% of FinBERT's accuracy
- Half the training time

## Trade-off Analysis

```
Accuracy vs Speed:
FinBERT:    [##########] 87.2%  |  Speed: [###-------] 0.7ms
DistilBERT: [########--] 81.4%  |  Speed: [##########] 0.2ms
```

### Decision Matrix

| Requirement | Recommended Model |
|-------------|-------------------|
| Maximum accuracy | FinBERT |
| Real-time trading signals | DistilBERT |
| Research/analysis reports | FinBERT |
| High-volume news processing | DistilBERT |
| Mobile/edge deployment | DistilBERT |
| Financial document analysis | FinBERT |

## Conclusion

For this project's trading signal pipeline, we recommend:

1. **Primary model**: FinBERT - for generating accurate sentiment scores
2. **Fallback option**: DistilBERT - if latency becomes a bottleneck

The 5.8 percentage point accuracy difference (87.2% vs 81.4%) translates to meaningful improvements in trading signal quality, making FinBERT the default choice unless speed constraints require otherwise.

---

*Document generated: December 6, 2025*
*Based on evaluation of 485 test samples from Financial PhraseBank*
