"""
Placeholder Streamlit dashboard.
Will be expanded in Week 4.
"""
import streamlit as st

st.set_page_config(
    page_title="Financial Sentiment Analysis",
    page_icon="",
    layout="wide"
)

st.title("Financial Sentiment Analysis")
st.markdown("### Multi-Model Comparison Dashboard")

st.info("""
**Status:** Environment setup complete!

This dashboard will show:
- Real-time sentiment by stock
- Model comparison visualizations  
- Backtest results

Coming in Week 4...
""")

# Quick environment check
st.subheader("Environment Check")

try:
    import torch
    st.success(f"PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
except ImportError:
    st.error("❌ PyTorch not installed")

try:
    import transformers
    st.success(f"Transformers {transformers.__version__}")
except ImportError:
    st.error("❌ Transformers not installed")

try:
    from datasets import load_dataset
    st.success("Datasets library installed")
except ImportError:
    st.error("❌ Datasets not installed")