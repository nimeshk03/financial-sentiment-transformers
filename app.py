"""
Financial Sentiment Analysis Dashboard

Interactive dashboard showing:
- Model comparison results
- Real-time sentiment predictions
- Backtest performance
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="Financial Sentiment Analysis",
    layout="wide",
)

# Title
st.title("Financial Sentiment Analysis Dashboard")
st.markdown("Comparing BERT, RoBERTa, DistilBERT, and FinBERT for financial text classification")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Model Comparison", "Live Sentiment", "Backtest Results"]
)

def model_comparison_page():
    """Display model comparison results."""
    st.header("Model Comparison")
    
    # Model results data
    results = {
        'Model': ['FinBERT', 'RoBERTa', 'BERT', 'DistilBERT'],
        'Accuracy': [87.2, 85.2, 83.3, 81.4],
        'F1 Score': [0.873, 0.846, 0.842, 0.816],
        'Inference (ms)': [0.7, 0.5, 0.5, 0.2],
        'Parameters (M)': [109, 125, 109, 66],
    }
    df = pd.DataFrame(results)
    
    # Two columns layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Accuracy Comparison")
        fig = px.bar(
            df, x='Model', y='Accuracy',
            color='Model',
            title='Test Accuracy by Model'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.subheader("Speed vs Accuracy Trade-off")
        fig = px.scatter(
            df, x='Inference (ms)', y='Accuracy',
            size='Parameters (M)', color='Model',
            title='Inference Speed vs Accuracy',
            hover_data=['F1 Score']
        )
        st.plotly_chart(fig, width="stretch")
    
    # Results table
    st.subheader("Full Results")
    st.dataframe(df, width="stretch")
    
    # Key insights
    st.subheader("Key Insights")
    st.markdown("""
    - **FinBERT** achieves the highest accuracy (87.2%) due to domain-specific pre-training
    - **DistilBERT** offers the best speed (0.2ms) with only 5.8% accuracy drop
    - **RoBERTa** slightly outperforms BERT despite similar architecture
    - Domain adaptation provides ~4% improvement over general-purpose models
    """)

def live_sentiment_page():
    """Display live sentiment predictions."""
    st.header("Live Sentiment Analysis")
    
    # Check if we have news data
    news_path = Path("outputs/news/sentiment_results.csv")
    
    if news_path.exists():
        df = pd.read_csv(news_path)
        df['published'] = pd.to_datetime(df['published'])
        
        # Ticker selector
        tickers = df['ticker'].unique().tolist()
        selected_ticker = st.selectbox("Select Ticker", tickers)
        
        # Filter data
        ticker_df = df[df['ticker'] == selected_ticker]
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        avg_sentiment = ticker_df['sentiment_score'].mean()
        with col1:
            st.metric(
                "Average Sentiment",
                f"{avg_sentiment:+.2f}",
                delta="Bullish" if avg_sentiment > 0.1 else "Bearish" if avg_sentiment < -0.1 else "Neutral"
            )
        
        with col2:
            st.metric("Total Articles", len(ticker_df))
        
        with col3:
            positive_pct = (ticker_df['sentiment'] == 'positive').mean() * 100
            st.metric("Positive %", f"{positive_pct:.1f}%")
        
        # Sentiment distribution
        st.subheader("Sentiment Distribution")
        fig = px.pie(
            ticker_df, names='sentiment',
            title=f'{selected_ticker} Sentiment Breakdown',
            color='sentiment',
            color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
        )
        st.plotly_chart(fig, width="stretch")
        
        # Recent headlines
        st.subheader("Recent Headlines")
        for _, row in ticker_df.head(10).iterrows():
            sentiment_marker = {'positive': '[+]', 'neutral': '[=]', 'negative': '[-]'}
            st.markdown(f"{sentiment_marker.get(row['sentiment'], '[=]')} **{row['sentiment'].upper()}** - {row['title']}")
    
    else:
        st.warning("No news data found. Run the news pipeline first:")
        st.code("python scripts/run_news_pipeline.py")
    
    # Manual input
    st.subheader("Try Your Own Text")
    user_text = st.text_area("Enter a financial headline:", "Apple reports record quarterly earnings")
    
    if st.button("Analyze Sentiment"):
        st.info("Note: Live prediction requires model checkpoint. See instructions below.")
        st.code("docker compose run --rm app python scripts/run_news_pipeline.py")

def backtest_page():
    """Display backtest results."""
    st.header("Backtest Results")
    
    # Load backtest results
    backtest_path = Path("outputs/backtest/backtest_results.json")
    
    if backtest_path.exists():
        with open(backtest_path) as f:
            results = json.load(f)
        
        # Strategy comparison
        st.subheader("Strategy Performance")
        
        strategies = list(results.keys())
        returns = [results[s]['total_return'] * 100 for s in strategies]
        sharpes = [results[s]['sharpe_ratio'] for s in strategies]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=strategies, y=returns,
                title='Total Return by Strategy (%)',
                labels={'x': 'Strategy', 'y': 'Return (%)'}
            )
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            fig = px.bar(
                x=strategies, y=sharpes,
                title='Sharpe Ratio by Strategy',
                labels={'x': 'Strategy', 'y': 'Sharpe Ratio'}
            )
            st.plotly_chart(fig, width="stretch")
        
        # Detailed metrics table
        st.subheader("Detailed Metrics")
        metrics_df = pd.DataFrame(results).T
        metrics_df['total_return'] = metrics_df['total_return'].apply(lambda x: f"{x:.2%}")
        metrics_df['win_rate'] = metrics_df['win_rate'].apply(lambda x: f"{x:.1%}")
        st.dataframe(metrics_df, width="stretch")
        
        # Best strategy
        best = max(results.items(), key=lambda x: x[1]['total_return'])
        st.success(f"Best Strategy: **{best[0]}** with {best[1]['total_return']:.2%} return")
    
    else:
        st.warning("No backtest results found. Run the backtest first:")
        st.code("python scripts/run_backtest.py")
    
    # Strategy explanations
    st.subheader("Strategy Descriptions")
    st.markdown("""
    | Strategy | Rule |
    |----------|------|
    | **Sentiment Threshold** | Long if sentiment > 0.2, Short if < -0.2 |
    | **Long Only** | Long if sentiment > 0.1, else flat |
    | **Momentum** | Trade based on sentiment momentum (improving/deteriorating) |
    | **Rolling 3d** | Long if 3-day rolling sentiment > 0.15 |
    """)

# Main routing
if page == "Model Comparison":
    model_comparison_page()
elif page == "Live Sentiment":
    live_sentiment_page()
elif page == "Backtest Results":
    backtest_page()

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | [GitHub](https://github.com/nimeshk03/financial-sentiment-transformers)")