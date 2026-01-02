"""
Financial Sentiment Analysis - Gradio API

Provides a callable API for sentiment analysis and trading signal generation.
Deploy to HuggingFace Spaces for remote access.
"""
import gradio as gr
import pandas as pd
from typing import List, Dict, Any
import json

# Use HuggingFace FinBERT for inference (no local model needed)
from transformers import pipeline

# Initialize sentiment pipeline with FinBERT (ProsusAI version works on HF router)
sentiment_pipeline = None

def get_pipeline():
    """Lazy load the sentiment pipeline."""
    global sentiment_pipeline
    if sentiment_pipeline is None:
        sentiment_pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            top_k=None  # Return all scores
        )
    return sentiment_pipeline


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment of a single text.
    
    Returns:
        dict with label, score, and all probabilities
    """
    pipe = get_pipeline()
    results = pipe(text)[0]
    
    # Find top label
    top = max(results, key=lambda x: x['score'])
    
    # Convert to numeric score (-1 to 1) - ProsusAI/finbert uses lowercase labels
    label_scores = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0,
                    'Positive': 1.0, 'Negative': -1.0, 'Neutral': 0.0}
    numeric_score = label_scores.get(top['label'], 0.0)
    
    return {
        'label': top['label'],
        'confidence': round(top['score'], 4),
        'numeric_score': numeric_score,
        'probabilities': {r['label']: round(r['score'], 4) for r in results}
    }


def analyze_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Analyze sentiment of multiple texts.
    
    Returns:
        List of sentiment results
    """
    pipe = get_pipeline()
    all_results = pipe(texts)
    
    parsed = []
    for results in all_results:
        top = max(results, key=lambda x: x['score'])
        label_scores = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0,
                        'Positive': 1.0, 'Negative': -1.0, 'Neutral': 0.0}
        numeric_score = label_scores.get(top['label'], 0.0)
        
        parsed.append({
            'label': top['label'],
            'confidence': round(top['score'], 4),
            'numeric_score': numeric_score,
            'probabilities': {r['label']: round(r['score'], 4) for r in results}
        })
    
    return parsed


def generate_trading_signal(
    sentiment_score: float,
    sentiment_momentum: float = 0.0,
    threshold_long: float = 0.2,
    threshold_short: float = -0.2
) -> Dict[str, Any]:
    """
    Generate trading signal based on sentiment.
    
    Strategies from financial-sentiment-transformers backtest:
    - Sentiment Threshold: Long if > 0.2, Short if < -0.2
    - Momentum: Consider sentiment trend
    
    Returns:
        dict with signal, strength, and reasoning
    """
    # Base signal from threshold
    if sentiment_score > threshold_long:
        signal = "LONG"
        strength = min((sentiment_score - threshold_long) / 0.8, 1.0)
    elif sentiment_score < threshold_short:
        signal = "SHORT"
        strength = min((threshold_short - sentiment_score) / 0.8, 1.0)
    else:
        signal = "NEUTRAL"
        strength = 0.0
    
    # Adjust for momentum
    if sentiment_momentum > 0.1 and signal == "LONG":
        strength = min(strength * 1.2, 1.0)
    elif sentiment_momentum < -0.1 and signal == "SHORT":
        strength = min(strength * 1.2, 1.0)
    elif sentiment_momentum < -0.1 and signal == "LONG":
        strength *= 0.8  # Weakening bullish sentiment
    elif sentiment_momentum > 0.1 and signal == "SHORT":
        strength *= 0.8  # Weakening bearish sentiment
    
    return {
        'signal': signal,
        'strength': round(strength, 4),
        'sentiment_score': sentiment_score,
        'momentum': sentiment_momentum,
        'reasoning': f"Sentiment {sentiment_score:.2f} {'>' if signal == 'LONG' else '<' if signal == 'SHORT' else 'between'} threshold"
    }


def analyze_with_signal(text: str) -> str:
    """
    Full analysis: sentiment + trading signal.
    Returns JSON string for API consumption.
    """
    sentiment = analyze_sentiment(text)
    signal = generate_trading_signal(sentiment['numeric_score'])
    
    result = {
        'text': text,
        'sentiment': sentiment,
        'trading_signal': signal
    }
    
    return json.dumps(result, indent=2)


def batch_analyze_with_signals(texts_json: str) -> str:
    """
    Batch analysis for multiple texts.
    Input: JSON array of strings
    Output: JSON array of results
    """
    try:
        texts = json.loads(texts_json)
        if not isinstance(texts, list):
            texts = [texts]
    except:
        texts = [texts_json]
    
    sentiments = analyze_batch(texts)
    
    results = []
    for text, sentiment in zip(texts, sentiments):
        signal = generate_trading_signal(sentiment['numeric_score'])
        results.append({
            'text': text,
            'sentiment': sentiment,
            'trading_signal': signal
        })
    
    return json.dumps(results, indent=2)


def aggregate_portfolio_sentiment(texts_json: str) -> str:
    """
    Aggregate sentiment across multiple headlines for portfolio-level signal.
    
    Input: JSON with structure {"AAPL": ["headline1", "headline2"], "MSFT": [...]}
    Output: Aggregated sentiment and signals per ticker
    """
    try:
        ticker_headlines = json.loads(texts_json)
    except:
        return json.dumps({"error": "Invalid JSON input"})
    
    portfolio_results = {}
    
    for ticker, headlines in ticker_headlines.items():
        if not headlines:
            portfolio_results[ticker] = {
                'avg_sentiment': 0.0,
                'signal': 'NEUTRAL',
                'headline_count': 0
            }
            continue
            
        sentiments = analyze_batch(headlines)
        scores = [s['numeric_score'] for s in sentiments]
        
        avg_score = sum(scores) / len(scores)
        signal = generate_trading_signal(avg_score)
        
        portfolio_results[ticker] = {
            'avg_sentiment': round(avg_score, 4),
            'signal': signal['signal'],
            'signal_strength': signal['strength'],
            'headline_count': len(headlines),
            'positive_pct': round(sum(1 for s in sentiments if s['label'] == 'Positive') / len(sentiments), 4),
            'negative_pct': round(sum(1 for s in sentiments if s['label'] == 'Negative') / len(sentiments), 4),
        }
    
    return json.dumps(portfolio_results, indent=2)


# Gradio Interface
with gr.Blocks(title="Financial Sentiment API") as demo:
    gr.Markdown("# Financial Sentiment Analysis API")
    gr.Markdown("FinBERT-powered sentiment analysis with trading signal generation")
    
    with gr.Tab("Single Text"):
        text_input = gr.Textbox(
            label="Financial Headline",
            placeholder="Apple reports record quarterly earnings",
            lines=2
        )
        analyze_btn = gr.Button("Analyze", variant="primary")
        output_single = gr.JSON(label="Result")
        
        analyze_btn.click(
            fn=lambda t: json.loads(analyze_with_signal(t)),
            inputs=text_input,
            outputs=output_single
        )
    
    with gr.Tab("Batch Analysis"):
        batch_input = gr.Textbox(
            label="Headlines (JSON array)",
            placeholder='["Apple beats earnings", "Tesla recalls vehicles"]',
            lines=4
        )
        batch_btn = gr.Button("Analyze Batch", variant="primary")
        output_batch = gr.JSON(label="Results")
        
        batch_btn.click(
            fn=lambda t: json.loads(batch_analyze_with_signals(t)),
            inputs=batch_input,
            outputs=output_batch
        )
    
    with gr.Tab("Portfolio Aggregation"):
        portfolio_input = gr.Textbox(
            label="Portfolio Headlines (JSON)",
            placeholder='{"AAPL": ["Apple beats earnings"], "MSFT": ["Microsoft cloud grows"]}',
            lines=6
        )
        portfolio_btn = gr.Button("Aggregate", variant="primary")
        output_portfolio = gr.JSON(label="Portfolio Signals")
        
        portfolio_btn.click(
            fn=lambda t: json.loads(aggregate_portfolio_sentiment(t)),
            inputs=portfolio_input,
            outputs=output_portfolio
        )


if __name__ == "__main__":
    demo.launch()
