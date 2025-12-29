"""
Portfolio-Grade Financial Analytics Platform:
Market Interaction-Based Performance Predictor
"""
import streamlit as st
import pandas as pd
import torch
import time
import networkx as nx
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

from data_processor import (
    fetch_market_data,
    add_technical_indicators,
    get_sentiment,
    get_feature_matrix,
    create_dynamic_graph
)
from gat_model import GATModel

# --- Configuration ---
TICKERS = ['NVDA', 'AMD', 'MSFT', 'AAPL', 'GOOGL', 'META', 'TSLA', 'AMZN', 'INTC', 'QCOM']
LOOKBACK_MINUTES = 60
CORRELATION_THRESHOLD = 0.6

# --- Page Configuration ---
st.set_page_config(page_title="🔴 QuantPlatform: GAT Predictor", layout="wide")

# --- Educational Sidebar ---
st.sidebar.title("🎓 University Mode")
show_math = st.sidebar.checkbox("Show Formulas & Explanations", value=False)
st.sidebar.divider()
st.sidebar.title("System Controls")
# This key will be used to reset the auto-refresh counter
if st.sidebar.button('Force Refresh Data'):
    st.session_state.refresh_counter = 0
st.sidebar.info(
    "The dashboard auto-refreshes every 60 seconds. "
    "Use the button above for an immediate data refresh."
)

# --- Auto-Refresh ---
st_autorefresh(interval=60 * 1000, key="refresh_counter")

# --- Main Application ---
def main():
    st.title("📈 Market Interaction-Based Performance Predictor")
    st.caption(f"Lead Quantitative Architect Edition | Last updated: {time.strftime('%H:%M:%S')}")
    
    # --- Data Loading and Processing ---
    with st.spinner("Fetching real-time market data..."):
        raw_data = fetch_market_data(TICKERS)
        if raw_data is None:
            st.error("Failed to fetch market data. The yfinance API may be temporarily down or markets are closed.")
            return
        
        tech_data = add_technical_indicators(raw_data.copy(), TICKERS)
        sentiment_data = get_sentiment(TICKERS)
    
    with st.spinner("Building feature matrix and running GAT model..."):
        feature_df = get_feature_matrix(tech_data, sentiment_data, TICKERS, LOOKBACK_MINUTES)
        correlation_matrix, adjacency_matrix = create_dynamic_graph(
            raw_data.xs('Close', level=1, axis=1), LOOKBACK_MINUTES, CORRELATION_THRESHOLD
        )
        
        # --- Model Inference ---
        features_tensor = torch.tensor(feature_df.values, dtype=torch.float32)
        adj_tensor = torch.tensor(adjacency_matrix.values, dtype=torch.float32)
        
        model = GATModel(in_features=features_tensor.shape[1], hidden_features=features_tensor.shape[1])
        scores = model(features_tensor, adj_tensor)
        attention_weights = model.get_attention_weights()
        
        scores_df = pd.DataFrame(
            {'ticker': feature_df.index, 'score': scores.detach().numpy()}
        ).sort_values('score', ascending=False).set_index('ticker')

        # Convert attention tensor to a DataFrame for easier handling in the UI
        attention_df = pd.DataFrame(
            attention_weights.detach().numpy(),
            columns=feature_df.index,
            index=feature_df.index
        )

    # --- UI Rendering ---
    render_ticker_tape(raw_data)
    st.divider()

    col1, col2 = st.columns([2, 1.5])
    with col1:
        render_network_graph(correlation_matrix, adjacency_matrix, scores_df, show_math)
    with col2:
        render_candlestick_chart(raw_data, tech_data, show_math)

    st.divider()
    render_prediction_section(scores_df, sentiment_data, attention_df, show_math)


def render_ticker_tape(data: pd.DataFrame):
    last_row = data.xs('Close', level=1, axis=1).iloc[-1]
    prev_row = data.xs('Close', level=1, axis=1).iloc[-2]
    cols = st.columns(5)
    for i, ticker in enumerate(TICKERS[:5]):
        with cols[i]:
            price = last_row[ticker]
            change = ((price - prev_row[ticker]) / prev_row[ticker]) * 100
            st.metric(label=ticker, value=f"${price:,.2f}", delta=f"{change:.3f}%")

def render_network_graph(correlation_matrix, adjacency_matrix, scores_df, show_math):
    st.subheader("🌐 Live Market Interaction Network")
    G = nx.from_pandas_adjacency(adjacency_matrix)
    pos = nx.spring_layout(G, seed=42)

    edge_traces, node_trace = [], None
    if G.nodes():
        for edge in G.edges():
            x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
            corr = correlation_matrix.loc[edge[0], edge[1]]
            width = 1 + (abs(corr) - CORRELATION_THRESHOLD) * 10
            color = '#00FF00' if corr > 0 else '#FF0033'
            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                line=dict(width=width, color=color), mode='lines'
            ))
        
        node_x, node_y, node_text, node_color = [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x); node_y.append(y); node_text.append(node)
            node_color.append(scores_df.loc[node, 'score'])

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center",
            marker=dict(showscale=True, colorscale='viridis', color=node_color, size=25,
                        colorbar=dict(thickness=15, title="Perf. Score"))
        )

    fig = go.Figure(data=edge_traces + ([node_trace] if node_trace else []),
                 layout=go.Layout(showlegend=False, margin=dict(b=0,l=0,r=0,t=0),
                                  plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                  xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    st.plotly_chart(fig, use_container_width=True)
    if show_math:
        st.info("This graph is an Adjacency Matrix where an edge exists if Pearson Correlation > threshold.")
        st.latex(r'''
        \rho(X, Y) = \frac{\text{cov}(X, Y)}{\sigma_X \sigma_Y}
        ''')


def render_candlestick_chart(raw_data, tech_data, show_math):
    st.subheader("🕯️ Live Price Chart")
    selected_ticker = st.selectbox("Select a stock to analyze:", TICKERS, key="candlestick_ticker")
    
    df = raw_data[selected_ticker].tail(100)
    df_ta = tech_data[selected_ticker].tail(100)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['volatility_bbh'], line=dict(color='gray', width=1), name='Upper Band'))
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['volatility_bbl'], line=dict(color='gray', width=1), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', name='Lower Band'))
    
    fig.update_layout(xaxis_rangeslider_visible=False, showlegend=False,
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      yaxis=dict(gridcolor='#444'), xaxis=dict(gridcolor='#444'),
                      font=dict(color='white'), margin=dict(b=20,l=20,r=20,t=20))
    st.plotly_chart(fig, use_container_width=True)

    if show_math:
        st.info("Shows 1-minute price candles with Bollinger Bands (a volatility indicator).")
        st.latex(r'''
        \text{Upper Band} = \text{MA}(20) + 2 \times \sigma[\text{Price}] \\
        \text{Lower Band} = \text{MA}(20) - 2 \times \sigma[\text{Price}]
        ''')

def render_prediction_section(scores_df, sentiment, attention, show_math):
    st.subheader("🤖 GAT Model Predictions")
    top_performer = scores_df.index[0]
    
    # "Why?" Box
    top_sentiment = sentiment.loc[top_performer, 'sentiment']
    sentiment_text = "Bullish" if top_sentiment > 0.05 else "Bearish" if top_sentiment < -0.05 else "Neutral"
    
    top_influencers = attention.loc[top_performer].nlargest(4)
    top_influencers = top_influencers[top_influencers.index != top_performer]
    
    prediction_state = "Bullish 🟢" if scores_df.iloc[0]['score'] > 0 else "Bearish 🔴"
    
    st.info(f"""
    **Top Prediction:** {top_performer} | **Stance:** {prediction_state}

    **Reasoning:** The model's prediction is based on a combination of factors:
    - **Sentiment:** News sentiment for {top_performer} is currently **{sentiment_text} ({top_sentiment:.2f})**.
    - **Key Influencers:** The model is paying most attention to **{', '.join(top_influencers.index[:2])}**.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Performance Score Ranking:")
        st.dataframe(scores_df.style.format("{:.3f}").background_gradient(cmap='viridis'), use_container_width=True)
    with col2:
        st.write(f"Attention on {top_performer}:")
        st.dataframe(top_influencers.to_frame(name="Attention Score").style.format("{:.3f}"), use_container_width=True)
    
    if show_math:
        st.info("The GAT model computes scores using an attention mechanism.")
        st.latex(r"""
        h'_i = \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} \mathbf{W}h_j\right) \quad \text{where} \quad \alpha_{ij} = \text{softmax}_j(e_{ij})
        """)

if __name__ == "__main__":
    main()