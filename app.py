"""
Real-Time Market Interaction Monitor
A live, "Bloomberg style" Streamlit dashboard for monitoring intraday stock
performance and interactions using a real-time graph model.
"""
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import time

# --- Project-specific imports ---
from config import TICKERS, LOOKBACK_MINUTES, CORRELATION_THRESHOLD, CUSTOM_CSS
from data_engine import fetch_live_data, get_live_features, get_live_correlation_matrix
from model_logic import InteractionScorer

# --- Page Configuration ---
st.set_page_config(
    page_title="🔴 LIVE Market Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for styling
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Auto-Refresh ---
# Trigger a refresh every 60 seconds (60,000 milliseconds)
st_autorefresh(interval=60 * 1000, key="data_refresh")

# --- Main Application Logic ---
def render_ticker_tape(data: pd.DataFrame):
    """Renders the top 'ticker tape' of metrics."""
    st.header("Live Ticker Tape")
    
    # Get the last two rows to calculate change
    last_row = data.iloc[-1]
    prev_row = data.iloc[-2]

    # Display metrics for the top 4 tickers
    cols = st.columns(4)
    for i, ticker in enumerate(TICKERS[:4]):
        with cols[i]:
            price = last_row[ticker]
            change = ((price - prev_row[ticker]) / prev_row[ticker]) * 100
            st.metric(
                label=f"{ticker}",
                value=f"${price:,.2f}",
                delta=f"{change:.2f}%"
            )

def render_live_graph(scores_df, correlation_matrix, adjacency_matrix):
    """Renders the live, interactive network graph."""
    st.subheader("Live Market Interaction Graph")
    
    # Create graph from adjacency matrix; add correlation as an edge attribute
    G = nx.from_pandas_adjacency(adjacency_matrix)
    
    # Check if graph is empty
    if not G.nodes():
        st.warning("No strong correlations found with the current threshold. The interaction graph is empty.")
        return

    # Add correlation weights to edges
    for i, j in G.edges():
        G.edges[i, j]['weight'] = correlation_matrix.loc[i, j]

    pos = nx.spring_layout(G, seed=42, k=0.9, iterations=50)
    
    edge_traces = []
    # Create a trace for each edge to control color and width individually
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        corr = edge[2]['weight']
        
        # Thicker lines for stronger correlations
        width = 1 + (abs(corr) - CORRELATION_THRESHOLD) * 10
        # Green for positive, red for negative correlation
        color = '#00f900' if corr > 0 else '#ff0000'

        edge_trace = go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            line=dict(width=width, color=color),
            hoverinfo='none',
            mode='lines')
        edge_traces.append(edge_trace)

    node_x, node_y, node_text, node_scores = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_scores.append(scores_df.loc[node, 'score'])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='viridis',
            reversescale=False,
            color=node_scores,
            size=25,
            colorbar=dict(
                thickness=15,
                title=dict(text='Performance Score', side='right'),
                xanchor='left'
            ),
            line_width=2))

    layout = go.Layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0,l=0,r=0,t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    fig = go.Figure(data=edge_traces + [node_trace], layout=layout)
    st.plotly_chart(fig, use_container_width=True)


def render_heatmap(correlation_matrix):
    """Renders a heatmap of the live correlation matrix."""
    st.subheader("Live Correlation Heatmap")
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#1E1E1E',
        font=dict(color='white')
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main function to run the Streamlit app."""
    last_update_time = time.strftime('%H:%M:%S')
    st.title(f"🔴 LIVE Market Interaction Monitor")
    st.caption(f"Last updated: {last_update_time}")

    # --- Data Loading and Feature Engineering ---
    price_data = fetch_live_data(TICKERS)

    # Handle case where markets are closed or data is unavailable
    if price_data is None or len(price_data) < LOOKBACK_MINUTES:
        st.error("Could not retrieve sufficient live market data. Markets may be closed or data source is unavailable.")
        return

    feature_df = get_live_features(price_data, LOOKBACK_MINUTES)
    correlation_matrix, adjacency_matrix = get_live_correlation_matrix(
        price_data, LOOKBACK_MINUTES, CORRELATION_THRESHOLD
    )

    # --- Model Prediction ---
    try:
        scorer = InteractionScorer(feature_matrix=feature_df, adjacency_matrix=adjacency_matrix)
        scores_df, attention_df = scorer.calculate_scores()
    except ValueError as e:
        st.error(f"Model Error: {e}. This can happen at market open when data is sparse.")
        return

    # --- UI Rendering ---
    render_ticker_tape(price_data)
    
    st.divider()

    col1, col2 = st.columns([3, 2]) # Give more space to the graph

    with col1:
        render_live_graph(scores_df, correlation_matrix, adjacency_matrix)

    with col2:
        st.subheader("Predicted Short-Term Movers")
        st.dataframe(
            scores_df.style.format({'score': "{:.3f}"}).background_gradient(cmap='viridis'),
            use_container_width=True
        )

    st.divider()
    render_heatmap(correlation_matrix)


if __name__ == "__main__":
    main()