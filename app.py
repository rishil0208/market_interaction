"""
Market Interaction-Based Performance Predictor
A Streamlit dashboard for visualizing predicted stock performance based on
a graph-based interaction model.
"""
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

# --- Project-specific imports ---
from config import TICKERS, LOOKBACK_DAYS, CORRELATION_THRESHOLD, CUSTOM_CSS
from data_engine import fetch_data, calculate_features, get_adjacency_matrix
from model_logic import InteractionScorer

# --- Page Configuration ---
st.set_page_config(
    page_title="Market Interaction Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply custom CSS for styling
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Main Application Logic ---
def main():
    """Main function to run the Streamlit app."""
    st.title("Market Interaction-Based Performance Predictor")
    st.markdown("Predicting next-day relative stock performance by modeling inter-company relationships.")

    try:
        # --- Data Loading and Feature Engineering ---
        with st.spinner("Fetching live market data and building interaction graph..."):
            price_data = fetch_data(TICKERS)
            log_returns, volatility, rsi = calculate_features(price_data, LOOKBACK_DAYS)
            
            # Use the latest available data for features
            latest_rsi = rsi.iloc[-1]
            latest_volatility = volatility.iloc[-1]
            latest_returns = log_returns.iloc[-1]
            
            feature_df = pd.concat([latest_returns, latest_volatility, latest_rsi], axis=1)
            feature_df.columns = ['Return', 'Volatility', 'RSI']
            
            adjacency_matrix = get_adjacency_matrix(price_data, LOOKBACK_DAYS, CORRELATION_THRESHOLD)

        # --- Model Prediction ---
        with st.spinner("Running interaction model to calculate performance scores..."):
            scorer = InteractionScorer(feature_matrix=feature_df, adjacency_matrix=adjacency_matrix)
            scores_df, attention_df = scorer.calculate_scores()

        # --- UI Rendering ---
        render_dashboard(scores_df, adjacency_matrix, attention_df)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.warning("Could not fetch data. Please check your internet connection or the ticker symbols.")

def render_dashboard(scores_df, adjacency_matrix, attention_df):
    """Renders the main dashboard components."""
    
    # --- Section 1: The High-Level Signal ---
    st.header("Predicted Performance Ranking")
    top_3_col, mid_3_col, last_3_col = st.columns(3)
    
    ranked_tickers = scores_df.index.tolist()
    
    with top_3_col:
        st.metric(label=f"🥇 Rank 1: {ranked_tickers[0]}", value=f"{scores_df.iloc[0]['score']:.2f}")
    with mid_3_col:
        st.metric(label=f"🥈 Rank 2: {ranked_tickers[1]}", value=f"{scores_df.iloc[1]['score']:.2f}")
    with last_3_col:
        st.metric(label=f"🥉 Rank 3: {ranked_tickers[2]}", value=f"{scores_df.iloc[2]['score']:.2f}")

    # --- Section 2: The Market Interaction Map ---
    st.header("Today's Market Interaction Map")
    
    # Create a graph from the adjacency matrix
    G = nx.from_pandas_adjacency(adjacency_matrix)
    
    # Check if graph is empty
    if not G.nodes():
        st.warning("No strong correlations found with the current threshold. The interaction graph is empty.")
    else:
        # Create an interactive plot with Plotly
        pos = nx.spring_layout(G, seed=42, k=0.9)
        
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#444'),
            hoverinfo='none',
            mode='lines')

        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale='Greens',
                reversescale=False,
                color=[],
                size=20,
                colorbar=dict(
                    thickness=15,
                    title=dict(text='Performance Score', side='right'),
                    xanchor='left'
                ),
                line_width=2))

        # Color nodes by their performance score
        node_adjacencies = []
        node_scores = []
        for node in G.nodes():
            score = scores_df.loc[node]['score']
            node_scores.append(score)
        
        node_trace.marker.color = node_scores

        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0,l=0,r=0,t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                        ))
        
        st.plotly_chart(fig, use_container_width=True)

    # --- Section 3: Explainability Deep Dive ---
    st.header("Why The Model Made Its Predictions")
    
    selected_ticker = st.selectbox(
        "Select a stock to see its influences:",
        options=ranked_tickers
    )
    
    if selected_ticker:
        # Get the top 3 influencers for the selected stock
        influencers = attention_df.loc[selected_ticker].drop(selected_ticker).nlargest(3)
        
        explanation = f"The model's prediction for **{selected_ticker}** is primarily driven by its interaction with the following companies:"
        st.markdown(explanation)
        
        for influencer, weight in influencers.items():
            if weight > 0.01: # Only show meaningful influences
                st.info(f"**{influencer}**: Contributed **{weight:.1%}** of the influence score. This is due to their strong recent market correlation and {influencer}'s own feature profile (e.g., momentum, volatility).")

if __name__ == "__main__":
    main()
