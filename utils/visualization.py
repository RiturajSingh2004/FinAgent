import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64

class Visualizer:
    """Class for creating portfolio and market visualizations"""
    
    @staticmethod
    def plot_asset_allocation(allocations, title="Asset Allocation"):
        """
        Create a pie chart of asset allocations
        
        Args:
            allocations (list): List of allocation dictionaries with asset_class and percentage
            title (str): Chart title
            
        Returns:
            fig: Plotly figure object
        """
        # Extract labels and values
        labels = [alloc["asset_class"] for alloc in allocations]
        values = [alloc["percentage"] for alloc in allocations]
        
        # Create color map based on asset class
        color_map = {
            'US Large Cap': '#1f77b4',
            'US Mid Cap': '#2ca02c',
            'US Small Cap': '#7f7f7f',
            'International Developed': '#9467bd',
            'Emerging Markets': '#17becf',
            'US Bonds': '#ff7f0e',
            'Real Estate': '#d62728',
            'Commodities': '#8c564b',
            'Cash': '#e377c2',
            'Cryptocurrency': '#bcbd22'
        }
        
        colors = [color_map.get(label, '#1f77b4') for label in labels]
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors,
            hoverinfo="label+percent",
            textinfo="label+percent",
            textfont_size=12,
            hole=0.4
        )])
        
        fig.update_layout(
            title=title,
            showlegend=True,
            height=500,
            width=700
        )
        
        return fig
    
    @staticmethod
    def plot_risk_return_profile(tickers_data, portfolio_data=None):
        """
        Create a risk-return scatter plot
        
        Args:
            tickers_data (dict): Dictionary with ticker data including returns and volatility
            portfolio_data (dict): Optional portfolio statistics
            
        Returns:
            fig: Plotly figure object
        """
        # Extract data
        tickers = list(tickers_data.keys())
        returns = [tickers_data[t].get('annual_return', 0) * 100 for t in tickers]  # Convert to %
        risks = [tickers_data[t].get('volatility', 0) * 100 for t in tickers]  # Convert to %
        
        # Create scatter plot
        fig = go.Figure()
        
        # Add individual assets
        fig.add_trace(go.Scatter(
            x=risks,
            y=returns,
            mode='markers',
            marker=dict(
                size=10,
                color='blue',
                opacity=0.7
            ),
            text=tickers,
            hovertemplate="<b>%{text}</b><br>Return: %{y:.2f}%<br>Risk: %{x:.2f}%<extra></extra>"
        ))
        
        # Add portfolio if provided
        if portfolio_data:
            portfolio_return = portfolio_data.get('expected_return', 0) * 100  # Convert to %
            portfolio_risk = portfolio_data.get('volatility', 0) * 100  # Convert to %
            
            fig.add_trace(go.Scatter(
                x=[portfolio_risk],
                y=[portfolio_return],
                mode='markers',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='star'
                ),
                name='Portfolio',
                hovertemplate="<b>Portfolio</b><br>Return: %{y:.2f}%<br>Risk: %{x:.2f}%<extra></extra>"
            ))
        
        # Add capital market line
        risk_free_rate = 2.0  # 2% risk-free rate
        max_risk = max(risks + [portfolio_risk] if portfolio_data else risks) * 1.2
        fig.add_trace(go.Scatter(
            x=[0, max_risk],
            y=[risk_free_rate, risk_free_rate + max_risk * 0.5],
            mode='lines',
            line=dict(color='green', dash='dash'),
            name='Capital Market Line'
        ))
        
        # Update layout
        fig.update_layout(
            title="Risk-Return Profile",
            xaxis_title="Risk (Volatility, %)",
            yaxis_title="Expected Return (%)",
            height=500,
            width=700,
            hovermode="closest",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    @staticmethod
    def plot_historical_performance(price_data, tickers, title="Historical Performance"):
        """
        Create a line chart of historical performance
        
        Args:
            price_data (dict): Dictionary with ticker as key and DataFrame as value
            tickers (list): List of tickers to include
            title (str): Chart title
            
        Returns:
            fig: Plotly figure object
        """
        fig = go.Figure()
        
        for ticker in tickers:
            if ticker in price_data:
                df = price_data[ticker]
                if not df.empty:
                    # Normalize to 100 for comparison
                    normalized = df['Close'] / df['Close'].iloc[0] * 100
                    
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=normalized,
                        mode='lines',
                        name=ticker
                    ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Normalized Price (100 = Start)",
            height=500,
            width=800,
            hovermode="x unified"
        )
        
        return fig
    
    @staticmethod
    def plot_correlation_matrix(correlation_matrix):
        """
        Create a heatmap of the correlation matrix
        
        Args:
            correlation_matrix (pd.DataFrame): Correlation matrix
            
        Returns:
            fig: Plotly figure object
        """
        fig = px.imshow(
            correlation_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1
        )
        
        fig.update_layout(
            title="Asset Correlation Matrix",
            height=600,
            width=700
        )
        
        return fig
    
    @staticmethod
    def plot_dividend_yield(ticker_data, title="Dividend Yield Comparison"):
        """
        Create a bar chart of dividend yields
        
        Args:
            ticker_data (dict): Dictionary with ticker data including dividend yields
            title (str): Chart title
            
        Returns:
            fig: Plotly figure object
        """
        tickers = []
        yields = []
        
        for ticker, data in ticker_data.items():
            if 'dividend_yield' in data:
                tickers.append(ticker)
                yields.append(data['dividend_yield'] * 100)  # Convert to percentage
        
        # Sort by yield
        sorted_indices = np.argsort(yields)[::-1]  # Descending
        tickers = [tickers[i] for i in sorted_indices]
        yields = [yields[i] for i in sorted_indices]
        
        fig = go.Figure(data=[
            go.Bar(
                x=tickers,
                y=yields,
                marker_color='darkblue'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Ticker",
            yaxis_title="Dividend Yield (%)",
            height=400,
            width=700
        )
        
        return fig
    
    @staticmethod
    def plot_sector_breakdown(ticker_allocation, sector_map, title="Sector Breakdown"):
        """
        Create a pie chart of sector allocations
        
        Args:
            ticker_allocation (dict): Ticker allocations
            sector_map (dict): Mapping of tickers to sectors
            title (str): Chart title
            
        Returns:
            fig: Plotly figure object
        """
        # Calculate sector weights
        sector_weights = {}
        for ticker, weight in ticker_allocation.items():
            sector = sector_map.get(ticker, 'Unknown')
            if sector not in sector_weights:
                sector_weights[sector] = 0
            sector_weights[sector] += weight
        
        # Prepare data for pie chart
        sectors = list(sector_weights.keys())
        weights = list(sector_weights.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=sectors,
            values=weights,
            hoverinfo="label+percent",
            textinfo="label+percent",
            marker=dict(
                colors=px.colors.qualitative.Plotly
            )
        )])
        
        fig.update_layout(
            title=title,
            height=500,
            width=700
        )
        
        return fig
    
    @staticmethod
    def plot_sentiment_analysis(sentiment_data, title="Market Sentiment Analysis"):
        """
        Create a visualization of sentiment analysis results
        
        Args:
            sentiment_data (dict): Sentiment analysis results
            title (str): Chart title
            
        Returns:
            fig: Plotly figure object
        """
        if not sentiment_data or 'sentiment_percentages' not in sentiment_data:
            # Create dummy data if no sentiment data is available
            sentiment_data = {
                'sentiment_percentages': {
                    'positive': 40,
                    'neutral': 40, 
                    'negative': 20
                }
            }
        
        # Extract sentiment percentages
        sentiments = list(sentiment_data['sentiment_percentages'].keys())
        percentages = list(sentiment_data['sentiment_percentages'].values())
        
        # Define colors
        colors = {
            'positive': 'green',
            'neutral': 'gray',
            'negative': 'red'
        }
        
        # Create color list
        bar_colors = [colors.get(s, 'blue') for s in sentiments]
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=sentiments,
            x=percentages,
            orientation='h',
            marker_color=bar_colors
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Percentage (%)",
            height=300,
            width=700,
            xaxis=dict(range=[0, 100])
        )
        
        return fig

    @staticmethod
    def fig_to_base64(fig):
        """
        Convert Plotly figure to base64 image for display
        
        Args:
            fig: Plotly figure object
            
        Returns:
            str: Base64 encoded image
        """
        img_bytes = fig.to_image(format="png")
        encoded = base64.b64encode(img_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"