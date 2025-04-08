import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
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
        
        # Add hover text with ticker information
        hover_text = []
        for alloc in allocations:
            tickers_text = ""
            if 'tickers' in alloc and alloc['tickers']:
                tickers_text = f"<br>Tickers: {', '.join(alloc['tickers'])}"
            hover_text.append(f"{alloc['asset_class']}: {alloc['percentage']:.1f}%{tickers_text}")
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors,
            hoverinfo="text",
            text=hover_text,
            textinfo="label+percent",
            textfont_size=12,
            hole=0.4
        )])
        
        fig.update_layout(
            title=title,
            showlegend=True,
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
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
        tickers = []
        returns = []
        risks = []
        names = []
        
        for ticker, data in tickers_data.items():
            if 'annual_return' in data and data['annual_return'] is not None and \
               'volatility' in data and data['volatility'] is not None:
                tickers.append(ticker)
                returns.append(data['annual_return'] * 100)  # Convert to %
                risks.append(data['volatility'] * 100)  # Convert to %
                
                # Get company name if available
                company_name = ticker
                if 'company_name' in data and data['company_name']:
                    company_name = data['company_name']
                names.append(company_name)
        
        # Create scatter plot
        fig = go.Figure()
        
        # Add individual assets
        if tickers:
            fig.add_trace(go.Scatter(
                x=risks,
                y=returns,
                mode='markers',
                marker=dict(
                    size=10,
                    color='blue',
                    opacity=0.7
                ),
                text=[f"{n} ({t})" for n, t in zip(names, tickers)],
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
        max_risk = max(risks + [portfolio_risk] if portfolio_data else risks) * 1.2 if risks else 20
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
            hovermode="closest",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    @staticmethod
    def plot_historical_performance(price_data, tickers, title="Historical Performance", display_names=None):
        """
        Create a line chart of historical performance
        
        Args:
            price_data (dict): Dictionary with ticker as key and DataFrame as value
            tickers (list): List of tickers to include
            title (str): Chart title
            display_names (dict): Optional dictionary mapping tickers to display names
            
        Returns:
            fig: Plotly figure object
        """
        fig = go.Figure()
        
        for ticker in tickers:
            if ticker in price_data:
                df = price_data[ticker]
                if not df.empty and 'Close' in df.columns:
                    # Normalize to 100 for comparison
                    normalized = df['Close'] / df['Close'].iloc[0] * 100
                    
                    # Use display name if available
                    display_name = ticker
                    if display_names and ticker in display_names:
                        display_name = display_names[ticker]
                    
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=normalized,
                        mode='lines',
                        name=display_name,
                        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{y:.2f}<extra></extra>"
                    ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Normalized Price (100 = Start)",
            height=500,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
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
        # Ensure we have valid data
        if correlation_matrix.empty:
            # Create a dummy correlation matrix
            correlation_matrix = pd.DataFrame([[1.0]], index=['No data'], columns=['No data'])
        
        fig = px.imshow(
            correlation_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            aspect="auto"
        )
        
        fig.update_layout(
            title="Asset Correlation Matrix",
            height=600,
            margin=dict(l=20, r=20, t=40, b=20)
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
        names = []
        
        for ticker, data in ticker_data.items():
            dividend_yield = None
            
            # Try to get dividend yield from stats if available
            if 'stats' in data and 'dividendYield' in data['stats']:
                dividend_yield = data['stats']['dividendYield']
                if dividend_yield is not None:
                    dividend_yield = dividend_yield * 100  # Convert to percentage
            
            # Get company name if available
            company_name = ticker
            if 'company_name' in data:
                company_name = data['company_name']
            
            if dividend_yield is not None and dividend_yield > 0:
                tickers.append(ticker)
                yields.append(dividend_yield)
                names.append(company_name)
        
        # Sort by yield
        if tickers:
            sorted_indices = np.argsort(yields)[::-1]  # Descending
            tickers = [tickers[i] for i in sorted_indices]
            yields = [yields[i] for i in sorted_indices]
            names = [names[i] for i in sorted_indices]
            
            hover_text = [f"{n} ({t})<br>Yield: {y:.2f}%" for n, t, y in zip(names, tickers, yields)]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=tickers,
                    y=yields,
                    marker_color='darkblue',
                    text=hover_text,
                    hoverinfo="text"
                )
            ])
            
            fig.update_layout(
                title=title,
                xaxis_title="Ticker",
                yaxis_title="Dividend Yield (%)",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
        else:
            # Create empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5,
                text="No dividend data available",
                font=dict(size=14),
                showarrow=False
            )
            
            fig.update_layout(
                title=title,
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
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
        
        if sectors:
            # Create hover text with tickers
            sector_tickers = {}
            for ticker, sector in sector_map.items():
                if ticker in ticker_allocation:
                    if sector not in sector_tickers:
                        sector_tickers[sector] = []
                    sector_tickers[sector].append(ticker)
            
            hover_text = []
            for sector in sectors:
                tickers_text = ""
                if sector in sector_tickers and sector_tickers[sector]:
                    tickers_text = f"<br>Tickers: {', '.join(sector_tickers[sector])}"
                hover_text.append(f"{sector}: {sector_weights[sector]:.1f}%{tickers_text}")
            
            fig = go.Figure(data=[go.Pie(
                labels=sectors,
                values=weights,
                hoverinfo="text",
                text=hover_text,
                textinfo="label+percent",
                marker=dict(
                    colors=px.colors.qualitative.Plotly
                )
            )])
            
            fig.update_layout(
                title=title,
                height=500,
                margin=dict(l=20, r=20, t=40, b=20)
            )
        else:
            # Create empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5,
                text="No sector data available",
                font=dict(size=14),
                showarrow=False
            )
            
            fig.update_layout(
                title=title,
                height=500,
                margin=dict(l=20, r=20, t=40, b=20)
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
                },
                'overall_sentiment_score': 0.2
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
            marker_color=bar_colors,
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>"
        ))
        
        # Add overall sentiment score
        overall_score = sentiment_data.get('overall_sentiment_score', 0)
        sentiment_label = "Neutral"
        if overall_score > 0.2:
            sentiment_label = "Positive"
        elif overall_score < -0.2:
            sentiment_label = "Negative"
        
        fig.add_annotation(
            x=0.5, y=-0.15,
            xref="paper", yref="paper",
            text=f"Overall Market Sentiment: {sentiment_label} (Score: {overall_score:.2f})",
            showarrow=False,
            font=dict(size=14)
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Percentage (%)",
            height=300,
            margin=dict(l=20, r=20, t=40, b=40),
            xaxis=dict(range=[0, 100])
        )
        
        return fig

    @staticmethod
    def plot_sector_performance(sector_data, title="Sector Performance (Last 30 Days)"):
        """
        Create a bar chart of sector performance
        
        Args:
            sector_data (list): List of dictionaries with sector performance data
            title (str): Chart title
            
        Returns:
            fig: Plotly figure object
        """
        if not sector_data:
            # Create empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5,
                text="No sector performance data available",
                font=dict(size=14),
                showarrow=False
            )
            
            fig.update_layout(
                title=title,
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            return fig
        
        # Extract data
        sectors = [item['Sector'] for item in sector_data]
        performances = [item['Performance (%)'] for item in sector_data]
        etfs = [item['ETF'] for item in sector_data]
        
        # Create hover text
        hover_text = [f"{s} ({e})<br>Performance: {p:.2f}%" for s, e, p in zip(sectors, etfs, performances)]
        
        # Define colors based on performance
        colors = ['green' if p >= 0 else 'red' for p in performances]
        
        # Create horizontal bar chart, sorted by performance
        fig = go.Figure()
        
        # Sort data by performance
        sorted_indices = np.argsort(performances)
        sorted_sectors = [sectors[i] for i in sorted_indices]
        sorted_performances = [performances[i] for i in sorted_indices]
        sorted_hover_text = [hover_text[i] for i in sorted_indices]
        sorted_colors = [colors[i] for i in sorted_indices]
        
        fig.add_trace(go.Bar(
            y=sorted_sectors,
            x=sorted_performances,
            orientation='h',
            marker_color=sorted_colors,
            text=sorted_hover_text,
            hoverinfo="text"
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Performance (%)",
            height=500,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    @staticmethod
    def plot_portfolio_metrics(portfolio_stats, benchmark_stats=None, title="Portfolio Risk Metrics"):
        """
        Create visualization of portfolio risk metrics compared to benchmark
        
        Args:
            portfolio_stats (dict): Portfolio statistics
            benchmark_stats (dict): Optional benchmark statistics
            title (str): Chart title
            
        Returns:
            fig: Plotly figure object
        """
        # Key metrics to display
        metrics = ['expected_return', 'volatility', 'sharpe_ratio', 'max_drawdown']
        metric_names = ['Expected Return (%)', 'Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)']
        
        # Extract portfolio values
        portfolio_values = []
        for metric in metrics:
            value = portfolio_stats.get(metric, 0)
            # Convert to percentage for return, volatility, and drawdown
            if metric in ['expected_return', 'volatility']:
                value = value * 100 if value else 0
            elif metric == 'max_drawdown':
                value = value * 100 if value else 0
                value = abs(value)  # Make positive for display
            portfolio_values.append(value)
        
        # Create figure
        fig = go.Figure()
        
        # Add portfolio trace
        fig.add_trace(go.Bar(
            x=metric_names,
            y=portfolio_values,
            name='Portfolio',
            marker_color='blue'
        ))
        
        # Add benchmark trace if provided
        if benchmark_stats:
            benchmark_values = []
            for metric in metrics:
                value = benchmark_stats.get(metric, 0)
                # Convert to percentage for return, volatility, and drawdown
                if metric in ['expected_return', 'volatility']:
                    value = value * 100 if value else 0
                elif metric == 'max_drawdown':
                    value = value * 100 if value else 0
                    value = abs(value)  # Make positive for display
                benchmark_values.append(value)
            
            fig.add_trace(go.Bar(
                x=metric_names,
                y=benchmark_values,
                name='Benchmark',
                marker_color='gray'
            ))
        
        fig.update_layout(
            title=title,
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            barmode='group'
        )
        
        return fig
    
    @staticmethod
    def plot_asset_growth(investment_amount, expected_return, years=30, title="Projected Growth"):
        """
        Create chart showing projected investment growth over time
        
        Args:
            investment_amount (float): Initial investment amount
            expected_return (float): Expected annual return (decimal, e.g., 0.07 for 7%)
            years (int): Number of years to project
            title (str): Chart title
            
        Returns:
            fig: Plotly figure object
        """
        # Calculate growth year by year
        year_values = list(range(years + 1))
        investment_values = [investment_amount * (1 + expected_return) ** year for year in year_values]
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=year_values,
            y=investment_values,
            mode='lines+markers',
            marker=dict(size=6),
            line=dict(width=2),
            name='Investment Value'
        ))
        
        # Add annotations for key years
        for year in [5, 10, 20, years]:
            if year <= years:
                value = investment_amount * (1 + expected_return) ** year
                fig.add_annotation(
                    x=year,
                    y=value,
                    text=f"${value:,.0f}",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-30
                )
        
        fig.update_layout(
            title=title,
            xaxis_title="Years",
            yaxis_title="Value ($)",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(tickformat="$,.0f")
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