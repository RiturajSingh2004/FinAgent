import streamlit as st
import pandas as pd
import numpy as np
import os
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import project modules
from data.market_data import MarketDataProvider
from data.fundamentals import FundamentalDataProvider
from data.data_utils import DataUtils
from models.sentiment import SentimentAnalyzer
from models.text_generator import TextGenerator
from models.portfolio import PortfolioGenerator, PortfolioOptimizer
from utils.visualization import Visualizer
from utils.risk_profile import RiskProfiler

# Set page config
st.set_page_config(
    page_title="FinAgent - ML Investment Strategist",
    page_icon="📈",
    layout="wide"
)

# Initialize data providers
market_data = MarketDataProvider()
fundamental_data = FundamentalDataProvider()
    
# Initialize models
sentiment_analyzer = SentimentAnalyzer()
text_generator = TextGenerator()
portfolio_generator = PortfolioGenerator(market_data, fundamental_data)

# Initialize utilities
visualizer = Visualizer()
risk_profiler = RiskProfiler()

# App title and description
st.title("FinAgent - ML Investment Strategist")
st.subheader("AI-Powered Investment Portfolio Generator")
st.markdown("""
This application helps you create a personalized investment portfolio based on your financial profile and risk tolerance.
It leverages machine learning models for market sentiment analysis and portfolio optimization.
""")

# Sidebar for user inputs
st.sidebar.header("Your Financial Profile")

# User profile form
with st.sidebar.form("user_profile_form"):
    st.subheader("Personal Information")
    
    # Basic info
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    income = st.number_input("Annual Income ($)", min_value=0, value=70000, step=5000)
    
    # Investment details
    st.subheader("Investment Details")
    investment_amount = st.number_input("Initial Investment Amount ($)", min_value=1000, value=10000, step=1000)
    investment_horizon_options = {
        "short_term": "Short-term (1-3 years)",
        "medium_term": "Medium-term (3-10 years)",
        "long_term": "Long-term (10+ years)"
    }
    investment_horizon = st.selectbox(
        "Investment Time Horizon", 
        options=list(investment_horizon_options.keys()),
        format_func=lambda x: investment_horizon_options[x],
        index=1
    )
    
    # Risk assessment
    st.subheader("Risk Assessment")
    
    risk_comfort = st.select_slider(
        "Risk Comfort Level",
        options=["very_low", "low", "moderate", "high", "very_high"],
        value="moderate",
        format_func=lambda x: x.replace("_", " ").title()
    )
    
    loss_reaction = st.radio(
        "If your investment lost 20% in a month, what would you do?",
        options=[
            "sell_all", 
            "sell_some", 
            "do_nothing", 
            "buy_some_more", 
            "buy_significantly_more"
        ],
        format_func=lambda x: {
            "sell_all": "Sell everything",
            "sell_some": "Sell some investments",
            "do_nothing": "Do nothing and wait it out",
            "buy_some_more": "Buy a little more at lower prices",
            "buy_significantly_more": "Buy significantly more at lower prices"
        }[x],
        index=2
    )
    
    investment_knowledge = st.select_slider(
        "Investment Knowledge",
        options=["none", "basic", "average", "advanced", "expert"],
        value="average",
        format_func=lambda x: x.title()
    )
    
    # Investment goals
    st.subheader("Investment Goals")
    investment_goal = st.radio(
        "Primary Investment Goal",
        options=[
            "capital_preservation", 
            "income", 
            "balanced", 
            "growth", 
            "aggressive_growth"
        ],
        format_func=lambda x: {
            "capital_preservation": "Capital Preservation (minimize risk of loss)",
            "income": "Income Generation (regular dividends/interest)",
            "balanced": "Balanced Growth and Income",
            "growth": "Long-term Growth",
            "aggressive_growth": "Maximum Growth Potential"
        }[x],
        index=2
    )
    
    income_stability = st.select_slider(
        "Income Stability",
        options=["unstable", "somewhat_stable", "stable", "very_stable", "extremely_stable"],
        value="stable",
        format_func=lambda x: x.replace("_", " ").title()
    )
    
    # Financial situation
    st.subheader("Financial Situation")
    has_emergency_fund = st.checkbox("I have an emergency fund (3-6 months of expenses)", value=True)
    has_high_debt = st.checkbox("I have high-interest debt (credit cards, personal loans)", value=False)
    
    # Sector preferences
    st.subheader("Investment Preferences (Optional)")
    exclude_sectors = st.multiselect(
        "Sectors to Exclude (if any)",
        options=["Technology", "Financial Services", "Healthcare", "Energy", "Consumer Cyclical", 
                "Consumer Defensive", "Industrials", "Basic Materials", "Communication Services", 
                "Utilities", "Real Estate"]
    )
    
    custom_preferences = st.text_area(
        "Any specific investment preferences or constraints?",
        placeholder="e.g., interested in sustainable investments, prefer dividend stocks, etc."
    )
    
    # Submit button
    submitted = st.form_submit_button("Generate Portfolio")

# Process form submission
if submitted:
    # Create user profile dictionary
    risk_answers = {
        'investment_timeline': investment_horizon,
        'risk_comfort': risk_comfort,
        'loss_reaction': loss_reaction,
        'investment_goal': investment_goal,
        'income_stability': income_stability,
        'investment_knowledge': investment_knowledge
    }
    
    # Calculate risk score and profile
    risk_score = risk_profiler.calculate_risk_score(risk_answers)
    risk_profile = risk_profiler.classify_risk_profile(risk_score)
    risk_summary = risk_profiler.get_risk_profile_summary(risk_profile)
    
    # Age-based suggestion
    age_suggestion = risk_profiler.get_age_based_suggestion(age)
    
    # Financial situation
    financial_situation = {
        'emergency_fund': has_emergency_fund,
        'high_debt': has_high_debt,
        'annual_income': income
    }
    
    # Create final user profile
    user_profile = {
        'age': age,
        'income': income,
        'investment_amount': investment_amount,
        'risk_tolerance': risk_profile,
        'risk_score': risk_score,
        'investment_horizon': investment_horizon,
        'investment_goal': investment_goal,
        'exclude_sectors': exclude_sectors,
        'custom_preferences': custom_preferences,
        'financial_situation': financial_situation
    }
    
    # Display progress message
    with st.spinner("Analyzing market data and generating your personalized portfolio..."):
        try:
            # Generate portfolio
            portfolio = portfolio_generator.generate_portfolio(user_profile)
            
            # Enhance with actual market data
            portfolio = portfolio_generator.enhance_portfolio_with_market_data(portfolio)
            
            # Get up-to-date market news from Yahoo Finance
            market_news = []
            try:
                # Try to get real market news using yfinance
                news_data = yf.Ticker("^GSPC").news  # S&P 500 news
                if news_data and len(news_data) >= 5:
                    for i in range(min(5, len(news_data))):
                        if 'title' in news_data[i]:
                            market_news.append(news_data[i]['title'])
                
                # If we couldn't get enough news, add some default items
                if len(market_news) < 3:
                    raise Exception("Not enough news items retrieved")
                    
            except Exception as e:
                st.warning(f"Could not retrieve live market news (Error: {str(e)}). Using recent general market news instead.")
                # Use recent general market news as fallback
                market_news = [
                    "Federal Reserve maintains current interest rates in latest meeting",
                    "Tech sector reports quarterly earnings above analyst expectations",
                    "Consumer confidence index shows improvement in economic outlook",
                    "International markets respond to latest economic indicators",
                    "Supply chain improvements reported across multiple industries"
                ]
            
            # Analyze market sentiment using the news
            sentiment_results = sentiment_analyzer.analyze_multiple_texts(market_news)
            market_sentiment = sentiment_analyzer.get_sentiment_summary(sentiment_results)
            
            # Generate investment rationale
            try:
                investment_rationale = text_generator.generate_investment_rationale(
                    portfolio, user_profile, market_sentiment
                )
                
                # Check if the investment rationale is valid (not the repetitive text issue)
                if (investment_rationale and 
                    "Write in a professional" in investment_rationale and 
                    "financial advice" in investment_rationale):
                    # Fall back to template if we got the repetitive text
                    st.warning("Using template-based investment rationale due to generation issue.")
                    investment_rationale = text_generator._generate_template_rationale(
                        risk_profile, 
                        investment_horizon,
                        "neutral" if not market_sentiment else 
                            ("positive" if market_sentiment.get('overall_sentiment_score', 0) > 0.2 else
                            "negative" if market_sentiment.get('overall_sentiment_score', 0) < -0.2 else "neutral"),
                        portfolio['allocations'],
                        investment_goal,
                        "neutral, with mixed signals that suggest a balanced approach",
                        "3-10 years" if investment_horizon == "medium_term" else 
                            "1-3 years" if investment_horizon == "short_term" else "10+ years"
                    )
            except Exception as e:
                st.warning(f"Could not generate custom investment rationale: {str(e)}")
                # Fall back to template
                investment_rationale = text_generator._generate_template_rationale(
                    risk_profile, 
                    investment_horizon,
                    "neutral" if not market_sentiment else 
                        ("positive" if market_sentiment.get('overall_sentiment_score', 0) > 0.2 else
                        "negative" if market_sentiment.get('overall_sentiment_score', 0) < -0.2 else "neutral"),
                    portfolio['allocations'],
                    investment_goal,
                    "neutral, with mixed signals that suggest a balanced approach",
                    "3-10 years" if investment_horizon == "medium_term" else 
                        "1-3 years" if investment_horizon == "short_term" else "10+ years"
                )
            
            # Get real market data for historical performance
            tickers = list(portfolio.get('ticker_allocation', {}).keys())
            
            # Add benchmark indexes for comparison
            benchmark_tickers = ['SPY', 'AGG']  # S&P 500 ETF and Bond Aggregate ETF
            all_tickers = list(set(tickers + benchmark_tickers))
            
            # Get one year of historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            # Create real price data for portfolio tickers
            price_data = {}
            for ticker in all_tickers:
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(start=start_date, end=end_date)
                    if not hist.empty:
                        price_data[ticker] = hist
                except Exception as e:
                    st.warning(f"Could not retrieve data for {ticker}: {str(e)}")
            
            # Create portfolio performance data based on actual ticker allocations
            if tickers and all(t in price_data for t in tickers):
                # Calculate weighted portfolio performance
                portfolio_df = pd.DataFrame(index=price_data[tickers[0]].index)
                portfolio_df['Close'] = 0
                
                # Add weighted price of each ticker
                for ticker, weight in portfolio['ticker_allocation'].items():
                    if ticker in price_data:
                        ticker_data = price_data[ticker]['Close']
                        if len(ticker_data) > 0:
                            # Normalize to start at 100 * weight
                            normalized = ticker_data / ticker_data.iloc[0] * 100 * weight
                            # Add to portfolio
                            if len(normalized) == len(portfolio_df):
                                portfolio_df['Close'] += normalized
                            else:
                                st.warning(f"Data length mismatch for {ticker}, skipping in portfolio calculation")
                
                price_data['Portfolio'] = portfolio_df
            
            # Create benchmark names dictionary for chart
            display_names = {
                'SPY': 'S&P 500 ETF', 
                'AGG': 'US Bond Aggregate',
                'Portfolio': 'Your Portfolio'
            }
            
            # Prepare chart display tickers
            chart_tickers = ['Portfolio']
            if 'SPY' in price_data:
                chart_tickers.append('SPY')
            if 'AGG' in price_data:
                chart_tickers.append('AGG')
                
        except Exception as e:
            st.error(f"An error occurred while generating your portfolio: {str(e)}")
            st.error("Please try again or contact support if the problem persists.")
            # Still display the form
            st.stop()
    
    # Display results
    st.header("Your Investment Profile")
    
    # Create columns for profile display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Profile")
        st.markdown(f"**Category:** {risk_profile.title()}")
        st.markdown(f"**Risk Score:** {risk_score:.1f}/100")
        st.markdown(f"**Description:** {risk_summary['description']}")
        st.markdown(f"**Recommended Investment Horizon:** {risk_summary['recommended_horizon']}")
    
    with col2:
        st.subheader("Age-Based Recommendation")
        st.markdown(f"**Suggested Profile:** {age_suggestion['suggested_profile'].title()}")
        st.markdown(f"**Suggested Stock/Bond Ratio:** {age_suggestion['stock_bond_ratio']}")
        st.markdown(f"**Rationale:** {age_suggestion['rationale']}")
    
    # Display portfolio allocation
    st.header("Recommended Portfolio")
    
    # Create columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Asset allocation pie chart
        allocation_fig = visualizer.plot_asset_allocation(portfolio['allocations'])
        st.plotly_chart(allocation_fig, use_container_width=True)
    
    with col2:
        # Get asset class mapping for tickers
        ticker_asset_class_map = DataUtils.create_asset_class_mappings()
        
        # Get sector mappings
        sector_map = {ticker: DataUtils.map_ticker_to_sector(ticker) 
                     for ticker in portfolio.get('ticker_allocation', {})}
        
        # Sector breakdown chart
        sector_fig = visualizer.plot_sector_breakdown(
            portfolio.get('ticker_allocation', {}), 
            sector_map, 
            title="Sector Breakdown"
        )
        st.plotly_chart(sector_fig, use_container_width=True)
    
    # Display market sentiment
    st.header("Market Sentiment Analysis")
    
    # Create columns for news and sentiment
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Latest Market News")
        for i, news in enumerate(market_news):
            st.markdown(f"- {news}")
    
    with col2:
        sentiment_fig = visualizer.plot_sentiment_analysis(market_sentiment)
        st.plotly_chart(sentiment_fig, use_container_width=True)
    
    # Display investment rationale
    st.header("Investment Strategy Rationale")
    st.markdown(investment_rationale)
    
    # Display portfolio details
    st.header("Portfolio Details")
    
    # Create table of allocations with tickers
    allocation_data = []
    for alloc in portfolio['allocations']:
        asset_class = alloc['asset_class']
        percentage = alloc['percentage']
        tickers = ', '.join(alloc.get('tickers', []))
        allocation_data.append({
            'Asset Class': asset_class,
            'Allocation (%)': f"{percentage:.1f}%",
            'Representative Tickers': tickers
        })
    
    allocation_df = pd.DataFrame(allocation_data)
    st.table(allocation_df)
    
    # Historical performance from actual market data
    st.header("Historical Performance Analysis")
    
    if len(price_data) > 0:
        # Plot historical performance using real data
        performance_fig = visualizer.plot_historical_performance(
            price_data, chart_tickers, 
            title="One-Year Historical Performance"
        )
        st.plotly_chart(performance_fig, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        This chart shows the actual historical performance of your recommended portfolio compared to common market benchmarks 
        over the past year. The portfolio line represents the weighted performance of your allocated assets.
        """)
        
        # Display return metrics
        returns_data = []
        for ticker in chart_tickers:
            if ticker in price_data:
                df = price_data[ticker]
                if not df.empty:
                    # Calculate return metrics
                    total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
                    
                    # Calculate volatility (standard deviation of daily returns)
                    daily_returns = df['Close'].pct_change().dropna()
                    volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized, in percent
                    
                    # Add to returns data
                    returns_data.append({
                        'Investment': display_names.get(ticker, ticker),
                        'Total Return (%)': f"{total_return:.2f}%",
                        'Annualized Volatility (%)': f"{volatility:.2f}%"
                    })
        
        # Display returns table
        if returns_data:
            st.subheader("Performance Metrics")
            returns_df = pd.DataFrame(returns_data)
            st.table(returns_df)
    else:
        st.info("Historical performance data could not be retrieved at this time. Please try again later.")
    
    # Calculate and display portfolio risk metrics
    st.header("Portfolio Risk Analysis")
    
    # Use only stocks that have data available
    valid_tickers = [t for t in tickers if t in price_data]
    
    if valid_tickers:
        # Calculate correlation matrix
        returns_data = {}
        for ticker in valid_tickers:
            if ticker in price_data:
                df = price_data[ticker]
                if not df.empty:
                    returns_data[ticker] = df['Close'].pct_change().dropna()
        
        # If we have returns data for at least 2 stocks, create a correlation matrix
        if len(returns_data) >= 2:
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()
            
            # Plot correlation matrix
            corr_fig = visualizer.plot_correlation_matrix(correlation_matrix)
            st.plotly_chart(corr_fig, use_container_width=True)
            
            st.markdown("""
            This heatmap shows the correlation between different assets in your portfolio. 
            Values close to 1 indicate strong positive correlation (assets move together), 
            values close to -1 indicate strong negative correlation (assets move in opposite directions),
            and values close to 0 indicate little correlation.
            A well-diversified portfolio typically includes assets with lower correlation to each other.
            """)
    
    # Rebalancing recommendations
    st.header("Rebalancing Recommendations")
    
    # Get age-based recommendation
    age_rec = risk_profiler.get_age_based_suggestion(age)
    
    # Compare recommended portfolio with age-based suggestion
    current_alloc = {a['asset_class']: a['percentage'] for a in portfolio['allocations']}
    target_alloc = PortfolioOptimizer.allocate_by_risk_profile(age_rec['suggested_profile'])
    
    # Get rebalancing recommendations
    rebalance_actions = PortfolioOptimizer.rebalance_portfolio(current_alloc, target_alloc, threshold=5)
    
    if rebalance_actions:
        st.markdown("Based on your age profile, you might consider the following adjustments:")
        for asset, action in rebalance_actions.items():
            direction = "increase" if action['action'] == 'increase' else "decrease"
            st.markdown(f"- **{asset}**: {direction} allocation from {action['current']:.1f}% to {action['target']:.1f}% ({abs(action['difference']):.1f}% difference)")
    else:
        st.markdown("Your current allocation aligns well with your age profile. No significant rebalancing is needed at this time.")
    
    # Add disclaimer
    st.markdown("---")
    st.caption("Disclaimer: This application is for educational purposes only and does not constitute financial advice. Past performance is not indicative of future results. Always consult with a financial advisor before making investment decisions.")

else:
    # Show welcome message and instructions when the app first loads
    st.info("👈 Please fill out your financial profile in the sidebar to generate a personalized investment portfolio.")
    
    # Show overview of what the app does with real example tickers
    st.header("How FinAgent Works")
    
    st.markdown("""
    ### Generate a Personalized Investment Portfolio

    FinAgent analyzes your financial profile and risk tolerance to create a custom investment portfolio using real market data. 
    The app leverages machine learning models to determine your optimal asset allocation and provide detailed investment rationales.
    
    Here's what you'll get:
    
    1. **Risk Profile Assessment**: Understand your investment style and appropriate risk level
    2. **Custom Asset Allocation**: Receive a detailed breakdown of recommended investments across asset classes
    3. **Market Sentiment Analysis**: See how current market news might impact your investments
    4. **Historical Performance Analysis**: View how your recommended portfolio would have performed over the past year
    5. **Rebalancing Recommendations**: Get suggestions for optimizing your portfolio based on your age and time horizon
    """)
    
    # Show example of a dashboard with sample portfolio
    st.header("Example Portfolio Visualization")
    
    # Create sample data for demonstration
    sample_allocations = [
        {"asset_class": "US Large Cap", "percentage": 35, "tickers": ["AAPL", "MSFT"]},
        {"asset_class": "US Bonds", "percentage": 25, "tickers": ["AGG", "BND"]},
        {"asset_class": "International Developed", "percentage": 15, "tickers": ["BABA", "TSM"]},
        {"asset_class": "US Mid Cap", "percentage": 10, "tickers": ["FTNT", "ROKU"]},
        {"asset_class": "Emerging Markets", "percentage": 10, "tickers": ["BIDU", "PBR"]},
        {"asset_class": "Real Estate", "percentage": 5, "tickers": ["VNQ"]},
    ]
    
    # Display sample chart
    sample_fig = visualizer.plot_asset_allocation(sample_allocations, title="Sample Asset Allocation")
    st.plotly_chart(sample_fig, use_container_width=True)
    
    # Show how to get started
    st.header("Getting Started")
    st.markdown("""
    1. **Complete Your Profile**: Fill out the financial profile form in the sidebar with your personal information, investment goals, and risk tolerance.
    
    2. **Generate Your Portfolio**: Click the "Generate Portfolio" button to create your personalized investment strategy.
    
    3. **Review Your Results**: Explore your portfolio recommendations, including asset allocation, sector breakdown, and historical performance.
    
    4. **Take Action**: Use the insights provided to inform your investment decisions or discuss them with your financial advisor.
    """)

# Footer
st.markdown("---")
st.markdown("FinAgent - ML Investment Strategist | © 2025 Rituraj Singh | MIT License")
st.caption("Disclaimer: This application is for educational purposes only and does not constitute financial advice. Always consult with a financial advisor before making investment decisions.")