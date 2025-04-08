import streamlit as st
import pandas as pd
import numpy as np
import os
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
        # Generate portfolio
        portfolio = portfolio_generator.generate_portfolio(user_profile)
        
        # Enhance with market data
        portfolio = portfolio_generator.enhance_portfolio_with_market_data(portfolio)
        
        # Get market sentiment
        market_news = [
            "Federal Reserve maintains current interest rates",
            "Tech sector reports strong quarterly earnings",
            "Consumer confidence index rises for third consecutive month",
            "International markets show mixed performance amid geopolitical tensions",
            "Supply chain issues starting to resolve according to latest economic indicators"
        ]
        sentiment_results = sentiment_analyzer.analyze_multiple_texts(market_news)
        market_sentiment = sentiment_analyzer.get_sentiment_summary(sentiment_results)
        
        # Generate investment rationale
        investment_rationale = text_generator.generate_investment_rationale(
            portfolio, user_profile, market_sentiment
        )
    
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
    
    # Historical performance and returns (for a complete app, this would use actual market data)
    st.header("Historical Performance Analysis")
    st.info("This section would show the historical performance of your recommended portfolio compared to benchmarks. In a production version, this would use actual historical market data.")
    
    # Create a random historical performance chart for demonstration
    dates = pd.date_range(start='2022-01-01', end='2023-01-01', freq='B')
    price_data = {}
    
    # S&P 500 proxy
    sp500 = pd.DataFrame(index=dates)
    sp500['Close'] = 100 * (1 + np.cumsum(np.random.normal(0.0005, 0.01, len(dates))))
    price_data['S&P 500'] = sp500
    
    # Portfolio proxy with slightly better returns
    portfolio_perf = pd.DataFrame(index=dates)
    portfolio_perf['Close'] = 100 * (1 + np.cumsum(np.random.normal(0.0007, 0.009, len(dates))))
    price_data['Portfolio'] = portfolio_perf
    
    # Bond index proxy with lower volatility
    bond_index = pd.DataFrame(index=dates)
    bond_index['Close'] = 100 * (1 + np.cumsum(np.random.normal(0.0002, 0.003, len(dates))))
    price_data['Bond Index'] = bond_index
    
    # Plot historical performance
    performance_fig = visualizer.plot_historical_performance(
        price_data, ['Portfolio', 'S&P 500', 'Bond Index'], 
        title="Simulated 1-Year Performance (For Demonstration)"
    )
    st.plotly_chart(performance_fig, use_container_width=True)
    
    # Add disclaimer
    st.caption("Disclaimer: This is a simulated performance chart for demonstration purposes only. Past performance is not indicative of future results.")

else:
    # Show welcome message and instructions when the app first loads
    st.info("👈 Please fill out your financial profile in the sidebar to generate a personalized investment portfolio.")
    
    # Show sample visualizations
    st.header("Sample Portfolio Visualization")
    
    # Create sample data for demonstration
    sample_allocations = [
        {"asset_class": "US Large Cap", "percentage": 30},
        {"asset_class": "US Bonds", "percentage": 25},
        {"asset_class": "International Developed", "percentage": 15},
        {"asset_class": "US Mid Cap", "percentage": 10},
        {"asset_class": "Emerging Markets", "percentage": 10},
        {"asset_class": "Real Estate", "percentage": 5},
        {"asset_class": "Commodities", "percentage": 5}
    ]
    
    # Sample allocation chart
    sample_fig = visualizer.plot_asset_allocation(sample_allocations, title="Sample Asset Allocation")
    st.plotly_chart(sample_fig, use_container_width=True)
    
    # Add explanatory text
    st.markdown("""
    ### How FinAgent Works
    
    1. **Complete Your Profile**: Fill out the financial profile form in the sidebar with your personal information, investment goals, and risk tolerance.
    
    2. **ML-Powered Analysis**: Our system analyzes your inputs using machine learning models to determine your optimal risk profile and asset allocation.
    
    3. **Market Sentiment Analysis**: We use natural language processing to analyze market news and sentiment to inform investment decisions.
    
    4. **Personalized Recommendations**: Receive a tailored investment strategy with detailed rationale and visualizations.
    
    5. **Portfolio Insights**: View detailed breakdowns of your recommended portfolio, including asset allocation, sector distribution, and historical performance analysis.
    """)

# Footer
st.markdown("---")
st.markdown("FinAgent - ML Investment Strategist | Built with free, open-source technologies")
st.caption("Disclaimer: This application is for educational purposes only and does not constitute financial advice. Always consult with a financial advisor before making investment decisions.")