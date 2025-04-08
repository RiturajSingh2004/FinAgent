# FinAgent - ML Investment Strategist

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.22.0+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Sentiment: FinBERT](https://img.shields.io/badge/Sentiment-FinBERT-green.svg)](https://huggingface.co/ProsusAI/finbert)
[![Text: Gemma](https://img.shields.io/badge/Text-Gemma-purple.svg)](https://huggingface.co/google/gemma-3-1b)

AI-powered investment portfolio generator that creates personalized investment strategies based on your financial profile and risk tolerance.

![FinAgent Screenshot](https://via.placeholder.com/800x450?text=FinAgent+Dashboard)

## 📊 Features

- **Personalized Investment Portfolios**: Tailored recommendations based on financial profile, risk tolerance, and investment goals
- **Market Sentiment Analysis**: Real-time analysis of market news using FinBERT
- **Risk Profiling**: Comprehensive assessment of investor risk tolerance
- **Interactive Visualizations**: Portfolio allocation, sector breakdown, historical performance, risk metrics
- **Portfolio Optimization**: Asset allocation based on modern portfolio theory
- **AI-Generated Investment Rationales**: Clear explanations of investment strategies using Gemma language model
- **Rebalancing Recommendations**: Suggestions for portfolio adjustments
- **Real Market Data**: Powered by yfinance with actual market performance

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip for package installation

### Installation

1. Clone the repository:
```bash
git clone https://github.com/riturajsingh/finagent.git
cd finagent
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (optional):
```bash
cp .env.example .env
# Edit .env file with your API keys if you have them
```

5. Run the application:
```bash
streamlit run app.py
```

6. Open your browser and navigate to `http://localhost:8501`

## 🔧 Project Structure

```
finagent/
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
├── data/                    # Data providers and utilities
│   ├── __init__.py
│   ├── market_data.py       # Market data using yfinance
│   ├── fundamentals.py      # Fundamental data provider
│   └── data_utils.py        # Data processing utilities
├── models/                  # ML models and algorithms
│   ├── __init__.py
│   ├── portfolio.py         # Portfolio optimization and generation
│   ├── sentiment.py         # Sentiment analysis using FinBERT
│   └── text_generator.py    # Investment rationale generation using Gemma
└── utils/                   # Helper utilities
    ├── __init__.py
    ├── risk_profile.py      # Risk assessment system
    └── visualization.py     # Visualization functions
```

## ⚙️ Core Components

### Data Providers

#### MarketDataProvider (`data/market_data.py`)

Serves as the interface for retrieving market data using the `yfinance` library.

**Key Functions:**
- `get_stock_data()`: Fetches historical stock data with robust error handling
- `get_multiple_stocks_data()`: Retrieves data for multiple stocks simultaneously
- `calculate_returns()`: Computes daily and annualized returns with volatility metrics
- `get_key_stats()`: Extracts essential metrics like P/E ratio, market cap, and beta
- `create_portfolio_performance()`: Calculates historical portfolio performance based on ticker weights
- `get_dividend_data()`: Retrieves dividend history and yield metrics
- `get_sectors_performance()`: Collects performance data for major market sectors

**Benefits:**
- Abstracts away the complexity of retrieving financial data
- Implements robust error handling to gracefully manage API failures
- Provides a unified interface for different types of market data
- Caches results to minimize redundant API calls

#### FundamentalDataProvider (`data/fundamentals.py`)

Retrieves company fundamental data using the Financial Modeling Prep API.

**Key Functions:**
- `get_company_profile()`: Retrieves company information and sector classification
- `get_financial_ratios()`: Fetches key financial ratios like P/E, P/B, and debt-to-equity
- `get_income_statement()`: Retrieves income statement data for financial analysis
- `get_news_sentiment()`: Provides latest news sentiment for a ticker

**Benefits:**
- Separates fundamental data concerns from market data
- Provides structured access to complex financial information
- Implements fallbacks for when the API is unavailable

#### DataUtils (`data/data_utils.py`)

Utility class for data processing and transformation operations.

**Key Functions:**
- `calculate_correlation_matrix()`: Computes correlation between multiple stocks
- `calculate_risk_metrics()`: Calculates key risk metrics including Sharpe ratio and max drawdown
- `normalize_fundamentals_data()`: Standardizes data from different sources
- `create_asset_class_mappings()`: Maps tickers to asset classes for portfolio construction
- `map_ticker_to_sector()`: Associates tickers with their market sectors

**Benefits:**
- Centralizes common data manipulation functions
- Ensures consistent data transformations across the application
- Provides reusable components for portfolio analysis

### ML Models

#### PortfolioOptimizer (`models/portfolio.py`)

Implements portfolio optimization algorithms based on modern portfolio theory.

**Key Functions:**
- `allocate_by_risk_profile()`: Generates asset allocation based on risk tolerance
- `optimize_portfolio()`: Implements mean-variance optimization with practical constraints
- `get_portfolio_statistics()`: Calculates expected return, volatility, Sharpe ratio, and max drawdown
- `select_tickers_by_asset_class()`: Selects representative tickers for each asset class
- `rebalance_portfolio()`: Determines rebalancing actions based on drift thresholds

**Benefits:**
- Implements sophisticated portfolio optimization techniques
- Uses Ledoit-Wolf shrinkage for robust covariance estimation
- Incorporates constraints for practical portfolio construction
- Provides actionable rebalancing recommendations

#### PortfolioGenerator (`models/portfolio.py`)

Bridges the gap between user profiles and optimized portfolios.

**Key Functions:**
- `generate_portfolio()`: Creates a portfolio based on user risk profile and preferences
- `enhance_portfolio_with_market_data()`: Enriches portfolios with live market data and performance metrics

**Benefits:**
- Transforms user inputs into concrete investment recommendations
- Integrates risk profiles with asset allocation models
- Provides context-aware portfolio construction based on market conditions

#### SentimentAnalyzer (`models/sentiment.py`)

Performs financial sentiment analysis using transformer-based models.

**Key Functions:**
- `analyze_sentiment()`: Analyzes sentiment of financial text using FinBERT
- `analyze_multiple_texts()`: Processes multiple text items in batch
- `get_sentiment_summary()`: Generates aggregated sentiment metrics and scores

**Benefits:**
- Uses FinBERT, a financial-domain-specific transformer model
- Implements lazy loading to reduce memory footprint
- Provides fallback mechanisms when model loading fails
- Offers both granular and aggregated sentiment analysis

#### TextGenerator (`models/text_generator.py`)

Generates natural language explanations for investment rationales.

**Key Functions:**
- `generate_investment_rationale()`: Creates detailed investment strategy explanations
- `generate_market_analysis()`: Produces market analysis based on news and sentiment
- `_generate_template_rationale()`: Template-based generation for consistency and reliability

**Benefits:**
- Uses transformer-based language models (Gemma)
- Implements efficient memory management with 8-bit quantization
- Provides template fallbacks for reliability
- Generates contextual investment explanations based on portfolio characteristics

### Utilities

#### RiskProfiler (`utils/risk_profile.py`)

Evaluates and classifies user risk profiles based on questionnaire responses.

**Key Functions:**
- `calculate_risk_score()`: Computes a numerical risk score from questionnaire answers
- `classify_risk_profile()`: Maps numerical scores to risk categories (conservative, moderate, aggressive)
- `get_risk_profile_summary()`: Provides descriptive information about risk profiles
- `get_age_based_suggestion()`: Recommends allocations based on age and life stage
- `recommend_allocation_adjustments()`: Suggests modifications based on financial situation

**Benefits:**
- Provides systematic risk profiling based on financial best practices
- Incorporates multiple dimensions of risk tolerance
- Offers age-appropriate recommendations
- Adjusts for special financial circumstances like high debt or lack of emergency fund

#### Visualizer (`utils/visualization.py`)

Creates interactive visualizations for portfolio analysis and market data.

**Key Functions:**
- `plot_asset_allocation()`: Generates pie charts for asset class breakdown
- `plot_risk_return_profile()`: Displays risk vs. return for various assets
- `plot_historical_performance()`: Shows comparative historical performance
- `plot_correlation_matrix()`: Visualizes asset correlations
- `plot_dividend_yield()`: Compares dividend yields across securities
- `plot_sector_breakdown()`: Shows sector allocation within the portfolio
- `plot_sentiment_analysis()`: Displays sentiment analysis results
- `plot_portfolio_metrics()`: Visualizes key portfolio risk and return metrics
- `plot_asset_growth()`: Projects investment growth over time

**Benefits:**
- Uses Plotly for interactive visualizations
- Implements consistent styling across all charts
- Provides rich hover information for enhanced user experience
- Handles edge cases gracefully (e.g., missing data)

## 📈 How It Works

### Workflow and Data Flow

1. **User Input Collection**:
   - Users complete a detailed financial profile form in the sidebar
   - Information collected includes age, income, investment amount, investment horizon, risk comfort level, loss reaction, investment knowledge, financial situation (emergency fund, debt), and investment goals
   - Optional sector exclusions and custom preferences can be specified

2. **Risk Profile Assessment**:
   - The RiskProfiler calculates a risk score (0-100) based on weighted questionnaire responses
   - Risk scores are mapped to three profiles: conservative (<35), moderate (35-65), or aggressive (>65)
   - Age-based recommendations are generated independently for comparison
   - The system evaluates special financial situations (high debt, missing emergency fund)

3. **Portfolio Generation**:
   - The PortfolioGenerator creates an initial asset allocation based on the determined risk profile
   - Asset classes like US Large Cap, US Bonds, International Developed, etc. are assigned percentage allocations
   - Specific tickers are selected for each asset class based on performance data or random selection if data is unavailable
   - The portfolio is enhanced with actual market data including returns, volatility, and key statistics

4. **Market Analysis**:
   - Current market news is retrieved using yfinance (with fallback to general news if unavailable)
   - The SentimentAnalyzer evaluates news sentiment using the FinBERT model
   - A market sentiment summary is generated with positive/neutral/negative classification

5. **Investment Rationale Creation**:
   - The TextGenerator creates a detailed investment rationale using Gemma or template-based generation
   - Explanations are customized to the user's risk profile, investment horizon, and market sentiment
   - Detailed rationales for each asset class are provided

6. **Visualization and Performance Analysis**:
   - Interactive visualizations show portfolio allocation and sector breakdown
   - Historical performance is compared to market benchmarks (S&P 500, Bonds)
   - Risk metrics and correlation analysis highlight diversification benefits
   - Return metrics are calculated from actual market data

7. **Rebalancing Recommendations**:
   - Comparison between recommended portfolio and age-based suggestions
   - Specific rebalancing actions with target percentages for each asset class
   - Threshold-based recommendations (for differences >5%)

## 🔍 Technical Implementation

### Data Retrieval and Processing

The application uses Yahoo Finance (`yfinance`) as the primary data source for real-time and historical market data, offering:

- **Free access** to extensive market data without API keys or rate limits
- **Broad coverage** of stocks, ETFs, indices, and other securities worldwide
- **Rich data points** including OHLC pricing, volume, dividends, fundamental data, and news
- **Historical data** with adjustable time periods for performance analysis

For fundamental data, the application optionally uses the Financial Modeling Prep API, which provides:

- Company profiles and sector classifications
- Financial ratios (P/E, P/B, debt-to-equity, etc.)
- Income statements and balance sheets
- News sentiment aggregation

Data processing involves several sophisticated techniques:

- **Robust error handling**: All data retrieval functions implement comprehensive error handling to manage API failures or missing data
- **Normalization**: Data from different sources is standardized for consistent analysis
- **Correlation analysis**: Asset relationships are analyzed using robust statistical methods
- **Performance calculation**: Returns, volatility, Sharpe ratios, and drawdowns are computed using industry-standard approaches

### Machine Learning Models

The application leverages several machine learning models:

1. **Sentiment Analysis (SentimentAnalyzer)**:
   - Uses FinBERT, a BERT model fine-tuned on financial text
   - Pre-trained on financial news and reports for domain-specific understanding
   - Classifies text as positive, neutral, or negative with confidence scores
   - Lazy loading implementation to reduce memory usage until needed
   - Fallback to simpler sentiment model if FinBERT fails to load

2. **Text Generation (TextGenerator)**:
   - Implements Google's Gemma 1B parameter model
   - 8-bit quantization for efficient memory usage on consumer hardware
   - Template-based fallbacks for reliability and consistency
   - Context-aware rationales that incorporate user profile and market data

3. **Portfolio Optimization (PortfolioOptimizer)**:
   - Implements mean-variance optimization from modern portfolio theory
   - Uses Ledoit-Wolf shrinkage for robust covariance estimation
   - Incorporates constraints like maximum allocation per asset
   - Optimizes for risk-adjusted returns based on user risk profile
   - Includes practical constraints like no-short-selling and diversification requirements

### Risk Profiling System

The risk profiling system implements a multi-dimensional assessment with weighted factors:

- **Investment Timeline (15%)**: Measures the user's time horizon (1-3 years to 10+ years)
- **Risk Comfort (25%)**: Assesses subjective comfort with volatility (very low to very high)
- **Loss Reaction (20%)**: Evaluates behavioral response to market downturns (sell all to buy significantly more)
- **Investment Goal (15%)**: Determines primary financial objectives (capital preservation to aggressive growth)
- **Income Stability (10%)**: Considers stability of income sources (unstable to extremely stable)
- **Investment Knowledge (15%)**: Accounts for financial literacy level (none to expert)

These dimensions are combined into a comprehensive risk score (0-100), which maps to:
- **Conservative**: Scores < 35
- **Moderate**: Scores 35-65
- **Aggressive**: Scores > 65

Age-based suggestions provide an additional reference point, with recommended stock/bond allocations adjusted by life stage.

### Portfolio Construction

Portfolio construction follows a hierarchical approach:

1. **Asset Class Allocation**:
   - Conservative profile focuses on bonds (60%), large caps (15%), with minimal exposure to higher-risk assets
   - Moderate profile balances stocks (50% total across market caps) with bonds (30%)
   - Aggressive profile emphasizes stocks (80% total) with minimal bond allocation (5%)
   - Additional asset classes like Real Estate, Commodities, and Cash are incorporated based on risk profile

2. **Ticker Selection**:
   - Specific securities are selected to represent each asset class
   - When market data is available, selections are based on performance metrics
   - Sector restrictions are applied based on user preferences
   - Equal weighting within asset classes for simplicity and diversification

3. **Portfolio Enhancement**:
   - Live market data is incorporated for all selected securities
   - Historical performance is calculated using actual price data
   - Risk metrics like volatility, Sharpe ratio, and max drawdown are computed
   - Correlation analysis highlights diversification benefits

## 📋 Usage Example

### Using FinAgent as a Library

```python
# Initialize data providers
market_data = MarketDataProvider()
fundamental_data = FundamentalDataProvider()

# Initialize models
sentiment_analyzer = SentimentAnalyzer()
text_generator = TextGenerator()
portfolio_generator = PortfolioGenerator(market_data, fundamental_data)
risk_profiler = RiskProfiler()
visualizer = Visualizer()

# Create user profile
user_profile = {
    'age': 35,
    'income': 70000,
    'investment_amount': 10000,
    'risk_tolerance': 'moderate',
    'investment_horizon': 'medium_term',
    'investment_goal': 'balanced',
    'exclude_sectors': ['Energy']
}

# Calculate risk score and profile
risk_answers = {
    'investment_timeline': 'medium_term',
    'risk_comfort': 'moderate',
    'loss_reaction': 'do_nothing',
    'investment_goal': 'balanced',
    'income_stability': 'stable',
    'investment_knowledge': 'average'
}
risk_score = risk_profiler.calculate_risk_score(risk_answers)
risk_profile = risk_profiler.classify_risk_profile(risk_score)
risk_summary = risk_profiler.get_risk_profile_summary(risk_profile)

# Generate portfolio
portfolio = portfolio_generator.generate_portfolio(user_profile)

# Enhance with market data
portfolio = portfolio_generator.enhance_portfolio_with_market_data(portfolio)

# Analyze market sentiment
news = market_data.get_market_news(ticker="^GSPC", max_items=5)
sentiment_results = sentiment_analyzer.analyze_multiple_texts(news)
market_sentiment = sentiment_analyzer.get_sentiment_summary(sentiment_results)

# Generate investment rationale
investment_rationale = text_generator.generate_investment_rationale(
    portfolio, user_profile, market_sentiment
)

# Get historical performance data
tickers = list(portfolio.get('ticker_allocation', {}).keys())
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
price_data = {}

for ticker in tickers:
    hist = market_data.get_stock_data(ticker, start=start_date, end=end_date)
    if not hist.empty:
        price_data[ticker] = hist

# Create visualizations
allocation_fig = visualizer.plot_asset_allocation(portfolio['allocations'])
# Display the figure in your application
```

### Using the Streamlit Application

1. Complete the user profile form in the sidebar:
   - Enter basic information (age, income, investment amount)
   - Select investment horizon (short, medium, or long-term)
   - Answer risk assessment questions
   - Specify investment goals and preferences

2. Click "Generate Portfolio" to create your personalized investment strategy

3. Explore the results:
   - Review your risk profile and age-based recommendations
   - Examine the recommended asset allocation and sector breakdown
   - Analyze market sentiment and its impact on your strategy
   - Study the detailed investment rationale
   - Evaluate historical performance against benchmarks
   - Consider the rebalancing recommendations

## 🛣️ Roadmap

### Immediate Enhancements
- **Model Fine-tuning**: Fine-tune FinBERT on more recent financial news for improved sentiment accuracy
- **Performance Optimization**: Implement caching strategies to improve application responsiveness
- **Mobile UX**: Enhance the mobile user experience for on-the-go investors
- **User Feedback Integration**: Add feedback mechanism to improve recommendations

### Medium-term Goals
- **Enhanced ML Models**: Implement more sophisticated sentiment analysis and text generation
- **Additional Data Sources**: Integrate alternative data for improved market insights
- **Tax-Aware Recommendations**: Add tax efficiency considerations to portfolio construction
- **User Accounts**: Implement persistent user profiles for tracking over time
- **Personalized Benchmarks**: Create custom benchmarks based on risk profile

### Long-term Vision
- **Automated Rebalancing**: Provide automated rebalancing recommendations based on portfolio drift
- **ESG Integration**: Add environmental, social, and governance criteria to investment selection
- **Scenario Analysis**: Implement stress testing for different market conditions
- **API Integration**: Connect with brokerage APIs for actual portfolio implementation
- **Retirement Planning**: Extend functionality to include retirement goal planning

## 🔐 Security and Ethical Considerations

The application implements several security and ethical safeguards:

- **No financial advice disclaimer**: Clearly communicates that the application is for educational purposes only
- **API key protection**: Uses environment variables to secure API keys, not hard-coded in source
- **Data privacy**: Keeps user data within the session, without persistent storage
- **Realistic expectations**: Avoids promising unrealistic returns or performance guarantees
- **Age-appropriate recommendations**: Adjusts suggestions based on user age and time horizon
- **Risk awareness**: Clearly communicates the risks associated with different investment strategies
- **Educational context**: Provides explanations of financial concepts to improve financial literacy
- **Transparent methodology**: Clearly explains how recommendations are generated

## 💻 Deployment and Scalability

The application is designed for deployment on Streamlit's cloud platform, offering:

- **Easy deployment**: Simple GitHub integration for continuous deployment
- **Scalable infrastructure**: Handles multiple concurrent users efficiently
- **Authentication options**: Can implement user authentication if needed
- **Monitoring and analytics**: Provides usage statistics and performance metrics

The modular design facilitates easy extension and modification:

- **New data sources**: Additional data providers can be integrated with minimal changes
- **Enhanced models**: ML models can be upgraded or replaced as better options become available
- **Additional visualizations**: New visualization types can be added to the Visualizer class
- **Custom asset classes**: The asset class mapping can be extended for specialized investments

## ⚠️ Disclaimer

This application is for educational purposes only and does not constitute financial advice. Past performance is not indicative of future results. The projections or other information generated regarding the likelihood of various investment outcomes are hypothetical in nature, do not reflect actual investment results, and are not guarantees of future results. Always consult with a qualified financial advisor before making investment decisions.

## 📄 License

FinAgent is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution

- **Additional asset classes**: Expand the available investment categories
- **Improved ticker selection**: Enhance the algorithm for selecting representative securities
- **Advanced optimization**: Implement additional portfolio optimization techniques
- **Enhanced visualizations**: Create new visualizations for better understanding
- **Documentation**: Improve documentation, tutorials, and examples
- **Testing**: Add unit tests and integration tests for reliability
- **UI enhancements**: Improve the user interface and experience

## 👏 Acknowledgements

- [Yahoo Finance](https://finance.yahoo.com/) for market data through yfinance
- [Financial Modeling Prep](https://financialmodelingprep.com/) for fundamental data
- [Streamlit](https://streamlit.io/) for the web application framework
- [HuggingFace](https://huggingface.co/) for hosting pre-trained models
- [FinBERT](https://huggingface.co/ProsusAI/finbert) for financial sentiment analysis
- [Gemma](https://huggingface.co/google/gemma-3-1b) for text generation
- [Plotly](https://plotly.com/) for interactive visualizations
- [scikit-learn](https://scikit-learn.org/) for machine learning utilities
- [PyTorch](https://pytorch.org/) for deep learning capabilities
- [Transformers](https://huggingface.co/docs/transformers/index) for NLP model implementations

---

© 2025 Rituraj Singh | [GitHub](https://github.com/riturajsingh) | [LinkedIn](https://linkedin.com/in/riturajsingh)
