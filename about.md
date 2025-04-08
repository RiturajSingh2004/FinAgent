# FinAgent - ML Investment Strategist

## Project Overview

FinAgent is an AI-powered investment portfolio generator that creates personalized investment recommendations based on a user's financial profile, risk tolerance, and investment goals. The application leverages machine learning models for market sentiment analysis and portfolio optimization, providing users with data-driven investment strategies.

## Technical Architecture

The project follows a modular architecture with clearly separated concerns:

1. **Data Layer**: Handles market data retrieval, fundamental data collection, and data utilities
2. **Models Layer**: Implements machine learning models for sentiment analysis, text generation, and portfolio optimization
3. **Utilities Layer**: Provides risk profiling, visualization, and other helper functions
4. **Application Layer**: Streamlit frontend that ties everything together

## Core Components

### 1. Data Providers

#### MarketDataProvider (`data/market_data.py`)

This class serves as the interface for retrieving market data using the `yfinance` library. 

**Key Functions:**

- `get_stock_data()`: Fetches historical stock data with robust error handling
- `get_multiple_stocks_data()`: Retrieves data for multiple stocks in parallel
- `calculate_returns()`: Computes daily and annualized returns with volatility
- `get_key_stats()`: Extracts essential metrics like P/E ratio, market cap, etc.
- `create_portfolio_performance()`: Calculates historical portfolio performance based on ticker weights
- `get_dividend_data()`: Retrieves dividend history and metrics
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
- `get_financial_ratios()`: Fetches key financial ratios like P/E, P/B, etc.
- `get_income_statement()`: Retrieves income statement data
- `get_news_sentiment()`: Provides latest news sentiment for a ticker

**Benefits:**
- Separates fundamental data concerns from market data
- Provides structured access to complex financial information
- Implements fallbacks for when the API is unavailable

#### DataUtils (`data/data_utils.py`)

Utility class for data processing and transformation operations.

**Key Functions:**

- `calculate_correlation_matrix()`: Computes correlation between multiple stocks
- `calculate_risk_metrics()`: Calculates key risk metrics like Sharpe ratio and max drawdown
- `normalize_fundamentals_data()`: Standardizes data from different sources
- `create_asset_class_mappings()`: Maps tickers to asset classes
- `map_ticker_to_sector()`: Associates tickers with their market sectors

**Benefits:**
- Centralizes common data manipulation functions
- Ensures consistent data transformations across the application
- Provides reusable components for portfolio analysis

### 2. Model Components

#### PortfolioOptimizer (`models/portfolio.py`)

Implements portfolio optimization algorithms based on modern portfolio theory.

**Key Functions:**

- `allocate_by_risk_profile()`: Generates asset allocation based on risk tolerance
- `optimize_portfolio()`: Implements mean-variance optimization with constraints
- `get_portfolio_statistics()`: Calculates expected return, volatility, and other metrics
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
- `enhance_portfolio_with_market_data()`: Enriches portfolios with live market data

**Benefits:**
- Transforms user inputs into concrete investment recommendations
- Integrates risk profiles with asset allocation models
- Provides context-aware portfolio construction

#### SentimentAnalyzer (`models/sentiment.py`)

Performs financial sentiment analysis using transformer-based models.

**Key Functions:**

- `analyze_sentiment()`: Analyzes sentiment of financial text
- `analyze_multiple_texts()`: Processes multiple text items in batch
- `get_sentiment_summary()`: Generates aggregated sentiment metrics

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
- `_generate_template_rationale()`: Template-based generation for consistency

**Benefits:**
- Uses transformer-based language models (Gemma)
- Implements efficient memory management with 8-bit quantization
- Provides template fallbacks for reliability
- Generates contextual investment explanations based on portfolio characteristics

### 3. Utility Components

#### RiskProfiler (`utils/risk_profile.py`)

Evaluates and classifies user risk profiles based on questionnaire responses.

**Key Functions:**

- `calculate_risk_score()`: Computes a numerical risk score from questionnaire answers
- `classify_risk_profile()`: Maps scores to risk categories
- `get_risk_profile_summary()`: Provides descriptive information about risk profiles
- `get_age_based_suggestion()`: Recommends allocations based on age
- `recommend_allocation_adjustments()`: Suggests modifications based on financial situation

**Benefits:**
- Provides systematic risk profiling based on financial best practices
- Incorporates multiple dimensions of risk tolerance
- Offers age-appropriate recommendations
- Adjusts for special financial circumstances

#### Visualizer (`utils/visualization.py`)

Creates interactive visualizations for portfolio analysis and market data.

**Key Functions:**

- `plot_asset_allocation()`: Generates pie charts for asset class breakdown
- `plot_historical_performance()`: Shows comparative historical performance
- `plot_correlation_matrix()`: Visualizes asset correlations
- `plot_sentiment_analysis()`: Displays sentiment analysis results
- `plot_sector_breakdown()`: Shows sector allocation
- Plus many other visualization functions for risk-return profiles, dividends, etc.

**Benefits:**
- Uses Plotly for interactive visualizations
- Implements consistent styling across all charts
- Provides rich hover information for enhanced user experience
- Handles edge cases gracefully (e.g., missing data)

### 4. Application Layer

#### Streamlit Web App (`app.py`)

The main application that ties all components together in a user-friendly interface.

**Key Sections:**

- User profile collection form
- Risk profiling and assessment
- Portfolio generation and visualization
- Market sentiment analysis
- Historical performance analysis
- Portfolio details and recommendations

**Benefits:**
- Provides an intuitive, interactive user experience
- Implements responsive design for different screen sizes
- Offers step-by-step guidance through the investment process
- Visualizes complex financial data in an accessible manner

## Workflow and Data Flow

1. **User Input Collection**:
   - Users complete a detailed financial profile form in the sidebar
   - Information includes age, income, investment amount, risk comfort, and goals

2. **Risk Profile Assessment**:
   - The RiskProfiler calculates a risk score based on user responses
   - Risk scores are mapped to conservative, moderate, or aggressive profiles
   - Age-based recommendations are generated

3. **Portfolio Generation**:
   - The PortfolioGenerator creates an initial asset allocation based on the risk profile
   - Specific tickers are selected for each asset class
   - The portfolio is enhanced with actual market data

4. **Market Analysis**:
   - Current market news is retrieved using yfinance
   - The SentimentAnalyzer evaluates news sentiment
   - A market outlook is generated

5. **Investment Rationale**:
   - The TextGenerator creates a detailed investment rationale
   - Explanations are customized to the user's risk profile and portfolio

6. **Visualization and Presentation**:
   - Multiple interactive visualizations show portfolio allocation, sector breakdown
   - Historical performance is compared to benchmarks
   - Risk metrics and correlation analysis are displayed
   - Rebalancing recommendations are provided

## Technical Implementation Details

### Data Retrieval and Processing

The application uses Yahoo Finance (`yfinance`) as the primary data source for real-time and historical market data. This choice offers several advantages:

- **Free access to extensive market data**: Eliminates the need for paid API subscriptions
- **Broad coverage**: Provides data for a wide range of assets including stocks, ETFs, and indices
- **Rich data points**: Offers pricing, fundamental, and news data in a unified interface

For fundamental data, the application uses the Financial Modeling Prep API, which provides company profiles, financial ratios, and income statements. This dual approach allows the application to present a comprehensive view of investment opportunities.

Data processing involves several sophisticated techniques:

- **Robust error handling**: All data retrieval functions implement comprehensive error handling to manage API failures or missing data
- **Normalization**: Data from different sources is standardized for consistent analysis
- **Correlation analysis**: Asset relationships are analyzed using robust statistical methods
- **Performance calculation**: Returns and risk metrics are computed using industry-standard approaches

### Machine Learning Models

The application leverages several machine learning models:

1. **Sentiment Analysis**: Uses FinBERT, a BERT model fine-tuned on financial text
   - Pre-trained on financial news and reports
   - Classifies text as positive, neutral, or negative
   - Provides confidence scores for each classification

2. **Text Generation**: Implements Google's Gemma 1B parameter model
   - Generates natural language investment rationales
   - Uses 8-bit quantization for efficient memory usage
   - Includes template-based fallbacks for reliability

3. **Portfolio Optimization**: Implements mean-variance optimization
   - Uses Ledoit-Wolf shrinkage for robust covariance estimation
   - Incorporates constraints for practical portfolio construction
   - Optimizes for risk-adjusted returns based on user preferences

### Risk Profiling System

The risk profiling system implements a multi-dimensional assessment of risk tolerance:

- **Investment Timeline**: Measures the user's time horizon
- **Risk Comfort**: Assesses subjective comfort with volatility
- **Loss Reaction**: Evaluates behavioral response to market downturns
- **Investment Goal**: Determines primary financial objectives
- **Income Stability**: Considers stability of income sources
- **Investment Knowledge**: Accounts for financial literacy level

Each dimension is weighted to calculate a comprehensive risk score, which is then mapped to one of three risk profiles: conservative, moderate, or aggressive. These profiles drive the asset allocation process, ensuring that portfolios align with user risk tolerance.

### Portfolio Construction

Portfolio construction follows a hierarchical approach:

1. **Asset Class Allocation**: Determines the breakdown among major asset classes (e.g., stocks, bonds, real estate)
2. **Ticker Selection**: Identifies specific securities to represent each asset class
3. **Weighting**: Assigns weights to individual securities within asset classes

This approach balances simplicity with sophistication, making the portfolio construction process understandable while still leveraging modern portfolio theory.

### Visualization System

The visualization system uses Plotly to create interactive, web-based charts. Key features include:

- **Consistent styling**: Maintains visual coherence across all charts
- **Rich hover information**: Provides detailed information on hover
- **Responsive design**: Adapts to different screen sizes
- **Error handling**: Gracefully handles missing or incomplete data

Each visualization is carefully designed to communicate specific aspects of the portfolio or market analysis, enhancing user understanding of complex financial concepts.

## Security and Ethical Considerations

The application implements several security and ethical safeguards:

- **No financial advice disclaimer**: Clearly states that the application is for educational purposes only
- **API key protection**: Uses environment variables to secure API keys
- **Data privacy**: Keeps user data within the session, without persistent storage
- **Realistic expectations**: Avoids promising unrealistic returns or guarantees
- **Age-appropriate recommendations**: Adjusts suggestions based on user age and time horizon
- **Risk awareness**: Clearly communicates the risks associated with different investment strategies

## Deployment and Scalability

The application is designed for deployment on Streamlit's cloud platform, which offers several advantages:

- **Easy deployment**: Simple GitHub integration for continuous deployment
- **Scalable infrastructure**: Handles multiple concurrent users
- **Authentication options**: Can implement user authentication if needed
- **Monitoring and analytics**: Provides usage statistics and performance metrics

The modular design also facilitates easy extension and modification:

- **New data sources**: Additional data providers can be integrated with minimal changes
- **Enhanced models**: ML models can be upgraded or replaced as better options become available
- **Additional visualizations**: New visualization types can be added to the Visualizer class
- **Custom asset classes**: The asset class mapping can be extended for specialized investments

## Benefits and Limitations

### Benefits

- **Personalized recommendations**: Tailors portfolios to individual financial situations
- **Data-driven approach**: Bases recommendations on real market data
- **Educational value**: Helps users understand investment concepts and portfolio construction
- **Visual explanations**: Makes complex financial information accessible
- **Comprehensive analysis**: Considers multiple dimensions of portfolio performance

### Limitations

- **Market data dependency**: Relies on free API services with potential limitations
- **Simplified models**: Uses approximations rather than full institutional-grade models
- **No account integration**: Cannot directly implement recommendations in user accounts
- **Limited asset universe**: Covers major asset classes but not all investment options
- **No tax optimization**: Does not account for tax implications of investment decisions

## Future Enhancement Opportunities

1. **Enhanced ML Models**: Implement more sophisticated sentiment analysis and text generation
2. **Additional Data Sources**: Integrate alternative data for improved market insights
3. **Tax-Aware Recommendations**: Add tax efficiency considerations to portfolio construction
4. **User Accounts**: Implement persistent user profiles for tracking over time
5. **Automated Rebalancing**: Provide automated rebalancing recommendations based on portfolio drift
6. **ESG Integration**: Add environmental, social, and governance criteria to investment selection
7. **Scenario Analysis**: Implement stress testing for different market conditions
8. **Mobile Optimization**: Enhance mobile user experience for on-the-go investing

## Conclusion

FinAgent represents a sophisticated application of machine learning to personal investment management. By combining modern portfolio theory with natural language processing and an intuitive user interface, it democratizes access to investment strategies previously available only through financial advisors. The modular architecture and thoughtful design ensure that the application can evolve with advances in financial technology while maintaining its core value proposition of personalized, data-driven investment guidance.