import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class MarketDataProvider:
    """Class to fetch market data using yfinance"""
    
    @staticmethod
    def get_stock_data(ticker, period="1y", start=None, end=None):
        """
        Fetch historical stock data with error handling
        
        Args:
            ticker (str): Stock ticker symbol
            period (str): Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            start (datetime): Start date (if specified, period is ignored)
            end (datetime): End date (if specified, period is ignored)
            
        Returns:
            pd.DataFrame: DataFrame with stock price data
        """
        try:
            stock = yf.Ticker(ticker)
            if start is not None and end is not None:
                hist = stock.history(start=start, end=end)
            else:
                hist = stock.history(period=period)
                
            if hist.empty:
                print(f"Warning: No data retrieved for {ticker}")
                
            return hist
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_multiple_stocks_data(tickers, period="1y", start=None, end=None):
        """
        Fetch data for multiple stocks with error handling
        
        Args:
            tickers (list): List of ticker symbols
            period (str): Time period
            start (datetime): Start date
            end (datetime): End date
            
        Returns:
            dict: Dictionary with ticker as key and DataFrame as value
        """
        data = {}
        for ticker in tickers:
            data[ticker] = MarketDataProvider.get_stock_data(ticker, period, start, end)
        return data
    
    @staticmethod
    def get_key_stats(ticker):
        """
        Get key statistics for a ticker
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Dictionary of key stats
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract relevant metrics
            key_stats = {}
            metrics = ['marketCap', 'trailingPE', 'forwardPE', 'dividendYield', 
                      'beta', '52WeekChange', 'shortPercentOfFloat', 'fiveYearAvgDividendYield',
                      'fiftyTwoWeekHigh', 'fiftyTwoWeekLow', 'priceToBook']
            
            for metric in metrics:
                if metric in info:
                    key_stats[metric] = info[metric]
            
            return key_stats
        except Exception as e:
            print(f"Error fetching stats for {ticker}: {e}")
            return {}
    
    @staticmethod
    def calculate_returns(stock_data):
        """
        Calculate daily and annualized returns
        
        Args:
            stock_data (pd.DataFrame): DataFrame with stock price data
            
        Returns:
            tuple: (daily_returns, annualized_return, volatility)
        """
        if stock_data.empty or len(stock_data) < 2:
            return None, None, None
        
        try:
            # Calculate daily returns
            daily_returns = stock_data['Close'].pct_change().dropna()
            
            # Calculate annualized return
            total_days = (stock_data.index[-1] - stock_data.index[0]).days
            if total_days > 0:
                total_return = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1
                annualized_return = ((1 + total_return) ** (365 / total_days)) - 1
            else:
                annualized_return = 0
            
            # Calculate volatility (annualized standard deviation)
            volatility = daily_returns.std() * (252 ** 0.5)  # 252 trading days in a year
            
            return daily_returns, annualized_return, volatility
        except Exception as e:
            print(f"Error calculating returns: {e}")
            return None, None, None
    
    @staticmethod
    def get_index_data(index_ticker="^GSPC", period="1y"):
        """
        Get market index data (default: S&P 500)
        
        Args:
            index_ticker (str): Index ticker
            period (str): Time period
            
        Returns:
            pd.DataFrame: DataFrame with index data
        """
        return MarketDataProvider.get_stock_data(index_ticker, period)
    
    @staticmethod
    def get_market_news(ticker="^GSPC", max_items=5):
        """
        Get latest market news
        
        Args:
            ticker (str): Ticker to get news for (default: S&P 500)
            max_items (int): Maximum number of news items to return
            
        Returns:
            list: List of news headlines
        """
        try:
            stock = yf.Ticker(ticker)
            news_data = stock.news
            
            headlines = []
            if news_data:
                for i in range(min(max_items, len(news_data))):
                    if 'title' in news_data[i]:
                        headlines.append(news_data[i]['title'])
            
            return headlines
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []
    
    @staticmethod
    def create_portfolio_performance(tickers, weights, start_date, end_date=None):
        """
        Calculate historical portfolio performance based on ticker weights
        
        Args:
            tickers (dict): Dictionary with ticker symbols as keys and weights as values
            weights (dict): Dictionary with ticker weights
            start_date (datetime): Start date
            end_date (datetime): End date (defaults to today)
            
        Returns:
            pd.DataFrame: DataFrame with portfolio performance
        """
        if end_date is None:
            end_date = datetime.now()
        
        # Get data for all tickers
        all_data = {}
        for ticker in tickers:
            data = MarketDataProvider.get_stock_data(ticker, start=start_date, end=end_date)
            if not data.empty:
                all_data[ticker] = data
        
        # Ensure we have data for all tickers
        if not all_data or len(all_data) == 0:
            return pd.DataFrame()
        
        # Find common dates
        common_dates = None
        for ticker, data in all_data.items():
            if common_dates is None:
                common_dates = set(data.index)
            else:
                common_dates = common_dates.intersection(set(data.index))
        
        if not common_dates:
            return pd.DataFrame()
        
        # Create portfolio DataFrame
        portfolio_df = pd.DataFrame(index=sorted(common_dates))
        
        # Calculate normalized returns for each ticker
        for ticker, data in all_data.items():
            weight = weights.get(ticker, 0)
            prices = data.loc[portfolio_df.index, 'Close']
            normalized = prices / prices.iloc[0] * 100 * weight
            ticker_col = f"{ticker}_contrib"
            portfolio_df[ticker_col] = normalized
        
        # Sum contributions to get portfolio value
        portfolio_df['Close'] = portfolio_df.sum(axis=1)
        
        return portfolio_df
    
    @staticmethod
    def get_dividend_data(tickers):
        """
        Get dividend data for a list of tickers
        
        Args:
            tickers (list): List of ticker symbols
            
        Returns:
            dict: Dictionary with ticker as key and dividend data as value
        """
        dividend_data = {}
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                # Get dividend history
                dividends = stock.dividends
                
                # Get key dividend metrics
                info = stock.info
                
                dividend_yield = info.get('dividendYield', None)
                if dividend_yield:
                    dividend_yield = round(dividend_yield * 100, 2)  # Convert to percentage
                
                dividend_rate = info.get('dividendRate', None)
                
                # Get dividend dates if available
                ex_dividend_date = info.get('exDividendDate', None)
                if ex_dividend_date:
                    ex_dividend_date = datetime.fromtimestamp(ex_dividend_date).strftime('%Y-%m-%d')
                
                dividend_data[ticker] = {
                    'dividend_yield': dividend_yield,
                    'dividend_rate': dividend_rate,
                    'ex_dividend_date': ex_dividend_date,
                    'dividend_history': dividends.to_dict() if not dividends.empty else {}
                }
            except Exception as e:
                print(f"Error fetching dividend data for {ticker}: {e}")
                dividend_data[ticker] = {
                    'dividend_yield': None,
                    'dividend_rate': None,
                    'ex_dividend_date': None,
                    'dividend_history': {}
                }
        
        return dividend_data
        
    @staticmethod
    def get_sectors_performance():
        """
        Get performance data for major market sectors
        
        Returns:
            pd.DataFrame: DataFrame with sector performance data
        """
        try:
            # Major sector ETFs
            sector_etfs = {
                'XLK': 'Technology',
                'XLF': 'Financial',
                'XLV': 'Healthcare',
                'XLE': 'Energy',
                'XLY': 'Consumer Cyclical',
                'XLP': 'Consumer Staples',
                'XLI': 'Industrials',
                'XLB': 'Materials',
                'XLC': 'Communication',
                'XLRE': 'Real Estate',
                'XLU': 'Utilities'
            }
            
            # Get performance data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Last month performance
            
            sector_data = []
            
            for etf, sector in sector_etfs.items():
                try:
                    data = MarketDataProvider.get_stock_data(etf, start=start_date, end=end_date)
                    if not data.empty:
                        # Calculate performance
                        initial_price = data['Close'].iloc[0]
                        final_price = data['Close'].iloc[-1]
                        performance = ((final_price / initial_price) - 1) * 100  # As percentage
                        
                        sector_data.append({
                            'Sector': sector,
                            'ETF': etf,
                            'Performance (%)': round(performance, 2),
                            'Initial Price': round(initial_price, 2),
                            'Current Price': round(final_price, 2)
                        })
                except Exception as e:
                    print(f"Error processing sector ETF {etf}: {e}")
            
            # Convert to DataFrame and sort by performance
            if sector_data:
                df = pd.DataFrame(sector_data)
                return df.sort_values('Performance (%)', ascending=False)
            
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching sector performance: {e}")
            return pd.DataFrame()