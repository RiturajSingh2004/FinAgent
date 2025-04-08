import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class MarketDataProvider:
    """Class to fetch market data using yfinance"""
    
    @staticmethod
    def get_stock_data(ticker, period="1y"):
        """
        Fetch historical stock data
        
        Args:
            ticker (str): Stock ticker symbol
            period (str): Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            pd.DataFrame: DataFrame with stock price data
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            return hist
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_multiple_stocks_data(tickers, period="1y"):
        """
        Fetch data for multiple stocks
        
        Args:
            tickers (list): List of ticker symbols
            period (str): Time period
            
        Returns:
            dict: Dictionary with ticker as key and DataFrame as value
        """
        data = {}
        for ticker in tickers:
            data[ticker] = MarketDataProvider.get_stock_data(ticker, period)
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
                      'beta', '52WeekChange', 'shortPercentOfFloat']
            
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
        if stock_data.empty:
            return None, None, None
        
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