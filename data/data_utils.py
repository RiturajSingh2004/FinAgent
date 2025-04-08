import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataUtils:
    """Utilities for data processing and transformation"""
    
    @staticmethod
    def calculate_correlation_matrix(stock_data_dict):
        """
        Calculate correlation matrix for multiple stocks
        
        Args:
            stock_data_dict (dict): Dictionary with ticker as key and DataFrame as value
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        # Extract closing prices for each stock
        close_prices = pd.DataFrame()
        
        for ticker, data in stock_data_dict.items():
            if not data.empty:
                close_prices[ticker] = data['Close']
        
        # Calculate correlation matrix
        return close_prices.pct_change().dropna().corr()
    
    @staticmethod
    def calculate_risk_metrics(returns):
        """
        Calculate various risk metrics
        
        Args:
            returns (pd.Series): Series of returns
            
        Returns:
            dict: Dictionary of risk metrics
        """
        if returns is None or len(returns) < 2:
            return {
                'volatility': None,
                'sharpe_ratio': None,
                'max_drawdown': None
            }
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Sharpe Ratio (assuming risk-free rate of 0.02 or 2%)
        risk_free_rate = 0.02
        mean_return = returns.mean() * 252  # Annualized
        sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum Drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    @staticmethod
    def normalize_fundamentals_data(fundamentals_dict):
        """
        Normalize fundamentals data for multiple stocks
        
        Args:
            fundamentals_dict (dict): Dictionary with ticker as key and fundamentals as value
            
        Returns:
            pd.DataFrame: Normalized fundamentals data
        """
        normalized_data = {}
        metrics = [
            'peRatioTTM', 
            'priceToBookRatioTTM', 
            'debtToEquityTTM', 
            'returnOnEquityTTM', 
            'dividendYieldTTM'
        ]
        
        for ticker, data in fundamentals_dict.items():
            normalized_data[ticker] = {}
            for metric in metrics:
                if metric in data:
                    normalized_data[ticker][metric] = data[metric]
                else:
                    normalized_data[ticker][metric] = None
        
        return pd.DataFrame.from_dict(normalized_data, orient='index')
    
    @staticmethod
    def create_asset_class_mappings():
        """
        Create mappings of tickers to asset classes
        
        Returns:
            dict: Dictionary mapping tickers to asset classes
        """
        # This is a simplified mapping for demonstration
        # In practice, this could be loaded from a database or external source
        return {
            # Large Cap Stocks
            'AAPL': 'US Large Cap',
            'MSFT': 'US Large Cap',
            'AMZN': 'US Large Cap',
            'GOOGL': 'US Large Cap',
            'BRK-B': 'US Large Cap',
            
            # Mid Cap Stocks
            'FTNT': 'US Mid Cap',
            'ROKU': 'US Mid Cap',
            'SNAP': 'US Mid Cap',
            
            # Small Cap Stocks
            'UVXY': 'US Small Cap',
            'SFIX': 'US Small Cap',
            
            # International Stocks
            'BABA': 'International Developed',
            'TSM': 'International Developed',
            'TM': 'International Developed',
            
            # Emerging Markets
            'BIDU': 'Emerging Markets',
            'PBR': 'Emerging Markets',
            
            # Bonds
            'AGG': 'US Bonds',
            'BND': 'US Bonds',
            'TLT': 'US Bonds',
            'VTEB': 'US Bonds',
            
            # REITs
            'VNQ': 'Real Estate',
            'SCHH': 'Real Estate',
            
            # Commodities
            'GLD': 'Commodities',
            'SLV': 'Commodities',
            'USO': 'Commodities',
            
            # Crypto (ETFs)
            'GBTC': 'Cryptocurrency',
            'ETHE': 'Cryptocurrency',
            
            # ETFs
            'SPY': 'US Index',
            'QQQ': 'US Index',
            'VTI': 'US Index',
            'VOO': 'US Index',
            'VEA': 'International Index',
            'VWO': 'Emerging Markets Index'
        }
    
    @staticmethod
    def map_ticker_to_sector(ticker):
        """
        Map ticker to sector
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            str: Sector name
        """
        # This is a simplified mapping for demonstration
        # In practice, this would use data from the fundamental data provider
        sector_map = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'AMZN': 'Consumer Cyclical',
            'GOOGL': 'Communication Services',
            'FB': 'Communication Services',
            'TSLA': 'Automotive',
            'BRK-B': 'Financial Services',
            'JNJ': 'Healthcare',
            'JPM': 'Financial Services',
            'V': 'Financial Services',
            'PG': 'Consumer Defensive',
            'UNH': 'Healthcare',
            'HD': 'Consumer Cyclical',
            'BAC': 'Financial Services',
            'MA': 'Financial Services',
            'DIS': 'Communication Services',
            'NVDA': 'Technology',
            'PYPL': 'Financial Services',
            'ADBE': 'Technology',
            'CMCSA': 'Communication Services',
            'NFLX': 'Communication Services',
            'XOM': 'Energy',
            'VZ': 'Communication Services',
            'KO': 'Consumer Defensive',
            'INTC': 'Technology'
        }
        
        return sector_map.get(ticker, 'Unknown')