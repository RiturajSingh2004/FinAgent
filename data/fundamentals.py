import requests
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv()

class FundamentalDataProvider:
    """Class to fetch fundamental data using Financial Modeling Prep API"""
    
    # Get API key from environment variables or use a default free key
    API_KEY = os.getenv('FMP_API_KEY', '')
    BASE_URL = 'https://financialmodelingprep.com/api/v3'
    
    @staticmethod
    def get_company_profile(ticker):
        """
        Get company profile data
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Company profile data
        """
        endpoint = f"{FundamentalDataProvider.BASE_URL}/profile/{ticker}"
        params = {'apikey': FundamentalDataProvider.API_KEY}
        
        try:
            response = requests.get(endpoint, params=params)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return data[0]
            return {}
        except Exception as e:
            print(f"Error fetching company profile for {ticker}: {e}")
            return {}
    
    @staticmethod
    def get_financial_ratios(ticker):
        """
        Get key financial ratios
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Dictionary of financial ratios
        """
        endpoint = f"{FundamentalDataProvider.BASE_URL}/ratios-ttm/{ticker}"
        params = {'apikey': FundamentalDataProvider.API_KEY}
        
        try:
            response = requests.get(endpoint, params=params)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return data[0]
            return {}
        except Exception as e:
            print(f"Error fetching financial ratios for {ticker}: {e}")
            return {}
    
    @staticmethod
    def get_income_statement(ticker, period="annual", limit=1):
        """
        Get income statement data
        
        Args:
            ticker (str): Stock ticker symbol
            period (str): annual or quarter
            limit (int): Number of periods to return
            
        Returns:
            dict: Income statement data
        """
        endpoint = f"{FundamentalDataProvider.BASE_URL}/income-statement/{ticker}"
        params = {
            'apikey': FundamentalDataProvider.API_KEY,
            'period': period,
            'limit': limit
        }
        
        try:
            response = requests.get(endpoint, params=params)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return data[0]
            return {}
        except Exception as e:
            print(f"Error fetching income statement for {ticker}: {e}")
            return {}

    @staticmethod
    def get_news_sentiment(ticker):
        """
        Get latest news sentiment (estimated based on available free APIs)
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: News sentiment data
        """
        # In a real implementation, this would use the FMP News Sentiment API
        # For this example, we'll return a placeholder structure
        # with data that would typically come from the API
        
        return {
            'ticker': ticker,
            'newsItems': 5,  # Number of recent news articles analyzed
            'overallSentiment': 'neutral',  # positive, negative, or neutral
            'sentimentScore': 0.2,  # Range from -1 (negative) to 1 (positive)
            'latestNewsDate': '2023-04-01'
        }
    
    @staticmethod
    def get_sector_performance():
        """
        Get sector performance data
        
        Returns:
            list: List of sector performance data
        """
        endpoint = f"{FundamentalDataProvider.BASE_URL}/sector-performance"
        params = {'apikey': FundamentalDataProvider.API_KEY}
        
        try:
            response = requests.get(endpoint, params=params)
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            print(f"Error fetching sector performance: {e}")
            return []