import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
import random

class PortfolioOptimizer:
    """Class for portfolio optimization and allocation"""
    
    def __init__(self):
        """Initialize portfolio optimizer"""
        pass
    
    @staticmethod
    def allocate_by_risk_profile(risk_profile, asset_classes=None):
        """
        Generate asset allocation based on risk profile
        
        Args:
            risk_profile (str): User risk profile (conservative, moderate, aggressive)
            asset_classes (list): Optional list of asset classes to include
            
        Returns:
            dict: Asset allocation percentages
        """
        # Default asset class allocations by risk profile
        default_allocations = {
            'conservative': {
                'US Large Cap': 15,
                'US Mid Cap': 5,
                'US Small Cap': 0,
                'International Developed': 5,
                'Emerging Markets': 0,
                'US Bonds': 60,
                'Real Estate': 5,
                'Commodities': 5,
                'Cash': 5,
            },
            'moderate': {
                'US Large Cap': 30,
                'US Mid Cap': 10,
                'US Small Cap': 5,
                'International Developed': 10,
                'Emerging Markets': 5,
                'US Bonds': 30,
                'Real Estate': 5,
                'Commodities': 5,
                'Cash': 0,
            },
            'aggressive': {
                'US Large Cap': 40,
                'US Mid Cap': 15,
                'US Small Cap': 10,
                'International Developed': 15,
                'Emerging Markets': 10,
                'US Bonds': 5,
                'Real Estate': 5,
                'Commodities': 0,
                'Cash': 0,
            }
        }
        
        # Adjust for actual available asset classes if provided
        if asset_classes:
            allocations = {}
            # Get the default allocation for the given risk profile
            profile_allocation = default_allocations.get(risk_profile, default_allocations['moderate'])
            
            # Filter by available asset classes
            available_classes = set(asset_classes).intersection(set(profile_allocation.keys()))
            
            # Allocate only to available asset classes
            total_weight = sum(profile_allocation[ac] for ac in available_classes)
            if total_weight > 0:
                # Normalize allocations to sum to 100%
                for ac in available_classes:
                    allocations[ac] = (profile_allocation[ac] / total_weight) * 100
            else:
                # Fallback: equal weighting
                weight = 100.0 / len(asset_classes)
                allocations = {ac: weight for ac in asset_classes}
        else:
            # Use default allocations
            allocations = default_allocations.get(risk_profile, default_allocations['moderate'])
        
        return allocations
    
    @staticmethod
    def optimize_portfolio(returns, expected_returns, risk_aversion=3.0, max_weight=0.3):
        """
        Mean-variance portfolio optimization
        
        Args:
            returns (pd.DataFrame): Historical returns
            expected_returns (pd.Series): Expected returns for each asset
            risk_aversion (float): Risk aversion parameter (higher means more risk-averse)
            max_weight (float): Maximum weight for any single asset
            
        Returns:
            pd.Series: Optimized portfolio weights
        """
        n_assets = len(expected_returns)
        
        # Use Ledoit-Wolf shrinkage for robust covariance estimation
        lw = LedoitWolf().fit(returns)
        cov_matrix = lw.covariance_
        
        # Define optimization objective: maximize utility (return - risk_aversion * variance)
        def portfolio_objective(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            utility = portfolio_return - risk_aversion * portfolio_variance
            return -utility  # Minimize negative utility (maximize utility)
        
        # Constraints: weights sum to 1, weights between 0 and max_weight
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = tuple((0.0, max_weight) for _ in range(n_assets))
        
        # Initial guess: equal weights
        initial_weights = np.ones(n_assets) / n_assets
        
        # Perform optimization
        result = minimize(
            portfolio_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Return optimized weights as a Series
        if result['success']:
            weights = pd.Series(result['x'], index=expected_returns.index)
            weights = weights / weights.sum()  # Normalize to ensure weights sum to 1
            return weights
        else:
            # Fallback to equal weighting if optimization fails
            return pd.Series(1.0/n_assets, index=expected_returns.index)
    
    @staticmethod
    def get_portfolio_statistics(weights, returns, expected_returns=None):
        """
        Calculate portfolio statistics
        
        Args:
            weights (pd.Series): Portfolio weights
            returns (pd.DataFrame): Historical returns
            expected_returns (pd.Series): Expected returns (optional)
            
        Returns:
            dict: Portfolio statistics
        """
        # Align weights and returns indices
        aligned_weights = weights.reindex(returns.columns, fill_value=0)
        
        # Calculate historical portfolio returns
        portfolio_returns = returns.dot(aligned_weights)
        
        # Calculate portfolio expected return
        if expected_returns is not None:
            expected_return = (aligned_weights * expected_returns).sum()
        else:
            expected_return = portfolio_returns.mean() * 252  # Annualized
        
        # Calculate portfolio volatility
        volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
        
        # Calculate Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Calculate maximum drawdown
        cum_returns = (1 + portfolio_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        return {
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    @staticmethod
    def select_tickers_by_asset_class(asset_class_allocation, ticker_asset_class_map, 
                                      n_per_class=2, market_data_provider=None, exclude_sectors=None):
        """
        Select tickers for each asset class based on allocation
        
        Args:
            asset_class_allocation (dict): Allocation percentages by asset class
            ticker_asset_class_map (dict): Mapping of tickers to asset classes
            n_per_class (int): Number of tickers to select per asset class
            market_data_provider: Optional provider for market data to rank tickers
            exclude_sectors (list): Optional list of sectors to exclude
            
        Returns:
            dict: Selected tickers with weights
        """
        # Group tickers by asset class
        asset_class_tickers = {}
        for ticker, asset_class in ticker_asset_class_map.items():
            if asset_class not in asset_class_tickers:
                asset_class_tickers[asset_class] = []
            asset_class_tickers[asset_class].append(ticker)
        
        # If we have market data provider, rank tickers by performance
        if market_data_provider:
            from data.data_utils import DataUtils
            sector_map = {}
            
            # Create sector mapping if excluding sectors
            if exclude_sectors:
                for ticker in ticker_asset_class_map.keys():
                    sector = DataUtils.map_ticker_to_sector(ticker)
                    sector_map[ticker] = sector
        
        # Select tickers for each asset class in the allocation
        selected_tickers = {}
        
        for asset_class, allocation in asset_class_allocation.items():
            if asset_class in asset_class_tickers and allocation > 0:
                # Get available tickers for this asset class
                available_tickers = asset_class_tickers[asset_class]
                
                # Filter out excluded sectors if provided
                if exclude_sectors and market_data_provider:
                    available_tickers = [t for t in available_tickers 
                                         if t in sector_map and sector_map[t] not in exclude_sectors]
                
                if not available_tickers:
                    continue
                
                selected_tickers_for_class = []
                
                # If market data provider is available, select based on performance
                if market_data_provider and len(available_tickers) > n_per_class:
                    try:
                        # Get performance data for tickers in this asset class
                        ticker_performance = {}
                        for ticker in available_tickers:
                            stock_data = market_data_provider.get_stock_data(ticker, period="1y")
                            if not stock_data.empty:
                                _, annual_return, _ = market_data_provider.calculate_returns(stock_data)
                                if annual_return is not None:
                                    ticker_performance[ticker] = annual_return
                        
                        # Sort by performance and select top performers
                        if ticker_performance:
                            sorted_tickers = sorted(ticker_performance.items(), 
                                                    key=lambda x: x[1], reverse=True)
                            selected_tickers_for_class = [t[0] for t in sorted_tickers[:n_per_class]]
                    except Exception as e:
                        print(f"Error selecting tickers by performance: {e}")
                
                # If we couldn't select by performance or don't have market data provider, select randomly
                if not selected_tickers_for_class:
                    # Select up to n_per_class tickers
                    n_select = min(n_per_class, len(available_tickers))
                    selected_tickers_for_class = random.sample(available_tickers, n_select)
                
                # Allocate evenly within this asset class
                weight_per_ticker = allocation / len(selected_tickers_for_class)
                for ticker in selected_tickers_for_class:
                    selected_tickers[ticker] = weight_per_ticker
        
        return selected_tickers
    
    @staticmethod
    def rebalance_portfolio(current_allocation, target_allocation, threshold=0.05):
        """
        Determine if and how to rebalance portfolio
        
        Args:
            current_allocation (dict): Current allocation percentages
            target_allocation (dict): Target allocation percentages
            threshold (float): Threshold for rebalancing
            
        Returns:
            dict: Rebalancing recommendations
        """
        rebalance_actions = {}
        
        # Calculate differences between current and target allocations
        for asset, target_weight in target_allocation.items():
            current_weight = current_allocation.get(asset, 0)
            diff = current_weight - target_weight
            
            # Check if difference exceeds threshold
            if abs(diff) >= threshold:
                if diff > 0:
                    action = 'reduce'
                else:
                    action = 'increase'
                
                rebalance_actions[asset] = {
                    'current': current_weight,
                    'target': target_weight,
                    'difference': diff,
                    'action': action
                }
        
        return rebalance_actions

class PortfolioGenerator:
    """Class for generating investment portfolios based on user profiles"""
    
    def __init__(self, market_data_provider=None, fundamental_data_provider=None):
        """
        Initialize the portfolio generator
        
        Args:
            market_data_provider: Provider for market data
            fundamental_data_provider: Provider for fundamental data
        """
        self.market_data_provider = market_data_provider
        self.fundamental_data_provider = fundamental_data_provider
        self.optimizer = PortfolioOptimizer()
    
    def generate_portfolio(self, user_profile, ticker_pool=None):
        """
        Generate a portfolio based on user profile
        
        Args:
            user_profile (dict): User profile with risk preferences and goals
            ticker_pool (list): Optional list of tickers to select from
            
        Returns:
            dict: Generated portfolio with allocations and rationale
        """
        # Extract risk profile and investment horizon
        risk_profile = user_profile.get('risk_tolerance', 'moderate')
        investment_horizon = user_profile.get('investment_horizon', 'medium-term')
        exclude_sectors = user_profile.get('exclude_sectors', [])
        
        # Get asset class allocation based on risk profile
        asset_class_allocation = self.optimizer.allocate_by_risk_profile(risk_profile)
        
        # Get asset class mappings
        from data.data_utils import DataUtils
        ticker_asset_class_map = DataUtils.create_asset_class_mappings()
        
        # If a ticker pool is provided, filter the mapping
        if ticker_pool:
            filtered_map = {t: ac for t, ac in ticker_asset_class_map.items() if t in ticker_pool}
            ticker_asset_class_map = filtered_map if filtered_map else ticker_asset_class_map
        
        # Select tickers for each asset class
        ticker_allocation = self.optimizer.select_tickers_by_asset_class(
            asset_class_allocation, 
            ticker_asset_class_map,
            n_per_class=2,  # Select 2 tickers per asset class
            market_data_provider=self.market_data_provider,
            exclude_sectors=exclude_sectors
        )
        
        # Convert to the format expected by the frontend
        allocations = []
        for asset_class, allocation in asset_class_allocation.items():
            if allocation > 0:
                # Find tickers for this asset class
                class_tickers = [t for t, ac in ticker_asset_class_map.items() 
                                if ac == asset_class and t in ticker_allocation]
                
                if class_tickers:
                    allocations.append({
                        'asset_class': asset_class,
                        'percentage': allocation,
                        'tickers': class_tickers
                    })
        
        # Sort allocations by percentage (descending)
        allocations = sorted(allocations, key=lambda x: x['percentage'], reverse=True)
        
        # Create the portfolio object
        portfolio = {
            'user_profile': user_profile,
            'asset_class_allocation': asset_class_allocation,
            'ticker_allocation': ticker_allocation,
            'allocations': allocations
        }
        
        return portfolio
    
    def enhance_portfolio_with_market_data(self, portfolio):
        """
        Enhance portfolio with market data
        
        Args:
            portfolio (dict): Portfolio to enhance
            
        Returns:
            dict: Enhanced portfolio
        """
        if not self.market_data_provider:
            return portfolio
        
        # Get tickers from the portfolio
        tickers = list(portfolio.get('ticker_allocation', {}).keys())
        
        if not tickers:
            return portfolio
        
        # Fetch market data for these tickers
        ticker_data = {}
        
        for ticker in tickers:
            try:
                # Get stock price data
                stock_data = self.market_data_provider.get_stock_data(ticker, period="1y")
                
                if not stock_data.empty:
                    daily_returns, annual_return, volatility = (
                        self.market_data_provider.calculate_returns(stock_data)
                    )
                    
                    # Get additional stats if available
                    stats = self.market_data_provider.get_key_stats(ticker)
                    
                    ticker_data[ticker] = {
                        'annual_return': annual_return,
                        'volatility': volatility,
                        'stats': stats
                    }
                    
                    # If fundamental data provider is available, get additional data
                    if self.fundamental_data_provider:
                        try:
                            # Get company profile data
                            profile = self.fundamental_data_provider.get_company_profile(ticker)
                            if profile:
                                ticker_data[ticker]['company_name'] = profile.get('companyName', '')
                                ticker_data[ticker]['sector'] = profile.get('sector', '')
                                ticker_data[ticker]['industry'] = profile.get('industry', '')
                            
                            # Get financial ratios
                            ratios = self.fundamental_data_provider.get_financial_ratios(ticker)
                            if ratios:
                                ticker_data[ticker]['ratios'] = ratios
                        except Exception as e:
                            print(f"Error getting fundamental data for {ticker}: {e}")
            except Exception as e:
                print(f"Error enhancing portfolio data for {ticker}: {e}")
        
        # Add market data to the portfolio
        portfolio['market_data'] = ticker_data
        
        # Add sector performance data if available
        try:
            if self.market_data_provider:
                sector_performance = self.market_data_provider.get_sectors_performance()
                if not sector_performance.empty:
                    portfolio['sector_performance'] = sector_performance.to_dict('records')
        except Exception as e:
            print(f"Error getting sector performance: {e}")
        
        return portfolio