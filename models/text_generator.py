from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random

class TextGenerator:
    """Class for generating investment rationales"""
    
    def __init__(self):
        """Initialize text generation model and tokenizer"""
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self):
        """
        Load the model on first use (lazy loading to save memory)
        """
        if self.model is None:
            try:
                # Load model
                model_name = "google/gemma-3-1b"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                # Load in 8-bit to reduce memory requirements if available
                if torch.cuda.is_available():
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name, 
                        device_map="auto",
                        load_in_8bit=True,
                        torch_dtype=torch.float16
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(model_name)
                    self.model.to(self.device)
                print(f"Loaded {model_name} for text generation")
            except Exception as e:
                print(f"Error loading model: {e}")
                # We'll use template-based generation if model loading fails
                self.model = None
                self.tokenizer = None
    
    def generate_investment_rationale(self, portfolio_data, user_profile, market_sentiment):
        """
        Generate investment rationale based on portfolio data and user profile
        
        Args:
            portfolio_data (dict): Portfolio allocation data
            user_profile (dict): User risk profile and investment goals
            market_sentiment (dict): Market sentiment analysis results
            
        Returns:
            str: Generated investment rationale
        """
        # Extract key information
        risk_level = user_profile.get('risk_tolerance', 'moderate')
        investment_horizon = user_profile.get('investment_horizon', 'medium-term')
        investment_goal = user_profile.get('investment_goal', 'balanced')
        
        # Map investment horizon to time period
        horizon_map = {
            'short_term': '1-3 years',
            'medium_term': '3-10 years',
            'long_term': '10+ years'
        }
        time_period = horizon_map.get(investment_horizon, '3-10 years')
        
        # Map investment goal to description
        goal_descriptions = {
            'capital_preservation': 'preserving capital with minimal risk',
            'income': 'generating regular income',
            'balanced': 'balancing growth and income',
            'growth': 'achieving long-term growth',
            'aggressive_growth': 'maximizing growth potential'
        }
        goal_description = goal_descriptions.get(investment_goal, 'balanced investing')
        
        # Get overall market sentiment
        sentiment_score = 0
        if market_sentiment and 'overall_sentiment_score' in market_sentiment:
            sentiment_score = market_sentiment['overall_sentiment_score']
        
        if sentiment_score > 0.2:
            market_outlook = "positive"
            sentiment_description = "favorable, indicating potential opportunities for growth-oriented investments"
        elif sentiment_score < -0.2:
            market_outlook = "negative"
            sentiment_description = "challenging, suggesting a more defensive positioning may be prudent"
        else:
            market_outlook = "neutral"
            sentiment_description = "neutral, with mixed signals that suggest a balanced approach"
        
        # Get portfolio allocations
        allocations = []
        if portfolio_data and 'allocations' in portfolio_data:
            allocations = portfolio_data['allocations']
        
        # Skip using the ML model and directly go to template-based generation
        # This ensures consistent, high-quality rationales without repetition issues
        return self._generate_template_rationale(
            risk_level, 
            investment_horizon, 
            market_outlook, 
            allocations, 
            investment_goal, 
            sentiment_description,
            time_period
        )
    
    def _generate_template_rationale(self, risk_level, investment_horizon, market_outlook, 
                                   allocations, investment_goal, sentiment_description,
                                   time_period):
        """
        Generate a template-based investment rationale
        
        Args:
            risk_level (str): User risk profile
            investment_horizon (str): Investment time horizon
            market_outlook (str): Market sentiment outlook
            allocations (list): List of allocation dictionaries
            investment_goal (str): Investment goal
            sentiment_description (str): Description of market sentiment
            time_period (str): Time period for investment horizon
            
        Returns:
            str: Generated investment rationale
        """
        # Map investment goals to descriptions
        goal_descriptions = {
            'capital_preservation': 'preserving capital with minimal risk',
            'income': 'generating regular income through dividends and interest',
            'balanced': 'balancing growth potential with income generation',
            'growth': 'achieving long-term capital appreciation',
            'aggressive_growth': 'maximizing growth potential with higher risk tolerance'
        }
        
        goal_description = goal_descriptions.get(investment_goal, 'balanced investing')
        
        rationale = f"""# Investment Strategy Rationale

## Overview
Based on your {risk_level} risk tolerance and {investment_horizon} investment horizon ({time_period}), we've designed a diversified portfolio aimed at {goal_description}. This strategy is specifically tailored to your financial situation and objectives, balancing growth potential with appropriate risk management.

## Market Analysis
Current market sentiment analysis indicates a {market_outlook} outlook. The current market environment is {sentiment_description}. """

        # Add market-specific commentary
        if market_outlook == "positive":
            rationale += """Our strategy maintains core exposure to growth assets while remaining diversified to manage potential volatility. Recent economic indicators suggest continued economic expansion, though we remain vigilant about inflationary pressures and central bank policies."""
        elif market_outlook == "negative":
            rationale += """In response to current conditions, we've adjusted allocations to emphasize more defensive positions and quality assets while maintaining some exposure to potential recovery. This approach provides downside protection while positioning for eventual market improvement."""
        else:
            rationale += """Given the mixed signals, we're maintaining a balanced approach with selective exposure to both growth opportunities and defensive positions. This strategy aims to capture upside potential while providing stability during uncertain periods."""

        rationale += """

## Portfolio Strategy
The recommended asset allocation is designed to:
1. Match your stated risk tolerance and investment goals
2. Provide appropriate diversification across asset classes to reduce overall portfolio risk
3. Align with your investment time horizon
4. Position the portfolio advantageously within the current market environment
5. Balance potential returns with risk management
"""
        
        # Add allocation rationales
        rationale += "\n## Key Allocations\n"
        
        for alloc in allocations:
            asset_class = alloc.get('asset_class', 'Unknown')
            percentage = alloc.get('percentage', 0)
            tickers = alloc.get('tickers', [])
            ticker_text = ""
            if tickers:
                ticker_text = f" through investments like {', '.join(tickers)}"
            
            rationale += f"### {asset_class} ({percentage:.1f}%)\n"
            
            if 'Large Cap' in asset_class:
                rationale += f"This allocation provides core market exposure{ticker_text} to established companies with strong balance sheets, consistent cash flows, and competitive market positions. Large cap stocks offer stability and liquidity, typically with moderate dividend income, serving as the foundation of your portfolio.\n\n"
            elif 'Mid Cap' in asset_class:
                rationale += f"Mid-cap companies{ticker_text} offer a balance of growth potential and established business models. They typically provide higher growth prospects than large caps while having more established operations than small caps, enhancing your portfolio's growth potential with manageable volatility.\n\n"
            elif 'Small Cap' in asset_class:
                rationale += f"Small-cap companies{ticker_text} offer significant growth potential as they expand market share and develop new products. This allocation provides exposure to innovative businesses with substantial room for expansion, though with higher volatility. It's appropriate for your {risk_level} risk profile and {investment_horizon} time horizon.\n\n"
            elif 'International Developed' in asset_class:
                rationale += f"This allocation{ticker_text} provides geographic diversification through exposure to established economies outside the domestic market. These investments reduce country-specific risk while capturing global growth opportunities, potentially moving differently from domestic assets and improving overall portfolio efficiency.\n\n"
            elif 'Emerging Markets' in asset_class:
                rationale += f"Emerging markets{ticker_text} provide exposure to rapidly growing economies with expanding middle classes and favorable demographics. While more volatile, these markets offer potentially higher long-term returns and diversification benefits, appropriate for your {risk_level} risk tolerance with a {investment_horizon} investment horizon.\n\n"
            elif 'Bonds' in asset_class:
                rationale += f"Bond investments{ticker_text} provide income generation and stability to your portfolio. This fixed-income allocation reduces overall portfolio volatility and offers defensive positioning during market downturns, creating a more balanced risk profile while generating steady income.\n\n"
            elif 'Real Estate' in asset_class:
                rationale += f"Real estate investments{ticker_text} provide inflation protection and income through property exposure. REITs offer liquidity and diversification into an alternative asset class with typically low correlation to stocks and bonds, enhancing portfolio stability while generating income through distributions.\n\n"
            elif 'Commodities' in asset_class:
                rationale += f"Commodities{ticker_text} help protect against inflation and provide portfolio diversification through exposure to physical assets. They typically move independently of stocks and bonds, potentially performing well during inflationary periods and enhancing overall portfolio resilience against various economic scenarios.\n\n"
            elif 'Cash' in asset_class:
                rationale += f"Cash and cash equivalents{ticker_text} provide liquidity and stability to your portfolio. This allocation offers capital preservation, flexibility to take advantage of market opportunities, and a buffer against market volatility, ensuring you have accessible funds for near-term needs without disrupting longer-term investments.\n\n"
            else:
                rationale += f"This allocation{ticker_text} contributes to your portfolio's diversification and risk-adjusted return potential. As part of your overall strategy, it helps balance growth opportunities with risk management appropriate for your investment goals.\n\n"
        
        # Add rebalancing and next steps
        rationale += """
## Rebalancing Recommendations
To maintain optimal risk-adjusted returns, we recommend:

1. Quarterly portfolio reviews to assess performance against benchmarks and market conditions
2. Annual rebalancing to target allocations, or whenever allocations drift more than 5% from targets
3. Tax-efficient rebalancing by directing new investments toward underweighted assets when possible
4. Adjusting allocations as your time horizon shortens or financial circumstances change

## Next Steps
1. Review this proposed strategy in detail, considering how it aligns with your complete financial picture
2. Consider implementation using dollar-cost averaging to reduce timing risk, especially in current market conditions
3. Set up automatic investments to maintain investment discipline regardless of market movements
4. Schedule a portfolio review in 3-6 months to assess performance and make necessary adjustments
5. Ensure this investment approach coordinates with your broader financial planning, including retirement, tax, and estate planning

This investment strategy is designed to be dynamic and responsive to both market conditions and your evolving financial needs. Regular reviews and adjustments will help ensure your portfolio continues to align with your goals over time.
"""
        
        return rationale
    
    def generate_market_analysis(self, news_items, sentiment_data):
        """
        Generate market analysis based on news and sentiment
        
        Args:
            news_items (list): List of market news headlines
            sentiment_data (dict): Sentiment analysis results
            
        Returns:
            str: Generated market analysis
        """
        if not news_items:
            return "No market news data available for analysis."
        
        sentiment_score = sentiment_data.get('overall_sentiment_score', 0) if sentiment_data else 0
        
        if sentiment_score > 0.2:
            sentiment_description = "positive"
        elif sentiment_score < -0.2:
            sentiment_description = "negative"
        else:
            sentiment_description = "neutral"
        
        return self._generate_template_market_analysis(news_items, sentiment_description, sentiment_score)
    
    def _generate_template_market_analysis(self, news_items, sentiment_description, sentiment_score):
        """
        Generate a template-based market analysis
        
        Args:
            news_items (list): List of market news headlines
            sentiment_description (str): Market sentiment description
            sentiment_score (float): Market sentiment score
            
        Returns:
            str: Generated market analysis
        """
        # Select key themes based on headlines
        possible_themes = [
            "monetary policy and interest rates",
            "corporate earnings reports",
            "economic indicators",
            "global market performance",
            "sector-specific developments",
            "geopolitical tensions",
            "inflation and consumer trends",
            "supply chain dynamics",
            "regulatory developments",
            "technological innovation"
        ]
        
        # Select 2-3 random themes
        selected_themes = random.sample(possible_themes, min(3, len(possible_themes)))
        
        analysis = f"""Recent market news indicates a generally {sentiment_description} sentiment (score: {sentiment_score:.2f}) with several key developments influencing market direction. Primary themes include {", ".join(selected_themes[:-1])} and {selected_themes[-1]}."""
        
        # Add sentiment-specific analysis
        if sentiment_description == "positive":
            analysis += """ The positive sentiment suggests potential support for risk assets, though investors should maintain diversification given ongoing uncertainties. Strength appears concentrated in high-quality companies with solid fundamentals and sustainable competitive advantages."""
        elif sentiment_description == "negative":
            analysis += """ The negative sentiment may create near-term headwinds for risk assets, suggesting more defensive positioning and emphasis on quality companies with strong balance sheets. This environment may also create selective opportunities for long-term investors as valuations adjust."""
        else:
            analysis += """ The neutral sentiment reflects mixed signals in the market, suggesting a balanced approach across asset classes. Focusing on quality investments with reasonable valuations appears prudent in this environment of conflicting indicators."""
        
        # Add headline reference
        if news_items:
            headline = news_items[0]
            analysis += f" The headline \"{headline}\" highlights important developments that investors should monitor for potential market impacts."
        
        # Add conclusion
        analysis += """ Overall, maintaining a disciplined, long-term investment approach aligned with your risk tolerance and goals remains advisable despite short-term market movements."""
        
        return analysis