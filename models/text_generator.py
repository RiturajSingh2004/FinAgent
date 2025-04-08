from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class TextGenerator:
    """Class for generating investment rationales using Gemma 3 1B model"""
    
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
                # Load Gemma 3 1B model
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
                print(f"Error loading Gemma model: {e}")
                # Fallback to a much smaller model if Gemma fails to load
                try:
                    model_name = "distilgpt2"
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModelForCausalLM.from_pretrained(model_name)
                    self.model.to(self.device)
                    print(f"Loaded fallback model {model_name} for text generation")
                except:
                    print("Failed to load any model for text generation")
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
        # Format risk profile
        risk_level = user_profile.get('risk_tolerance', 'moderate')
        investment_horizon = user_profile.get('investment_horizon', 'medium-term')
        
        # Get overall market sentiment
        if market_sentiment and 'overall_sentiment_score' in market_sentiment:
            sentiment_score = market_sentiment['overall_sentiment_score']
            if sentiment_score > 0.2:
                market_outlook = "positive"
            elif sentiment_score < -0.2:
                market_outlook = "negative"
            else:
                market_outlook = "neutral"
        else:
            market_outlook = "neutral"
        
        # Get top allocated sectors/assets
        allocations = []
        if portfolio_data and 'allocations' in portfolio_data:
            allocations = portfolio_data['allocations'][:3] if len(portfolio_data['allocations']) > 3 else portfolio_data['allocations']
        
        # Extract allocation details for the prompt
        allocation_text = ""
        for alloc in allocations:
            asset_class = alloc.get('asset_class', 'Unknown')
            percentage = alloc.get('percentage', 0)
            allocation_text += f"{asset_class}: {percentage:.1f}%, "
        
        # Create a detailed prompt for the model
        prompt = f"""Write a detailed investment strategy rationale for a portfolio with the following characteristics:

Risk profile: {risk_level}
Investment horizon: {investment_horizon}
Market sentiment: {market_outlook}
Top allocations: {allocation_text}

The rationale should include:
1. An overview explaining how the portfolio matches the investor's risk tolerance
2. Market analysis incorporating the current sentiment
3. Portfolio strategy explaining the asset allocation approach
4. Key allocation explanations
5. Rebalancing recommendations
"""

        # Try to use the model for generation
        try:
            # Load model if not already loaded
            self.load_model()
            
            if self.model and self.tokenizer:
                # Tokenize input
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                # Generate text
                outputs = self.model.generate(
                    **inputs,
                    max_length=1024,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
                
                # Decode generated text
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove the prompt from the generated text
                rationale = generated_text[len(prompt):].strip()
                
                # If generation is too short or empty, fall back to template
                if len(rationale.split()) < 50:
                    return self._generate_template_rationale(risk_level, investment_horizon, market_outlook, allocations)
                
                return "# Investment Strategy Rationale\n\n" + rationale
            else:
                return self._generate_template_rationale(risk_level, investment_horizon, market_outlook, allocations)
        except Exception as e:
            print(f"Error generating text: {e}")
            return self._generate_template_rationale(risk_level, investment_horizon, market_outlook, allocations)
    
    def _generate_template_rationale(self, risk_level, investment_horizon, market_outlook, allocations):
        """
        Generate a template-based investment rationale as a fallback
        """
        rationale = f"""# Investment Strategy Rationale

## Overview
Based on your {risk_level} risk tolerance and {investment_horizon} investment horizon, we've created a diversified portfolio that aims to balance growth potential with appropriate risk management.

## Market Analysis
Current market sentiment analysis indicates a {market_outlook} outlook. """

        if market_outlook == "positive":
            rationale += "This suggests favorable conditions for growth-oriented investments, though we maintain diversification to guard against potential volatility."
        elif market_outlook == "negative":
            rationale += "In response, we've adjusted allocations to emphasize more defensive positions while maintaining exposure to potential recovery."
        else:
            rationale += "We're maintaining a balanced approach with selective exposure to both growth opportunities and defensive positions."

        rationale += """

## Portfolio Strategy
The recommended asset allocation is designed to:
1. Match your stated risk tolerance
2. Provide appropriate diversification across asset classes
3. Align with your investment time horizon
4. Adapt to current market conditions

## Key Allocations
"""
        
        # Add allocation rationales
        for alloc in allocations:
            asset_class = alloc.get('asset_class', 'Unknown')
            percentage = alloc.get('percentage', 0)
            rationale += f"- **{asset_class} ({percentage:.1f}%)**: "
            
            if 'Large Cap' in asset_class or 'Index' in asset_class:
                rationale += "Provides core market exposure and stability from established companies with strong fundamentals.\n"
            elif 'Small Cap' in asset_class or 'Mid Cap' in asset_class:
                rationale += "Offers growth potential through companies with room for expansion, balanced with higher volatility.\n"
            elif 'Bond' in asset_class:
                rationale += "Provides income and helps reduce overall portfolio volatility.\n"
            elif 'International' in asset_class or 'Emerging' in asset_class:
                rationale += "Adds geographic diversification and exposure to markets with different economic cycles.\n"
            elif 'Real Estate' in asset_class:
                rationale += "Offers potential inflation protection and income through property exposure.\n"
            elif 'Commodities' in asset_class:
                rationale += "Helps protect against inflation and provides portfolio diversification through non-correlated assets.\n"
            else:
                rationale += "Contributes to overall portfolio diversification and risk-adjusted returns.\n"
        
        rationale += """
## Rebalancing Recommendation
To maintain optimal risk-adjusted returns, consider reviewing this portfolio quarterly and rebalancing when allocations drift more than 5% from targets.

## Next Steps
1. Review the proposed allocations
2. Consider setting up automatic investments to implement dollar-cost averaging
3. Schedule a portfolio review in 3-6 months to assess performance and make adjustments
"""
        
        return rationale

    def generate_with_model(self, prompt):
        """
        Generate text using the loaded model
        
        Args:
            prompt (str): Input prompt for text generation
            
        Returns:
            str: Generated text
        """
        # Load model if not already loaded
        try:
            self.load_model()
            
            if not self.model or not self.tokenizer:
                return "Unable to generate text with the model. Model not loaded."
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate text
            outputs = self.model.generate(
                **inputs,
                max_length=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the generated text
            generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
        except Exception as e:
            print(f"Error generating text: {e}")
            return "Unable to generate text with the model. Please check the error logs."