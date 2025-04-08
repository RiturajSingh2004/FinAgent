class RiskProfiler:
    """Class for evaluating and classifying user risk profiles"""
    
    # Risk category definitions
    RISK_CATEGORIES = {
        'conservative': {
            'description': 'Conservative investors prioritize capital preservation and steady income over growth. They prefer lower-risk investments and are willing to accept lower returns to avoid volatility.',
            'recommended_horizon': 'Short to medium-term (1-5 years)',
            'suitable_assets': ['US Bonds', 'Large Cap Stocks', 'Cash', 'Treasury Bonds']
        },
        'moderate': {
            'description': 'Moderate investors seek a balance between growth and income, accepting moderate volatility for moderate returns. They aim for steady portfolio growth over time while limiting downside risk.',
            'recommended_horizon': 'Medium-term (5-10 years)',
            'suitable_assets': ['US Large Cap', 'International Developed', 'US Bonds', 'Real Estate']
        },
        'aggressive': {
            'description': 'Aggressive investors prioritize maximum long-term growth and are comfortable with significant volatility. They are willing to accept larger drawdowns for potentially higher returns.',
            'recommended_horizon': 'Long-term (10+ years)',
            'suitable_assets': ['US Small Cap', 'Emerging Markets', 'International Stocks', 'Sector-specific investments']
        }
    }
    
    @staticmethod
    def calculate_risk_score(answers):
        """
        Calculate risk score based on questionnaire answers
        
        Args:
            answers (dict): Answers to risk questionnaire
            
        Returns:
            float: Risk score (0-100)
        """
        # Define question weights
        weights = {
            'investment_timeline': 0.15,
            'risk_comfort': 0.25,
            'loss_reaction': 0.20,
            'investment_goal': 0.15,
            'income_stability': 0.10,
            'investment_knowledge': 0.15
        }
        
        # Question scoring
        scores = {
            'investment_timeline': {
                'less_than_1_year': 0,
                '1_to_3_years': 25,
                '3_to_5_years': 50,
                '5_to_10_years': 75,
                'more_than_10_years': 100
            },
            'risk_comfort': {
                'very_low': 0,
                'low': 25,
                'moderate': 50,
                'high': 75,
                'very_high': 100
            },
            'loss_reaction': {
                'sell_all': 0,
                'sell_some': 25,
                'do_nothing': 50,
                'buy_some_more': 75,
                'buy_significantly_more': 100
            },
            'investment_goal': {
                'capital_preservation': 0,
                'income': 25,
                'balanced': 50,
                'growth': 75,
                'aggressive_growth': 100
            },
            'income_stability': {
                'unstable': 0,
                'somewhat_stable': 25,
                'stable': 50,
                'very_stable': 75,
                'extremely_stable': 100
            },
            'investment_knowledge': {
                'none': 0,
                'basic': 25,
                'average': 50,
                'advanced': 75,
                'expert': 100
            }
        }
        
        # Calculate weighted score
        total_score = 0
        
        for question, answer in answers.items():
            if question in weights and question in scores and answer in scores[question]:
                total_score += weights[question] * scores[question][answer]
        
        return total_score
    
    @staticmethod
    def classify_risk_profile(risk_score):
        """
        Classify user into a risk profile category
        
        Args:
            risk_score (float): Risk score (0-100)
            
        Returns:
            str: Risk profile category (conservative, moderate, aggressive)
        """
        if risk_score < 35:
            return 'conservative'
        elif risk_score < 65:
            return 'moderate'
        else:
            return 'aggressive'
    
    @staticmethod
    def get_risk_profile_summary(risk_profile):
        """
        Get summary of a risk profile
        
        Args:
            risk_profile (str): Risk profile category
            
        Returns:
            dict: Summary of risk profile
        """
        if risk_profile in RiskProfiler.RISK_CATEGORIES:
            return RiskProfiler.RISK_CATEGORIES[risk_profile]
        else:
            return RiskProfiler.RISK_CATEGORIES['moderate']  # Default to moderate
    
    @staticmethod
    def get_age_based_suggestion(age):
        """
        Get age-based investment suggestion
        
        Args:
            age (int): User age
            
        Returns:
            dict: Age-based suggestion
        """
        if age < 30:
            return {
                'suggested_profile': 'aggressive',
                'stock_bond_ratio': '90/10',
                'rationale': 'Young investors have a long time horizon and can take more risk for potentially higher returns.'
            }
        elif age < 40:
            return {
                'suggested_profile': 'aggressive',
                'stock_bond_ratio': '80/20',
                'rationale': 'Investors in their 30s have time to recover from market downturns and should focus on growth.'
            }
        elif age < 50:
            return {
                'suggested_profile': 'moderate',
                'stock_bond_ratio': '70/30',
                'rationale': 'Investors in their 40s should start balancing growth with some capital preservation.'
            }
        elif age < 60:
            return {
                'suggested_profile': 'moderate',
                'stock_bond_ratio': '60/40',
                'rationale': 'As retirement approaches, a more balanced portfolio helps protect against market volatility.'
            }
        else:
            return {
                'suggested_profile': 'conservative',
                'stock_bond_ratio': '40/60',
                'rationale': 'Near or in retirement, focus shifts toward income and capital preservation.'
            }
    
    @staticmethod
    def recommend_allocation_adjustments(risk_profile, financial_situation, investment_horizon):
        """
        Recommend adjustments to standard allocations based on specific factors
        
        Args:
            risk_profile (str): Risk profile category
            financial_situation (dict): Financial situation details
            investment_horizon (str): Investment time horizon
            
        Returns:
            dict: Recommended adjustments
        """
        adjustments = {}
        
        # Adjust based on financial situation
        if financial_situation.get('high_debt', False):
            adjustments['bonds'] = 'Increase bond allocation by 5-10% for stability'
            adjustments['small_cap'] = 'Reduce small cap exposure to lower portfolio volatility'
        
        if financial_situation.get('emergency_fund', False) == False:
            adjustments['cash'] = 'Consider holding 3-6 months of expenses in cash before fully investing'
        
        # Adjust based on investment horizon
        if investment_horizon == 'short_term':
            adjustments['bonds'] = 'Increase bond allocation to match shorter time horizon'
            adjustments['stocks'] = 'Reduce stock allocation, particularly in more volatile segments'
        elif investment_horizon == 'long_term' and risk_profile == 'conservative':
            adjustments['stocks'] = 'Consider slightly higher equity allocation given long time horizon'
        
        return adjustments