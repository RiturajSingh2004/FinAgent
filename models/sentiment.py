import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

class SentimentAnalyzer:
    """Class for financial sentiment analysis using FinBERT"""
    
    def __init__(self):
        """Initialize the FinBERT model and tokenizer"""
        # Load FinBERT model and tokenizer
        model_name = "ProsusAI/finbert"
        self.tokenizer = None
        self.model = None
        self.labels = ["negative", "neutral", "positive"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self):
        """
        Load the model on first use (lazy loading to save memory)
        """
        if self.model is None:
            try:
                model_name = "ProsusAI/finbert"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.model.to(self.device)
            except Exception as e:
                print(f"Error loading FinBERT model: {e}")
                # Fallback to a simpler model if FinBERT fails to load
                model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.model.to(self.device)
                self.labels = ["negative", "positive"]
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of a given text
        
        Args:
            text (str): Input text for sentiment analysis
            
        Returns:
            dict: Dictionary with sentiment prediction and scores
        """
        # Load model if not already loaded
        self.load_model()
        
        # Tokenize text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # Get sentiment prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            scores = scores.cpu().numpy()[0]
        
        # Get predicted label and score
        predicted_label = self.labels[np.argmax(scores)]
        
        # Create result dictionary
        result = {
            "sentiment": predicted_label,
            "confidence": float(np.max(scores)),
            "scores": {label: float(score) for label, score in zip(self.labels, scores)}
        }
        
        return result
    
    def analyze_multiple_texts(self, texts):
        """
        Analyze sentiment for multiple texts
        
        Args:
            texts (list): List of input texts
            
        Returns:
            list: List of sentiment results
        """
        results = []
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)
        
        return results
    
    def get_sentiment_summary(self, results):
        """
        Generate a summary of sentiment results
        
        Args:
            results (list): List of sentiment results
            
        Returns:
            dict: Summary of sentiments
        """
        sentiment_counts = {label: 0 for label in self.labels}
        total_confidence = 0
        
        for result in results:
            sentiment_counts[result["sentiment"]] += 1
            total_confidence += result["confidence"]
        
        total = len(results)
        avg_confidence = total_confidence / total if total > 0 else 0
        
        # Calculate sentiment percentages
        sentiment_percentages = {
            label: count / total * 100 if total > 0 else 0 
            for label, count in sentiment_counts.items()
        }
        
        # Calculate overall sentiment score (-1 to 1)
        if len(self.labels) == 3:  # FinBERT model
            sentiment_score = (
                sentiment_percentages["positive"] / 100 - 
                sentiment_percentages["negative"] / 100
            )
        else:  # Binary sentiment model
            sentiment_score = (2 * sentiment_percentages["positive"] / 100) - 1
        
        return {
            "total_analyzed": total,
            "sentiment_counts": sentiment_counts,
            "sentiment_percentages": sentiment_percentages,
            "avg_confidence": avg_confidence,
            "overall_sentiment_score": sentiment_score
        }