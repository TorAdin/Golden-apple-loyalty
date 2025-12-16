"""
GoldenApple Loyalty Analysis Package
Анализ лояльности клиентов по отзывам Darling
"""

from .data_loader import load_data, get_data_info
from .preprocessing import preprocess_text, clean_dataframe
from .sentiment_analyzer import SentimentAnalyzer
from .loyalty_scorer import LoyaltyScorer

__all__ = [
    'load_data',
    'get_data_info',
    'preprocess_text',
    'clean_dataframe',
    'SentimentAnalyzer',
    'LoyaltyScorer'
]
