"""
Модуль расчёта Loyalty Score - метрики лояльности клиента
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LoyaltyWeights:
    """Веса компонентов Loyalty Score"""
    is_recommended: float = 0.30
    sentiment: float = 0.35
    stars: float = 0.25
    engagement: float = 0.10

    def validate(self) -> bool:
        """Проверка, что веса в сумме дают 1.0"""
        total = self.is_recommended + self.sentiment + self.stars + self.engagement
        return abs(total - 1.0) < 0.001


class LoyaltyScorer:
    """
    Расчёт Loyalty Score на основе данных отзывов

    Формула:
    Loyalty = w1 * IsRecommended + w2 * Sentiment + w3 * Stars_norm + w4 * Engagement

    Где:
    - IsRecommended: 0 или 1
    - Sentiment: комбинированный score [0, 1]
    - Stars_norm: (Stars - 1) / 4 для шкалы 1-5
    - Engagement: нормализованная вовлечённость в отзыв [0, 1]
    """

    # Границы сегментов лояльности
    LOYAL_THRESHOLD = 0.90      # >= 0.9 = loyal
    NEUTRAL_THRESHOLD = 0.70    # 0.7-0.9 = neutral, < 0.7 = at_risk

    def __init__(self, weights: Optional[LoyaltyWeights] = None):
        """
        Инициализация скорера

        Args:
            weights: Кастомные веса или None для дефолтных
        """
        self.weights = weights or LoyaltyWeights()

        if not self.weights.validate():
            raise ValueError("Сумма весов должна быть равна 1.0")

    def calculate_engagement(self, row: pd.Series) -> float:
        """
        Расчёт вовлечённости на основе заполненности полей

        Args:
            row: Строка DataFrame

        Returns:
            Engagement score [0, 1]
        """
        score = 0.0

        # Наличие полей (по 0.3 за каждое)
        if 'has_pros' in row and row['has_pros']:
            score += 0.30
        elif 'pros' in row and row['pros'] and len(str(row['pros'])) > 0:
            score += 0.30

        if 'has_cons' in row and row['has_cons']:
            score += 0.30
        elif 'cons' in row and row['cons'] and len(str(row['cons'])) > 0:
            score += 0.30

        if 'has_comment' in row and row['has_comment']:
            score += 0.20
        elif 'comment' in row and row['comment'] and len(str(row['comment'])) > 0:
            score += 0.20

        # Бонус за длинный текст (до 0.2)
        total_length = 0
        for col in ['pros', 'cons', 'comment']:
            if col in row and row[col]:
                total_length += len(str(row[col]))

        # Нормализация длины: 500+ символов = максимальный бонус
        length_bonus = min(total_length / 500, 1.0) * 0.20
        score += length_bonus

        return min(score, 1.0)

    def normalize_stars(self, stars: float, min_stars: int = 1, max_stars: int = 5) -> float:
        """
        Нормализация оценки звёзд к диапазону [0, 1]

        Args:
            stars: Оценка в звёздах
            min_stars: Минимальная оценка
            max_stars: Максимальная оценка

        Returns:
            Нормализованная оценка [0, 1]
        """
        if pd.isna(stars):
            return 0.5  # Среднее значение при отсутствии данных

        stars = float(stars)
        normalized = (stars - min_stars) / (max_stars - min_stars)
        return max(0, min(1, normalized))

    def calculate_loyalty_score(self, row: pd.Series) -> float:
        """
        Расчёт Loyalty Score для одной записи

        Args:
            row: Строка DataFrame с нужными полями

        Returns:
            Loyalty Score [0, 1]
        """
        # IsRecommended
        is_recommended = 0.0
        if 'is_recommended' in row:
            is_recommended = float(row['is_recommended']) if pd.notna(row['is_recommended']) else 0.0

        # Sentiment (если уже рассчитан)
        sentiment = 0.5
        if 'combined_sentiment' in row and pd.notna(row['combined_sentiment']):
            sentiment = float(row['combined_sentiment'])

        # Stars
        stars_norm = 0.5
        if 'stars' in row and pd.notna(row['stars']):
            stars_norm = self.normalize_stars(row['stars'])

        # Engagement
        engagement = self.calculate_engagement(row)

        # Итоговый score
        loyalty_score = (
            self.weights.is_recommended * is_recommended +
            self.weights.sentiment * sentiment +
            self.weights.stars * stars_norm +
            self.weights.engagement * engagement
        )

        return max(0, min(1, loyalty_score))

    def get_loyalty_segment(self, score: float) -> str:
        """
        Определение сегмента лояльности

        Args:
            score: Loyalty Score

        Returns:
            Название сегмента: 'loyal', 'neutral', 'at_risk'
        """
        if score >= self.LOYAL_THRESHOLD:
            return 'loyal'
        elif score >= self.NEUTRAL_THRESHOLD:
            return 'neutral'
        else:
            return 'at_risk'

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Расчёт Loyalty Score для всего датафрейма

        Args:
            df: DataFrame с данными отзывов

        Returns:
            DataFrame с добавленными колонками loyalty_score и loyalty_segment
        """
        df = df.copy()

        print(f"Расчёт Loyalty Score для {len(df):,} записей...")

        # Рассчитываем score для каждой строки
        df['loyalty_score'] = df.apply(self.calculate_loyalty_score, axis=1)

        # Определяем сегменты
        df['loyalty_segment'] = df['loyalty_score'].apply(self.get_loyalty_segment)

        # Добавляем компоненты для анализа
        df['engagement_score'] = df.apply(self.calculate_engagement, axis=1)
        if 'stars' in df.columns:
            df['stars_normalized'] = df['stars'].apply(self.normalize_stars)

        return df

    def get_loyalty_summary(self, df: pd.DataFrame) -> Dict:
        """
        Получение сводки по лояльности

        Args:
            df: DataFrame с рассчитанным loyalty_score

        Returns:
            Словарь со статистикой
        """
        if 'loyalty_score' not in df.columns:
            raise ValueError("Сначала вызовите score_dataframe()")

        summary = {
            'total_reviews': len(df),
            'mean_loyalty': df['loyalty_score'].mean(),
            'median_loyalty': df['loyalty_score'].median(),
            'std_loyalty': df['loyalty_score'].std(),
            'min_loyalty': df['loyalty_score'].min(),
            'max_loyalty': df['loyalty_score'].max(),
        }

        # Распределение по сегментам
        if 'loyalty_segment' in df.columns:
            segment_counts = df['loyalty_segment'].value_counts()
            segment_pcts = df['loyalty_segment'].value_counts(normalize=True) * 100

            summary['segments'] = {
                'loyal': {
                    'count': segment_counts.get('loyal', 0),
                    'percent': round(segment_pcts.get('loyal', 0), 1)
                },
                'neutral': {
                    'count': segment_counts.get('neutral', 0),
                    'percent': round(segment_pcts.get('neutral', 0), 1)
                },
                'at_risk': {
                    'count': segment_counts.get('at_risk', 0),
                    'percent': round(segment_pcts.get('at_risk', 0), 1)
                }
            }

        # Анализ по категориям (если есть)
        if 'product_type' in df.columns:
            category_loyalty = df.groupby('product_type')['loyalty_score'].agg(['mean', 'count'])
            category_loyalty = category_loyalty.sort_values('mean', ascending=False)
            summary['top_categories'] = category_loyalty.head(10).to_dict('index')
            summary['bottom_categories'] = category_loyalty.tail(5).to_dict('index')

        return summary

    def print_loyalty_report(self, df: pd.DataFrame) -> None:
        """Вывод отчёта по лояльности"""
        summary = self.get_loyalty_summary(df)

        print("\n" + "=" * 60)
        print("ОТЧЁТ ПО ЛОЯЛЬНОСТИ КЛИЕНТОВ DARLING")
        print("=" * 60)

        print(f"\nВсего отзывов: {summary['total_reviews']:,}")
        print(f"\n--- Loyalty Score ---")
        print(f"  Средний: {summary['mean_loyalty']:.3f}")
        print(f"  Медиана: {summary['median_loyalty']:.3f}")
        print(f"  Стд. отклонение: {summary['std_loyalty']:.3f}")
        print(f"  Диапазон: [{summary['min_loyalty']:.3f}, {summary['max_loyalty']:.3f}]")

        if 'segments' in summary:
            print(f"\n--- Сегментация ---")
            for segment, data in summary['segments'].items():
                emoji = {'loyal': '+', 'neutral': '~', 'at_risk': '-'}[segment]
                print(f"  [{emoji}] {segment:10}: {data['count']:,} ({data['percent']:.1f}%)")

        if 'top_categories' in summary:
            print(f"\n--- Топ категории по лояльности ---")
            for cat, data in list(summary['top_categories'].items())[:5]:
                print(f"  {cat[:30]:30} : {data['mean']:.3f} (n={data['count']})")

        print("\n" + "=" * 60)


def analyze_loyalty_drivers(df: pd.DataFrame) -> Dict:
    """
    Анализ драйверов лояльности - какие факторы больше влияют

    Args:
        df: DataFrame с рассчитанным loyalty_score

    Returns:
        Словарь с корреляциями
    """
    correlations = {}

    target = 'loyalty_score'
    if target not in df.columns:
        return correlations

    # Корреляция с компонентами
    components = ['is_recommended', 'stars', 'combined_sentiment', 'engagement_score']

    for col in components:
        if col in df.columns:
            corr = df[target].corr(df[col])
            if pd.notna(corr):
                correlations[col] = round(corr, 3)

    return dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))


if __name__ == "__main__":
    # Тестовый пример
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from pathlib import Path

    # Создаём тестовые данные
    test_data = pd.DataFrame({
        'pros': ['Отличный продукт', 'Хорошо', ''],
        'cons': ['', 'Дороговато', 'Не помогает вообще'],
        'comment': ['Рекомендую всем!', '', 'Разочарование'],
        'is_recommended': [1, 1, 0],
        'stars': [5, 4, 2],
        'combined_sentiment': [0.9, 0.6, 0.2]
    })

    scorer = LoyaltyScorer()
    result = scorer.score_dataframe(test_data)

    print("Тестовые результаты:")
    print(result[['loyalty_score', 'loyalty_segment', 'engagement_score']])
