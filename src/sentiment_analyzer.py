"""
Модуль анализа тональности текстов отзывов
Поддержка русского и английского языков
ОПТИМИЗИРОВАННАЯ ВЕРСИЯ - векторизованная обработка
"""

import pandas as pd
import numpy as np
import re
from typing import Dict


# Словари для keyword-based анализа (компилируем regex один раз)
RU_POSITIVE = [
    'отлично', 'супер', 'класс', 'прекрасно', 'замечательно', 'восторг',
    'нравится', 'понравил', 'рекомендую', 'советую', 'лучший', 'идеально',
    'качествен', 'эффективн', 'работает', 'помогает', 'хорошо', 'круто',
    'обожаю', 'люблю', 'божествен', 'потрясающ', 'волшебн', 'шикарн',
    'мастхэв', 'вау', 'топ', 'бомба', 'огонь', 'великолепн', 'чудесн'
]

RU_NEGATIVE = [
    'плохо', 'ужасн', 'разочаров', 'не понравил', 'не рекомендую',
    'отвратительн', 'ужас', 'на ветер', 'трата', 'бесполезн',
    'не работает', 'не помогает', 'аллерги', 'раздражен', 'жалею',
    'обман', 'подделк', 'фейк', 'ерунд', 'фигн', 'никакого эффект',
    'зря', 'хуже', 'не советую', 'разочаровал'
]

EN_POSITIVE = [
    'excellent', 'great', 'amazing', 'wonderful', 'fantastic', 'love',
    'perfect', 'best', 'recommend', 'good', 'nice', 'awesome', 'beautiful',
    'effective', 'works', 'helped', 'lovely', 'brilliant', 'superb',
    'outstanding', 'incredible', 'favorite', 'favourite', 'happy', 'pleased'
]

EN_NEGATIVE = [
    'bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disappoint',
    'waste', 'useless', 'not recommend', 'poor', 'cheap', 'broke', 'broken',
    'fake', 'scam', 'allergic', 'irritation', 'regret', 'refund', 'never again'
]

# Объединённые паттерны
ALL_POSITIVE = RU_POSITIVE + EN_POSITIVE
ALL_NEGATIVE = RU_NEGATIVE + EN_NEGATIVE

# Компилируем regex паттерны для скорости
POSITIVE_PATTERN = re.compile('|'.join(ALL_POSITIVE), re.IGNORECASE)
NEGATIVE_PATTERN = re.compile('|'.join(ALL_NEGATIVE), re.IGNORECASE)


def fast_detect_language(text: str) -> str:
    """Быстрое определение языка по символам"""
    if not text or not isinstance(text, str) or len(text) < 3:
        return 'unknown'

    text = str(text)[:200]  # Берём только начало для скорости
    cyrillic = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    latin = sum(1 for c in text if 'a' <= c.lower() <= 'z')

    if cyrillic > latin:
        return 'ru'
    elif latin > cyrillic:
        return 'en'
    return 'ru'  # По умолчанию русский


def fast_sentiment_score(text: str) -> float:
    """
    Быстрый расчёт sentiment score [0, 1]
    Использует regex для поиска ключевых слов
    """
    if not text or not isinstance(text, str) or len(str(text).strip()) == 0:
        return 0.5

    text = str(text).lower()

    # Считаем совпадения
    pos_count = len(POSITIVE_PATTERN.findall(text))
    neg_count = len(NEGATIVE_PATTERN.findall(text))

    if pos_count == 0 and neg_count == 0:
        return 0.5

    # Score от 0 до 1
    total = pos_count + neg_count
    score = (pos_count - neg_count) / total / 2 + 0.5

    return max(0.0, min(1.0, score))


def fast_cons_negativity(text: str) -> float:
    """Оценка негативности в минусах (0 = нет негатива, 1 = много)"""
    if not text or not isinstance(text, str) or len(str(text).strip()) == 0:
        return 0.0

    text = str(text).lower()
    neg_count = len(NEGATIVE_PATTERN.findall(text))

    # Также учитываем длину текста - длинные минусы = больше проблем
    length_factor = min(len(text) / 200, 1.0) * 0.3
    keyword_factor = min(neg_count / 3, 1.0) * 0.7

    return min(length_factor + keyword_factor, 1.0)


class SentimentAnalyzer:
    """
    Оптимизированный анализатор тональности
    Использует векторизованные pandas операции
    """

    def __init__(self):
        print("SentimentAnalyzer инициализирован (оптимизированная версия)")

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Быстрый анализ всего датафрейма через векторизацию
        ~95K записей за 30-60 секунд
        """
        df = df.copy()
        total = len(df)
        print(f"Анализ тональности {total:,} отзывов (оптимизированный)...")

        # Векторизованный sentiment для pros
        if 'pros' in df.columns:
            print("  - Анализ Pros...")
            df['pros_sentiment'] = df['pros'].fillna('').apply(fast_sentiment_score)
        else:
            df['pros_sentiment'] = 0.5

        # Векторизованный sentiment для cons (инвертированный)
        if 'cons' in df.columns:
            print("  - Анализ Cons...")
            df['cons_negativity'] = df['cons'].fillna('').apply(fast_cons_negativity)
            df['cons_sentiment'] = 1 - df['cons_negativity']
        else:
            df['cons_sentiment'] = 1.0
            df['cons_negativity'] = 0.0

        # Векторизованный sentiment для comment
        if 'comment' in df.columns:
            print("  - Анализ Comments...")
            df['comment_sentiment'] = df['comment'].fillna('').apply(fast_sentiment_score)
        else:
            df['comment_sentiment'] = 0.5

        # Комбинированный sentiment (векторизованно)
        print("  - Расчёт combined sentiment...")
        df['combined_sentiment'] = (
            0.30 * df['pros_sentiment'] +
            0.50 * df['cons_sentiment'] +
            0.20 * df['comment_sentiment']
        )

        # Определение языка (по первому непустому полю)
        print("  - Определение языка...")
        df['detected_language'] = 'ru'  # По умолчанию

        # Определяем язык только для сэмпла (для скорости)
        sample_size = min(1000, len(df))
        sample_idx = df.sample(sample_size).index

        for idx in sample_idx:
            for col in ['pros', 'comment', 'cons']:
                if col in df.columns:
                    text = df.loc[idx, col]
                    if text and str(text).strip():
                        df.loc[idx, 'detected_language'] = fast_detect_language(str(text))
                        break

        print(f"[OK] Sentiment analysis завершён!")
        return df

    def get_sentiment_score(self, text: str) -> float:
        """Для совместимости со старым API"""
        return fast_sentiment_score(text)

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Для совместимости со старым API"""
        score = fast_sentiment_score(text)
        return {
            'positive': max(0, score - 0.5) * 2,
            'negative': max(0, 0.5 - score) * 2,
            'neutral': 1 - abs(score - 0.5) * 2,
            'language': fast_detect_language(text)
        }


if __name__ == "__main__":
    # Тест скорости
    import time

    analyzer = SentimentAnalyzer()

    test_texts = [
        "Отличный продукт, очень понравился!",
        "Ужасное качество, не рекомендую никому",
        "Amazing product, highly recommend!",
        "Terrible quality, waste of money",
        "Нормально, ничего особенного",
    ]

    print("\nТест анализатора:")
    for text in test_texts:
        score = fast_sentiment_score(text)
        lang = fast_detect_language(text)
        print(f"[{lang}] {text[:35]:35} -> {score:.2f}")

    # Тест скорости на большом объёме
    print("\n\nТест скорости:")
    large_list = test_texts * 20000  # 100K текстов

    start = time.time()
    scores = [fast_sentiment_score(t) for t in large_list]
    elapsed = time.time() - start

    print(f"100K текстов обработано за {elapsed:.2f} сек")
