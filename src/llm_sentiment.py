"""
LLM-based Sentiment Analysis
Использует OpenAI API для более точного анализа тональности

Требует: pip install openai
Требует: OPENAI_API_KEY в переменных окружения или .env файле
"""

import os
import time
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class LLMSentimentResult:
    """Результат LLM анализа"""
    sentiment_score: float  # 0-1, где 1 = очень позитивный
    sentiment_label: str    # positive, neutral, negative
    confidence: float       # 0-1
    reasoning: str          # краткое объяснение


class LLMSentimentAnalyzer:
    """
    Анализатор тональности на основе LLM (OpenAI GPT)

    Особенности:
    - Более точный анализ, чем keyword-based
    - Понимает контекст и сарказм
    - Работает с русским языком нативно
    - Batch-обработка для экономии токенов
    """

    SYSTEM_PROMPT = """Ты эксперт по анализу тональности отзывов на косметику.
Твоя задача - оценить тональность отзыва клиента.

Оцени каждый отзыв по шкале от 0 до 1:
- 0.0-0.3: негативный (недоволен, жалуется, не рекомендует)
- 0.3-0.7: нейтральный (смешанные эмоции, есть плюсы и минусы)
- 0.7-1.0: позитивный (доволен, хвалит, рекомендует)

Учитывай:
- Общий тон текста
- Сарказм и иронию
- Скрытое недовольство
- Намерение купить повторно"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Инициализация анализатора

        Args:
            api_key: OpenAI API ключ (или из переменной OPENAI_API_KEY)
            model: Модель для анализа (gpt-4o-mini рекомендуется для баланса цена/качество)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None

        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                print(f"LLM Sentiment Analyzer инициализирован (модель: {model})")
            except ImportError:
                print("Ошибка: установите openai (pip install openai)")
        else:
            print("Предупреждение: OPENAI_API_KEY не найден. LLM анализ недоступен.")

    def is_available(self) -> bool:
        """Проверка доступности LLM"""
        return self.client is not None

    def analyze_single(self, pros: str, cons: str, comment: str) -> LLMSentimentResult:
        """
        Анализ одного отзыва

        Args:
            pros: Плюсы товара
            cons: Минусы товара
            comment: Комментарий

        Returns:
            LLMSentimentResult с оценкой
        """
        if not self.is_available():
            return LLMSentimentResult(0.5, "neutral", 0.0, "LLM недоступен")

        # Формируем текст отзыва
        review_parts = []
        if pros and str(pros).strip():
            review_parts.append(f"Плюсы: {pros}")
        if cons and str(cons).strip():
            review_parts.append(f"Минусы: {cons}")
        if comment and str(comment).strip():
            review_parts.append(f"Комментарий: {comment}")

        if not review_parts:
            return LLMSentimentResult(0.5, "neutral", 0.0, "Пустой отзыв")

        review_text = "\n".join(review_parts)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Оцени тональность этого отзыва:\n\n{review_text}\n\nВерни JSON: {{\"score\": 0.0-1.0, \"label\": \"positive/neutral/negative\", \"confidence\": 0.0-1.0, \"reason\": \"краткое объяснение\"}}"}
                ],
                temperature=0.1,
                max_tokens=150,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            return LLMSentimentResult(
                sentiment_score=float(result.get("score", 0.5)),
                sentiment_label=result.get("label", "neutral"),
                confidence=float(result.get("confidence", 0.5)),
                reasoning=result.get("reason", "")
            )

        except Exception as e:
            print(f"Ошибка LLM: {e}")
            return LLMSentimentResult(0.5, "neutral", 0.0, f"Ошибка: {str(e)[:50]}")

    def analyze_batch(self, reviews: List[Dict], batch_size: int = 10) -> List[LLMSentimentResult]:
        """
        Batch-анализ нескольких отзывов

        Args:
            reviews: Список словарей с ключами pros, cons, comment
            batch_size: Размер батча для одного запроса

        Returns:
            Список LLMSentimentResult
        """
        if not self.is_available():
            return [LLMSentimentResult(0.5, "neutral", 0.0, "LLM недоступен") for _ in reviews]

        results = []
        total_batches = (len(reviews) + batch_size - 1) // batch_size

        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i + batch_size]
            batch_num = i // batch_size + 1
            print(f"  Обработка батча {batch_num}/{total_batches}...")

            # Формируем текст для батча
            batch_text = ""
            for idx, review in enumerate(batch, 1):
                parts = []
                if review.get('pros'):
                    parts.append(f"Плюсы: {review['pros']}")
                if review.get('cons'):
                    parts.append(f"Минусы: {review['cons']}")
                if review.get('comment'):
                    parts.append(f"Комментарий: {review['comment']}")

                review_text = "\n".join(parts) if parts else "(пусто)"
                batch_text += f"\n---\nОтзыв #{idx}:\n{review_text}\n"

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": f"Оцени тональность {len(batch)} отзывов. Для каждого верни JSON объект.\n{batch_text}\n\nВерни JSON массив: [{{\"id\": 1, \"score\": 0.0-1.0, \"label\": \"positive/neutral/negative\"}}]"}
                    ],
                    temperature=0.1,
                    max_tokens=50 * len(batch),
                    response_format={"type": "json_object"}
                )

                batch_results = json.loads(response.choices[0].message.content)

                # Парсим результаты
                if isinstance(batch_results, dict) and 'reviews' in batch_results:
                    batch_results = batch_results['reviews']
                elif isinstance(batch_results, dict):
                    batch_results = [batch_results]

                for item in batch_results:
                    results.append(LLMSentimentResult(
                        sentiment_score=float(item.get("score", 0.5)),
                        sentiment_label=item.get("label", "neutral"),
                        confidence=0.8,
                        reasoning=""
                    ))

                # Добавляем недостающие результаты
                while len(results) < i + len(batch):
                    results.append(LLMSentimentResult(0.5, "neutral", 0.5, ""))

            except Exception as e:
                print(f"  Ошибка батча: {e}")
                for _ in batch:
                    results.append(LLMSentimentResult(0.5, "neutral", 0.0, f"Ошибка: {str(e)[:30]}"))

            # Rate limiting
            time.sleep(0.5)

        return results

    def analyze_dataframe(self, df: pd.DataFrame, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Анализ датафрейма с LLM

        Args:
            df: DataFrame с колонками pros, cons, comment
            sample_size: Если задано, анализируем только сэмпл (для экономии)

        Returns:
            DataFrame с добавленными колонками llm_sentiment_*
        """
        df = df.copy()

        if not self.is_available():
            print("LLM недоступен. Пропускаем анализ.")
            return df

        # Определяем размер выборки
        if sample_size and sample_size < len(df):
            print(f"LLM Sentiment: анализ сэмпла {sample_size:,} из {len(df):,} записей")
            indices = df.sample(sample_size).index
        else:
            print(f"LLM Sentiment: анализ всех {len(df):,} записей")
            indices = df.index

        # Подготавливаем данные
        reviews = []
        for idx in indices:
            reviews.append({
                'pros': df.loc[idx, 'pros'] if 'pros' in df.columns else '',
                'cons': df.loc[idx, 'cons'] if 'cons' in df.columns else '',
                'comment': df.loc[idx, 'comment'] if 'comment' in df.columns else ''
            })

        # Анализируем
        results = self.analyze_batch(reviews, batch_size=5)

        # Добавляем результаты
        df['llm_sentiment_score'] = pd.NA
        df['llm_sentiment_label'] = pd.NA

        for idx, result in zip(indices, results):
            df.loc[idx, 'llm_sentiment_score'] = result.sentiment_score
            df.loc[idx, 'llm_sentiment_label'] = result.sentiment_label

        # Статистика
        analyzed = df['llm_sentiment_score'].notna().sum()
        print(f"[OK] LLM анализ завершён для {analyzed:,} записей")

        if analyzed > 0:
            mean_score = df['llm_sentiment_score'].mean()
            print(f"  Средний LLM sentiment: {mean_score:.3f}")

        return df


def compare_sentiments(df: pd.DataFrame) -> Dict:
    """
    Сравнение keyword-based и LLM sentiment

    Args:
        df: DataFrame с обоими типами sentiment

    Returns:
        Словарь со статистикой сравнения
    """
    if 'combined_sentiment' not in df.columns or 'llm_sentiment_score' not in df.columns:
        return {}

    # Только записи с обоими значениями
    mask = df['llm_sentiment_score'].notna()
    if mask.sum() == 0:
        return {}

    comparison = {
        'records_compared': int(mask.sum()),
        'keyword_mean': round(df.loc[mask, 'combined_sentiment'].mean(), 3),
        'llm_mean': round(df.loc[mask, 'llm_sentiment_score'].mean(), 3),
        'correlation': round(df.loc[mask, 'combined_sentiment'].corr(df.loc[mask, 'llm_sentiment_score']), 3),
        'mean_difference': round(abs(df.loc[mask, 'combined_sentiment'] - df.loc[mask, 'llm_sentiment_score']).mean(), 3)
    }

    return comparison


if __name__ == "__main__":
    # Тест
    analyzer = LLMSentimentAnalyzer()

    if analyzer.is_available():
        result = analyzer.analyze_single(
            pros="Отличный крем, кожа стала мягкой",
            cons="Немного дорого",
            comment="Буду покупать ещё!"
        )
        print(f"\nРезультат: {result}")
    else:
        print("\nДля тестирования установите OPENAI_API_KEY")
        print("Пример: export OPENAI_API_KEY='your-key-here'")
