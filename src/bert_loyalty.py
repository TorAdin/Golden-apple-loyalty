"""
BERT-based Loyalty Analyzer
Модуль для анализа лояльности с использованием обученной BERT модели
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class BertLoyaltyAnalyzer:
    """
    Анализатор лояльности на основе BERT модели
    Использует модель, обученную другом на 600 размеченных отзывах
    """

    def __init__(self,
                 model_path: str = "Golden-apple-loyalty/models_binary_fixed_v2",
                 threshold: float = 0.718):
        """
        Инициализация анализатора

        Args:
            model_path: Путь к обученной BERT модели
            threshold: Порог для классификации (0.718 = высокая точность)
        """
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Разные пороги для разных режимов анализа
        self.threshold_high = 0.718    # Высокая точность
        self.threshold_medium = 0.55   # Средняя
        self.threshold_low = 0.40      # Низкая (для исследований)

        self.model = None
        self.tokenizer = None
        self._model_loaded = False

    def load_model(self) -> bool:
        """
        Загрузка BERT модели

        Returns:
            True если модель загружена успешно
        """
        try:
            print(f"Загрузка BERT модели из {self.model_path}...")

            if not self.model_path.exists():
                print(f"[ERROR] Модель не найдена: {self.model_path}")
                return False

            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
            self.model.to(self.device)
            self.model.eval()

            print(f"[OK] BERT модель загружена на {self.device}")
            self._model_loaded = True
            return True

        except Exception as e:
            print(f"[ERROR] Ошибка загрузки модели: {e}")
            return False

    def is_available(self) -> bool:
        """Проверка доступности модели"""
        return self._model_loaded or self.model_path.exists()

    def _prepare_text(self, row: pd.Series) -> str:
        """
        Подготовка текста для анализа (объединение pros, cons, comment)

        Args:
            row: Строка DataFrame

        Returns:
            Объединённый текст
        """
        text_parts = []

        for field in ['pros', 'cons', 'comment']:
            if field in row and pd.notna(row[field]):
                value = str(row[field]).strip()
                if value:
                    text_parts.append(value)

        text = ' '.join(text_parts)

        # Если текст пустой, используем заглушку
        if not text or len(text) < 10:
            return "нет текста"

        return text

    def predict_batch(self,
                     texts: List[str],
                     batch_size: int = 32) -> np.ndarray:
        """
        Предсказание вероятностей лояльности для батча текстов

        Args:
            texts: Список текстов для анализа
            batch_size: Размер батча

        Returns:
            Массив вероятностей [0, 1]
        """
        if not self._model_loaded:
            raise RuntimeError("Модель не загружена. Вызовите load_model() сначала")

        all_probs = []

        for i in tqdm(range(0, len(texts), batch_size),
                     desc="BERT анализ",
                     unit="batch"):
            batch_texts = texts[i:i+batch_size]

            # Токенизация
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors='pt'
            ).to(self.device)

            # Предсказание
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

                # Обработка выхода модели (бинарная классификация)
                if logits.shape[-1] == 1:
                    # Один выход - используем sigmoid
                    probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                elif logits.shape[-1] == 2:
                    # Два выхода - берём вероятность класса 1 (лояльный)
                    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                else:
                    # Fallback
                    probs = torch.softmax(logits, dim=-1).max(dim=-1)[0].cpu().numpy()

            all_probs.extend(probs)

        return np.array(all_probs)

    def analyze_dataframe(self,
                         df: pd.DataFrame,
                         batch_size: int = 32) -> pd.DataFrame:
        """
        Анализ лояльности для всего DataFrame

        Args:
            df: DataFrame с отзывами
            batch_size: Размер батча для обработки

        Returns:
            DataFrame с добавленными колонками:
            - bert_loyalty_prob: вероятность лояльности [0, 1]
            - bert_loyalty_high: лояльный по строгому порогу (0.718)
            - bert_loyalty_medium: лояльный по среднему порогу (0.55)
            - bert_loyalty_low: лояльный по мягкому порогу (0.40)
        """
        if not self._model_loaded:
            if not self.load_model():
                print("⚠️ Не удалось загрузить BERT модель")
                return df

        df = df.copy()

        print(f"Анализ лояльности BERT для {len(df):,} отзывов...")

        # Подготовка текстов
        texts = df.apply(self._prepare_text, axis=1).tolist()

        # Предсказание
        probabilities = self.predict_batch(texts, batch_size=batch_size)

        # Добавление результатов в DataFrame
        df['bert_loyalty_prob'] = probabilities
        df['bert_loyalty_high'] = probabilities >= self.threshold_high
        df['bert_loyalty_medium'] = probabilities >= self.threshold_medium
        df['bert_loyalty_low'] = probabilities >= self.threshold_low

        # Категория по умолчанию (средний порог)
        df['bert_loyalty_class'] = df['bert_loyalty_high'].map({
            True: 'loyal',
            False: 'not_loyal'
        })

        print(f"[OK] BERT анализ завершён")
        print(f"   Лояльных (строгий порог >={self.threshold_high}): "
              f"{df['bert_loyalty_high'].sum():,} ({df['bert_loyalty_high'].mean()*100:.1f}%)")
        print(f"   Лояльных (средний порог >={self.threshold_medium}): "
              f"{df['bert_loyalty_medium'].sum():,} ({df['bert_loyalty_medium'].mean()*100:.1f}%)")

        return df

    def calculate_product_stats(self,
                                df: pd.DataFrame,
                                min_reviews: int = 100) -> pd.DataFrame:
        """
        Расчёт статистики лояльности по продуктам (как у друга)

        Args:
            df: DataFrame с результатами BERT анализа
            min_reviews: Минимальное количество отзывов для анализа продукта

        Returns:
            DataFrame со статистикой по продуктам
        """
        if 'bert_loyalty_prob' not in df.columns:
            raise ValueError("Сначала вызовите analyze_dataframe()")

        if 'product_name' not in df.columns:
            print("⚠️ Колонка product_name не найдена")
            return pd.DataFrame()

        print(f"\n[INFO] Расчёт статистики по продуктам (минимум {min_reviews} отзывов)...")

        # Группировка по продуктам
        product_stats = df.groupby('product_name').agg({
            'bert_loyalty_prob': ['count', 'mean', 'std'],
            'bert_loyalty_high': 'sum',
            'bert_loyalty_medium': 'sum',
            'bert_loyalty_low': 'sum'
        })

        # Flatten column names
        product_stats.columns = [
            'total_reviews',
            'avg_bert_prob',
            'std_bert_prob',
            'loyal_high',
            'loyal_medium',
            'loyal_low'
        ]

        product_stats = product_stats.reset_index()

        # Фильтрация по минимальному количеству отзывов
        product_stats = product_stats[product_stats['total_reviews'] >= min_reviews].copy()

        # Расчёт относительных показателей
        product_stats['loyalty_rate_high'] = product_stats['loyal_high'] / product_stats['total_reviews']
        product_stats['loyalty_rate_medium'] = product_stats['loyal_medium'] / product_stats['total_reviews']
        product_stats['loyalty_rate_low'] = product_stats['loyal_low'] / product_stats['total_reviews']

        # Z-score (относительная оценка как у друга)
        global_loyalty_rate = product_stats['loyal_high'].sum() / product_stats['total_reviews'].sum()

        product_stats['z_score'] = (
            (product_stats['loyalty_rate_high'] - global_loyalty_rate) /
            np.sqrt(global_loyalty_rate * (1 - global_loyalty_rate) / product_stats['total_reviews'])
        )

        # Байесовский балл (как на IMDb)
        C = global_loyalty_rate
        m = product_stats['total_reviews'].median()

        product_stats['bayesian_score'] = (
            (product_stats['total_reviews'] / (product_stats['total_reviews'] + m)) * product_stats['loyalty_rate_high'] +
            (m / (product_stats['total_reviews'] + m)) * C
        )

        # Процентиль
        product_stats['percentile_rank'] = product_stats['loyalty_rate_high'].rank(pct=True) * 100

        # Категоризация (как у друга)
        def categorize_z_score(z):
            if z >= 2:
                return "🚀 Выдающийся"
            elif z >= 1:
                return "📈 Выше среднего"
            elif z >= -1:
                return "📊 Средний"
            elif z >= -2:
                return "⚠️ Ниже среднего"
            else:
                return "🔥 Проблемный"

        product_stats['relative_category'] = product_stats['z_score'].apply(categorize_z_score)

        # Сортировка по z-score
        product_stats = product_stats.sort_values('z_score', ascending=False)

        print(f"[OK] Проанализировано {len(product_stats)} продуктов")
        print(f"   Средняя лояльность: {global_loyalty_rate:.1%}")
        print(f"   Лучший продукт (z-score): {product_stats.iloc[0]['z_score']:.2f}")

        return product_stats

    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Получение статистики по BERT анализу

        Args:
            df: DataFrame с результатами анализа

        Returns:
            Словарь со статистикой
        """
        if 'bert_loyalty_prob' not in df.columns:
            return {}

        stats = {
            'total_reviews': len(df),
            'avg_probability': df['bert_loyalty_prob'].mean(),
            'median_probability': df['bert_loyalty_prob'].median(),
            'std_probability': df['bert_loyalty_prob'].std(),
            'loyal_high': {
                'count': int(df['bert_loyalty_high'].sum()),
                'percent': df['bert_loyalty_high'].mean() * 100
            },
            'loyal_medium': {
                'count': int(df['bert_loyalty_medium'].sum()),
                'percent': df['bert_loyalty_medium'].mean() * 100
            },
            'loyal_low': {
                'count': int(df['bert_loyalty_low'].sum()),
                'percent': df['bert_loyalty_low'].mean() * 100
            }
        }

        return stats


def compare_with_keyword_method(df: pd.DataFrame) -> pd.DataFrame:
    """
    Сравнение BERT анализа с keyword-based методом

    Args:
        df: DataFrame с обоими типами анализа

    Returns:
        DataFrame со статистикой сравнения
    """
    required_cols = ['bert_loyalty_prob', 'loyalty_score']

    if not all(col in df.columns for col in required_cols):
        print("⚠️ Отсутствуют необходимые колонки для сравнения")
        return pd.DataFrame()

    comparison = pd.DataFrame({
        'metric': ['Средняя вероятность/score', 'Медиана', 'Стд. отклонение'],
        'BERT': [
            df['bert_loyalty_prob'].mean(),
            df['bert_loyalty_prob'].median(),
            df['bert_loyalty_prob'].std()
        ],
        'Keyword': [
            df['loyalty_score'].mean(),
            df['loyalty_score'].median(),
            df['loyalty_score'].std()
        ]
    })

    # Корреляция
    corr = df['bert_loyalty_prob'].corr(df['loyalty_score'])

    print("\n📊 Сравнение методов:")
    print(comparison.to_string(index=False))
    print(f"\nКорреляция между методами: {corr:.3f}")

    return comparison


if __name__ == "__main__":
    # Тест
    print("BERT Loyalty Analyzer - Test")
    print("=" * 60)

    analyzer = BertLoyaltyAnalyzer()

    if analyzer.is_available():
        print("[OK] BERT модель доступна")

        # Создаём тестовые данные
        test_df = pd.DataFrame({
            'pros': [
                'Отличный продукт, куплю ещё!',
                'Нормально',
                'Ужасное качество'
            ],
            'cons': ['', 'Дороговато', 'Всё плохо'],
            'comment': [
                'Рекомендую всем, буду покупать снова',
                'Обычный товар',
                'Больше не куплю'
            ],
            'product_name': ['Продукт А', 'Продукт Б', 'Продукт В']
        })

        print(f"\nТестируем на {len(test_df)} отзывах...")

        result = analyzer.analyze_dataframe(test_df)

        print("\nРезультаты:")
        print(result[['product_name', 'bert_loyalty_prob', 'bert_loyalty_class']])

        stats = analyzer.get_statistics(result)
        print(f"\nСтатистика:")
        print(f"  Средняя вероятность: {stats['avg_probability']:.3f}")
        print(f"  Лояльных (строгий порог): {stats['loyal_high']['count']} ({stats['loyal_high']['percent']:.1f}%)")
    else:
        print("[ERROR] BERT модель недоступна")
        print(f"   Ожидаемый путь: Golden-apple-loyalty/models_binary_fixed_v2")
