"""
GoldenApple Loyalty Analysis
Главный скрипт для анализа лояльности клиентов Darling

Использование:
    python main.py                    # Полный пайплайн
    python main.py --eda-only         # Только EDA
    python main.py --skip-sentiment   # Пропустить sentiment (быстрее)
"""

import argparse
import sys
from pathlib import Path

# Добавляем src в path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_data, print_data_summary
from src.preprocessing import clean_dataframe, get_text_statistics
from src.sentiment_analyzer import SentimentAnalyzer
from src.loyalty_scorer import LoyaltyScorer, analyze_loyalty_drivers


def run_eda(df):
    """Exploratory Data Analysis"""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print_data_summary(df)

    # Статистика текстов
    print("\n--- Статистика текстовых полей ---")
    text_stats = get_text_statistics(df)
    for field, stats in text_stats.items():
        print(f"\n{field}:")
        print(f"  Средняя длина: {stats['mean_length']:.0f} символов")
        print(f"  Пустых: {stats['empty_percent']:.1f}%")


def run_sentiment_analysis(df, use_dostoevsky=True):
    """Анализ тональности"""
    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS")
    print("=" * 60)

    analyzer = SentimentAnalyzer(use_dostoevsky=use_dostoevsky)
    df = analyzer.analyze_dataframe(df)

    print("\nСтатистика sentiment:")
    print(f"  Средний combined_sentiment: {df['combined_sentiment'].mean():.3f}")
    print(f"  Медиана: {df['combined_sentiment'].median():.3f}")

    return df


def run_loyalty_scoring(df):
    """Расчёт Loyalty Score"""
    print("\n" + "=" * 60)
    print("LOYALTY SCORING")
    print("=" * 60)

    scorer = LoyaltyScorer()
    df = scorer.score_dataframe(df)

    # Отчёт
    scorer.print_loyalty_report(df)

    # Драйверы лояльности
    drivers = analyze_loyalty_drivers(df)
    print("\n--- Драйверы лояльности (корреляции) ---")
    for driver, corr in drivers.items():
        print(f"  {driver}: {corr:+.3f}")

    return df


def save_results(df, output_path):
    """Сохранение результатов"""
    # Сохраняем в разных форматах
    df.to_pickle(output_path / 'results_loyalty.pkl')
    df.to_csv(output_path / 'results_loyalty.csv', index=False)

    # Сохраняем сводку
    summary_cols = ['loyalty_score', 'loyalty_segment', 'combined_sentiment',
                    'stars', 'is_recommended', 'product_type', 'product_name']
    summary_cols = [c for c in summary_cols if c in df.columns]

    df[summary_cols].to_csv(output_path / 'loyalty_summary.csv', index=False)

    print(f"\nРезультаты сохранены в {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Анализ лояльности клиентов Darling')
    parser.add_argument('--data', type=str, default=None, help='Путь к файлу данных')
    parser.add_argument('--eda-only', action='store_true', help='Только EDA')
    parser.add_argument('--skip-sentiment', action='store_true', help='Пропустить sentiment analysis')
    parser.add_argument('--no-dostoevsky', action='store_true', help='Использовать keyword-based sentiment')
    parser.add_argument('--output', type=str, default='data', help='Папка для результатов')

    args = parser.parse_args()

    # Определяем пути
    project_root = Path(__file__).parent
    output_path = project_root / args.output

    print("=" * 60)
    print("GOLDENAPPLE LOYALTY ANALYSIS")
    print("Анализ лояльности клиентов Darling")
    print("=" * 60)

    # 1. Загрузка данных
    print("\n[1/4] Загрузка данных...")
    df = load_data(args.data)
    print(f"Загружено {len(df):,} отзывов")

    # 2. Очистка и EDA
    print("\n[2/4] Очистка и EDA...")
    df = clean_dataframe(df)
    run_eda(df)

    if args.eda_only:
        print("\n[!] Режим EDA-only. Завершение.")
        return

    # 3. Sentiment Analysis
    if not args.skip_sentiment:
        print("\n[3/4] Sentiment Analysis...")
        df = run_sentiment_analysis(df, use_dostoevsky=not args.no_dostoevsky)
    else:
        print("\n[3/4] Sentiment Analysis пропущен")
        # Используем дефолтные значения
        df['combined_sentiment'] = 0.5

    # 4. Loyalty Scoring
    print("\n[4/4] Loyalty Scoring...")
    df = run_loyalty_scoring(df)

    # Сохранение
    save_results(df, output_path)

    print("\n" + "=" * 60)
    print("АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 60)
    print(f"\nРезультаты: {output_path}/results_loyalty.csv")
    print(f"Используйте notebooks/01_eda.ipynb для визуализации")


if __name__ == "__main__":
    main()
