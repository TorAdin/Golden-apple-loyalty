"""
Модуль детекции кэтч-фраз - индикаторов повторной покупки
"""

import pandas as pd
import re
from typing import List, Tuple, Optional


# Паттерны, указывающие на намерение повторной покупки
REPEAT_PURCHASE_PATTERNS = [
    # Прямые указания на повторную покупку
    r'куплю\s*(ещё|еще|снова|опять|повторно)',
    r'закажу\s*(ещё|еще|снова|опять|повторно)',
    r'возьму\s*(ещё|еще|снова|опять|повторно)',
    r'буду\s*(брать|покупать|заказывать)',
    r'куплю\s+обязательно',
    r'закажу\s+обязательно',

    # Уже повторная покупка
    r'покупаю\s*(не\s*первый\s*раз|второй|третий|уже|постоянно|регулярно)',
    r'беру\s*(не\s*первый\s*раз|второй|третий|уже|постоянно|регулярно)',
    r'заказываю\s*(не\s*первый\s*раз|второй|третий|уже|постоянно|регулярно)',
    r'(второй|третий|четвёртый|пятый)\s*(раз|тюбик|флакон|баночк)',
    r'уже\s*(несколько|много)\s*(раз|штук)',
    r'постоянн(о|ый)\s*(покупаю|беру|заказываю|клиент)',

    # Лояльность к продукту
    r'мой\s+фаворит',
    r'моя\s+любовь',
    r'must\s*have',
    r'мастхэв',
    r'маст\s*хэв',
    r'не\s+изменяю',
    r'верн(усь|а|ый)\s+(к|этому|этой)',
    r'на\s+постоянк[у|е]',
    r'в\s+постоянном\s+использовании',

    # Рекомендации с сильной эмоцией
    r'всем\s+советую',
    r'всем\s+рекомендую',
    r'однозначно\s+рекомендую',
    r'горячо\s+рекомендую',
    r'настоятельно\s+рекомендую',

    # Запасы
    r'закупаю\s+(про\s*)?запас',
    r'взя(л|ла)\s+про\s*запас',
    r'куп(ил|ила)\s+несколько',
    r'беру\s+(сразу\s+)?несколько',

    # Привычка
    r'без\s+(него|неё|этого)\s+не\s+(могу|представляю)',
    r'подсе(л|ла)\s+на',
    r'привык(ла)?',
]

# Компилируем паттерны для скорости
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in REPEAT_PURCHASE_PATTERNS]


def detect_catch_phrases(text: str) -> Tuple[bool, List[str]]:
    """
    Обнаруживает кэтч-фразы в тексте

    Args:
        text: Текст для анализа

    Returns:
        Tuple (has_catch_phrase, list_of_matches)
    """
    if not text or not isinstance(text, str) or len(str(text).strip()) == 0:
        return False, []

    text = str(text).lower()
    matches = []

    for pattern in COMPILED_PATTERNS:
        found = pattern.findall(text)
        if found:
            # Находим полное совпадение в тексте
            match = pattern.search(text)
            if match:
                matches.append(match.group(0).strip())

    return len(matches) > 0, matches


def analyze_catch_phrases_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет колонки с кэтч-фразами для всего датафрейма

    Args:
        df: DataFrame с колонками pros, cons, comment

    Returns:
        DataFrame с добавленными колонками:
        - has_catch_phrase: bool
        - catch_phrases: list of found phrases
        - catch_phrase_count: int
    """
    df = df.copy()
    print(f"Анализ кэтч-фраз для {len(df):,} записей...")

    # Объединяем все текстовые поля
    def get_all_text(row):
        texts = []
        for col in ['pros', 'cons', 'comment']:
            if col in row and row[col] and pd.notna(row[col]):
                texts.append(str(row[col]))
        return ' '.join(texts)

    # Анализируем каждую строку
    results = []
    for idx, row in df.iterrows():
        all_text = get_all_text(row)
        has_phrase, phrases = detect_catch_phrases(all_text)
        results.append({
            'has_catch_phrase': has_phrase,
            'catch_phrases': phrases,
            'catch_phrase_count': len(phrases)
        })

    # Добавляем результаты
    results_df = pd.DataFrame(results)
    df['has_catch_phrase'] = results_df['has_catch_phrase'].values
    df['catch_phrases'] = results_df['catch_phrases'].values
    df['catch_phrase_count'] = results_df['catch_phrase_count'].values

    # Статистика
    total_with_phrases = df['has_catch_phrase'].sum()
    print(f"[OK] Найдено {total_with_phrases:,} отзывов с кэтч-фразами ({total_with_phrases/len(df)*100:.1f}%)")

    return df


def get_catch_phrase_summary(df: pd.DataFrame) -> dict:
    """
    Сводка по кэтч-фразам
    """
    if 'has_catch_phrase' not in df.columns:
        return {}

    summary = {
        'total_reviews': len(df),
        'reviews_with_phrases': int(df['has_catch_phrase'].sum()),
        'percent_with_phrases': round(df['has_catch_phrase'].mean() * 100, 2),
    }

    # Подсчёт популярных фраз
    all_phrases = []
    for phrases in df['catch_phrases']:
        if phrases:
            all_phrases.extend(phrases)

    if all_phrases:
        from collections import Counter
        phrase_counts = Counter(all_phrases)
        summary['top_phrases'] = dict(phrase_counts.most_common(10))

    # Корреляция с лояльностью
    if 'loyalty_score' in df.columns:
        with_phrase = df[df['has_catch_phrase'] == True]['loyalty_score'].mean()
        without_phrase = df[df['has_catch_phrase'] == False]['loyalty_score'].mean()
        summary['avg_loyalty_with_phrase'] = round(with_phrase, 3)
        summary['avg_loyalty_without_phrase'] = round(without_phrase, 3)

    return summary


if __name__ == "__main__":
    # Тест
    test_texts = [
        "Отличный крем, куплю ещё обязательно!",
        "Покупаю уже третий раз, очень нравится",
        "Мой фаворит, без него не могу",
        "Нормальный продукт, ничего особенного",
        "Буду заказывать постоянно, must have!",
        "Взяла про запас несколько штук",
        "Разочаровал, больше не куплю",
        "Подсела на этот шампунь, уже несколько раз покупаю",
    ]

    print("Тест детекции кэтч-фраз:\n")
    for text in test_texts:
        has_phrase, phrases = detect_catch_phrases(text)
        status = "[+]" if has_phrase else "[-]"
        print(f"[{status}] {text[:50]:50} -> {phrases if phrases else '-'}")
