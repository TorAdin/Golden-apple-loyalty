"""
Модуль предобработки текстовых данных отзывов
"""

import re
import pandas as pd
from typing import Optional


def preprocess_text(text: str) -> str:
    """
    Очистка и нормализация текста отзыва

    Args:
        text: Исходный текст

    Returns:
        Очищенный текст
    """
    if not isinstance(text, str) or text in ('', 'nan', 'None'):
        return ''

    # Приводим к нижнему регистру
    text = text.lower()

    # Удаляем URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Нормализуем повторяющиеся символы (ооооочень -> очень)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Удаляем спецсимволы, оставляем буквы, цифры, пробелы и знаки препинания
    text = re.sub(r'[^\w\s.,!?;:\-()а-яёa-z]', ' ', text)

    # Нормализуем пробелы
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очистка всего датафрейма

    Args:
        df: Исходный DataFrame

    Returns:
        Очищенный DataFrame с дополнительными колонками
    """
    df = df.copy()

    # Очищаем текстовые поля
    text_columns = ['pros', 'cons', 'comment']

    for col in text_columns:
        if col in df.columns:
            # Сохраняем оригинал
            df[f'{col}_original'] = df[col]
            # Очищаем
            df[col] = df[col].apply(preprocess_text)

    # Добавляем вспомогательные признаки
    df = _add_text_features(df)

    return df


def _add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавление признаков на основе текста"""

    # Длина текстов
    if 'pros' in df.columns:
        df['pros_length'] = df['pros'].str.len().fillna(0).astype(int)
        df['has_pros'] = (df['pros_length'] > 0).astype(int)

    if 'cons' in df.columns:
        df['cons_length'] = df['cons'].str.len().fillna(0).astype(int)
        df['has_cons'] = (df['cons_length'] > 0).astype(int)

    if 'comment' in df.columns:
        df['comment_length'] = df['comment'].str.len().fillna(0).astype(int)
        df['has_comment'] = (df['comment_length'] > 0).astype(int)

    # Общая длина текста
    length_cols = [c for c in ['pros_length', 'cons_length', 'comment_length'] if c in df.columns]
    if length_cols:
        df['total_text_length'] = df[length_cols].sum(axis=1)

    # Количество заполненных полей
    has_cols = [c for c in ['has_pros', 'has_cons', 'has_comment'] if c in df.columns]
    if has_cols:
        df['fields_filled'] = df[has_cols].sum(axis=1)

    return df


def extract_keywords(texts: pd.Series, top_n: int = 20) -> dict:
    """
    Извлечение частотных ключевых слов из серии текстов

    Args:
        texts: Серия текстов
        top_n: Количество топ слов

    Returns:
        Словарь {слово: частота}
    """
    from collections import Counter

    # Русские стоп-слова (базовый список)
    stop_words = {
        'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как',
        'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к',
        'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'её', 'ее', 'мне',
        'было', 'вот', 'от', 'меня', 'еще', 'ещё', 'нет', 'о', 'из',
        'ему', 'теперь', 'когда', 'уже', 'для', 'это', 'этот', 'эта',
        'очень', 'нравится', 'хорошо', 'хороший', 'отличный', 'отлично',
        'просто', 'класс', 'супер', 'товар', 'продукт', 'средство'
    }

    all_words = []
    for text in texts.dropna():
        if isinstance(text, str) and text:
            words = text.lower().split()
            words = [w for w in words if len(w) > 2 and w not in stop_words]
            all_words.extend(words)

    word_counts = Counter(all_words)
    return dict(word_counts.most_common(top_n))


def get_text_statistics(df: pd.DataFrame) -> dict:
    """
    Получение статистики по текстовым полям

    Args:
        df: DataFrame с данными

    Returns:
        Словарь со статистикой
    """
    stats = {}

    for col in ['pros', 'cons', 'comment']:
        if col in df.columns:
            lengths = df[col].str.len().fillna(0)
            stats[col] = {
                'mean_length': lengths.mean(),
                'median_length': lengths.median(),
                'max_length': lengths.max(),
                'empty_percent': (lengths == 0).sum() / len(df) * 100,
                'non_empty_count': (lengths > 0).sum()
            }

    return stats
