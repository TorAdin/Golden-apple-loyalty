"""
Модуль загрузки и валидации данных отзывов Darling
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any


def load_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Загрузка данных из xlsx файла

    Args:
        file_path: Путь к файлу. По умолчанию ищет в data/data_darling.xlsx

    Returns:
        DataFrame с отзывами
    """
    if file_path is None:
        # Ищем файл относительно корня проекта
        project_root = Path(__file__).parent.parent
        file_path = project_root / "data" / "data_darling.xlsx"

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    # Загружаем данные
    df = pd.read_excel(file_path, engine='openpyxl')

    # Проверяем и стандартизируем колонки
    df = _standardize_columns(df)

    # Преобразуем типы данных
    df = _convert_dtypes(df)

    return df


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Стандартизация названий колонок"""
    expected_columns = {
        'Pros': 'pros',
        'Cons': 'cons',
        'Comment': 'comment',
        'IsRecommended': 'is_recommended',
        'Stars': 'stars',
        'CatalogName': 'product_name',
        'ProductType': 'product_type',
        'CreatedDate': 'created_date'
    }

    # Переименовываем колонки если они совпадают
    rename_map = {}
    for old_name, new_name in expected_columns.items():
        if old_name in df.columns:
            rename_map[old_name] = new_name
        elif old_name.lower() in [c.lower() for c in df.columns]:
            # Case-insensitive match
            for col in df.columns:
                if col.lower() == old_name.lower():
                    rename_map[col] = new_name
                    break

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def _convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Преобразование типов данных"""
    # Текстовые колонки
    text_cols = ['pros', 'cons', 'comment', 'product_name', 'product_type']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('nan', '')

    # Булевые/числовые колонки
    if 'is_recommended' in df.columns:
        df['is_recommended'] = pd.to_numeric(df['is_recommended'], errors='coerce').fillna(0).astype(int)

    if 'stars' in df.columns:
        df['stars'] = pd.to_numeric(df['stars'], errors='coerce')

    # Дата
    if 'created_date' in df.columns:
        df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')

    return df


def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Получение сводной информации о датасете

    Args:
        df: DataFrame с данными

    Returns:
        Словарь с информацией о данных
    """
    info = {
        'total_reviews': len(df),
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percent': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
    }

    # Статистика по числовым колонкам
    if 'stars' in df.columns:
        info['stars_distribution'] = df['stars'].value_counts().sort_index().to_dict()
        info['stars_mean'] = df['stars'].mean()

    if 'is_recommended' in df.columns:
        info['recommendation_rate'] = df['is_recommended'].mean() * 100

    # Статистика по категориям
    if 'product_type' in df.columns:
        info['product_types_count'] = df['product_type'].nunique()
        info['top_product_types'] = df['product_type'].value_counts().head(10).to_dict()

    if 'product_name' in df.columns:
        info['products_count'] = df['product_name'].nunique()

    # Временной диапазон
    if 'created_date' in df.columns:
        valid_dates = df['created_date'].dropna()
        if len(valid_dates) > 0:
            info['date_range'] = {
                'min': valid_dates.min().isoformat() if pd.notna(valid_dates.min()) else None,
                'max': valid_dates.max().isoformat() if pd.notna(valid_dates.max()) else None
            }

    return info


def print_data_summary(df: pd.DataFrame) -> None:
    """Вывод сводки по данным в консоль"""
    info = get_data_info(df)

    print("=" * 50)
    print("СВОДКА ПО ДАТАСЕТУ DARLING")
    print("=" * 50)
    print(f"\nВсего отзывов: {info['total_reviews']:,}")
    print(f"Колонок: {len(info['columns'])}")

    print("\n--- Пропуски ---")
    for col, pct in info['missing_percent'].items():
        if pct > 0:
            print(f"  {col}: {pct:.1f}%")

    if 'stars_mean' in info:
        print(f"\n--- Оценки ---")
        print(f"  Средняя оценка: {info['stars_mean']:.2f}")
        print(f"  Распределение: {info['stars_distribution']}")

    if 'recommendation_rate' in info:
        print(f"\n--- Рекомендации ---")
        print(f"  Рекомендуют: {info['recommendation_rate']:.1f}%")

    if 'product_types_count' in info:
        print(f"\n--- Продукты ---")
        print(f"  Категорий: {info['product_types_count']}")
        print(f"  Продуктов: {info.get('products_count', 'N/A')}")

    if 'date_range' in info:
        print(f"\n--- Период ---")
        print(f"  От: {info['date_range']['min']}")
        print(f"  До: {info['date_range']['max']}")

    print("=" * 50)


if __name__ == "__main__":
    # Тестовый запуск
    df = load_data()
    print_data_summary(df)
