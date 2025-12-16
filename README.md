# Golden Apple Loyalty Analysis

Анализ лояльности клиентов по отзывам на продукцию Darling в магазине "Золотое Яблоко".

## Цель проекта

Построить **Loyalty Score** — proxy-метрику вероятности повторной покупки на основе данных отзывов (без прямых данных о повторных покупках).

## Формула Loyalty Score

```python
Loyalty_Score = 0.30 * IsRecommended +
                0.25 * Stars_normalized +
                0.35 * Sentiment_score +
                0.10 * Engagement
```

| Компонент | Вес | Описание |
|-----------|-----|----------|
| IsRecommended | 30% | Готовность рекомендовать товар |
| Sentiment | 35% | Тональность текста (Pros/Cons/Comment) |
| Stars | 25% | Оценка в звёздах (1-5) |
| Engagement | 10% | Вовлечённость (полнота отзыва) |

### Сегментация

- **Loyal** (>0.7) — лояльные клиенты
- **Neutral** (0.4-0.7) — нейтральные
- **At-risk** (<0.4) — под риском оттока

## Структура проекта

```
├── app.py                    # Streamlit дашборд
├── main.py                   # CLI для анализа
├── data/
│   └── data_darling.xlsx     # Данные (не в репозитории)
├── src/
│   ├── data_loader.py        # Загрузка данных
│   ├── preprocessing.py      # Очистка текста
│   ├── sentiment_analyzer.py # Анализ тональности (RU/EN)
│   └── loyalty_scorer.py     # Расчёт Loyalty Score
└── notebooks/
    └── 01_eda.ipynb          # Exploratory Data Analysis
```

## Установка

```bash
# Клонировать репозиторий
git clone https://github.com/TorAdin/Golden-apple-loyalty.git
cd Golden-apple-loyalty

# Установить зависимости
pip install -r requirements.txt
```

## Данные

Поместите файл `data_darling.xlsx` в папку `data/`.

**Ожидаемые колонки:**
- `Pros` — плюсы товара
- `Cons` — минусы товара
- `Comment` — комментарий
- `IsRecommended` — рекомендует ли (0/1)
- `Stars` — оценка (1-5)
- `CatalogName` — название товара
- `ProductType` — категория
- `CreatedDate` — дата отзыва

## Запуск

### Streamlit дашборд (интерактивный)

```bash
streamlit run app.py
```

Откроется в браузере: http://localhost:8501

### CLI

```bash
# Полный анализ
python main.py

# Только EDA
python main.py --eda-only

# Быстрый режим (без sentiment)
python main.py --skip-sentiment
```

## Технологии

- **Python 3.10+**
- **pandas** — работа с данными
- **Streamlit** — интерактивный дашборд
- **Plotly** — визуализации
- **TextBlob** — sentiment analysis (EN)
- **Keyword-based** — sentiment analysis (RU)

## Sentiment Analysis

Поддерживает **русский** и **английский** языки:

- Автоматическое определение языка
- TextBlob для английского
- Keyword-based анализ для русского
- Оптимизированная обработка (~100K отзывов за минуту)

## Результаты

После анализа в папке `data/` появятся:
- `results_loyalty.csv` — полные результаты
- `loyalty_summary.csv` — сводка по сегментам

## Лицензия

MIT
