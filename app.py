"""
GoldenApple Loyalty Dashboard
Интерактивный дашборд для анализа лояльности клиентов Darling

Запуск: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Настройка страницы
st.set_page_config(
    page_title="Darling Loyalty Analysis",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Импорт модулей проекта
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_data, get_data_info
from src.preprocessing import clean_dataframe
from src.sentiment_analyzer import SentimentAnalyzer
from src.loyalty_scorer import LoyaltyScorer


@st.cache_data
def load_and_process_data(file_path=None):
    """Загрузка и обработка данных (с кэшированием)"""
    df = load_data(file_path)
    df = clean_dataframe(df)
    return df


@st.cache_data
def run_sentiment_analysis(_df):
    """Sentiment analysis (с кэшированием)"""
    analyzer = SentimentAnalyzer()
    return analyzer.analyze_dataframe(_df)


@st.cache_data
def calculate_loyalty(_df):
    """Расчёт Loyalty Score (с кэшированием)"""
    scorer = LoyaltyScorer()
    return scorer.score_dataframe(_df)


def main():
    # Заголовок
    st.title("🍎 Darling Loyalty Analysis")
    st.markdown("**Анализ лояльности клиентов по отзывам в Золотом Яблоке**")

    # Сайдбар
    st.sidebar.header("⚙️ Настройки")

    # Загрузка данных
    data_path = Path(__file__).parent / "data" / "data_darling.xlsx"

    if not data_path.exists():
        st.error(f"Файл данных не найден: {data_path}")
        st.info("Поместите data_darling.xlsx в папку data/")
        return

    # Загружаем данные
    with st.spinner("Загрузка данных..."):
        df = load_and_process_data(str(data_path))

    st.sidebar.success(f"✅ Загружено {len(df):,} отзывов")

    # Опции анализа
    run_sentiment = st.sidebar.checkbox("Запустить Sentiment Analysis", value=True)
    run_loyalty = st.sidebar.checkbox("Рассчитать Loyalty Score", value=True)

    # Обработка
    if run_sentiment:
        with st.spinner("Анализ тональности... (это может занять несколько минут)"):
            df = run_sentiment_analysis(df)

    if run_loyalty and 'combined_sentiment' in df.columns:
        with st.spinner("Расчёт Loyalty Score..."):
            df = calculate_loyalty(df)

    # Фильтры
    st.sidebar.header("🔍 Фильтры")

    # Фильтр по категории
    if 'product_type' in df.columns:
        categories = ['Все'] + sorted(df['product_type'].dropna().unique().tolist())
        selected_category = st.sidebar.selectbox("Категория товара", categories)
        if selected_category != 'Все':
            df = df[df['product_type'] == selected_category]

    # Фильтр по оценке
    if 'stars' in df.columns:
        star_range = st.sidebar.slider("Оценка (Stars)", 1, 5, (1, 5))
        df = df[(df['stars'] >= star_range[0]) & (df['stars'] <= star_range[1])]

    # Фильтр по лояльности
    if 'loyalty_segment' in df.columns:
        segments = st.sidebar.multiselect(
            "Сегмент лояльности",
            ['loyal', 'neutral', 'at_risk'],
            default=['loyal', 'neutral', 'at_risk']
        )
        df = df[df['loyalty_segment'].isin(segments)]

    st.sidebar.markdown(f"**Отфильтровано: {len(df):,} отзывов**")

    # === ОСНОВНОЙ КОНТЕНТ ===

    # Вкладки
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Обзор", "📈 Лояльность", "💬 Отзывы", "📋 Данные"])

    # === TAB 1: ОБЗОР ===
    with tab1:
        st.header("Общая статистика")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Всего отзывов", f"{len(df):,}")

        with col2:
            if 'stars' in df.columns:
                avg_stars = df['stars'].mean()
                st.metric("Средняя оценка", f"{avg_stars:.2f} ⭐")

        with col3:
            if 'is_recommended' in df.columns:
                rec_rate = df['is_recommended'].mean() * 100
                st.metric("Рекомендуют", f"{rec_rate:.1f}%")

        with col4:
            if 'loyalty_score' in df.columns:
                avg_loyalty = df['loyalty_score'].mean()
                st.metric("Avg Loyalty Score", f"{avg_loyalty:.3f}")

        # Графики
        col1, col2 = st.columns(2)

        with col1:
            if 'stars' in df.columns:
                fig = px.histogram(
                    df, x='stars',
                    title="Распределение оценок",
                    color_discrete_sequence=['#FFD700']
                )
                fig.update_layout(bargap=0.2)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if 'is_recommended' in df.columns:
                rec_data = df['is_recommended'].value_counts()
                fig = px.pie(
                    values=rec_data.values,
                    names=['Не рекомендуют', 'Рекомендуют'],
                    title="Рекомендации",
                    color_discrete_sequence=['#ff6b6b', '#51cf66']
                )
                st.plotly_chart(fig, use_container_width=True)

        # Топ категорий
        if 'product_type' in df.columns:
            st.subheader("Топ категорий по количеству отзывов")
            top_cats = df['product_type'].value_counts().head(10)
            fig = px.bar(
                x=top_cats.values,
                y=top_cats.index,
                orientation='h',
                title="",
                color=top_cats.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # === TAB 2: ЛОЯЛЬНОСТЬ ===
    with tab2:
        st.header("Анализ лояльности")

        if 'loyalty_score' not in df.columns:
            st.warning("Включите расчёт Loyalty Score в настройках")
        else:
            # Метрики по сегментам
            col1, col2, col3 = st.columns(3)

            segment_counts = df['loyalty_segment'].value_counts()

            with col1:
                loyal_pct = segment_counts.get('loyal', 0) / len(df) * 100
                st.metric("🟢 Loyal (>0.7)", f"{loyal_pct:.1f}%",
                         f"{segment_counts.get('loyal', 0):,} отзывов")

            with col2:
                neutral_pct = segment_counts.get('neutral', 0) / len(df) * 100
                st.metric("🟡 Neutral (0.4-0.7)", f"{neutral_pct:.1f}%",
                         f"{segment_counts.get('neutral', 0):,} отзывов")

            with col3:
                atrisk_pct = segment_counts.get('at_risk', 0) / len(df) * 100
                st.metric("🔴 At Risk (<0.4)", f"{atrisk_pct:.1f}%",
                         f"{segment_counts.get('at_risk', 0):,} отзывов")

            # Распределение Loyalty Score
            col1, col2 = st.columns(2)

            with col1:
                fig = px.histogram(
                    df, x='loyalty_score',
                    nbins=50,
                    title="Распределение Loyalty Score",
                    color_discrete_sequence=['#4ecdc4']
                )
                fig.add_vline(x=0.7, line_dash="dash", line_color="green", annotation_text="Loyal")
                fig.add_vline(x=0.4, line_dash="dash", line_color="red", annotation_text="At Risk")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Сегменты pie chart
                fig = px.pie(
                    values=segment_counts.values,
                    names=segment_counts.index,
                    title="Сегментация клиентов",
                    color=segment_counts.index,
                    color_discrete_map={'loyal': '#51cf66', 'neutral': '#ffd43b', 'at_risk': '#ff6b6b'}
                )
                st.plotly_chart(fig, use_container_width=True)

            # Scatter: Sentiment vs Stars
            if 'combined_sentiment' in df.columns and 'stars' in df.columns:
                st.subheader("Sentiment vs Stars")
                sample_df = df.sample(min(5000, len(df)))  # Ограничиваем для производительности
                fig = px.scatter(
                    sample_df,
                    x='stars',
                    y='combined_sentiment',
                    color='loyalty_segment',
                    opacity=0.5,
                    title="Соотношение оценок и тональности",
                    color_discrete_map={'loyal': '#51cf66', 'neutral': '#ffd43b', 'at_risk': '#ff6b6b'}
                )
                st.plotly_chart(fig, use_container_width=True)

            # Loyalty по категориям
            if 'product_type' in df.columns:
                st.subheader("Лояльность по категориям")
                cat_loyalty = df.groupby('product_type').agg({
                    'loyalty_score': 'mean',
                    'stars': 'count'
                }).rename(columns={'stars': 'count'})
                cat_loyalty = cat_loyalty[cat_loyalty['count'] >= 50].sort_values('loyalty_score', ascending=False)

                fig = px.bar(
                    cat_loyalty.head(15),
                    x=cat_loyalty.head(15).index,
                    y='loyalty_score',
                    title="Топ-15 категорий по Loyalty Score (мин. 50 отзывов)",
                    color='loyalty_score',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

    # === TAB 3: ОТЗЫВЫ ===
    with tab3:
        st.header("Примеры отзывов")

        # Фильтр по сегменту
        segment_filter = st.selectbox(
            "Показать отзывы сегмента:",
            ['Все', 'loyal', 'neutral', 'at_risk']
        )

        display_df = df.copy()
        if segment_filter != 'Все' and 'loyalty_segment' in df.columns:
            display_df = display_df[display_df['loyalty_segment'] == segment_filter]

        # Показываем отзывы
        n_reviews = st.slider("Количество отзывов", 5, 50, 10)
        sample = display_df.sample(min(n_reviews, len(display_df)))

        for idx, row in sample.iterrows():
            with st.expander(f"⭐ {row.get('stars', 'N/A')} | {row.get('product_name', 'Продукт')[:50]}..."):
                cols = st.columns([1, 1, 2])

                with cols[0]:
                    st.markdown(f"**Оценка:** {row.get('stars', 'N/A')} ⭐")
                    st.markdown(f"**Рекомендует:** {'Да ✅' if row.get('is_recommended') else 'Нет ❌'}")
                    if 'loyalty_score' in row:
                        st.markdown(f"**Loyalty Score:** {row['loyalty_score']:.3f}")
                    if 'loyalty_segment' in row:
                        segment_emoji = {'loyal': '🟢', 'neutral': '🟡', 'at_risk': '🔴'}.get(row['loyalty_segment'], '')
                        st.markdown(f"**Сегмент:** {segment_emoji} {row['loyalty_segment']}")

                with cols[1]:
                    if 'combined_sentiment' in row:
                        st.markdown(f"**Sentiment:** {row['combined_sentiment']:.3f}")
                    if 'detected_language' in row:
                        st.markdown(f"**Язык:** {row['detected_language']}")
                    st.markdown(f"**Категория:** {row.get('product_type', 'N/A')}")

                with cols[2]:
                    st.markdown("**Плюсы:**")
                    st.write(row.get('pros', '-') or '-')
                    st.markdown("**Минусы:**")
                    st.write(row.get('cons', '-') or '-')
                    if row.get('comment'):
                        st.markdown("**Комментарий:**")
                        st.write(row.get('comment', '-'))

    # === TAB 4: ДАННЫЕ ===
    with tab4:
        st.header("Данные")

        # Выбор колонок
        available_cols = df.columns.tolist()
        default_cols = ['product_name', 'stars', 'is_recommended', 'loyalty_score', 'loyalty_segment', 'pros', 'cons']
        default_cols = [c for c in default_cols if c in available_cols]

        selected_cols = st.multiselect("Выберите колонки:", available_cols, default=default_cols)

        if selected_cols:
            st.dataframe(df[selected_cols].head(1000), use_container_width=True)

        # Скачать данные
        st.subheader("Экспорт")

        col1, col2 = st.columns(2)

        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Скачать CSV",
                csv,
                "loyalty_analysis.csv",
                "text/csv"
            )

        with col2:
            if 'loyalty_score' in df.columns:
                summary = df.groupby('loyalty_segment').agg({
                    'loyalty_score': ['count', 'mean'],
                    'stars': 'mean',
                    'is_recommended': 'mean'
                }).round(3)
                summary_csv = summary.to_csv().encode('utf-8')
                st.download_button(
                    "📥 Скачать сводку",
                    summary_csv,
                    "loyalty_summary.csv",
                    "text/csv"
                )


if __name__ == "__main__":
    main()
