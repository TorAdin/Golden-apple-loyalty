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
from src.catch_phrases import analyze_catch_phrases_dataframe, get_catch_phrase_summary
from src.llm_sentiment import LLMSentimentAnalyzer, compare_sentiments


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


@st.cache_data
def detect_catch_phrases(_df):
    """Детекция кэтч-фраз (с кэшированием)"""
    return analyze_catch_phrases_dataframe(_df)


@st.cache_data
def run_llm_sentiment(_df, sample_size, api_key):
    """LLM Sentiment Analysis (с кэшированием)"""
    analyzer = LLMSentimentAnalyzer(api_key=api_key)
    if analyzer.is_available():
        return analyzer.analyze_dataframe(_df, sample_size=sample_size)
    return _df


def main():
    # Заголовок
    st.title("🍎 Darling Loyalty Analysis")
    st.markdown("**Анализ лояльности клиентов по отзывам в Золотом Яблоке**")

    # Сайдбар
    st.sidebar.header("⚙️ Настройки")

    # Загрузка данных - поддержка локального файла и upload
    data_path = Path(__file__).parent / "data" / "data_darling.xlsx"

    df = None

    # Проверяем локальный файл
    if data_path.exists():
        with st.spinner("Загрузка данных..."):
            df = load_and_process_data(str(data_path))
    else:
        # Cloud mode - показываем uploader
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📁 Загрузка данных")
        uploaded_file = st.sidebar.file_uploader(
            "Загрузите data_darling.xlsx",
            type=['xlsx'],
            help="Файл с отзывами в формате Excel"
        )

        if uploaded_file is not None:
            with st.spinner("Загрузка данных..."):
                # Читаем напрямую из uploaded file
                df = pd.read_excel(uploaded_file)

                # Стандартизация колонок (как в data_loader.py)
                column_mapping = {
                    'pros': 'pros',
                    'cons': 'cons',
                    'comment': 'comment',
                    'isrecommended': 'is_recommended',
                    'stars': 'stars',
                    'catalogname': 'product_name',
                    'producttype': 'product_type',
                    'createddate': 'created_date'
                }
                df.columns = df.columns.str.lower().str.strip()
                df = df.rename(columns=column_mapping)

                # Преобразование типов данных
                if 'is_recommended' in df.columns:
                    df['is_recommended'] = pd.to_numeric(df['is_recommended'], errors='coerce').fillna(0).astype(int)
                if 'stars' in df.columns:
                    df['stars'] = pd.to_numeric(df['stars'], errors='coerce')

                # Очищаем данные
                df = clean_dataframe(df)
        else:
            st.info("👆 Загрузите файл data_darling.xlsx через сайдбар слева")
            st.markdown("""
            **Ожидаемые колонки в файле:**
            - `pros` - плюсы товара
            - `cons` - минусы товара
            - `comment` - комментарий
            - `stars` - оценка (1-5)
            - `isrecommended` - рекомендует ли (True/False)
            - `product_type` - тип товара
            - `catalog_name` - название каталога
            """)
            return

    if df is None:
        st.error("Не удалось загрузить данные")
        return

    st.sidebar.success(f"✅ Загружено {len(df):,} отзывов")

    # Опции анализа
    run_sentiment = st.sidebar.checkbox("Запустить Sentiment Analysis", value=True)
    run_loyalty = st.sidebar.checkbox("Рассчитать Loyalty Score", value=True)
    run_catch_phrases = st.sidebar.checkbox("Детекция кэтч-фраз", value=True)

    # Обработка
    if run_sentiment:
        with st.spinner("Анализ тональности..."):
            df = run_sentiment_analysis(df)

    if run_loyalty and 'combined_sentiment' in df.columns:
        with st.spinner("Расчёт Loyalty Score..."):
            df = calculate_loyalty(df)

    if run_catch_phrases:
        with st.spinner("Детекция кэтч-фраз..."):
            df = detect_catch_phrases(df)

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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Обзор", "📈 Лояльность", "📦 Товары", "🔄 Кэтч-фразы", "💬 Отзывы", "📋 Данные"])

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
            # Метрики по сегментам (обновлённые трешхолды)
            col1, col2, col3 = st.columns(3)

            segment_counts = df['loyalty_segment'].value_counts()

            with col1:
                loyal_pct = segment_counts.get('loyal', 0) / len(df) * 100
                st.metric("🟢 Loyal (≥0.9)", f"{loyal_pct:.1f}%",
                         f"{segment_counts.get('loyal', 0):,} отзывов")

            with col2:
                neutral_pct = segment_counts.get('neutral', 0) / len(df) * 100
                st.metric("🟡 Neutral (0.7-0.9)", f"{neutral_pct:.1f}%",
                         f"{segment_counts.get('neutral', 0):,} отзывов")

            with col3:
                atrisk_pct = segment_counts.get('at_risk', 0) / len(df) * 100
                st.metric("🔴 At Risk (<0.7)", f"{atrisk_pct:.1f}%",
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
                fig.add_vline(x=0.9, line_dash="dash", line_color="green", annotation_text="Loyal ≥0.9")
                fig.add_vline(x=0.7, line_dash="dash", line_color="orange", annotation_text="Neutral ≥0.7")
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
                sample_df = df.sample(min(5000, len(df)))
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

            # LLM Sentiment Analysis
            st.subheader("🤖 LLM Sentiment Analysis (OpenAI)")

            with st.expander("Настройки LLM анализа"):
                st.markdown("""
                **LLM анализ** использует GPT для более точной оценки тональности.
                Он лучше понимает контекст, сарказм и скрытое недовольство.

                Требуется: OpenAI API ключ
                """)

                api_key = st.text_input("OpenAI API Key:", type="password",
                                       help="Введите ваш API ключ или установите OPENAI_API_KEY")

                llm_sample = st.slider("Размер выборки для LLM:", 10, 1000, 100,
                                       help="LLM анализ дорогой, рекомендуется начать с небольшой выборки")

                if st.button("Запустить LLM анализ"):
                    if api_key:
                        with st.spinner(f"LLM анализ {llm_sample} отзывов..."):
                            df_llm = run_llm_sentiment(df, llm_sample, api_key)
                            st.session_state['df_with_llm'] = df_llm
                        st.rerun()
                    else:
                        st.error("Введите OpenAI API ключ")

            # Показываем результаты LLM, если есть
            if 'df_with_llm' in st.session_state:
                df_llm = st.session_state['df_with_llm']
                if 'llm_sentiment_score' in df_llm.columns:
                    # Считаем статистику напрямую
                    llm_analyzed = df_llm['llm_sentiment_score'].notna().sum()

                    if llm_analyzed > 0:
                        st.markdown(f"**Результаты LLM анализа ({llm_analyzed} отзывов):**")
                        col1, col2, col3, col4 = st.columns(4)

                        mask = df_llm['llm_sentiment_score'].notna()
                        llm_mean = df_llm.loc[mask, 'llm_sentiment_score'].mean()
                        keyword_mean = df_llm.loc[mask, 'combined_sentiment'].mean() if 'combined_sentiment' in df_llm.columns else 0

                        with col1:
                            st.metric("Проанализировано", f"{llm_analyzed}")
                        with col2:
                            st.metric("LLM Sentiment (avg)", f"{llm_mean:.3f}")
                        with col3:
                            st.metric("Keyword Sentiment (avg)", f"{keyword_mean:.3f}")
                        with col4:
                            diff = llm_mean - keyword_mean
                            st.metric("Разница", f"{diff:+.3f}")

                        # Scatter: Keyword vs LLM
                        mask = df_llm['llm_sentiment_score'].notna()
                        if mask.sum() > 0:
                            col1, col2 = st.columns(2)

                            with col1:
                                fig = px.scatter(
                                    df_llm[mask],
                                    x='combined_sentiment',
                                    y='llm_sentiment_score',
                                    color='loyalty_segment' if 'loyalty_segment' in df_llm.columns else None,
                                    title="Keyword vs LLM Sentiment",
                                    labels={'combined_sentiment': 'Keyword Sentiment', 'llm_sentiment_score': 'LLM Sentiment'},
                                    opacity=0.6
                                )
                                fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                                             line=dict(color="gray", dash="dash"))
                                st.plotly_chart(fig, use_container_width=True)

                            with col2:
                                # Распределение LLM sentiment
                                fig = px.histogram(
                                    df_llm[mask],
                                    x='llm_sentiment_score',
                                    nbins=20,
                                    title="Распределение LLM Sentiment",
                                    color_discrete_sequence=['#4ecdc4']
                                )
                                st.plotly_chart(fig, use_container_width=True)

                            # Примеры отзывов с LLM оценками
                            st.markdown("**Примеры LLM анализа:**")
                            sample_llm = df_llm[mask].sample(min(10, mask.sum()))
                            display_cols = ['product_name', 'llm_sentiment_score', 'combined_sentiment', 'pros', 'cons']
                            display_cols = [c for c in display_cols if c in sample_llm.columns]
                            st.dataframe(sample_llm[display_cols].round(3), use_container_width=True, hide_index=True)

    # === TAB 3: ТОВАРЫ (НОВАЯ ВКЛАДКА) ===
    with tab3:
        st.header("📦 Агрегация по товарам")

        if 'product_name' not in df.columns:
            st.warning("Колонка product_name не найдена")
        elif 'stars' not in df.columns:
            st.warning("Колонка stars не найдена")
        else:
            # Агрегация по товарам - только существующие колонки
            agg_dict = {'stars': ['mean', 'count']}
            if 'is_recommended' in df.columns:
                agg_dict['is_recommended'] = 'mean'
            if 'product_type' in df.columns:
                agg_dict['product_type'] = 'first'
            if 'loyalty_score' in df.columns:
                agg_dict['loyalty_score'] = ['mean', 'std']
            if 'combined_sentiment' in df.columns:
                agg_dict['combined_sentiment'] = 'mean'

            product_agg = df.groupby('product_name').agg(agg_dict).round(3)

            # Flatten column names
            product_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in product_agg.columns.values]
            product_agg = product_agg.reset_index()

            # Rename columns
            rename_map = {
                'loyalty_score_mean': 'avg_loyalty',
                'loyalty_score_std': 'std_loyalty',
                'stars_mean': 'avg_stars',
                'stars_count': 'reviews_count',
                'is_recommended_mean': 'recommend_rate',
                'combined_sentiment_mean': 'avg_sentiment',
                'product_type_first': 'category'
            }
            product_agg = product_agg.rename(columns={k: v for k, v in rename_map.items() if k in product_agg.columns})

            # Фильтр по минимальному количеству отзывов
            min_reviews = st.slider("Минимум отзывов на товар", 1, 100, 10)
            if 'reviews_count' in product_agg.columns:
                product_agg = product_agg[product_agg['reviews_count'] >= min_reviews]

            st.markdown(f"**Товаров с ≥{min_reviews} отзывов: {len(product_agg):,}**")

            # Метрики
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Всего товаров", f"{len(product_agg):,}")
            with col2:
                if 'avg_loyalty' in product_agg.columns:
                    st.metric("Средний Loyalty", f"{product_agg['avg_loyalty'].mean():.3f}")
            with col3:
                if 'avg_stars' in product_agg.columns:
                    st.metric("Средний Stars", f"{product_agg['avg_stars'].mean():.2f}")
            with col4:
                if 'recommend_rate' in product_agg.columns:
                    st.metric("Средний Recommend", f"{product_agg['recommend_rate'].mean()*100:.1f}%")

            # Топ и худшие товары
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🏆 Топ товары по лояльности")
                if 'avg_loyalty' in product_agg.columns:
                    display_cols = [c for c in ['product_name', 'avg_loyalty', 'avg_stars', 'reviews_count', 'category'] if c in product_agg.columns]
                    top_products = product_agg.nlargest(10, 'avg_loyalty')[display_cols]
                    st.dataframe(top_products, use_container_width=True, hide_index=True)

            with col2:
                st.subheader("⚠️ Проблемные товары")
                if 'avg_loyalty' in product_agg.columns:
                    display_cols = [c for c in ['product_name', 'avg_loyalty', 'avg_stars', 'reviews_count', 'category'] if c in product_agg.columns]
                    bottom_products = product_agg.nsmallest(10, 'avg_loyalty')[display_cols]
                    st.dataframe(bottom_products, use_container_width=True, hide_index=True)

            # График: Loyalty vs Stars по товарам
            if 'avg_loyalty' in product_agg.columns and 'avg_stars' in product_agg.columns:
                st.subheader("Loyalty vs Stars по товарам")
                fig = px.scatter(
                    product_agg,
                    x='avg_stars',
                    y='avg_loyalty',
                    size='reviews_count' if 'reviews_count' in product_agg.columns else None,
                    color='category' if 'category' in product_agg.columns else None,
                    hover_name='product_name',
                    title="Каждая точка — товар (размер = кол-во отзывов)",
                    opacity=0.6
                )
                fig.add_hline(y=0.9, line_dash="dash", line_color="green", annotation_text="Loyal")
                fig.add_hline(y=0.7, line_dash="dash", line_color="orange", annotation_text="Neutral")
                st.plotly_chart(fig, use_container_width=True)

            # Поиск товара
            st.subheader("🔍 Поиск товара")
            search_query = st.text_input("Введите название товара:")
            if search_query:
                found = product_agg[product_agg['product_name'].str.contains(search_query, case=False, na=False)]
                if len(found) > 0:
                    st.dataframe(found, use_container_width=True, hide_index=True)
                else:
                    st.info("Товар не найден")

            # Скачать агрегацию
            st.subheader("📥 Экспорт")
            csv = product_agg.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Скачать агрегацию по товарам (CSV)",
                csv,
                "products_loyalty.csv",
                "text/csv"
            )

    # === TAB 4: КЭТЧ-ФРАЗЫ ===
    with tab4:
        st.header("🔄 Кэтч-фразы (индикаторы повторной покупки)")

        if 'has_catch_phrase' not in df.columns:
            st.warning("Включите детекцию кэтч-фраз в настройках")
        else:
            # Сводка
            summary = get_catch_phrase_summary(df)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Отзывов с кэтч-фразами",
                         f"{summary.get('reviews_with_phrases', 0):,}",
                         f"{summary.get('percent_with_phrases', 0):.1f}%")

            with col2:
                if 'avg_loyalty_with_phrase' in summary:
                    diff = summary['avg_loyalty_with_phrase'] - summary['avg_loyalty_without_phrase']
                    st.metric("Лояльность с фразами",
                             f"{summary['avg_loyalty_with_phrase']:.3f}",
                             f"+{diff:.3f}" if diff > 0 else f"{diff:.3f}")

            with col3:
                if 'avg_loyalty_without_phrase' in summary:
                    st.metric("Лояльность без фраз",
                             f"{summary['avg_loyalty_without_phrase']:.3f}")

            with col4:
                total_phrases = sum(len(p) for p in df['catch_phrases'] if p)
                st.metric("Всего найдено фраз", f"{total_phrases:,}")

            # Топ фразы
            if 'top_phrases' in summary:
                st.subheader("🏆 Топ кэтч-фразы")
                phrase_df = pd.DataFrame([
                    {'Фраза': phrase, 'Количество': count}
                    for phrase, count in summary['top_phrases'].items()
                ])
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig = px.bar(
                        phrase_df,
                        x='Количество',
                        y='Фраза',
                        orientation='h',
                        title="Популярные индикаторы повторной покупки",
                        color='Количество',
                        color_continuous_scale='Greens'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.dataframe(phrase_df, use_container_width=True, hide_index=True)

            # Примеры отзывов с кэтч-фразами
            st.subheader("📝 Примеры отзывов с кэтч-фразами")
            catch_df = df[df['has_catch_phrase'] == True]

            if len(catch_df) > 0:
                n_samples = st.slider("Показать примеров:", 5, 30, 10, key="catch_samples")
                sample_catch = catch_df.sample(min(n_samples, len(catch_df)))

                for idx, row in sample_catch.iterrows():
                    phrases_str = ', '.join(row['catch_phrases']) if row['catch_phrases'] else '-'
                    with st.expander(f"⭐ {row.get('stars', 'N/A')} | Фразы: {phrases_str}"):
                        cols = st.columns([1, 2])
                        with cols[0]:
                            st.markdown(f"**Продукт:** {row.get('product_name', 'N/A')[:50]}")
                            st.markdown(f"**Категория:** {row.get('product_type', 'N/A')}")
                            st.markdown(f"**Loyalty:** {row.get('loyalty_score', 0):.3f}")
                            st.markdown(f"**Найденные фразы:** `{phrases_str}`")
                        with cols[1]:
                            st.markdown("**Плюсы:**")
                            st.write(row.get('pros', '-') or '-')
                            st.markdown("**Комментарий:**")
                            st.write(row.get('comment', '-') or '-')

            # Связь с лояльностью
            if 'loyalty_score' in df.columns:
                st.subheader("📊 Связь кэтч-фраз с лояльностью")

                col1, col2 = st.columns(2)

                with col1:
                    # Box plot
                    fig = px.box(
                        df,
                        x='has_catch_phrase',
                        y='loyalty_score',
                        color='has_catch_phrase',
                        title="Распределение Loyalty Score",
                        labels={'has_catch_phrase': 'Есть кэтч-фраза', 'loyalty_score': 'Loyalty Score'},
                        color_discrete_map={True: '#51cf66', False: '#868e96'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Распределение по сегментам
                    if 'loyalty_segment' in df.columns:
                        seg_phrase = df.groupby(['loyalty_segment', 'has_catch_phrase']).size().unstack(fill_value=0)
                        seg_phrase_pct = seg_phrase.div(seg_phrase.sum(axis=1), axis=0) * 100
                        fig = px.bar(
                            seg_phrase_pct.reset_index(),
                            x='loyalty_segment',
                            y=[True, False] if True in seg_phrase_pct.columns else seg_phrase_pct.columns.tolist(),
                            title="% отзывов с кэтч-фразами по сегментам",
                            barmode='group',
                            labels={'value': '% отзывов', 'loyalty_segment': 'Сегмент'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

    # === TAB 5: ОТЗЫВЫ ===
    with tab5:
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
                    if 'has_catch_phrase' in row and row['has_catch_phrase']:
                        phrases = ', '.join(row['catch_phrases']) if row.get('catch_phrases') else ''
                        st.markdown(f"**🔄 Кэтч-фразы:** `{phrases}`")

                with cols[2]:
                    st.markdown("**Плюсы:**")
                    st.write(row.get('pros', '-') or '-')
                    st.markdown("**Минусы:**")
                    st.write(row.get('cons', '-') or '-')
                    if row.get('comment'):
                        st.markdown("**Комментарий:**")
                        st.write(row.get('comment', '-'))

    # === TAB 6: ДАННЫЕ ===
    with tab6:
        st.header("Данные")

        # Выбор колонок
        available_cols = df.columns.tolist()
        default_cols = ['product_name', 'stars', 'is_recommended', 'loyalty_score', 'loyalty_segment', 'has_catch_phrase', 'catch_phrases', 'pros', 'cons']
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
            if 'loyalty_score' in df.columns and 'loyalty_segment' in df.columns:
                # Строим agg dict только из существующих колонок
                agg_dict = {'loyalty_score': ['count', 'mean']}
                if 'stars' in df.columns:
                    agg_dict['stars'] = 'mean'
                if 'is_recommended' in df.columns:
                    agg_dict['is_recommended'] = 'mean'

                summary = df.groupby('loyalty_segment').agg(agg_dict).round(3)
                summary_csv = summary.to_csv().encode('utf-8')
                st.download_button(
                    "📥 Скачать сводку",
                    summary_csv,
                    "loyalty_summary.csv",
                    "text/csv"
                )


if __name__ == "__main__":
    main()
