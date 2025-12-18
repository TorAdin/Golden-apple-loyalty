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
from src.bert_loyalty import BertLoyaltyAnalyzer, compare_with_keyword_method


@st.cache_data
def load_and_process_data(file_path=None):
    """Загрузка и обработка данных (с кэшированием)"""
    df = load_data(file_path)
    df = clean_dataframe(df)
    return df


def run_sentiment_analysis(df):
    """Sentiment analysis (БЕЗ кэширования для корректной работы с uploaded files)"""
    analyzer = SentimentAnalyzer()
    return analyzer.analyze_dataframe(df)


def calculate_loyalty(df):
    """Расчёт Loyalty Score (БЕЗ кэширования)"""
    scorer = LoyaltyScorer()
    return scorer.score_dataframe(df)


def detect_catch_phrases_func(df):
    """Детекция кэтч-фраз (БЕЗ кэширования)"""
    return analyze_catch_phrases_dataframe(df)


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

    # Загрузка данных - базовый XLSX + мерж с расширенными фичами из CSV
    xlsx_path = Path(__file__).parent / "data" / "data_darling.xlsx"
    csv_advanced_path = Path(__file__).parent / "final_data_darling.csv"

    df = None

    # Грузим базовый файл
    if xlsx_path.exists():
        with st.spinner("Загрузка данных..."):
            df = load_and_process_data(str(xlsx_path))

        # МЕРЖИМ расширенные фичи из CSV если он есть
        if csv_advanced_path.exists():
            try:
                csv_df = pd.read_csv(csv_advanced_path)

                # Список расширенных колонок для мержа
                advanced_cols = [
                    'Repurchase_Intent_Tag', 'Abandonment_Tag', 'Misexpectation_Type',
                    'Advocacy_Strength', 'Price_Sensitivity_Tag', 'Alternative_Brand_Mentioned',
                    'Affection_Trigger', 'Review_Purpose', 'Review_Emotion_Class'
                ]

                # Берём только те колонки которые есть в CSV
                merge_cols = [col for col in advanced_cols if col in csv_df.columns]

                if merge_cols and len(csv_df) == len(df):
                    # ПРЯМОЕ ДОБАВЛЕНИЕ колонок по индексу (если количество строк совпадает)
                    for col in merge_cols:
                        df[col] = csv_df[col].values

                    st.sidebar.success(f"✅ Добавлено {len(merge_cols)} расширенных фич")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Не удалось загрузить расширенные фичи: {e}")
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
            # Используем session_state для кэширования обработанных данных
            file_key = f"processed_{uploaded_file.name}_{uploaded_file.size}"

            if file_key not in st.session_state:
                with st.spinner("Загрузка и обработка данных..."):
                    # Читаем напрямую из uploaded file
                    raw_df = pd.read_excel(uploaded_file)

                    # Приводим к lowercase
                    raw_df.columns = raw_df.columns.str.lower().str.strip()

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
                    raw_df = raw_df.rename(columns=column_mapping)

                    # Преобразование типов данных
                    if 'is_recommended' in raw_df.columns:
                        # Конвертируем True/False/1/0 в числа
                        raw_df['is_recommended'] = raw_df['is_recommended'].map({True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0}).fillna(0).astype(int)

                    if 'stars' in raw_df.columns:
                        raw_df['stars'] = pd.to_numeric(raw_df['stars'], errors='coerce')

                    # Очищаем данные
                    raw_df = clean_dataframe(raw_df)

                    # Сохраняем в session_state
                    st.session_state[file_key] = raw_df

            df = st.session_state[file_key].copy()
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

    # НОВОЕ: BERT анализ
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 BERT Анализ")
    run_bert = st.sidebar.checkbox(
        "🧠 Запустить BERT анализ лояльности",
        value=False,
        help="Использует обученную BERT модель для более точного анализа лояльности"
    )

    # Обработка с кэшированием в session_state
    # Создаём уникальный ключ для текущего состояния данных
    data_hash = hash(tuple(df.columns.tolist()) + (len(df),))

    if run_sentiment:
        sentiment_key = f"sentiment_{data_hash}"
        if sentiment_key not in st.session_state:
            with st.spinner("Анализ тональности..."):
                st.session_state[sentiment_key] = run_sentiment_analysis(df)
        df = st.session_state[sentiment_key].copy()

    if run_loyalty and 'combined_sentiment' in df.columns:
        loyalty_key = f"loyalty_{data_hash}"
        if loyalty_key not in st.session_state:
            with st.spinner("Расчёт Loyalty Score..."):
                st.session_state[loyalty_key] = calculate_loyalty(df)
        df = st.session_state[loyalty_key].copy()

    if run_catch_phrases:
        catch_key = f"catch_{data_hash}"
        if catch_key not in st.session_state:
            with st.spinner("Детекция кэтч-фраз..."):
                st.session_state[catch_key] = detect_catch_phrases_func(df)
        df = st.session_state[catch_key].copy()

    # BERT анализ
    if run_bert:
        bert_key = f"bert_{data_hash}"
        if bert_key not in st.session_state:
            with st.spinner("🧠 BERT анализ лояльности (может занять несколько минут)..."):
                analyzer = BertLoyaltyAnalyzer()
                if analyzer.is_available():
                    st.session_state[bert_key] = analyzer.analyze_dataframe(df)
                    st.success("✅ BERT анализ завершён!")
                else:
                    st.error("❌ BERT модель недоступна. Убедитесь, что папка Golden-apple-loyalty/models_binary_fixed_v2 существует")
                    st.session_state[bert_key] = df
        df = st.session_state[bert_key].copy()

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

    # ФИКС: Нормализуем названия расширенных колонок (если есть в lowercase - переименовываем обратно)
    advanced_col_mapping = {
        'repurchase_intent_tag': 'Repurchase_Intent_Tag',
        'abandonment_tag': 'Abandonment_Tag',
        'misexpectation_type': 'Misexpectation_Type',
        'advocacy_strength': 'Advocacy_Strength',
        'price_sensitivity_tag': 'Price_Sensitivity_Tag',
        'alternative_brand_mentioned': 'Alternative_Brand_Mentioned',
        'affection_trigger': 'Affection_Trigger',
        'review_purpose': 'Review_Purpose',
        'review_emotion_class': 'Review_Emotion_Class'
    }

    # Переименовываем расширенные колонки из lowercase в правильный формат
    rename_advanced = {}
    for lower_name, proper_name in advanced_col_mapping.items():
        if lower_name in df.columns:
            rename_advanced[lower_name] = proper_name

    if rename_advanced:
        df = df.rename(columns=rename_advanced)

    # Проверяем наличие расширенных фич лояльности
    has_advanced_features = all(col in df.columns for col in [
        'Repurchase_Intent_Tag', 'Abandonment_Tag', 'Review_Purpose', 'Review_Emotion_Class'
    ])

    # Отладочная информация в sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Статус данных")
    st.sidebar.write(f"Всего колонок: {len(df.columns)}")

    # ПОКАЗЫВАЕМ ВСЕ КОЛОНКИ
    with st.sidebar.expander("📋 Все колонки в данных"):
        for col in sorted(df.columns):
            st.write(f"• {col}")

    if has_advanced_features:
        st.sidebar.success("✅ Расширенные фичи загружены")
    else:
        st.sidebar.warning("⚠️ Расширенные фичи не найдены")
        st.sidebar.write("Ищем колонки:")
        for col in ['Repurchase_Intent_Tag', 'Abandonment_Tag', 'Review_Purpose', 'Review_Emotion_Class']:
            if col in df.columns:
                st.sidebar.write(f"  ✅ {col}")
            else:
                st.sidebar.write(f"  ❌ {col}")

    # Вкладки (продвинутый анализ сразу после Обзора)
    tabs = ["📊 Обзор"]

    # Добавляем продвинутый анализ сразу после Обзора
    if has_advanced_features:
        tabs.append("🎯 Продвинутый анализ")

    tabs.extend(["📈 Лояльность", "📦 Товары", "🔄 Кэтч-фразы"])

    # Добавляем BERT вкладку если анализ запущен
    if 'bert_loyalty_prob' in df.columns:
        tabs.append("🧠 BERT Анализ")

    tabs.extend(["💬 Отзывы", "📋 Данные"])

    all_tabs = st.tabs(tabs)

    # Распаковываем вкладки (новый порядок: Обзор -> Продвинутый анализ -> Лояльность -> Товары -> Кэтч-фразы -> BERT -> Отзывы -> Данные)
    tab_idx = 0
    tab1 = all_tabs[tab_idx]; tab_idx += 1  # Обзор

    # Продвинутый анализ (сразу после Обзора, если есть)
    if has_advanced_features:
        tab_advanced = all_tabs[tab_idx]; tab_idx += 1

    tab2 = all_tabs[tab_idx]; tab_idx += 1  # Лояльность
    tab3 = all_tabs[tab_idx]; tab_idx += 1  # Товары
    tab4 = all_tabs[tab_idx]; tab_idx += 1  # Кэтч-фразы

    # BERT вкладка (если есть)
    if 'bert_loyalty_prob' in df.columns:
        tab_bert = all_tabs[tab_idx]; tab_idx += 1

    tab5 = all_tabs[tab_idx]; tab_idx += 1  # Отзывы
    tab6 = all_tabs[tab_idx]; tab_idx += 1  # Данные

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
                rec_data = df['is_recommended'].value_counts().sort_index()
                # 0 = Не рекомендуют (красный), 1 = Рекомендуют (зелёный)
                labels = ['Не рекомендуют', 'Рекомендуют'] if 0 in rec_data.index else ['Рекомендуют']
                colors = ['#ff6b6b', '#51cf66'] if len(rec_data) == 2 else ['#51cf66']

                fig = px.pie(
                    values=rec_data.values,
                    names=labels,
                    title="Рекомендации",
                    color_discrete_sequence=colors
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
                # Убираем ось Y - цвет уже показывает лояльность
                fig.update_layout(
                    xaxis_tickangle=-45,
                    yaxis_visible=False,
                    yaxis_showticklabels=False
                )
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
            # Агрегация по товарам - группируем по названию + категории, чтобы избежать дубликатов
            group_cols = ['product_name']
            if 'product_type' in df.columns:
                group_cols.append('product_type')

            agg_dict = {'stars': ['mean', 'count']}
            if 'is_recommended' in df.columns:
                agg_dict['is_recommended'] = 'mean'
            if 'loyalty_score' in df.columns:
                agg_dict['loyalty_score'] = ['mean', 'std']
            if 'combined_sentiment' in df.columns:
                agg_dict['combined_sentiment'] = 'mean'

            product_agg = df.groupby(group_cols).agg(agg_dict).round(3)

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
                'product_type': 'category'
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

    # === TAB ADVANCED: ПРОДВИНУТЫЙ АНАЛИЗ ЛОЯЛЬНОСТИ ===
    if has_advanced_features:
        with tab_advanced:
            st.header("🎯 Продвинутый анализ лояльности")

            st.markdown("""
            **Расширенный анализ факторов лояльности на основе GPT-разметки:**
            - Намерение повторной покупки и риски оттока
            - Причины несоответствия ожиданий
            - Сила адвокации и ценовая чувствительность
            - Эмоциональные триггеры и цель отзыва
            """)

            # === 1. REPURCHASE INTENT & ABANDONMENT ===
            st.subheader("🔄 Намерение повторной покупки vs Риск ухода")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Repurchase Intent (Намерение купить снова)**")
                if 'Repurchase_Intent_Tag' in df.columns:
                    repurchase_counts = df['Repurchase_Intent_Tag'].value_counts()
                    repurchase_pct = df['Repurchase_Intent_Tag'].value_counts(normalize=True) * 100

                    # Метрики
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Yes", f"{repurchase_counts.get('Yes', 0):,}",
                                f"{repurchase_pct.get('Yes', 0):.1f}%")
                    with col_b:
                        st.metric("Unclear", f"{repurchase_counts.get('Unclear', 0):,}",
                                f"{repurchase_pct.get('Unclear', 0):.1f}%")
                    with col_c:
                        st.metric("No", f"{repurchase_counts.get('No', 0):,}",
                                f"{repurchase_pct.get('No', 0):.1f}%")

                    # Pie chart
                    fig = px.pie(
                        values=repurchase_counts.values,
                        names=repurchase_counts.index,
                        title="Распределение намерений повторной покупки",
                        color=repurchase_counts.index,
                        color_discrete_map={'Yes': '#51cf66', 'Unclear': '#ffd43b', 'No': '#ff6b6b'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**Abandonment (Риск ухода)**")
                if 'Abandonment_Tag' in df.columns:
                    abandon_counts = df['Abandonment_Tag'].value_counts()
                    abandon_pct = df['Abandonment_Tag'].value_counts(normalize=True) * 100

                    # Метрики
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Stay", f"{abandon_counts.get('Stay', 0):,}",
                                f"{abandon_pct.get('Stay', 0):.1f}%")
                    with col_b:
                        st.metric("Considering", f"{abandon_counts.get('Considering_leave', 0):,}",
                                f"{abandon_pct.get('Considering_leave', 0):.1f}%")
                    with col_c:
                        st.metric("Leave", f"{abandon_counts.get('Leave', 0):,}",
                                f"{abandon_pct.get('Leave', 0):.1f}%")

                    # Pie chart
                    fig = px.pie(
                        values=abandon_counts.values,
                        names=abandon_counts.index,
                        title="Распределение рисков ухода",
                        color=abandon_counts.index,
                        color_discrete_map={'Stay': '#51cf66', 'Considering_leave': '#ffd43b', 'Leave': '#ff6b6b'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Матрица Repurchase vs Abandonment
            if 'Repurchase_Intent_Tag' in df.columns and 'Abandonment_Tag' in df.columns:
                st.subheader("🎯 Матрица: Повторная покупка vs Уход")

                # Создаём сводную таблицу
                matrix = pd.crosstab(
                    df['Repurchase_Intent_Tag'],
                    df['Abandonment_Tag'],
                    normalize='all'
                ) * 100

                # Heatmap
                fig = px.imshow(
                    matrix,
                    labels=dict(x="Abandonment", y="Repurchase Intent", color="% отзывов"),
                    x=matrix.columns,
                    y=matrix.index,
                    title="Комбинации намерений (% от всех отзывов)",
                    color_continuous_scale='RdYlGn',
                    text_auto='.1f'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Интерпретация ключевых сегментов
                st.markdown("**📊 Ключевые сегменты:**")
                col1, col2 = st.columns(2)

                with col1:
                    # Yes + Stay
                    yes_stay = ((df['Repurchase_Intent_Tag'] == 'Yes') &
                              (df['Abandonment_Tag'] == 'Stay')).sum()
                    st.success(f"✅ **Устойчивая лояльность:** {yes_stay:,} ({yes_stay/len(df)*100:.1f}%)")
                    st.caption("Купят снова и не ищут альтернативы")

                    # No + Leave
                    no_leave = ((df['Repurchase_Intent_Tag'] == 'No') &
                              (df['Abandonment_Tag'] == 'Leave')).sum()
                    st.error(f"❌ **Критический отток:** {no_leave:,} ({no_leave/len(df)*100:.1f}%)")
                    st.caption("Не купят и активно уходят")

                with col2:
                    # Yes + Considering_leave
                    yes_considering = ((df['Repurchase_Intent_Tag'] == 'Yes') &
                                     (df['Abandonment_Tag'] == 'Considering_leave')).sum()
                    st.warning(f"⚠️ **Лояльность под угрозой:** {yes_considering:,} ({yes_considering/len(df)*100:.1f}%)")
                    st.caption("Купят, но сомневаются")

                    # No + Stay
                    no_stay = ((df['Repurchase_Intent_Tag'] == 'No') &
                             (df['Abandonment_Tag'] == 'Stay')).sum()
                    st.info(f"🔒 **Вынужденная лояльность:** {no_stay:,} ({no_stay/len(df)*100:.1f}%)")
                    st.caption("Не купят снова, но и не уходят (нет альтернатив)")

            # === 2. MISEXPECTATION TYPE ===
            st.subheader("💥 Причины несоответствия ожиданий")

            if 'Misexpectation_Type' in df.columns:
                misexp_counts = df['Misexpectation_Type'].value_counts()

                col1, col2 = st.columns([2, 1])

                with col1:
                    # Bar chart
                    fig = px.bar(
                        x=misexp_counts.index,
                        y=misexp_counts.values,
                        title="Распределение причин разочарования",
                        labels={'x': 'Причина', 'y': 'Количество'},
                        color=misexp_counts.values,
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(xaxis_tickangle=-45, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("**Топ причин:**")
                    for idx, (reason, count) in enumerate(misexp_counts.head(5).items(), 1):
                        pct = count / len(df) * 100
                        st.write(f"{idx}. **{reason}**: {count:,} ({pct:.1f}%)")

                # Связь с Repurchase Intent
                if 'Repurchase_Intent_Tag' in df.columns:
                    st.markdown("**Влияние причин на намерение повторной покупки:**")

                    misexp_repurchase = pd.crosstab(
                        df['Misexpectation_Type'],
                        df['Repurchase_Intent_Tag'],
                        normalize='index'
                    ) * 100

                    fig = px.bar(
                        misexp_repurchase.reset_index(),
                        x='Misexpectation_Type',
                        y=['Yes', 'No', 'Unclear'],
                        title="% намерений по типам разочарования",
                        barmode='group',
                        labels={'value': '% отзывов', 'variable': 'Намерение'}
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)

            # === 3. ADVOCACY & PRICE SENSITIVITY ===
            st.subheader("📣 Сила адвокации и ценовая чувствительность")

            col1, col2 = st.columns(2)

            with col1:
                if 'Advocacy_Strength' in df.columns:
                    st.markdown("**Advocacy Strength (Сила рекомендации)**")

                    advocacy_counts = df['Advocacy_Strength'].value_counts()

                    fig = px.funnel(
                        x=advocacy_counts.values,
                        y=advocacy_counts.index,
                        title="Воронка адвокации"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Метрика NPS-style
                    promoters = advocacy_counts.get('Expert', 0) + advocacy_counts.get('Strong', 0)
                    detractors = advocacy_counts.get('Detractor', 0)
                    nps_style = (promoters - detractors) / len(df) * 100

                    st.metric("Advocacy Score (NPS-style)", f"{nps_style:+.1f}%",
                            help="(Promoters - Detractors) / Total")

            with col2:
                if 'Price_Sensitivity_Tag' in df.columns:
                    st.markdown("**Price Sensitivity (Ценовая чувствительность)**")

                    price_sens_counts = df['Price_Sensitivity_Tag'].value_counts()

                    fig = px.pie(
                        values=price_sens_counts.values,
                        names=price_sens_counts.index,
                        title="Распределение ценовой чувствительности",
                        color=price_sens_counts.index,
                        color_discrete_map={'low': '#51cf66', 'medium': '#ffd43b', 'high': '#ff6b6b'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Связь с Abandonment
                    if 'Abandonment_Tag' in df.columns:
                        high_price_leave = ((df['Price_Sensitivity_Tag'] == 'high') &
                                          (df['Abandonment_Tag'] == 'Leave')).sum()
                        st.warning(f"⚠️ High price + Leave: {high_price_leave:,} отзывов")

            # === 4. AFFECTION TRIGGERS ===
            st.subheader("❤️ Эмоциональные триггеры привязанности")

            if 'Affection_Trigger' in df.columns:
                # Разбираем множественные триггеры
                all_triggers = []
                for triggers_str in df['Affection_Trigger'].dropna():
                    if pd.notna(triggers_str) and triggers_str != 'none':
                        all_triggers.extend(str(triggers_str).split(';'))

                if all_triggers:
                    trigger_counts = pd.Series(all_triggers).value_counts()

                    col1, col2 = st.columns([3, 1])

                    with col1:
                        fig = px.bar(
                            x=trigger_counts.index,
                            y=trigger_counts.values,
                            title="Что вызывает привязанность к продукту",
                            labels={'x': 'Триггер', 'y': 'Упоминаний'},
                            color=trigger_counts.values,
                            color_continuous_scale='Greens'
                        )
                        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.markdown("**Топ триггеры:**")
                        for idx, (trigger, count) in enumerate(trigger_counts.head(5).items(), 1):
                            st.write(f"{idx}. **{trigger}**: {count:,}")

            # === 5. REVIEW PURPOSE & EMOTION ===
            st.subheader("💭 Цель отзыва и эмоции")

            col1, col2 = st.columns(2)

            with col1:
                if 'Review_Purpose' in df.columns:
                    st.markdown("**Review Purpose (Зачем написан отзыв)**")

                    purpose_counts = df['Review_Purpose'].value_counts()

                    fig = px.pie(
                        values=purpose_counts.values,
                        names=purpose_counts.index,
                        title="Распределение целей отзывов",
                        hole=0.4
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Метрики по типам
                    col_a, col_b = st.columns(2)
                    with col_a:
                        complain_count = purpose_counts.get('complain', 0)
                        st.metric("Жалобы", f"{complain_count:,}",
                                f"{complain_count/len(df)*100:.1f}%")
                    with col_b:
                        recommend_count = purpose_counts.get('recommend', 0)
                        st.metric("Рекомендации", f"{recommend_count:,}",
                                f"{recommend_count/len(df)*100:.1f}%")

            with col2:
                if 'Review_Emotion_Class' in df.columns:
                    st.markdown("**Review Emotion (Доминирующая эмоция)**")

                    emotion_counts = df['Review_Emotion_Class'].value_counts()

                    fig = px.bar(
                        x=emotion_counts.index,
                        y=emotion_counts.values,
                        title="Распределение эмоций в отзывах",
                        labels={'x': 'Эмоция', 'y': 'Количество'},
                        color=emotion_counts.index,
                        color_discrete_map={
                            'joy': '#51cf66',
                            'neutral': '#868e96',
                            'surprise': '#ffd43b',
                            'disappointment': '#ff922b',
                            'anger': '#ff6b6b'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # === 6. ALTERNATIVE BRANDS ===
            if 'Alternative_Brand_Mentioned' in df.columns:
                st.subheader("🔀 Упоминание альтернативных брендов")

                alt_brand_counts = df['Alternative_Brand_Mentioned'].value_counts()

                col1, col2, col3 = st.columns([1, 2, 1])

                with col2:
                    fig = px.pie(
                        values=alt_brand_counts.values,
                        names=alt_brand_counts.index,
                        title="Упоминаются ли конкуренты?",
                        color=alt_brand_counts.index,
                        color_discrete_map={'Yes': '#ff6b6b', 'No': '#51cf66'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Связь с Abandonment
                if 'Abandonment_Tag' in df.columns:
                    st.markdown("**Связь упоминания конкурентов с уходом:**")

                    brand_abandon = pd.crosstab(
                        df['Alternative_Brand_Mentioned'],
                        df['Abandonment_Tag'],
                        normalize='index'
                    ) * 100

                    fig = px.bar(
                        brand_abandon.reset_index(),
                        x='Alternative_Brand_Mentioned',
                        y=['Stay', 'Considering_leave', 'Leave'],
                        title="% риска ухода при упоминании конкурентов",
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # === 7. ЭКСПОРТ ===
            st.subheader("📥 Экспорт расширенных данных")

            advanced_cols = [
                'Repurchase_Intent_Tag', 'Abandonment_Tag', 'Misexpectation_Type',
                'Advocacy_Strength', 'Price_Sensitivity_Tag', 'Alternative_Brand_Mentioned',
                'Affection_Trigger', 'Review_Purpose', 'Review_Emotion_Class'
            ]
            advanced_cols = [col for col in advanced_cols if col in df.columns]

            if 'product_name' in df.columns:
                advanced_cols.insert(0, 'product_name')

            export_df = df[advanced_cols].copy()
            csv = export_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                "📥 Скачать расширенные фичи лояльности (CSV)",
                csv,
                "advanced_loyalty_features.csv",
                "text/csv"
            )

    # === TAB BERT: BERT АНАЛИЗ ===
    if 'bert_loyalty_prob' in df.columns:
        with tab_bert:
            st.header("🧠 BERT Анализ Лояльности")

            st.markdown("""
            **О BERT модели:**
            - Обучена на 600 размеченных вручную отзывах
            - Использует псевдолейблинг для улучшения качества
            - Бинарная классификация: лояльный / нелояльный
            - Три порога уверенности: строгий (0.718), средний (0.55), мягкий (0.40)
            """)

            # Статистика
            bert_analyzer = BertLoyaltyAnalyzer()
            bert_stats = bert_analyzer.get_statistics(df)

            st.subheader("📊 Общая статистика BERT анализа")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Всего отзывов", f"{bert_stats['total_reviews']:,}")

            with col2:
                st.metric(
                    "Лояльных (строгий)",
                    f"{bert_stats['loyal_high']['count']:,}",
                    f"{bert_stats['loyal_high']['percent']:.1f}%"
                )

            with col3:
                st.metric(
                    "Лояльных (средний)",
                    f"{bert_stats['loyal_medium']['count']:,}",
                    f"{bert_stats['loyal_medium']['percent']:.1f}%"
                )

            with col4:
                st.metric(
                    "Средняя вероятность",
                    f"{bert_stats['avg_probability']:.3f}"
                )

            # Графики распределения
            st.subheader("📈 Распределение вероятностей BERT")

            col1, col2 = st.columns(2)

            with col1:
                # Гистограмма вероятностей
                fig = px.histogram(
                    df,
                    x='bert_loyalty_prob',
                    nbins=50,
                    title="Распределение вероятностей лояльности (BERT)",
                    color_discrete_sequence=['#4ecdc4']
                )
                fig.add_vline(x=0.718, line_dash="dash", line_color="red", annotation_text="Строгий (0.718)")
                fig.add_vline(x=0.55, line_dash="dash", line_color="orange", annotation_text="Средний (0.55)")
                fig.add_vline(x=0.40, line_dash="dash", line_color="green", annotation_text="Мягкий (0.40)")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Сравнение порогов
                threshold_data = pd.DataFrame({
                    'Порог': ['Строгий\n(0.718)', 'Средний\n(0.55)', 'Мягкий\n(0.40)'],
                    'Лояльных': [
                        bert_stats['loyal_high']['count'],
                        bert_stats['loyal_medium']['count'],
                        bert_stats['loyal_low']['count']
                    ],
                    'Процент': [
                        bert_stats['loyal_high']['percent'],
                        bert_stats['loyal_medium']['percent'],
                        bert_stats['loyal_low']['percent']
                    ]
                })
                fig = px.bar(
                    threshold_data,
                    x='Порог',
                    y='Лояльных',
                    title="Влияние порога на количество лояльных",
                    color='Процент',
                    color_continuous_scale='RdYlGn',
                    text='Лояльных'
                )
                st.plotly_chart(fig, use_container_width=True)

            # Сравнение с keyword методом
            if 'loyalty_score' in df.columns:
                st.subheader("🔬 Сравнение: BERT vs Keyword метод")

                col1, col2 = st.columns(2)

                with col1:
                    # Scatter plot
                    sample_df = df.sample(min(5000, len(df)))
                    fig = px.scatter(
                        sample_df,
                        x='loyalty_score',
                        y='bert_loyalty_prob',
                        color='bert_loyalty_class',
                        opacity=0.5,
                        title="Keyword Loyalty Score vs BERT вероятность",
                        labels={
                            'loyalty_score': 'Keyword Loyalty Score',
                            'bert_loyalty_prob': 'BERT Вероятность',
                            'bert_loyalty_class': 'BERT Класс'
                        },
                        color_discrete_map={'loyal': '#51cf66', 'not_loyal': '#ff6b6b'}
                    )
                    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                                line=dict(color="gray", dash="dash"))
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Корреляция и статистика
                    comparison = compare_with_keyword_method(df)
                    if not comparison.empty:
                        st.markdown("**Сравнительная статистика:**")
                        st.dataframe(comparison, use_container_width=True, hide_index=True)

                    # Корреляция
                    corr = df['bert_loyalty_prob'].corr(df['loyalty_score'])
                    st.metric("Корреляция между методами", f"{corr:.3f}")

                    # Диссонансы
                    high_keyword_low_bert = df[
                        (df['loyalty_score'] > 0.8) &
                        (df['bert_loyalty_prob'] < 0.4)
                    ]
                    low_keyword_high_bert = df[
                        (df['loyalty_score'] < 0.5) &
                        (df['bert_loyalty_prob'] > 0.7)
                    ]

                    st.markdown("**Диссонансы между методами:**")
                    st.write(f"• High Keyword / Low BERT: {len(high_keyword_low_bert):,}")
                    st.write(f"• Low Keyword / High BERT: {len(low_keyword_high_bert):,}")

            # Относительный анализ по продуктам (z-score как у друга)
            if 'product_name' in df.columns:
                st.subheader("🏆 Относительный анализ продуктов (z-score)")

                min_reviews_product = st.slider(
                    "Минимум отзывов на продукт",
                    min_value=20,
                    max_value=200,
                    value=100,
                    step=10,
                    key="bert_min_reviews"
                )

                if st.button("🔄 Рассчитать относительные показатели"):
                    with st.spinner("Расчёт z-score и байесовских баллов..."):
                        product_stats = bert_analyzer.calculate_product_stats(df, min_reviews=min_reviews_product)

                        if not product_stats.empty:
                            st.session_state['bert_product_stats'] = product_stats
                            st.success(f"✅ Проанализировано {len(product_stats)} продуктов")
                        else:
                            st.warning("Нет продуктов с достаточным количеством отзывов")

                # Показываем результаты если есть
                if 'bert_product_stats' in st.session_state:
                    product_stats = st.session_state['bert_product_stats']

                    # Метрики
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Проанализировано продуктов", f"{len(product_stats):,}")

                    with col2:
                        global_loyalty = product_stats['loyal_high'].sum() / product_stats['total_reviews'].sum()
                        st.metric("Средняя лояльность", f"{global_loyalty:.1%}")

                    with col3:
                        best_z = product_stats['z_score'].max()
                        st.metric("Лучший z-score", f"{best_z:.2f}")

                    with col4:
                        above_avg = (product_stats['z_score'] >= 1).sum()
                        st.metric("Выше среднего", f"{above_avg}")

                    # Топ и худшие
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**🏆 Топ-10 по относительной лояльности (z-score):**")
                        top_10 = product_stats.head(10)[
                            ['product_name', 'total_reviews', 'loyalty_rate_high', 'z_score', 'relative_category']
                        ].copy()
                        top_10['loyalty_rate_high'] = top_10['loyalty_rate_high'].apply(lambda x: f"{x:.1%}")
                        top_10['z_score'] = top_10['z_score'].apply(lambda x: f"{x:.2f}")
                        st.dataframe(top_10, use_container_width=True, hide_index=True)

                    with col2:
                        st.markdown("**⚠️ Проблемные продукты (низкий z-score):**")
                        bottom_10 = product_stats.tail(10)[
                            ['product_name', 'total_reviews', 'loyalty_rate_high', 'z_score', 'relative_category']
                        ].copy()
                        bottom_10['loyalty_rate_high'] = bottom_10['loyalty_rate_high'].apply(lambda x: f"{x:.1%}")
                        bottom_10['z_score'] = bottom_10['z_score'].apply(lambda x: f"{x:.2f}")
                        st.dataframe(bottom_10, use_container_width=True, hide_index=True)

                    # График распределения z-scores
                    fig = px.histogram(
                        product_stats,
                        x='z_score',
                        nbins=30,
                        title="Распределение z-scores (относительная лояльность)",
                        color_discrete_sequence=['#4ecdc4']
                    )
                    fig.add_vline(x=0, line_dash="solid", line_color="red", annotation_text="Среднее")
                    fig.add_vline(x=1, line_dash="dash", line_color="green", annotation_text="Выше среднего")
                    fig.add_vline(x=-1, line_dash="dash", line_color="orange", annotation_text="Ниже среднего")
                    st.plotly_chart(fig, use_container_width=True)

                    # Scatter: z-score vs количество отзывов
                    fig = px.scatter(
                        product_stats,
                        x='total_reviews',
                        y='z_score',
                        color='relative_category',
                        size='loyalty_rate_high',
                        hover_name='product_name',
                        title="Z-score vs Количество отзывов",
                        log_x=True,
                        color_discrete_map={
                            '🚀 Выдающийся': '#2ecc71',
                            '📈 Выше среднего': '#27ae60',
                            '📊 Средний': '#f39c12',
                            '⚠️ Ниже среднего': '#e74c3c',
                            '🔥 Проблемный': '#c0392b'
                        }
                    )
                    fig.add_hline(y=0, line_dash="solid", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)

                    # Категории
                    st.markdown("**📊 Распределение по категориям относительной лояльности:**")
                    cat_counts = product_stats['relative_category'].value_counts()
                    cat_df = pd.DataFrame({
                        'Категория': cat_counts.index,
                        'Количество': cat_counts.values,
                        'Процент': (cat_counts.values / len(product_stats) * 100).round(1)
                    })
                    st.dataframe(cat_df, use_container_width=True, hide_index=True)

                    # Экспорт
                    st.markdown("**📥 Экспорт:**")
                    csv = product_stats.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Скачать относительный анализ продуктов (CSV)",
                        csv,
                        "bert_product_stats.csv",
                        "text/csv"
                    )

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
