"""
A/B Тестирование дизайна email-рассылки
Интерактивный дашборд на Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import beta
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="A/B Test Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Загрузка данных
@st.cache_data
def load_data():
    """Загрузка данных с кэшированием"""
    import os
    paths = ['data/results.csv', '../data/results.csv', 'results.csv']
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['date'] = pd.to_datetime(df['date'], dayfirst=True)
            return df
    return None

df = load_data()

if df is None:
    st.error("❌ Файл results.csv не найден!")
    st.stop()

# Боковая панель
with st.sidebar:
    st.title("📊 A/B Test Dashboard")
    st.markdown("---")
    
    # Информация о данных
    st.subheader("📁 Данные")
    st.info(
        f"""
        **Период:** {df['date'].min().strftime('%d.%m.%Y')} - {df['date'].max().strftime('%d.%m.%Y')}
        **Строк:** {len(df):,}
        """
    )
    
    st.markdown("---")
    
    # Фильтры
    st.subheader("🔍 Фильтры")
    
    # Тип пользователя
    user_types = ['Все'] + sorted(df['user_type'].unique().tolist())
    selected_user_type = st.selectbox(
        "Тип пользователя", 
        user_types,
        index=0,
        key='user_type_filter'
    )
    
    # Период
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    date_range = st.date_input(
        "Период",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key='date_range_filter'
    )
    
    st.markdown("---")
    
    # Статус эксперимента
    st.subheader("🎯 Статус")
    
    # Быстрый расчет для статуса
    control = df[df['group'] == 'control']
    treatment = df[df['group'] == 'treatment']
    n_control = len(control)
    n_treatment = len(treatment)
    clicks_control = control['converted'].sum()
    clicks_treatment = treatment['converted'].sum()
    cr_control = clicks_control / n_control
    cr_treatment = clicks_treatment / n_treatment
    
    p_pool = (clicks_control + clicks_treatment) / (n_control + n_treatment)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_control + 1/n_treatment))
    z_stat = (cr_treatment - cr_control) / se
    p_value = 1 - stats.norm.cdf(z_stat)
    
    if p_value < 0.05:
        st.success("✅ Статистически значим")
    else:
        st.error("❌ Не значим")
    
    st.markdown("---")
    
    # Кнопка обновления (на всякий случай)
    if st.button("🔄 Обновить данные", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Применение фильтров к данным
# Фильтруем данные на основе выбора в боковой панели
filtered_df = df.copy()

# Фильтр по типу пользователя
if selected_user_type != 'Все':
    filtered_df = filtered_df[filtered_df['user_type'] == selected_user_type]

# Фильтр по дате
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[
        (filtered_df['date'] >= pd.to_datetime(start_date)) &
        (filtered_df['date'] <= pd.to_datetime(end_date))
    ]

# Расчет метрик для отфильтрованных данных
@st.cache_data
def calculate_metrics(df_filtered):
    """Расчет всех метрик для отфильтрованных данных"""
    df = df_filtered.copy()
    df['week'] = df['date'].dt.isocalendar().week
    
    control = df[df['group'] == 'control']
    treatment = df[df['group'] == 'treatment']
    
    n_control = len(control)
    n_treatment = len(treatment)
    clicks_control = control['converted'].sum()
    clicks_treatment = treatment['converted'].sum()
    
    cr_control = clicks_control / n_control if n_control > 0 else 0
    cr_treatment = clicks_treatment / n_treatment if n_treatment > 0 else 0
    diff_abs = cr_treatment - cr_control
    diff_rel = (cr_treatment / cr_control - 1) * 100 if cr_control > 0 else 0
    
    # Z-test
    p_pool = (clicks_control + clicks_treatment) / (n_control + n_treatment) if (n_control + n_treatment) > 0 else 0.5
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_control + 1/n_treatment)) if n_control > 0 and n_treatment > 0 else 0
    z_stat = (cr_treatment - cr_control) / se if se > 0 else 0
    p_value = 1 - stats.norm.cdf(z_stat)
    
    # Доверительный интервал
    z_crit = stats.norm.ppf(0.975)
    ci_lower = diff_abs - z_crit * se
    ci_upper = diff_abs + z_crit * se
    
    # Байесовская вероятность
    np.random.seed(42)
    a_control, b_control = clicks_control + 1, n_control - clicks_control + 1
    a_treatment, b_treatment = clicks_treatment + 1, n_treatment - clicks_treatment + 1
    post_control = beta.rvs(a_control, b_control, size=100000)
    post_treatment = beta.rvs(a_treatment, b_treatment, size=100000)
    prob_better = np.mean(post_treatment > post_control)
    
    # Ежедневная статистика
    daily_stats = df.groupby(['date', 'group']).agg(
        users=('user_id', 'count'),
        clicks=('converted', 'sum'),
        cr=('converted', 'mean')
    ).reset_index()
    
    daily_pivot = daily_stats.pivot(index='date', columns='group', values='cr')
    daily_diff = daily_pivot['treatment'] - daily_pivot['control'] if 'treatment' in daily_pivot.columns and 'control' in daily_pivot.columns else pd.Series()
    
    # Недельная статистика
    weekly_stats = df.groupby(['week', 'group']).agg(
        users=('user_id', 'count'),
        clicks=('converted', 'sum'),
        cr=('converted', 'mean')
    ).reset_index()
    
    weekly_pivot = weekly_stats.pivot(index='week', columns='group', values='cr')
    weekly_pivot['diff'] = weekly_pivot['treatment'] - weekly_pivot['control'] if 'treatment' in weekly_pivot.columns and 'control' in weekly_pivot.columns else 0
    
    # Сегментация
    segmented = df.groupby(['group', 'user_type'])['converted'].mean().unstack(fill_value=0)
    
    # Баланс
    daily_balance = df.groupby(['date', 'group']).size().unstack(fill_value=0)
    
    # SRM
    total = n_control + n_treatment
    expected = total / 2
    if total > 0 and expected > 0:
        chi2 = ((n_control - expected) ** 2 + (n_treatment - expected) ** 2) / expected
        srm_p_value = 1 - stats.chi2.cdf(chi2, df=1)
    else:
        srm_p_value = 1.0
    
    return {
        'n_control': n_control,
        'n_treatment': n_treatment,
        'clicks_control': clicks_control,
        'clicks_treatment': clicks_treatment,
        'cr_control': cr_control,
        'cr_treatment': cr_treatment,
        'diff_abs': diff_abs,
        'diff_rel': diff_rel,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'prob_better': prob_better,
        'daily_pivot': daily_pivot,
        'daily_diff': daily_diff,
        'weekly_pivot': weekly_pivot,
        'segmented': segmented,
        'daily_balance': daily_balance,
        'srm_p_value': srm_p_value,
        'total_rows': len(df)
    }

metrics = calculate_metrics(filtered_df)

# Функции для графиков
def create_daily_cr_figure(daily_pivot):
    fig = go.Figure()
    
    if 'control' in daily_pivot.columns:
        fig.add_trace(go.Scatter(
            x=daily_pivot.index,
            y=daily_pivot['control'],
            name='Control',
            mode='lines+markers',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=6)
        ))
    
    if 'treatment' in daily_pivot.columns:
        fig.add_trace(go.Scatter(
            x=daily_pivot.index,
            y=daily_pivot['treatment'],
            name='Treatment',
            mode='lines+markers',
            line=dict(color='#E84855', width=3),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title='📈 Динамика конверсии по дням',
        yaxis_title='Конверсия',
        yaxis_tickformat='.0%',
        height=400,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    return fig


def create_daily_diff_figure(daily_diff):
    if len(daily_diff) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Нет данных для отображения", showarrow=False)
        fig.update_layout(height=400)
        return fig
    
    colors = ['#06A77D' if x > 0 else '#E84855' for x in daily_diff]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily_diff.index,
        y=daily_diff,
        marker_color=colors,
        opacity=0.8,
        name='Разница (T-C)'
    ))
    fig.add_hline(y=0, line=dict(color='gray', dash='solid', width=1.5))
    fig.update_layout(
        title='Разница конверсий (Treatment - Control)',
        yaxis_title='Разница, п.п.',
        yaxis_tickformat='+.1%',
        height=400,
        template='plotly_white',
        hovermode='x unified'
    )
    return fig


def create_balance_figure(daily_balance):
    fig = go.Figure()
    
    if 'control' in daily_balance.columns:
        fig.add_trace(go.Bar(
            x=daily_balance.index,
            y=daily_balance['control'],
            name='Control',
            marker_color='#2E86AB',
            opacity=0.6
        ))
    
    if 'treatment' in daily_balance.columns:
        fig.add_trace(go.Bar(
            x=daily_balance.index,
            y=daily_balance['treatment'],
            name='Treatment',
            marker_color='#E84855',
            opacity=0.6
        ))
    
    fig.update_layout(
        title='Баланс групп по дням',
        yaxis_title='Количество пользователей',
        height=350,
        template='plotly_white',
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    return fig


def create_segmentation_figure(segmented):
    fig = go.Figure()
    
    if len(segmented) == 0:
        fig.add_annotation(text="Нет данных для отображения", showarrow=False)
        fig.update_layout(height=350)
        return fig
    
    groups = ['control', 'treatment']
    group_labels = ['Control', 'Treatment']
    colors = {'new': '#F39C12', 'old': '#8E44AD'}
    
    for user_type in segmented.columns:
        values = []
        for group in groups:
            if group in segmented.index and user_type in segmented.columns:
                values.append(segmented.loc[group, user_type])
            else:
                values.append(0)
        
        fig.add_trace(go.Bar(
            x=group_labels,
            y=values,
            name=f'{user_type} users',
            marker_color=colors.get(user_type, '#95A5A6'),
            opacity=0.8,
            text=[f'{v:.2%}' for v in values],
            textposition='outside'
        ))
    
    fig.update_layout(
        title='Стратификация по типу пользователя',
        yaxis_title='Конверсия',
        yaxis_tickformat='.0%',
        height=350,
        template='plotly_white',
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    return fig

# Основной контент
# Заголовок
st.title("📊 A/B Тестирование дизайна email-рассылки")
st.caption("Анализ влияния нового дизайна на конверсию пользователей")

# KPI карточки
st.markdown("### 📈 Ключевые показатели")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Конверсия Control",
        value=f"{metrics['cr_control']:.2%}" if metrics['cr_control'] > 0 else "0.00%",
        delta=f"{metrics['n_control']:,} пользователей",
        delta_color="off"
    )

with col2:
    st.metric(
        label="Конверсия Treatment",
        value=f"{metrics['cr_treatment']:.2%}" if metrics['cr_treatment'] > 0 else "0.00%",
        delta=f"{metrics['n_treatment']:,} пользователей",
        delta_color="off"
    )

with col3:
    st.metric(
        label="Разница (T-C)",
        value=f"{metrics['diff_abs']:+.2%}",
        delta=f"{metrics['diff_rel']:+.1f}%",
        delta_color="normal" if metrics['diff_abs'] > 0 else "inverse"
    )

with col4:
    st.metric(
        label="p-value",
        value=f"{metrics['p_value']:.6f}",
        delta="✅ Значимо" if metrics['p_value'] < 0.05 else "❌ Не значимо",
        delta_color="normal" if metrics['p_value'] < 0.05 else "inverse"
    )

with col5:
    st.metric(
        label="Bayesian Probability",
        value=f"{metrics['prob_better']:.1%}",
        delta="Дизайн лучше контроля",
        delta_color="off"
    )

# Графики
col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(create_daily_cr_figure(metrics['daily_pivot']), use_container_width=True)

with col2:
    st.plotly_chart(create_daily_diff_figure(metrics['daily_diff']), use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(create_balance_figure(metrics['daily_balance']), use_container_width=True)

with col2:
    st.plotly_chart(create_segmentation_figure(metrics['segmented']), use_container_width=True)

# Итоговый вердикт
st.markdown("---")
st.markdown("### 🎯 Итоговый вердикт")

is_significant = metrics['p_value'] < 0.05
is_srm_valid = metrics['srm_p_value'] >= 0.05
has_data = metrics['n_control'] > 0 and metrics['n_treatment'] > 0

if has_data and is_significant and is_srm_valid and metrics['diff_abs'] > 0:
    st.success(
        f"""
        ### ✅ РЕКОМЕНДАЦИЯ: ВНЕДРИТЬ НОВЫЙ ДИЗАЙН
        
        **Обоснование:**
        - 📈 Прирост CTR: **{metrics['diff_abs']:+.2%} п.п. ({metrics['diff_rel']:+.1f}%)**
        - 📊 Статистическая значимость: **p = {metrics['p_value']:.6f}** ✅
        - 📏 Доверительный интервал: **[{metrics['ci_lower']:+.2%}, {metrics['ci_upper']:+.2%}]**
        - 🎲 Байесовская вероятность: **{metrics['prob_better']:.1%}**
        - 🛡️ SRM тест: **Пройден** ✅
        
        **Ожидаемый бизнес-эффект:** окупаемость разработки менее чем за 6 месяцев.
        """
    )
elif not has_data:
    st.warning("⚠️ Нет данных для отображения. Измените фильтры.")
elif not is_srm_valid:
    st.error(f"### ❌ ЭКСПЕРИМЕНТ НЕВАЛИДЕН\n\n**Причина:** SRM тест не пройден (p = {metrics['srm_p_value']:.6f})")
elif not is_significant:
    st.warning(f"### ⚠️ ЭФФЕКТ НЕ ПОДТВЕРЖДЕН\n\n**Причина:** p-value = {metrics['p_value']:.6f} >= 0.05")
else:
    st.info("### ℹ️ ЭФФЕКТ НЕ ОПРЕДЕЛЕН\n\nРекомендуется дополнительный анализ.")

# Футер
st.markdown("---")
st.caption(f"📊 Данные: {df['date'].min().strftime('%d.%m.%Y')} - {df['date'].max().strftime('%d.%m.%Y')} • Всего: {len(df):,} записей")            
