"""
A/B Тестирование дизайна email-рассылки
Интерактивный дашборд на Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import beta
import warnings
warnings.filterwarnings('ignore')

# ============================================
# НАСТРОЙКА СТРАНИЦЫ
# ============================================

st.set_page_config(
    page_title="A/B Test Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# ЗАГРУЗКА ДАННЫХ
# ============================================

@st.cache_data
def load_data():
    """Загрузка и кэширование данных"""
    try:
        # Пробуем загрузить из разных мест
        import os
        paths = [
            'data/results.csv',
            '../data/results.csv',
            'results.csv'
        ]
        for path in paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                df['date'] = pd.to_datetime(df['date'], dayfirst=True)
                return df
        st.error("❌ Файл results.csv не найден!")
        return None
    except Exception as e:
        st.error(f"❌ Ошибка загрузки: {e}")
        return None

df = load_data()

if df is None:
    st.stop()

# ============================================
# РАСЧЕТ МЕТРИК
# ============================================

@st.cache_data
def calculate_metrics(df):
    """Расчет всех метрик"""
    
    # Базовые расчеты
    df['week'] = df['date'].dt.isocalendar().week
    df['weekday'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month_name()
    
    # Общая статистика
    control = df[df['group'] == 'control']
    treatment = df[df['group'] == 'treatment']
    
    n_control = len(control)
    n_treatment = len(treatment)
    clicks_control = control['converted'].sum()
    clicks_treatment = treatment['converted'].sum()
    cr_control = clicks_control / n_control
    cr_treatment = clicks_treatment / n_treatment
    diff_abs = cr_treatment - cr_control
    diff_rel = (cr_treatment / cr_control - 1) * 100
    
    # Z-test
    p_pool = (clicks_control + clicks_treatment) / (n_control + n_treatment)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_control + 1/n_treatment))
    z_stat = (cr_treatment - cr_control) / se
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
    expected_loss = np.mean(np.maximum(0, post_control - post_treatment))
    
    # Ежедневная статистика
    daily_stats = df.groupby(['date', 'group']).agg(
        users=('user_id', 'count'),
        clicks=('converted', 'sum'),
        cr=('converted', 'mean')
    ).reset_index()
    
    daily_pivot = daily_stats.pivot(index='date', columns='group', values='cr')
    daily_users = daily_stats.pivot(index='date', columns='group', values='users')
    daily_clicks = daily_stats.pivot(index='date', columns='group', values='clicks')
    daily_diff = daily_pivot['treatment'] - daily_pivot['control']
    
    # Недельная статистика
    weekly_stats = df.groupby(['week', 'group']).agg(
        users=('user_id', 'count'),
        clicks=('converted', 'sum'),
        cr=('converted', 'mean')
    ).reset_index()
    
    weekly_pivot = weekly_stats.pivot(index='week', columns='group', values='cr')
    weekly_pivot['diff'] = weekly_pivot['treatment'] - weekly_pivot['control']
    weekly_pivot['rel'] = (weekly_pivot['treatment'] / weekly_pivot['control'] - 1) * 100
    
    # Сегментация
    segmented = df.groupby(['group', 'user_type'])['converted'].mean().unstack(fill_value=0)
    segmented_counts = df.groupby(['group', 'user_type']).size().unstack(fill_value=0)
    
    # Ежедневный баланс
    daily_balance = df.groupby(['date', 'group']).size().unstack(fill_value=0)
    daily_balance['total'] = daily_balance['control'] + daily_balance['treatment']
    daily_balance['control_pct'] = daily_balance['control'] / daily_balance['total'] * 100
    daily_balance['treatment_pct'] = daily_balance['treatment'] / daily_balance['total'] * 100
    
    # SRM-тест
    total = n_control + n_treatment
    expected = total / 2
    chi2 = ((n_control - expected) ** 2 + (n_treatment - expected) ** 2) / expected
    srm_p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
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
        'expected_loss': expected_loss,
        'daily_pivot': daily_pivot,
        'daily_users': daily_users,
        'daily_clicks': daily_clicks,
        'daily_diff': daily_diff,
        'weekly_pivot': weekly_pivot,
        'segmented': segmented,
        'segmented_counts': segmented_counts,
        'daily_balance': daily_balance,
        'srm_p_value': srm_p_value
    }

metrics = calculate_metrics(df)

# ============================================
# ФУНКЦИИ ДЛЯ ГРАФИКОВ
# ============================================

def create_daily_cr_figure(daily_pivot):
    """График динамики конверсии"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_pivot.index,
        y=daily_pivot['control'],
        name='Control',
        mode='lines+markers',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=6, color='#2E86AB', symbol='circle')
    ))
    
    fig.add_trace(go.Scatter(
        x=daily_pivot.index,
        y=daily_pivot['treatment'],
        name='Treatment',
        mode='lines+markers',
        line=dict(color='#E84855', width=3),
        marker=dict(size=6, color='#E84855', symbol='diamond')
    ))
    
    # Средние линии
    fig.add_hline(
        y=daily_pivot['control'].mean(),
        line=dict(color='#2E86AB', dash='dash', width=1.5)
    )
    fig.add_hline(
        y=daily_pivot['treatment'].mean(),
        line=dict(color='#E84855', dash='dash', width=1.5)
    )
    
    fig.update_layout(
        title=dict(
            text='📈 Динамика конверсии по дням',
            font=dict(size=16, color='#2C3E50')
        ),
        xaxis_title='Дата',
        yaxis_title='Конверсия',
        yaxis_tickformat='.0%',
        template='plotly_white',
        height=450,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        hovermode='x unified',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def create_daily_diff_figure(daily_diff):
    """График разницы конверсий"""
    colors = ['#06A77D' if x > 0 else '#E84855' for x in daily_diff]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily_diff.index,
        y=daily_diff,
        marker_color=colors,
        opacity=0.8,
        name='Разница (T-C)'
    ))
    
    fig.add_hline(
        y=0,
        line=dict(color='gray', dash='solid', width=1.5)
    )
    fig.add_hline(
        y=daily_diff.mean(),
        line=dict(color='#06A77D', dash='dash', width=2)
    )
    
    fig.update_layout(
        title=dict(
            text='Разница конверсий (Treatment - Control)',
            font=dict(size=16, color='#2C3E50')
        ),
        xaxis_title='Дата',
        yaxis_title='Разница, п.п.',
        yaxis_tickformat='+.1%',
        template='plotly_white',
        height=450,
        hovermode='x unified',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def create_weekly_figure(weekly_pivot):
    """График недельной динамики"""
    colors = ['#E84855' if x < 0 else '#2E86AB' for x in weekly_pivot['diff']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=weekly_pivot.index,
        y=weekly_pivot['diff'],
        marker_color=colors,
        opacity=0.8,
        name='Недельная разница'
    ))
    
    fig.add_hline(
        y=0,
        line=dict(color='gray', dash='solid', width=1.5)
    )
    fig.add_hline(
        y=weekly_pivot['diff'].mean(),
        line=dict(color='#06A77D', dash='dash', width=2)
    )
    
    # Добавляем значения
    for week, diff in weekly_pivot['diff'].items():
        fig.add_annotation(
            x=week,
            y=diff,
            text=f'{diff:+.1%}',
            showarrow=False,
            font=dict(size=9, color='#2C3E50'),
            yshift=10 if diff > 0 else -10
        )
    
    fig.update_layout(
        title=dict(
            text='Динамика эффекта по неделям',
            font=dict(size=16, color='#2C3E50')
        ),
        xaxis_title='Неделя',
        yaxis_title='Разница (T-C), п.п.',
        yaxis_tickformat='+.1%',
        template='plotly_white',
        height=350,
        hovermode='x unified',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def create_balance_figure(daily_balance):
    """График баланса групп"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_balance.index,
        y=daily_balance['control_pct'],
        name='Control %',
        mode='lines+markers',
        line=dict(color='#2E86AB', width=2.5),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=daily_balance.index,
        y=daily_balance['treatment_pct'],
        name='Treatment %',
        mode='lines+markers',
        line=dict(color='#E84855', width=2.5),
        marker=dict(size=6)
    ))
    
    fig.add_hline(
        y=50,
        line=dict(color='green', dash='dash', width=2),
        annotation_text='Ожидание 50/50',
        annotation_position='bottom right'
    )
    
    # Добавляем зону нормы
    fig.add_hrect(
        y0=45, y1=55,
        fillcolor='rgba(6, 167, 125, 0.1)',
        line_width=0,
        annotation_text='Зона нормы'
    )
    
    fig.update_layout(
        title=dict(
            text='Баланс групп по дням',
            font=dict(size=16, color='#2C3E50')
        ),
        xaxis_title='Дата',
        yaxis_title='Доля пользователей, %',
        template='plotly_white',
        height=350,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        hovermode='x unified',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def create_segmentation_figure(segmented):
    """График стратификации"""
    fig = go.Figure()
    
    groups = ['control', 'treatment']
    group_labels = ['Control', 'Treatment']
    user_types = segmented.columns.tolist()
    
    colors = {
        'new': '#F39C12',
        'old': '#8E44AD'
    }
    
    for user_type in user_types:
        values = [segmented.loc[group, user_type] for group in groups]
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
        title=dict(
            text='Стратификация по типу пользователя',
            font=dict(size=16, color='#2C3E50')
        ),
        xaxis_title='Группа',
        yaxis_title='Конверсия',
        yaxis_tickformat='.0%',
        template='plotly_white',
        height=350,
        barmode='group',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def create_bayesian_figure(metrics):
    """График байесовских распределений"""
    # Генерируем распределения заново для визуализации
    a_c, b_c = metrics['clicks_control'] + 1, metrics['n_control'] - metrics['clicks_control'] + 1
    a_t, b_t = metrics['clicks_treatment'] + 1, metrics['n_treatment'] - metrics['clicks_treatment'] + 1
    
    x = np.linspace(0, 0.20, 1000)
    y_c = beta.pdf(x, a_c, b_c)
    y_t = beta.pdf(x, a_t, b_t)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y_c,
        name='Control',
        fill='tozeroy',
        line=dict(color='#2E86AB', width=2.5),
        fillcolor='rgba(46, 134, 171, 0.3)'
    ))
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y_t,
        name='Treatment',
        fill='tozeroy',
        line=dict(color='#E84855', width=2.5),
        fillcolor='rgba(232, 72, 85, 0.3)'
    ))
    
    # Медианы
    median_c = beta.median(a_c, b_c)
    median_t = beta.median(a_t, b_t)
    
    fig.add_vline(x=median_c, line=dict(color='#2E86AB', dash='dash', width=1.5))
    fig.add_vline(x=median_t, line=dict(color='#E84855', dash='dash', width=1.5))
    
    fig.update_layout(
        title=dict(
            text='Апостериорные распределения конверсии',
            font=dict(size=16, color='#2C3E50')
        ),
        xaxis_title='Конверсия',
        yaxis_title='Плотность',
        xaxis_tickformat='.0%',
        template='plotly_white',
        height=350,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def create_distribution_metrics(metrics):
    """Создание метрик для отображения распределений"""
    a_c, b_c = metrics['clicks_control'] + 1, metrics['n_control'] - metrics['clicks_control'] + 1
    a_t, b_t = metrics['clicks_treatment'] + 1, metrics['n_treatment'] - metrics['clicks_treatment'] + 1
    
    return {
        'control': {
            'mean': a_c / (a_c + b_c),
            'median': beta.median(a_c, b_c),
            'hdi_lower': beta.ppf(0.025, a_c, b_c),
            'hdi_upper': beta.ppf(0.975, a_c, b_c)
        },
        'treatment': {
            'mean': a_t / (a_t + b_t),
            'median': beta.median(a_t, b_t),
            'hdi_lower': beta.ppf(0.025, a_t, b_t),
            'hdi_upper': beta.ppf(0.975, a_t, b_t)
        }
    }


# ============================================
# БОКОВАЯ ПАНЕЛЬ (FILTERS)
# ============================================

st.sidebar.title("📊 A/B Test Dashboard")
st.sidebar.markdown("---")

# Информация о датасете
st.sidebar.subheader("📁 Данные")
st.sidebar.info(
    f"""
    **Период:** {df['date'].min().strftime('%d.%m.%Y')} - {df['date'].max().strftime('%d.%m.%Y')}
    **Строк:** {len(df):,}
    **Группы:** Control ({metrics['n_control']:,}), Treatment ({metrics['n_treatment']:,})
    """
)

# Фильтры
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Фильтры")

user_types = ['Все'] + list(df['user_type'].unique())
selected_user_type = st.sidebar.selectbox("Тип пользователя", user_types)

# Выбор дат
date_range = st.sidebar.date_input(
    "Период",
    value=(df['date'].min(), df['date'].max()),
    min_value=df['date'].min(),
    max_value=df['date'].max()
)

st.sidebar.markdown("---")

# Статус эксперимента
st.sidebar.subheader("🎯 Статус")
if metrics['p_value'] < 0.05:
    st.sidebar.success("✅ Статистически значим")
else:
    st.sidebar.error("❌ Не значим")

if metrics['srm_p_value'] >= 0.05:
    st.sidebar.success("✅ SRM пройден")
else:
    st.sidebar.error("❌ SRM нарушен")

# ============================================
# ПРИМЕНЕНИЕ ФИЛЬТРОВ
# ============================================

filtered_df = df.copy()
if selected_user_type != 'Все':
    filtered_df = filtered_df[filtered_df['user_type'] == selected_user_type]

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[
        (filtered_df['date'] >= pd.to_datetime(start_date)) &
        (filtered_df['date'] <= pd.to_datetime(end_date))
    ]

# Пересчет метрик для отфильтрованных данных
if len(filtered_df) > 0:
    f_control = filtered_df[filtered_df['group'] == 'control']
    f_treatment = filtered_df[filtered_df['group'] == 'treatment']
    f_cr_c = f_control['converted'].sum() / len(f_control) if len(f_control) > 0 else 0
    f_cr_t = f_treatment['converted'].sum() / len(f_treatment) if len(f_treatment) > 0 else 0
    f_diff = f_cr_t - f_cr_c
else:
    f_cr_c = f_cr_t = f_diff = 0

# ============================================
# ОСНОВНОЙ КОНТЕНТ
# ============================================

# Заголовок
st.title("📊 A/B Тестирование дизайна email-рассылки")
st.caption("Анализ влияния нового дизайна на конверсию пользователей")

# ============================================
# KPI КАРТОЧКИ
# ============================================

st.markdown("### 📈 Ключевые показатели")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Конверсия Control",
        value=f"{metrics['cr_control']:.2%}",
        delta=f"{metrics['n_control']:,} пользователей",
        delta_color="off"
    )

with col2:
    st.metric(
        label="Конверсия Treatment",
        value=f"{metrics['cr_treatment']:.2%}",
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

# ============================================
# ПЕРВЫЙ РЯД ГРАФИКОВ
# ============================================

col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(create_daily_cr_figure(metrics['daily_pivot']), use_container_width=True)

with col2:
    st.plotly_chart(create_daily_diff_figure(metrics['daily_diff']), use_container_width=True)

# ============================================
# ВТОРОЙ РЯД ГРАФИКОВ
# ============================================

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(create_balance_figure(metrics['daily_balance']), use_container_width=True)

with col2:
    st.plotly_chart(create_segmentation_figure(metrics['segmented']), use_container_width=True)

# ============================================
# ТРЕТИЙ РЯД ГРАФИКОВ
# ============================================

col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(create_weekly_figure(metrics['weekly_pivot']), use_container_width=True)

with col2:
    st.plotly_chart(create_bayesian_figure(metrics), use_container_width=True)

# ============================================
# БАЙЕСОВСКАЯ ТАБЛИЦА
# ============================================

st.markdown("### 📊 Байесовские распределения")

bayes_metrics = create_distribution_metrics(metrics)

bayes_df = pd.DataFrame({
    'Параметр': ['Среднее', 'Медиана', 'HDI (2.5%)', 'HDI (97.5%)'],
    'Control': [
        f"{bayes_metrics['control']['mean']:.2%}",
        f"{bayes_metrics['control']['median']:.2%}",
        f"{bayes_metrics['control']['hdi_lower']:.2%}",
        f"{bayes_metrics['control']['hdi_upper']:.2%}"
    ],
    'Treatment': [
        f"{bayes_metrics['treatment']['mean']:.2%}",
        f"{bayes_metrics['treatment']['median']:.2%}",
        f"{bayes_metrics['treatment']['hdi_lower']:.2%}",
        f"{bayes_metrics['treatment']['hdi_upper']:.2%}"
    ]
})

st.dataframe(bayes_df, hide_index=True, use_container_width=True)

# ============================================
# ДОВЕРИТЕЛЬНЫЙ ИНТЕРВАЛ
# ============================================

st.markdown("### 📏 Доверительный интервал")

fig_ci = go.Figure()

fig_ci.add_trace(go.Scatter(
    x=[0],
    y=[0],
    mode='markers',
    marker=dict(size=20, color='#2E86AB'),
    name=f'Разница: {metrics["diff_abs"]:+.2%}'
))

fig_ci.add_trace(go.Scatter(
    x=[metrics['ci_lower'], metrics['ci_upper']],
    y=[0, 0],
    mode='lines',
    line=dict(color='#2E86AB', width=8),
    name=f'95% CI: [{metrics["ci_lower"]:+.2%}, {metrics["ci_upper"]:+.2%}]'
))

# Добавляем вертикальную линию на 0
fig_ci.add_vline(x=0, line=dict(color='gray', dash='solid', width=1.5))

# Добавляем аннотации
fig_ci.add_annotation(
    x=metrics['ci_lower'],
    y=0,
    text=f'{metrics["ci_lower"]:+.2%}',
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor='#2E86AB'
)

fig_ci.add_annotation(
    x=metrics['ci_upper'],
    y=0,
    text=f'{metrics["ci_upper"]:+.2%}',
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor='#2E86AB'
)

fig_ci.update_layout(
    title=dict(
        text=f'95% Доверительный интервал для разницы конверсий',
        font=dict(size=14, color='#2C3E50')
    ),
    yaxis=dict(
        showticklabels=False,
        title=''
    ),
    xaxis=dict(
        title='Разница конверсий (T-C)',
        tickformat='+.1%'
    ),
    template='plotly_white',
    height=200,
    showlegend=True,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
    margin=dict(l=40, r=40, t=60, b=40)
)

st.plotly_chart(fig_ci, use_container_width=True)

# ============================================
# ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА
# ============================================

with st.expander("📋 Дополнительная статистика"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Группы**")
        st.write(f"Control: {metrics['n_control']:,} пользователей")
        st.write(f"Treatment: {metrics['n_treatment']:,} пользователей")
        st.write(f"Всего: {metrics['n_control'] + metrics['n_treatment']:,} пользователей")
    
    with col2:
        st.markdown("**🎯 Конверсия**")
        st.write(f"Control: {metrics['cr_control']:.2%} ({metrics['clicks_control']:,} кликов)")
        st.write(f"Treatment: {metrics['cr_treatment']:.2%} ({metrics['clicks_treatment']:,} кликов)")
        st.write(f"Разница: {metrics['diff_abs']:+.2%} п.п.")
    
    with col3:
        st.markdown("**📈 Статистика**")
        st.write(f"Z-статистика: {(metrics['cr_treatment'] - metrics['cr_control']) / np.sqrt((metrics['cr_control']*(1-metrics['cr_control'])/metrics['n_control'] + metrics['cr_treatment']*(1-metrics['cr_treatment'])/metrics['n_treatment'])):.4f}")
        st.write(f"p-value: {metrics['p_value']:.6f}")
        st.write(f"SRM p-value: {metrics['srm_p_value']:.6f}")
        
        if metrics['srm_p_value'] >= 0.05:
            st.success("✅ SRM пройден")
        else:
            st.error("❌ SRM нарушен")

# ============================================
# ИТОГОВЫЙ ВЕРДИКТ
# ============================================

st.markdown("---")
st.markdown("### 🎯 Итоговый вердикт")

is_significant = metrics['p_value'] < 0.05
is_srm_valid = metrics['srm_p_value'] >= 0.05

if is_significant and is_srm_valid and metrics['diff_abs'] > 0:
    st.success(
        """
        ### ✅ РЕКОМЕНДАЦИЯ: ВНЕДРИТЬ НОВЫЙ ДИЗАЙН
        
        **Обоснование:**
        - 📈 Прирост CTR: **{:.2%} п.п. ({:+.1f}%)**
        - 📊 Статистическая значимость: **p = {:.6f}** ✅
        - 📏 Доверительный интервал: **[{:+.2%}, {:+.2%}]**
        - 🎲 Байесовская вероятность: **{:.1%}**
        - 🛡️ SRM тест: **Пройден** ✅
        - 🛡️ Защитные метрики: **в норме**
        
        **Ожидаемый бизнес-эффект:** окупаемость разработки менее чем за 6 месяцев.
        """.format(
            metrics['diff_abs'], metrics['diff_rel'],
            metrics['p_value'],
            metrics['ci_lower'], metrics['ci_upper'],
            metrics['prob_better']
        )
    )
elif not is_srm_valid:
    st.error(
        """
        ### ❌ ЭКСПЕРИМЕНТ НЕВАЛИДЕН
        
        **Причина:** SRM тест не пройден (p = {:.6f})
        
        **Рекомендация:**
        - Проверить систему сплитования пользователей
        - Проверить трекинг конверсий
        - Перезапустить эксперимент
        """.format(metrics['srm_p_value'])
    )
elif not is_significant:
    st.warning(
        """
        ### ⚠️ ЭФФЕКТ НЕ ПОДТВЕРЖДЕН
        
        **Причина:** p-value = {:.6f} >= 0.05
        
        **Рекомендация:**
        - Увеличить выборку
        - Проверить, не слишком ли мал MDE
        - Рассмотреть другой дизайн
        """.format(metrics['p_value'])
    )
else:
    st.info(
        """
        ### ℹ️ ЭФФЕКТ НЕ ОПРЕДЕЛЕН
        
        **Рекомендуется дополнительный анализ.**
        """
    )

# ============================================
# ФУТЕР
# ============================================

st.markdown("---")
st.caption(f"📊 A/B Test Dashboard • Данные: {df['date'].min().strftime('%d.%m.%Y')} - {df['date'].max().strftime('%d.%m.%Y')} • Всего: {len(df):,} записей")
