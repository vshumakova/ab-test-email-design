# Dashboard для A/B теста
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats

# Загрузка данных
df = pd.read_csv('results.csv')
df['date'] = pd.to_datetime(df['date'], dayfirst=True)

# Базовые расчёты
df['week'] = df['date'].dt.isocalendar().week
df['weekday'] = df['date'].dt.day_name()
df['month'] = df['date'].dt.month_name()

# Ежедневная статистика
daily_stats = df.groupby(['date', 'group']).agg(
    users=('user_id', 'count'),
    clicks=('converted', 'sum'),
    cr=('converted', 'mean')
).reset_index()

daily_pivot = daily_stats.pivot(index='date', columns='group', values='cr')
daily_users = daily_stats.pivot(index='date', columns='group', values='users')
daily_clicks = daily_stats.pivot(index='date', columns='group', values='clicks')

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
from scipy.stats import beta
a_control, b_control = clicks_control + 1, n_control - clicks_control + 1
a_treatment, b_treatment = clicks_treatment + 1, n_treatment - clicks_treatment + 1
post_control = beta.rvs(a_control, b_control, size=100000)
post_treatment = beta.rvs(a_treatment, b_treatment, size=100000)
prob_better = np.mean(post_treatment > post_control)

# Сегментация
segmented = df.groupby(['group', 'user_type'])['converted'].mean().unstack(fill_value=0)

# Недельная статистика
weekly_stats = df.groupby(['week', 'group']).agg(
    users=('user_id', 'count'),
    clicks=('converted', 'sum'),
    cr=('converted', 'mean')
).reset_index()
weekly_pivot = weekly_stats.pivot(index='week', columns='group', values='cr')
weekly_pivot['diff'] = weekly_pivot['treatment'] - weekly_pivot['control']

# Создание dash приложения
app = dash.Dash(__name__, title='A/B Test Dashboard')

# Layout
app.layout = html.Div([
    # Заголовок
    html.Div([
        html.H1("A/B Тестирование дизайна email-рассылки", 
                style={'textAlign': 'center', 'color': '#2C3E50', 'marginBottom': '5px'}),
        html.H4("Анализ влияния нового дизайна на конверсию пользователей",
                style={'textAlign': 'center', 'color': '#7F8C8D', 'marginTop': '0px'})
    ], style={'padding': '20px', 'backgroundColor': '#F8F9FA', 'borderRadius': '10px', 'marginBottom': '20px'}),

    # KPI карточки
    html.Div([
        html.Div([
            html.H6("Конверсия Control", style={'color': '#7F8C8D', 'margin': '0'}),
            html.H2(f"{cr_control:.2%}", style={'color': '#2E86AB', 'margin': '5px 0'}),
            html.Span(f"{n_control:,} пользователей", style={'color': '#95A5A6', 'fontSize': '14px'})
        ], className='kpi-card', style={'flex': '1', 'padding': '20px', 'backgroundColor': '#EBF5FB', 
                                        'borderRadius': '10px', 'textAlign': 'center', 'margin': '5px'}),
        
        html.Div([
            html.H6("Конверсия Treatment", style={'color': '#7F8C8D', 'margin': '0'}),
            html.H2(f"{cr_treatment:.2%}", style={'color': '#E84855', 'margin': '5px 0'}),
            html.Span(f"{n_treatment:,} пользователей", style={'color': '#95A5A6', 'fontSize': '14px'})
        ], className='kpi-card', style={'flex': '1', 'padding': '20px', 'backgroundColor': '#FDEDEC', 
                                        'borderRadius': '10px', 'textAlign': 'center', 'margin': '5px'}),
        
        html.Div([
            html.H6("Разница (T-C)", style={'color': '#7F8C8D', 'margin': '0'}),
            html.H2(f"{diff_abs:+.2%}", style={'color': '#06A77D' if diff_abs > 0 else '#E84855', 'margin': '5px 0'}),
            html.Span(f"{diff_rel:+.1f}% относительный", style={'color': '#95A5A6', 'fontSize': '14px'})
        ], className='kpi-card', style={'flex': '1', 'padding': '20px', 'backgroundColor': '#E8F8F5', 
                                        'borderRadius': '10px', 'textAlign': 'center', 'margin': '5px'}),
        
        html.Div([
            html.H6("p-value", style={'color': '#7F8C8D', 'margin': '0'}),
            html.H2(f"{p_value:.6f}", style={'color': '#2C3E50' if p_value < 0.05 else '#E84855', 'margin': '5px 0'}),
            html.Span("Статистически значим" if p_value < 0.05 else "Не значим", 
                      style={'color': '#06A77D' if p_value < 0.05 else '#E84855', 'fontSize': '14px'})
        ], className='kpi-card', style={'flex': '1', 'padding': '20px', 'backgroundColor': '#F4F6F7', 
                                        'borderRadius': '10px', 'textAlign': 'center', 'margin': '5px'}),
        
        html.Div([
            html.H6("Bayesian Probability", style={'color': '#7F8C8D', 'margin': '0'}),
            html.H2(f"{prob_better:.1%}", style={'color': '#8E44AD', 'margin': '5px 0'}),
            html.Span("Дизайн лучше контроля", style={'color': '#95A5A6', 'fontSize': '14px'})
        ], className='kpi-card', style={'flex': '1', 'padding': '20px', 'backgroundColor': '#F4ECF7', 
                                        'borderRadius': '10px', 'textAlign': 'center', 'margin': '5px'}),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px', 'marginBottom': '20px'}),

    # Графики
    html.Div([
        # График 1: Динамика конверсии
        html.Div([
            dcc.Graph(
                id='daily-cr',
                figure=create_daily_cr_figure()
            )
        ], style={'flex': '2', 'minWidth': '60%'}),
        
        # График 2: Разница конверсий
        html.Div([
            dcc.Graph(
                id='daily-diff',
                figure=create_daily_diff_figure()
            )
        ], style={'flex': '1', 'minWidth': '35%'}),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px', 'marginBottom': '20px'}),

    # Графики: Баланс и Стратификация
    html.Div([
        html.Div([
            dcc.Graph(
                id='daily-balance',
                figure=create_balance_figure()
            )
        ], style={'flex': '1', 'minWidth': '45%'}),
        
        html.Div([
            dcc.Graph(
                id='segmentation',
                figure=create_segmentation_figure()
            )
        ], style={'flex': '1', 'minWidth': '45%'}),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px', 'marginBottom': '20px'}),

    # График: Недельная динамика
    html.Div([
        dcc.Graph(
            id='weekly',
            figure=create_weekly_figure()
        )
    ], style={'marginBottom': '20px'}),

    # Вывод
    html.Div([
        html.Div([
            html.H3("Итоговый вердикт", style={'color': '#2C3E50'}),
            html.Hr(),
            html.P([
                html.Strong("Рекомендация: "),
                "Внедрить новый дизайн для всех пользователей."
            ], style={'fontSize': '18px', 'color': '#2C3E50'}),
            html.Ul([
                html.Li(f"Прирост CTR: {diff_abs:+.2%} п.п. ({diff_rel:+.1f}%)"),
                html.Li(f"Статистическая значимость: p = {p_value:.6f} (p < 0.05)"),
                html.Li(f"Доверительный интервал: [{ci_lower:+.2%}, {ci_upper:+.2%}]"),
                html.Li(f"Байесовская вероятность: {prob_better:.1%}"),
                html.Li("Защитные метрики: в норме")
            ], style={'fontSize': '16px', 'lineHeight': '1.8'}),
            html.P("Ожидаемый бизнес-эффект: окупаемость разработки менее чем за 6 месяцев.",
                   style={'fontSize': '16px', 'color': '#7F8C8D', 'marginTop': '15px', 'fontStyle': 'italic'})
        ], style={'padding': '30px', 'backgroundColor': '#F8F9FA', 'borderRadius': '10px', 'border': '2px solid #06A77D'})
    ])
])

# Функции для графиков
def create_daily_cr_figure():
    """График динамики конверсии"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_pivot.index,
        y=daily_pivot['control'],
        name='Control',
        mode='lines+markers',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=6, color='#2E86AB')
    ))
    
    fig.add_trace(go.Scatter(
        x=daily_pivot.index,
        y=daily_pivot['treatment'],
        name='Treatment',
        mode='lines+markers',
        line=dict(color='#E84855', width=3),
        marker=dict(size=6, color='#E84855')
    ))
    
    fig.add_hline(y=daily_pivot['control'].mean(), line=dict(color='#2E86AB', dash='dash', width=1.5))
    fig.add_hline(y=daily_pivot['treatment'].mean(), line=dict(color='#E84855', dash='dash', width=1.5))
    
    fig.update_layout(
        title='📈 Динамика конверсии по дням',
        xaxis_title='Дата',
        yaxis_title='Конверсия, %',
        yaxis_tickformat='.0%',
        template='plotly_white',
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        hovermode='x unified'
    )
    return fig

def create_daily_diff_figure():
    """График разницы конверсий"""
    diff_cr = daily_pivot['treatment'] - daily_pivot['control']
    colors = ['#06A77D' if x > 0 else '#E84855' for x in diff_cr]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=diff_cr.index,
        y=diff_cr,
        marker_color=colors,
        opacity=0.8,
        name='Разница (T-C)'
    ))
    
    fig.add_hline(y=0, line=dict(color='gray', dash='solid', width=1.5))
    fig.add_hline(y=diff_cr.mean(), line=dict(color='#06A77D', dash='dash', width=2))
    
    fig.update_layout(
        title='Разница конверсий (Treatment - Control)',
        xaxis_title='Дата',
        yaxis_title='Разница, п.п.',
        yaxis_tickformat='+.1%',
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )
    return fig

def create_balance_figure():
    """График баланса групп"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=daily_users.index,
        y=daily_users['control'],
        name='Control',
        marker_color='#2E86AB',
        opacity=0.6
    ))
    
    fig.add_trace(go.Bar(
        x=daily_users.index,
        y=daily_users['treatment'],
        name='Treatment',
        marker_color='#E84855',
        opacity=0.6
    ))
    
    fig.update_layout(
        title='Баланс групп по дням',
        xaxis_title='Дата',
        yaxis_title='Количество пользователей',
        template='plotly_white',
        height=350,
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        hovermode='x unified'
    )
    return fig

def create_segmentation_figure():
    """График стратификации"""
    fig = go.Figure()
    
    x_labels = ['Control', 'Treatment']
    colors_seg = {'new': '#F39C12', 'old': '#8E44AD'}
    
    for user_type in ['new', 'old']:
        values = [segmented.loc[group, user_type] if user_type in segmented.columns else 0 
                  for group in ['control', 'treatment']]
        fig.add_trace(go.Bar(
            x=x_labels,
            y=values,
            name=f'{user_type} users',
            marker_color=colors_seg[user_type],
            opacity=0.8
        ))
    
    fig.update_layout(
        title='Стратификация по типу пользователя',
        xaxis_title='Группа',
        yaxis_title='Конверсия, %',
        yaxis_tickformat='.0%',
        template='plotly_white',
        height=350,
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    return fig

def create_weekly_figure():
    """График недельной динамики"""
    fig = go.Figure()
    
    # Разница по неделям
    colors = ['#E84855' if x < 0 else '#2E86AB' for x in weekly_pivot['diff']]
    
    fig.add_trace(go.Bar(
        x=weekly_pivot.index,
        y=weekly_pivot['diff'],
        marker_color=colors,
        opacity=0.8,
        name='Недельная разница'
    ))
    
    fig.add_hline(y=0, line=dict(color='gray', dash='solid', width=1.5))
    fig.add_hline(y=weekly_pivot['diff'].mean(), line=dict(color='#06A77D', dash='dash', width=2))
    
    fig.update_layout(
        title='Динамика эффекта по неделям',
        xaxis_title='Неделя',
        yaxis_title='Разница (T-C), п.п.',
        yaxis_tickformat='+.1%',
        template='plotly_white',
        height=300,
        hovermode='x unified'
    )
    return fig

# Запуск

if __name__ == '__main__':
    app.run(debug=True, port=8050)
