"""
Шаблоны для визуализации результатов A/B-тестирования
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats


def set_style():
    """Настройка стиля для всех графиков"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette(['#2E86AB', '#E84855', '#06A77D', '#F18F01'])
    
    # Настройка шрифтов
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10


def plot_daily_conversion(df_results, save_path=None):
    """
    График ежедневной конверсии по группам
    
    Parameters:
    -----------
    df_results : pd.DataFrame
        Результаты эксперимента
    save_path : str, optional
        Путь для сохранения
    """
    set_style()
    
    # Агрегируем по дням
    daily_stats = df_results.groupby(['date', 'group']).agg(
        users=('user_id', 'count'),
        clicks=('converted', 'sum'),
        cr=('converted', 'mean')
    ).reset_index()
    
    daily_pivot = daily_stats.pivot(index='date', columns='group', values='cr')
    
    # Создаем график
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Линии конверсии
    ax.plot(daily_pivot.index, daily_pivot['control'], 
            color='#2E86AB', linewidth=2.5, label='Control', marker='o', markersize=6)
    ax.plot(daily_pivot.index, daily_pivot['treatment'], 
            color='#E84855', linewidth=2.5, label='Treatment', marker='s', markersize=6)
    
    # Средние линии
    ax.axhline(daily_pivot['control'].mean(), color='#2E86AB', 
               linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(daily_pivot['treatment'].mean(), color='#E84855', 
               linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Оформление
    ax.set_xlabel('Дата', fontsize=12)
    ax.set_ylabel('Конверсия', fontsize=12)
    ax.set_title('Ежедневная динамика конверсии', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Форматирование оси Y
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Ротация подписей
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_weekly_difference(df_results, anomaly_weeks=None, save_path=None):
    """
    График разницы конверсий по неделям
    
    Parameters:
    -----------
    df_results : pd.DataFrame
        Результаты эксперимента
    anomaly_weeks : list, optional
        Список аномальных недель для выделения
    save_path : str, optional
        Путь для сохранения
    """
    set_style()
    
    # Добавляем недели
    df = df_results.copy()
    df['week'] = df['date'].dt.isocalendar().week
    
    # Агрегируем по неделям
    weekly_stats = df.groupby(['week', 'group']).agg(
        users=('user_id', 'count'),
        clicks=('converted', 'sum'),
        cr=('converted', 'mean')
    ).reset_index()
    
    weekly_pivot = weekly_stats.pivot(index='week', columns='group', values='cr')
    weekly_pivot['diff'] = weekly_pivot['treatment'] - weekly_pivot['control']
    
    # Создаем график
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Цвета для столбцов
    colors = ['#E84855' if (anomaly_weeks and week in anomaly_weeks) else '#2E86AB' 
              for week in weekly_pivot.index]
    
    # Столбцы разницы
    bars = ax.bar(weekly_pivot.index, weekly_pivot['diff'] * 100, 
                  color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Нулевая линия
    ax.axhline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5)
    
    # Средняя линия
    avg_diff = weekly_pivot['diff'].mean()
    ax.axhline(avg_diff * 100, color='#06A77D', linestyle='--', 
               linewidth=2, label=f'Средняя: {avg_diff*100:.1f} п.п.')
    
    # Оформление
    ax.set_xlabel('Неделя', fontsize=12)
    ax.set_ylabel('Разница (Treatment - Control), п.п.', fontsize=12)
    ax.set_title('Разница конверсии по неделям', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения на столбцы
    for bar, diff in zip(bars, weekly_pivot['diff'] * 100):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3 if height > 0 else height - 1.5,
                f'{diff:.1f}', ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_bayesian_posterior(control_clicks, control_users, 
                            treatment_clicks, treatment_users, 
                            save_path=None):
    """
    График байесовских апостериорных распределений
    
    Parameters:
    -----------
    control_clicks, treatment_clicks : int
        Количество кликов в группах
    control_users, treatment_users : int
        Количество пользователей в группах
    save_path : str, optional
        Путь для сохранения
    """
    set_style()
    
    from scipy.stats import beta
    
    # Бета-апостериоры
    a_c, b_c = control_clicks + 1, control_users - control_clicks + 1
    a_t, b_t = treatment_clicks + 1, treatment_users - treatment_clicks + 1
    
    x = np.linspace(0, 0.20, 1000)
    y_c = beta.pdf(x, a_c, b_c)
    y_t = beta.pdf(x, a_t, b_t)
    
    # Создаем график
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(x, y_c, color='#2E86AB', linewidth=2.5, label='Control')
    ax.fill_between(x, 0, y_c, color='#2E86AB', alpha=0.3)
    
    ax.plot(x, y_t, color='#E84855', linewidth=2.5, label='Treatment')
    ax.fill_between(x, 0, y_t, color='#E84855', alpha=0.3)
    
    # Медианы
    median_c = beta.median(a_c, b_c)
    median_t = beta.median(a_t, b_t)
    
    ax.axvline(median_c, color='#2E86AB', linestyle='--', linewidth=1.5)
    ax.axvline(median_t, color='#E84855', linestyle='--', linewidth=1.5)
    
    # Оформление
    ax.set_xlabel('Конверсия', fontsize=12)
    ax.set_ylabel('Плотность', fontsize=12)
    ax.set_title('Апостериорные распределения конверсии', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Форматирование оси X
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
    
    # Добавляем аннотацию с медианами
    ax.annotate(f'Control: {median_c:.2%}', xy=(median_c, 0.98), 
                xycoords=('data', 'axes fraction'), ha='center', va='top',
                color='#2E86AB', fontweight='bold')
    ax.annotate(f'Treatment: {median_t:.2%}', xy=(median_t, 0.95), 
                xycoords=('data', 'axes fraction'), ha='center', va='top',
                color='#E84855', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_stratification(df_results, save_path=None):
    """
    График стратификации конверсии по типам пользователей
    
    Parameters:
    -----------
    df_results : pd.DataFrame
        Результаты эксперимента
    save_path : str, optional
        Путь для сохранения
    """
    set_style()
    
    segmented_cr = df_results.groupby(['group', 'user_type'])['converted'].mean().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(segmented_cr.index))
    width = 0.35
    
    bars_new = ax.bar(x - width/2, segmented_cr['new'], width, 
                      label='Новые пользователи', color='#F39C12', alpha=0.8)
    bars_old = ax.bar(x + width/2, segmented_cr['old'], width, 
                      label='Старые пользователи', color='#8E44AD', alpha=0.8)
    
    ax.set_xlabel('Группа', fontsize=12)
    ax.set_ylabel('Конверсия', fontsize=12)
    ax.set_title('Стратификация конверсии по типу пользователя', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(segmented_cr.index)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Форматирование оси Y
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Добавляем значения
    for bar in bars_new + bars_old:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                f'{height:.2%}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_sample_size_vs_mde(baseline_rate=0.078, alpha=0.05, power=0.90, 
                            mde_values=None, save_path=None):
    """
    График зависимости размера выборки от MDE
    
    Parameters:
    -----------
    baseline_rate : float
        Базовый показатель
    alpha : float
        Уровень значимости
    power : float
        Мощность теста
    mde_values : list, optional
        Список MDE для расчета
    save_path : str, optional
        Путь для сохранения
    """
    set_style()
    
    from src.stats import calculate_sample_size
    
    if mde_values is None:
        mde_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    
    ns = [calculate_sample_size(baseline_rate, mde, alpha, power)['n_per_group'] 
          for mde in mde_values]
    days = [calculate_sample_size(baseline_rate, mde, alpha, power)['days'] 
            for mde in mde_values]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # График 1: Размер выборки
    color = '#2E86AB'
    ax1.set_xlabel('MDE (относительный прирост)', fontsize=12)
    ax1.set_ylabel('Размер выборки (на группу)', fontsize=12, color=color)
    line1 = ax1.plot([m*100 for m in mde_values], ns, color=color, 
                     linewidth=2.5, marker='o', markersize=8, label='Размер выборки')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # График 2: Длительность
    ax2 = ax1.twinx()
    color = '#E84855'
    ax2.set_ylabel('Длительность (дней)', fontsize=12, color=color)
    line2 = ax2.plot([m*100 for m in mde_values], days, color=color, 
                     linewidth=2.5, linestyle='--', marker='s', markersize=8, label='Длительность')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Отмечаем наш MDE = 20%
    ax1.axvline(20, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axhline(calculate_sample_size(baseline_rate, 0.20, alpha, power)['n_per_group'], 
                color='green', linestyle=':', linewidth=2, alpha=0.7)
    
    ax1.set_title('Зависимость размера выборки и длительности от MDE', 
                  fontsize=14, fontweight='bold')
    
    # Легенда
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    # Аннотация для MDE=20%
    ax1.annotate('MDE = 20%', xy=(20, calculate_sample_size(baseline_rate, 0.20, alpha, power)['n_per_group']),
                 xytext=(25, 6000), arrowprops=dict(arrowstyle='->', color='green'), 
                 fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confidence_intervals(control_rate, treatment_rate, se, save_path=None):
    """
    График доверительных интервалов
    
    Parameters:
    -----------
    control_rate, treatment_rate : float
        Конверсии в группах
    se : float
        Стандартная ошибка
    save_path : str, optional
        Путь для сохранения
    """
    set_style()
    
    from src.stats import confidence_interval
    
    diff = treatment_rate - control_rate
    ci = confidence_interval(diff, se)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Точка оценки
    ax.scatter(0, diff * 100, color='#2E86AB', s=100, zorder=3, 
               label=f'Разница: {diff*100:.2f} п.п.')
    
    # Доверительный интервал
    ax.hlines(0, ci['lower'] * 100, ci['upper'] * 100, 
              color='#2E86AB', linewidth=3)
    ax.scatter([ci['lower'] * 100, ci['upper'] * 100], [0, 0], 
               color='#2E86AB', s=50, zorder=3)
    
    # Вертикальная линия на 0
    ax.axvline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5)
    
    # Оформление
    ax.set_xlabel('Разница конверсий (Treatment - Control), п.п.', fontsize=12)
    ax.set_title(f'95% Доверительный интервал: [{ci["lower"]*100:.2f}; {ci["upper"]*100:.2f}] п.п.', 
                 fontsize=14, fontweight='bold')
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Добавляем аннотации
    ax.annotate(f'{ci["lower"]*100:.2f} п.п.', xy=(ci['lower'] * 100, 0), 
                xytext=(ci['lower'] * 100, 0.15), ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.annotate(f'{ci["upper"]*100:.2f} п.п.', xy=(ci['upper'] * 100, 0), 
                xytext=(ci['upper'] * 100, 0.15), ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    
    if ci['lower'] > 0:
        ax.annotate('Эффект положительный', xy=(0.05, 0.5), 
                    xycoords='axes fraction', fontsize=12, fontweight='bold', color='#06A77D')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
