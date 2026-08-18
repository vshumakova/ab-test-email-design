"""
Функции для расчета бизнес-метрик
"""
import numpy as np
import pandas as pd


def calculate_ctr(clicks, opens):
    """
    Расчет CTR (конверсия из открытия в клик)
    
    Parameters:
    -----------
    clicks : int
        Количество кликов
    opens : int
        Количество открытий
    
    Returns:
    --------
    float: CTR в долях (0-1)
    """
    if opens == 0:
        return 0.0
    return clicks / opens


def calculate_conversion_rate(converted, total):
    """
    Расчет конверсии
    
    Parameters:
    -----------
    converted : int
        Количество конверсий
    total : int
        Общее количество
    
    Returns:
    --------
    float: Конверсия в долях (0-1)
    """
    if total == 0:
        return 0.0
    return converted / total


def calculate_aov(revenue, orders):
    """
    Расчет среднего чека
    
    Parameters:
    -----------
    revenue : float
        Выручка
    orders : int
        Количество заказов
    
    Returns:
    --------
    float: Средний чек
    """
    if orders == 0:
        return 0.0
    return revenue / orders


def absolute_difference(rate_a, rate_b):
    """
    Абсолютная разница в процентных пунктах
    
    Parameters:
    -----------
    rate_a, rate_b : float
        Две конверсии (в долях)
    
    Returns:
    --------
    float: Разница в п.п.
    """
    return (rate_b - rate_a) * 100


def relative_difference(rate_a, rate_b):
    """
    Относительная разница в процентах
    
    Parameters:
    -----------
    rate_a, rate_b : float
        Две конверсии (в долях)
    
    Returns:
    --------
    float: Относительная разница в %
    """
    if rate_a == 0:
        return float('inf')
    return (rate_b / rate_a - 1) * 100


def calculate_mde_economic(development_cost, payback_months, current_monthly_revenue, 
                           ctr_elasticity=0.85, buffer=1.2):
    """
    Расчет экономически обоснованного MDE
    
    Parameters:
    -----------
    development_cost : float
        Стоимость разработки (руб)
    payback_months : int
        Желаемый период окупаемости (месяцев)
    current_monthly_revenue : float
        Текущая выручка в месяц (руб)
    ctr_elasticity : float
        Эластичность выручки по CTR (по умолчанию 0.85)
    buffer : float
        Буфер безопасности (по умолчанию 1.2)
    
    Returns:
    --------
    dict : Параметры MDE
    """
    required_monthly_growth = development_cost / payback_months
    required_growth_pct = required_monthly_growth / current_monthly_revenue
    mde = required_growth_pct / ctr_elasticity
    
    return {
        'development_cost': development_cost,
        'payback_months': payback_months,
        'required_monthly_growth': required_monthly_growth,
        'required_growth_pct': required_growth_pct,
        'required_growth_pct_pct': required_growth_pct * 100,
        'mde': mde,
        'mde_pct': mde * 100,
        'mde_with_buffer': mde * buffer,
        'mde_with_buffer_pct': mde * buffer * 100,
        'ctr_elasticity': ctr_elasticity,
        'buffer': buffer
    }


def calculate_economic_impact(current_revenue, current_ctr, new_ctr, 
                              total_emails, open_rate, conversion_to_purchase, aov):
    """
    Расчет экономического эффекта от изменения CTR
    
    Returns:
    --------
    dict : Экономические показатели
    """
    # Текущие показатели
    current_opens = total_emails * open_rate
    current_clicks = current_opens * current_ctr
    current_purchases = current_clicks * conversion_to_purchase
    current_revenue_calc = current_purchases * aov
    
    # Новые показатели
    new_clicks = current_opens * new_ctr
    new_purchases = new_clicks * conversion_to_purchase
    new_revenue = new_purchases * aov
    
    # Эффект
    revenue_increase = new_revenue - current_revenue
    revenue_increase_pct = (new_revenue / current_revenue - 1) * 100 if current_revenue > 0 else 0
    
    return {
        'current_opens': current_opens,
        'current_clicks': current_clicks,
        'current_purchases': current_purchases,
        'current_revenue': current_revenue_calc,
        'new_clicks': new_clicks,
        'new_purchases': new_purchases,
        'new_revenue': new_revenue,
        'revenue_increase': revenue_increase,
        'revenue_increase_pct': revenue_increase_pct,
        'additional_clicks': new_clicks - current_clicks,
        'additional_purchases': new_purchases - current_purchases
    }


def calculate_guardrail_metrics(control_df, treatment_df):
    """
    Расчет защитных метрик для эксперимента
    
    Parameters:
    -----------
    control_df, treatment_df : pd.DataFrame
        Данные по группам
    
    Returns:
    --------
    dict : Защитные метрики
    """
    metrics = {}
    
    # Если есть данные по заказам
    if 'revenue' in control_df.columns and 'orders' in control_df.columns:
        aov_control = calculate_aov(control_df['revenue'].sum(), control_df['orders'].sum())
        aov_treatment = calculate_aov(treatment_df['revenue'].sum(), treatment_df['orders'].sum())
        
        metrics['aov'] = {
            'control': aov_control,
            'treatment': aov_treatment,
            'diff': aov_treatment - aov_control,
            'diff_pct': (aov_treatment / aov_control - 1) * 100 if aov_control > 0 else 0
        }
    
    # Если есть данные по отказу
    if 'bounce' in control_df.columns:
        bounce_control = control_df['bounce'].mean()
        bounce_treatment = treatment_df['bounce'].mean()
        
        metrics['bounce'] = {
            'control': bounce_control,
            'treatment': bounce_treatment,
            'diff': bounce_treatment - bounce_control,
            'diff_pct': (bounce_treatment - bounce_control) * 100
        }
    
    return metrics
