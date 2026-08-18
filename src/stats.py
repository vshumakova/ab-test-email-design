"""
Статистические функции для A/B-тестирования
"""
import numpy as np
from scipy import stats
import pandas as pd


def calculate_sample_size(baseline_rate, mde, alpha=0.05, power=0.90, one_sided=True):
    """
    Расчет размера выборки для A/B теста
    
    Parameters:
    -----------
    baseline_rate : float
        Базовый показатель (например, CTR)
    mde : float
        Минимальный детектируемый эффект (относительный, например 0.20 для 20%)
    alpha : float
        Уровень значимости (по умолчанию 0.05)
    power : float
        Мощность теста (по умолчанию 0.90)
    one_sided : bool
        Односторонний (True) или двухсторонний (False) тест
    
    Returns:
    --------
    dict : Параметры расчета
    """
    mde_abs = baseline_rate * mde
    test_rate = baseline_rate + mde_abs
    p_pooled = (baseline_rate + test_rate) / 2
    
    if one_sided:
        z_alpha = stats.norm.ppf(1 - alpha)
    else:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    
    z_beta = stats.norm.ppf(power)
    
    n = ((z_alpha + z_beta) ** 2 * p_pooled * (1 - p_pooled) * 2) / (mde_abs ** 2)
    n = int(np.ceil(n))
    
    # Длительность при ежедневном трафике
    daily_traffic = 82  # открытий в день (из данных)
    days = int(np.ceil(n / daily_traffic))
    
    return {
        'baseline_rate': baseline_rate,
        'test_rate': test_rate,
        'mde_abs': mde_abs,
        'mde_rel': mde * 100,
        'p_pooled': p_pooled,
        'n_per_group': n,
        'n_total': n * 2,
        'alpha': alpha,
        'power': power,
        'z_alpha': z_alpha,
        'z_beta': z_beta,
        'one_sided': one_sided,
        'daily_traffic': daily_traffic,
        'days': days
    }


def z_test_proportions(successes_a, n_a, successes_b, n_b, one_sided=True):
    """
    Z-тест для сравнения двух пропорций
    
    Parameters:
    -----------
    successes_a, successes_b : int
        Количество успехов в группах
    n_a, n_b : int
        Размеры групп
    one_sided : bool
        Односторонний тест (right-tailed)
    
    Returns:
    --------
    dict : Результаты теста
    """
    p_a = successes_a / n_a
    p_b = successes_b / n_b
    p_pool = (successes_a + successes_b) / (n_a + n_b)
    
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z_stat = (p_b - p_a) / se
    
    if one_sided:
        p_value = 1 - stats.norm.cdf(z_stat)
    else:
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # Интерпретация
    if p_value < 0.001:
        significance = '***'
        significance_text = 'очень сильная значимость'
    elif p_value < 0.01:
        significance = '**'
        significance_text = 'сильная значимость'
    elif p_value < 0.05:
        significance = '*'
        significance_text = 'статистически значимо'
    else:
        significance = ''
        significance_text = 'не значимо'
    
    return {
        'p_a': p_a,
        'p_b': p_b,
        'diff': p_b - p_a,
        'diff_pct': (p_b / p_a - 1) * 100 if p_a > 0 else np.nan,
        'se': se,
        'z_stat': z_stat,
        'p_value': p_value,
        'one_sided': one_sided,
        'significance': significance,
        'significance_text': significance_text,
        'is_significant': p_value < 0.05
    }


def confidence_interval(diff, se, ci=95):
    """
    Расчет доверительного интервала для разницы пропорций
    
    Parameters:
    -----------
    diff : float
        Разница пропорций
    se : float
        Стандартная ошибка
    ci : int
        Уровень доверия (по умолчанию 95)
    
    Returns:
    --------
    dict : Доверительный интервал
    """
    z_crit = stats.norm.ppf(1 - (100 - ci) / 200)
    lower = diff - z_crit * se
    upper = diff + z_crit * se
    
    return {
        'lower': lower,
        'upper': upper,
        'ci': ci,
        'contains_zero': lower < 0 < upper,
        'all_positive': lower > 0,
        'all_negative': upper < 0
    }


def bayesian_beta(control_clicks, control_users, treatment_clicks, treatment_users, n_samples=100000):
    """
    Байесовский анализ с использованием бета-распределений
    
    Parameters:
    -----------
    control_clicks, treatment_clicks : int
        Количество кликов в группах
    control_users, treatment_users : int
        Количество пользователей в группах
    n_samples : int
        Количество сэмплов для MCMC
    
    Returns:
    --------
    dict : Результаты байесовского анализа
    """
    # Априорные распределения
    a_control, b_control = control_clicks + 1, control_users - control_clicks + 1
    a_treatment, b_treatment = treatment_clicks + 1, treatment_users - treatment_clicks + 1
    
    # Генерация апостериорных распределений
    np.random.seed(42)
    post_control = stats.beta.rvs(a_control, b_control, size=n_samples)
    post_treatment = stats.beta.rvs(a_treatment, b_treatment, size=n_samples)
    
    # Вероятность, что treatment лучше control
    prob_better = np.mean(post_treatment > post_control)
    
    # Ожидаемая потеря при внедрении
    expected_loss = np.mean(np.maximum(0, post_control - post_treatment))
    
    # Медианы и интервалы
    median_control = np.median(post_control)
    median_treatment = np.median(post_treatment)
    
    hdi_control = np.percentile(post_control, [2.5, 97.5])
    hdi_treatment = np.percentile(post_treatment, [2.5, 97.5])
    
    return {
        'prob_treatment_better': prob_better,
        'expected_loss': expected_loss,
        'expected_loss_pct': expected_loss * 100,
        'median_control': median_control,
        'median_treatment': median_treatment,
        'hdi_control': hdi_control,
        'hdi_treatment': hdi_treatment,
        'post_control': post_control,
        'post_treatment': post_treatment,
        'n_samples': n_samples
    }


def srm_test(control_n, treatment_n, expected_ratio=0.5):
    """
    SRM-тест (Sample Ratio Mismatch)
    Проверяет, соответствует ли распределение пользователей ожидаемому
    
    Parameters:
    -----------
    control_n, treatment_n : int
        Количество пользователей в группах
    expected_ratio : float
        Ожидаемая доля control (по умолчанию 0.5)
    
    Returns:
    --------
    dict : Результаты теста
    """
    total = control_n + treatment_n
    expected_control = total * expected_ratio
    expected_treatment = total * (1 - expected_ratio)
    
    chi2 = ((control_n - expected_control) ** 2 / expected_control +
            (treatment_n - expected_treatment) ** 2 / expected_treatment)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return {
        'chi2': chi2,
        'p_value': p_value,
        'control_n': control_n,
        'treatment_n': treatment_n,
        'control_pct': control_n / total * 100,
        'treatment_pct': treatment_n / total * 100,
        'expected_control': expected_control,
        'expected_treatment': expected_treatment,
        'is_valid': p_value >= 0.05,
        'is_significant': p_value < 0.05
    }


def mann_whitney_test(control_series, treatment_series):
    """
    U-тест Манна-Уитни для сравнения двух распределений
    
    Parameters:
    -----------
    control_series, treatment_series : array-like
        Значения в группах
    
    Returns:
    --------
    dict : Результаты теста
    """
    stat, p_value = stats.mannwhitneyu(
        treatment_series, control_series,
        alternative='greater'
    )
    
    return {
        'statistic': stat,
        'p_value': p_value,
        'is_significant': p_value < 0.05
    }
