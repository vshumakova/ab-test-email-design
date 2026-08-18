"""
Статистические функции для A/B-тестирования
"""
import numpy as np
from scipy import stats


def calculate_sample_size(baseline_rate, mde, alpha=0.05, power=0.90, one_sided=True):
    """
    Расчет размера выборки для A/B теста
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
    
    daily_traffic = 82
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
    """
    p_a = successes_a / n_a if n_a > 0 else 0
    p_b = successes_b / n_b if n_b > 0 else 0
    p_pool = (successes_a + successes_b) / (n_a + n_b) if (n_a + n_b) > 0 else 0.5
    
    # 🔧 ИСПРАВЛЕНО: Проверка на нулевое SE
    if p_pool == 0 or p_pool == 1 or n_a == 0 or n_b == 0:
        return {
            'p_a': p_a,
            'p_b': p_b,
            'diff': p_b - p_a,
            'diff_pct': (p_b / p_a - 1) * 100 if p_a > 0 else 0,
            'se': 0,
            'z_stat': 0,
            'p_value': 0.5,
            'one_sided': one_sided,
            'significance': '',
            'significance_text': 'не значимо (недостаточно данных)',
            'is_significant': False
        }
    
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
        'diff_pct': (p_b / p_a - 1) * 100 if p_a > 0 else 0,
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
    """
    a_control, b_control = control_clicks + 1, control_users - control_clicks + 1
    a_treatment, b_treatment = treatment_clicks + 1, treatment_users - treatment_clicks + 1
    
    np.random.seed(42)
    post_control = stats.beta.rvs(a_control, b_control, size=n_samples)
    post_treatment = stats.beta.rvs(a_treatment, b_treatment, size=n_samples)
    
    prob_better = np.mean(post_treatment > post_control)
    expected_loss = np.mean(np.maximum(0, post_control - post_treatment))
    
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
    """
    total = control_n + treatment_n
    expected_control = total * expected_ratio
    expected_treatment = total * (1 - expected_ratio)
    
    if expected_control == 0 or expected_treatment == 0:
        return {
            'chi2': 0,
            'p_value': 1.0,
            'control_n': control_n,
            'treatment_n': treatment_n,
            'control_pct': control_n / total * 100 if total > 0 else 50,
            'treatment_pct': treatment_n / total * 100 if total > 0 else 50,
            'expected_control': expected_control,
            'expected_treatment': expected_treatment,
            'is_valid': True,
            'is_significant': False
        }
    
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
