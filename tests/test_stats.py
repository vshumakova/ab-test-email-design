"""
Тесты для статистических функций
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from scipy import stats

from src.stats import (
    calculate_sample_size,
    z_test_proportions,
    confidence_interval,
    bayesian_beta,
    srm_test,
    mann_whitney_test
)


class TestSampleSize:
    """Тесты для расчета размера выборки"""
    
    def test_basic_calculation(self):
        result = calculate_sample_size(
            baseline_rate=0.078,
            mde=0.20,
            alpha=0.05,
            power=0.90,
            one_sided=True
        )
        
        assert result['n_per_group'] > 0
        assert result['n_total'] == result['n_per_group'] * 2
        assert result['test_rate'] == 0.078 * 1.20
        assert 0 < result['p_pooled'] < 1
        assert result['alpha'] == 0.05
        assert result['power'] == 0.90
        assert result['one_sided'] is True
        
    def test_one_vs_two_sided(self):
        one = calculate_sample_size(0.078, 0.20, one_sided=True)
        two = calculate_sample_size(0.078, 0.20, one_sided=False)
        
        assert one['n_per_group'] < two['n_per_group']
        assert one['z_alpha'] < two['z_alpha']
        
    def test_higher_power_requires_more_samples(self):
        power_80 = calculate_sample_size(0.078, 0.20, power=0.80)
        power_90 = calculate_sample_size(0.078, 0.20, power=0.90)
        
        assert power_90['n_per_group'] > power_80['n_per_group']
        
    def test_smaller_mde_requires_more_samples(self):
        mde_10 = calculate_sample_size(0.078, 0.10)
        mde_20 = calculate_sample_size(0.078, 0.20)
        
        assert mde_10['n_per_group'] > mde_20['n_per_group']
        
    def test_daily_traffic_calculation(self):
        result = calculate_sample_size(0.078, 0.20)
        expected_days = int(np.ceil(result['n_per_group'] / 82))
        assert result['days'] == expected_days


class TestZTest:
    """Тесты для Z-теста пропорций"""
    
    def test_significant_difference(self):
        result = z_test_proportions(
            successes_a=100, n_a=1000,
            successes_b=150, n_b=1000,
            one_sided=True
        )
        
        assert result['p_b'] > result['p_a']
        assert result['diff'] > 0
        assert result['p_value'] < 0.05
        assert result['is_significant'] == True
        assert result['significance'] in ['*', '**', '***']
        
    def test_no_difference(self):
        result = z_test_proportions(
            successes_a=100, n_a=1000,
            successes_b=100, n_b=1000,
            one_sided=True
        )
        
        assert abs(result['diff']) < 0.01
        assert result['p_value'] > 0.05
        assert result['is_significant'] == False
        
    def test_one_sided_vs_two_sided(self):
        one = z_test_proportions(100, 1000, 150, 1000, one_sided=True)
        two = z_test_proportions(100, 1000, 150, 1000, one_sided=False)
        
        assert one['p_value'] < two['p_value']
        
    def test_very_strong_significance(self):
        result = z_test_proportions(
            successes_a=50, n_a=1000,
            successes_b=200, n_b=1000
        )
        
        assert result['p_value'] < 0.001
        assert result['significance'] == '***'
        assert result['significance_text'] == 'очень сильная значимость'
        
    def test_calculation_with_edge_cases(self):
        result = z_test_proportions(0, 1000, 0, 1000)
        assert result['diff'] == 0
        assert result['p_value'] == 0.5 or result['p_value'] == 0.0
        
        result = z_test_proportions(1000, 1000, 1000, 1000)
        assert result['diff'] == 0
        assert result['p_value'] == 0.5 or result['p_value'] == 0.0


class TestConfidenceInterval:
    """Тесты для доверительных интервалов"""
    
    def test_basic_ci(self):
        result = confidence_interval(diff=0.03, se=0.01, ci=95)
        
        assert result['lower'] < result['upper']
        assert result['ci'] == 95

        z_crit = stats.norm.ppf(0.975)
        assert result['lower'] == pytest.approx(0.03 - z_crit * 0.01, rel=1e-9)
        assert result['upper'] == pytest.approx(0.03 + z_crit * 0.01, rel=1e-9)
        
    def test_ci_different_levels(self):
        ci_90 = confidence_interval(diff=0.03, se=0.01, ci=90)
        ci_95 = confidence_interval(diff=0.03, se=0.01, ci=95)
        ci_99 = confidence_interval(diff=0.03, se=0.01, ci=99)
        
        assert ci_90['lower'] > ci_95['lower']
        assert ci_90['upper'] < ci_95['upper']
        assert ci_95['lower'] > ci_99['lower']
        assert ci_95['upper'] < ci_99['upper']
        
    def test_ci_contains_zero(self):
        # Интервал выше нуля
        result = confidence_interval(diff=0.03, se=0.005)
        assert result['all_positive'] == True
        assert result['contains_zero'] == False
        
        # Интервал ниже нуля
        result = confidence_interval(diff=-0.03, se=0.005)
        assert result['all_negative'] == True
        assert result['contains_zero'] == False
        
        # Интервал пересекает ноль
        result = confidence_interval(diff=0.01, se=0.01)
        assert result['contains_zero'] == True
        assert result['all_positive'] == False
        assert result['all_negative'] == False


class TestBayesianBeta:
    """Тесты для байесовского анализа"""
    
    def test_basic_bayesian(self):
        result = bayesian_beta(
            control_clicks=100, control_users=1000,
            treatment_clicks=150, treatment_users=1000,
            n_samples=10000
        )
        
        assert 0 <= result['prob_treatment_better'] <= 1
        assert result['expected_loss'] >= 0
        assert result['median_control'] > 0
        assert result['median_treatment'] > 0
        assert len(result['post_control']) == 10000
        assert len(result['post_treatment']) == 10000
        
    def test_treatment_better(self):
        result = bayesian_beta(
            control_clicks=50, control_users=1000,
            treatment_clicks=200, treatment_users=1000,
            n_samples=10000
        )
        
        assert result['prob_treatment_better'] > 0.95
        assert result['median_treatment'] > result['median_control']
        
    def test_control_better(self):
        result = bayesian_beta(
            control_clicks=200, control_users=1000,
            treatment_clicks=50, treatment_users=1000,
            n_samples=10000
        )
        
        assert result['prob_treatment_better'] < 0.05
        assert result['median_control'] > result['median_treatment']
        
    def test_no_difference(self):
        result = bayesian_beta(
            control_clicks=100, control_users=1000,
            treatment_clicks=100, treatment_users=1000,
            n_samples=10000
        )
        
        assert 0.4 < result['prob_treatment_better'] < 0.6
        assert abs(result['median_control'] - result['median_treatment']) < 0.01


class TestSRM:
    """Тесты для SRM-теста"""
    
    def test_perfect_balance(self):
        result = srm_test(control_n=1000, treatment_n=1000)
        
        assert result['is_valid'] == True
        assert result['p_value'] == 1.0
        assert result['control_pct'] == 50.0
        assert result['treatment_pct'] == 50.0
        
    def test_small_imbalance(self):
        result = srm_test(control_n=510, treatment_n=490)
        
        assert result['is_valid'] == True
        assert result['p_value'] > 0.05
        
    def test_large_imbalance(self):
        result = srm_test(control_n=900, treatment_n=100)
        
        assert result['is_valid'] == False
        assert result['p_value'] < 0.001
        
    def test_different_expected_ratio(self):
        result = srm_test(control_n=600, treatment_n=400, expected_ratio=0.6)
        
        assert result['is_valid'] == True
        assert result['p_value'] > 0.05
        assert result['expected_control'] == 600
        assert result['expected_treatment'] == 400


class TestMannWhitney:
    """Тесты для U-теста Манна-Уитни"""
    
    def test_significant_difference(self):
        np.random.seed(42)
        control = np.random.normal(0, 1, 100)
        treatment = np.random.normal(1, 1, 100)
        
        result = mann_whitney_test(control, treatment)
        
        assert result['p_value'] < 0.05
        assert result['is_significant'] == True
        
    def test_no_difference(self):
        np.random.seed(42)
        control = np.random.normal(0, 1, 100)
        treatment = np.random.normal(0, 1, 100)
        
        result = mann_whitney_test(control, treatment)
        
        assert result['p_value'] > 0.05
        assert result['is_significant'] == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
