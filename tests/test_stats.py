"""
Тесты для статистических функций
"""
import pytest
import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.append('..')
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
        """Проверка базового расчета"""
        result = calculate_sample_size(
            baseline_rate=0.078,
            mde=0.20,
            alpha=0.05,
            power=0.90,
            one_sided=True
        )
        
        # Проверяем базовые параметры
        assert result['n_per_group'] > 0
        assert result['n_total'] == result['n_per_group'] * 2
        assert result['test_rate'] == 0.078 * 1.20
        assert 0 < result['p_pooled'] < 1
        assert result['alpha'] == 0.05
        assert result['power'] == 0.90
        assert result['one_sided'] is True
        
    def test_one_vs_two_sided(self):
        """Односторонний тест требует меньшей выборки"""
        one = calculate_sample_size(0.078, 0.20, one_sided=True)
        two = calculate_sample_size(0.078, 0.20, one_sided=False)
        
        assert one['n_per_group'] < two['n_per_group']
        assert one['z_alpha'] < two['z_alpha']
        
    def test_higher_power_requires_more_samples(self):
        """Большая мощность требует большей выборки"""
        power_80 = calculate_sample_size(0.078, 0.20, power=0.80)
        power_90 = calculate_sample_size(0.078, 0.20, power=0.90)
        
        assert power_90['n_per_group'] > power_80['n_per_group']
        
    def test_smaller_mde_requires_more_samples(self):
        """Меньший MDE требует большей выборки"""
        mde_10 = calculate_sample_size(0.078, 0.10)
        mde_20 = calculate_sample_size(0.078, 0.20)
        
        assert mde_10['n_per_group'] > mde_20['n_per_group']
        
    def test_daily_traffic_calculation(self):
        """Проверка расчета длительности"""
        result = calculate_sample_size(0.078, 0.20)
        
        # daily_traffic = 82 из константы
        expected_days = int(np.ceil(result['n_per_group'] / 82))
        assert result['days'] == expected_days


class TestZTest:
    """Тесты для Z-теста пропорций"""
    
    def test_significant_difference(self):
        """Тест должен обнаружить значимую разницу"""
        result = z_test_proportions(
            successes_a=100, n_a=1000,
            successes_b=150, n_b=1000,
            one_sided=True
        )
        
        assert result['p_b'] > result['p_a']
        assert result['diff'] > 0
        assert result['p_value'] < 0.05
        assert result['is_significant'] is True
        assert result['significance'] in ['*', '**', '***']
        
    def test_no_difference(self):
        """Тест не должен показывать разницу, если её нет"""
        result = z_test_proportions(
            successes_a=100, n_a=1000,
            successes_b=100, n_b=1000,
            one_sided=True
        )
        
        assert abs(result['diff']) < 0.01
        assert result['p_value'] > 0.05
        assert result['is_significant'] is False
        
    def test_one_sided_vs_two_sided(self):
        """Разные типы тестов дают разные p-value"""
        one = z_test_proportions(100, 1000, 150, 1000, one_sided=True)
        two = z_test_proportions(100, 1000, 150, 1000, one_sided=False)
        
        # Односторонний тест дает меньшее p-value
        assert one['p_value'] < two['p_value']
        
    def test_very_strong_significance(self):
        """Очень сильная значимость должна отмечаться ***"""
        result = z_test_proportions(
            successes_a=50, n_a=1000,
            successes_b=200, n_b=1000
        )
        
        assert result['p_value'] < 0.001
        assert result['significance'] == '***'
        assert result['significance_text'] == 'очень сильная значимость'
        
    def test_calculation_with_edge_cases(self):
        """Проверка крайних случаев"""
        # Нулевая конверсия в обеих группах
        result = z_test_proportions(0, 1000, 0, 1000)
        assert result['diff'] == 0
        assert result['p_value'] == 0.5
        
        # Полная конверсия в обеих группах
        result = z_test_proportions(1000, 1000, 1000, 1000)
        assert result['diff'] == 0
        assert result['p_value'] == 0.5


class TestConfidenceInterval:
    """Тесты для доверительных интервалов"""
    
    def test_basic_ci(self):
        """Проверка базового расчета"""
        result = confidence_interval(diff=0.03, se=0.01, ci=95)
        
        assert result['lower'] < result['upper']
        assert result['ci'] == 95
        assert result['lower'] == 0.03 - 1.96 * 0.01
        assert result['upper'] == 0.03 + 1.96 * 0.01
        
    def test_ci_different_levels(self):
        """Разные уровни доверия дают разные интервалы"""
        ci_90 = confidence_interval(diff=0.03, se=0.01, ci=90)
        ci_95 = confidence_interval(diff=0.03, se=0.01, ci=95)
        ci_99 = confidence_interval(diff=0.03, se=0.01, ci=99)
        
        # Больше доверие → шире интервал
        assert ci_90['lower'] > ci_95['lower']
        assert ci_90['upper'] < ci_95['upper']
        assert ci_95['lower'] > ci_99['lower']
        assert ci_95['upper'] < ci_99['upper']
        
    def test_ci_contains_zero(self):
        """Проверка пересечения с нулем"""
        # Интервал выше нуля
        result = confidence_interval(diff=0.03, se=0.005)
        assert result['all_positive'] is True
        assert result['contains_zero'] is False
        
        # Интервал ниже нуля
        result = confidence_interval(diff=-0.03, se=0.005)
        assert result['all_negative'] is True
        assert result['contains_zero'] is False
        
        # Интервал пересекает ноль
        result = confidence_interval(diff=0.01, se=0.01)
        assert result['contains_zero'] is True
        assert result['all_positive'] is False
        assert result['all_negative'] is False


class TestBayesianBeta:
    """Тесты для байесовского анализа"""
    
    def test_basic_bayesian(self):
        """Проверка базового байесовского анализа"""
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
        """Когда treatment лучше, вероятность должна быть высокой"""
        result = bayesian_beta(
            control_clicks=50, control_users=1000,
            treatment_clicks=200, treatment_users=1000,
            n_samples=10000
        )
        
        assert result['prob_treatment_better'] > 0.95
        assert result['median_treatment'] > result['median_control']
        
    def test_control_better(self):
        """Когда control лучше, вероятность должна быть низкой"""
        result = bayesian_beta(
            control_clicks=200, control_users=1000,
            treatment_clicks=50, treatment_users=1000,
            n_samples=10000
        )
        
        assert result['prob_treatment_better'] < 0.05
        assert result['median_control'] > result['median_treatment']
        
    def test_no_difference(self):
        """Когда разницы нет, вероятность ~50%"""
        result = bayesian_beta(
            control_clicks=100, control_users=1000,
            treatment_clicks=100, treatment_users=1000,
            n_samples=10000
        )
        
        # Из-за случайности может быть небольшое отклонение
        assert 0.4 < result['prob_treatment_better'] < 0.6
        assert abs(result['median_control'] - result['median_treatment']) < 0.01


class TestSRM:
    """Тесты для SRM-теста"""
    
    def test_perfect_balance(self):
        """Идеальный баланс"""
        result = srm_test(control_n=1000, treatment_n=1000)
        
        assert result['is_valid'] is True
        assert result['p_value'] == 1.0
        assert result['control_pct'] == 50.0
        assert result['treatment_pct'] == 50.0
        
    def test_small_imbalance(self):
        """Небольшой дисбаланс (должен пройти)"""
        result = srm_test(control_n=510, treatment_n=490)
        
        assert result['is_valid'] is True
        assert result['p_value'] > 0.05
        
    def test_large_imbalance(self):
        """Сильный дисбаланс (должен упасть)"""
        result = srm_test(control_n=900, treatment_n=100)
        
        assert result['is_valid'] is False
        assert result['p_value'] < 0.001
        
    def test_different_expected_ratio(self):
        """Тест с другим ожидаемым соотношением"""
        # Ожидаем 60/40
        result = srm_test(control_n=600, treatment_n=400, expected_ratio=0.6)
        
        assert result['is_valid'] is True
        assert result['p_value'] > 0.05
        assert result['expected_control'] == 600
        assert result['expected_treatment'] == 400


class TestMannWhitney:
    """Тесты для U-теста Манна-Уитни"""
    
    def test_significant_difference(self):
        """Значимая разница между группами"""
        control = np.random.normal(0, 1, 100)
        treatment = np.random.normal(1, 1, 100)
        
        result = mann_whitney_test(control, treatment)
        
        assert result['p_value'] < 0.05
        assert result['is_significant'] is True
        
    def test_no_difference(self):
        """Нет разницы между группами"""
        control = np.random.normal(0, 1, 100)
        treatment = np.random.normal(0, 1, 100)
        
        result = mann_whitney_test(control, treatment)
        
        assert result['p_value'] > 0.05
        assert result['is_significant'] is False


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
