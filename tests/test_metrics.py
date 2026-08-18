"""
Тесты для метрик и бизнес-расчетов
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd

from src.metrics import (
    calculate_ctr,
    calculate_conversion_rate,
    calculate_aov,
    absolute_difference,
    relative_difference,
    calculate_mde_economic,
    calculate_economic_impact,
    calculate_guardrail_metrics
)


class TestBasicMetrics:
    """Тесты для базовых метрик"""
    
    def test_calculate_ctr(self):
        assert calculate_ctr(clicks=100, opens=1000) == 0.1
        assert calculate_ctr(clicks=0, opens=1000) == 0.0
        assert calculate_ctr(clicks=100, opens=0) == 0.0
        
    def test_calculate_conversion_rate(self):
        assert calculate_conversion_rate(converted=100, total=1000) == 0.1
        assert calculate_conversion_rate(converted=0, total=1000) == 0.0
        assert calculate_conversion_rate(converted=100, total=0) == 0.0
        
    def test_calculate_aov(self):
        assert calculate_aov(revenue=10000, orders=100) == 100.0
        assert calculate_aov(revenue=0, orders=100) == 0.0
        assert calculate_aov(revenue=10000, orders=0) == 0.0


class TestDifferences:
    """Тесты для расчета разниц"""
    
    def test_absolute_difference(self):
        assert absolute_difference(0.05, 0.10) == 5.0
        assert absolute_difference(0.10, 0.05) == -5.0
        assert absolute_difference(0.05, 0.05) == 0.0
        
    def test_relative_difference(self):
        """Относительная разница в %"""
        assert relative_difference(0.05, 0.10) == 100.0
        assert relative_difference(0.10, 0.05) == -50.0
        assert relative_difference(0.05, 0.05) == 0.0
        assert relative_difference(0.05, 0.06) == pytest.approx(20.0, rel=1e-9)
        
    def test_relative_difference_zero_baseline(self):
        result = relative_difference(0, 0.10)
        assert result == float('inf')


class TestMDE:
    """Тесты для расчета MDE"""
    
    def test_basic_mde_calculation(self):
        result = calculate_mde_economic(
            development_cost=600000,
            payback_months=6,
            current_monthly_revenue=2000000,
            ctr_elasticity=0.85,
            buffer=1.2
        )
        
        assert result['development_cost'] == 600000
        assert result['payback_months'] == 6
        assert result['required_monthly_growth'] == 100000
        assert result['ctr_elasticity'] == 0.85
        assert result['buffer'] == 1.2
        
    def test_mde_with_different_costs(self):
        result_high = calculate_mde_economic(1000000, 6, 2000000)
        result_low = calculate_mde_economic(500000, 6, 2000000)
        assert result_high['mde'] > result_low['mde']
        
    def test_mde_with_different_payback(self):
        result_short = calculate_mde_economic(600000, 3, 2000000)
        result_long = calculate_mde_economic(600000, 12, 2000000)
        assert result_short['mde'] > result_long['mde']
        
    def test_mde_with_different_elasticity(self):
        result_low = calculate_mde_economic(600000, 6, 2000000, ctr_elasticity=0.5)
        result_high = calculate_mde_economic(600000, 6, 2000000, ctr_elasticity=0.9)
        assert result_low['mde'] > result_high['mde']
        
    def test_buffer_applied_correctly(self):
        """Буфер применяется корректно"""
        result = calculate_mde_economic(
            development_cost=600000,
            payback_months=6,
            current_monthly_revenue=2000000,
            buffer=1.2
        )
        
        assert result['mde_with_buffer'] == result['mde'] * 1.2
        assert result['mde_with_buffer_pct'] == pytest.approx(result['mde_pct'] * 1.2, rel=1e-9)


class TestEconomicImpact:
    """Тесты для экономического эффекта"""
    
    def test_basic_impact_calculation(self):
        result = calculate_economic_impact(
            current_revenue=1000000,
            current_ctr=0.05,
            new_ctr=0.07,
            total_emails=50000,
            open_rate=0.20,
            conversion_to_purchase=0.10,
            aov=1000
        )
        
        assert result['current_opens'] == 10000
        assert result['current_clicks'] == 500
        assert result['current_purchases'] == 50
        assert result['current_revenue'] == 50000
        assert result['new_clicks'] == pytest.approx(700, rel=1e-9)
        assert result['new_purchases'] == pytest.approx(70, rel=1e-9)
        assert result['new_revenue'] == pytest.approx(70000, rel=1e-9)
        assert result['revenue_increase'] == pytest.approx(20000, rel=1e-9)
        
    def test_impact_with_zero_conversion(self):
        result = calculate_economic_impact(
            current_revenue=1000000,
            current_ctr=0,
            new_ctr=0.05,
            total_emails=50000,
            open_rate=0.20,
            conversion_to_purchase=0.10,
            aov=1000
        )
        
        assert result['current_revenue'] == 0
        assert result['revenue_increase'] > 0
        
    def test_impact_with_no_change(self):
        result = calculate_economic_impact(
            current_revenue=1000000,
            current_ctr=0.05,
            new_ctr=0.05,
            total_emails=50000,
            open_rate=0.20,
            conversion_to_purchase=0.10,
            aov=1000
        )
        
        # 🔧 ИСПРАВЛЕНО: CTR не меняется → прирост 0
        assert result['revenue_increase'] == 0


class TestGuardrailMetrics:
    """Тесты для защитных метрик"""
    
    def test_aov_metrics(self):
        control = pd.DataFrame({
            'revenue': [100, 200, 150, 300],
            'orders': [1, 2, 1, 3]
        })
        treatment = pd.DataFrame({
            'revenue': [120, 180, 160, 280],
            'orders': [1, 2, 1, 3]
        })
        
        result = calculate_guardrail_metrics(control, treatment)
        
        assert 'aov' in result
        assert result['aov']['control'] == 750 / 7
        assert result['aov']['treatment'] == 740 / 7
        
    def test_bounce_metrics(self):
        control = pd.DataFrame({'bounce': [0, 1, 0, 0, 1, 1, 0]})
        treatment = pd.DataFrame({'bounce': [0, 0, 1, 0, 0, 0, 0]})
        
        result = calculate_guardrail_metrics(control, treatment)
        
        assert 'bounce' in result
        assert result['bounce']['control'] == 3/7
        assert result['bounce']['treatment'] == 1/7
        
    def test_missing_columns(self):
        control = pd.DataFrame({'col1': [1, 2, 3]})
        treatment = pd.DataFrame({'col1': [4, 5, 6]})
        
        result = calculate_guardrail_metrics(control, treatment)
        
        assert isinstance(result, dict)
        assert 'aov' not in result
        assert 'bounce' not in result


class TestIntegration:
    """Интеграционные тесты"""
    
    def test_end_to_end_mde_to_sample_size(self):
        from src.stats import calculate_sample_size
        
        mde_result = calculate_mde_economic(
            development_cost=600000,
            payback_months=6,
            current_monthly_revenue=2000000,
            ctr_elasticity=0.85,
            buffer=1.2
        )
        
        sample_result = calculate_sample_size(
            baseline_rate=0.078,
            mde=mde_result['mde_with_buffer'],
            alpha=0.05,
            power=0.90
        )
        
        assert sample_result['n_per_group'] > 0
        
    def test_dataframe_integration(self):
        np.random.seed(42)
        n = 1000
        
        df = pd.DataFrame({
            'user_id': range(n),
            'group': np.random.choice(['control', 'treatment'], n),
            'user_type': np.random.choice(['new', 'old'], n),
            'converted': np.random.binomial(1, 0.07, n)
        })
        
        control = df[df['group'] == 'control']
        treatment = df[df['group'] == 'treatment']
        
        cr_c = calculate_conversion_rate(control['converted'].sum(), len(control))
        cr_t = calculate_conversion_rate(treatment['converted'].sum(), len(treatment))
        
        assert 0 <= cr_c <= 1
        assert 0 <= cr_t <= 1
        
        diff_abs = absolute_difference(cr_c, cr_t)
        diff_rel = relative_difference(cr_c, cr_t)
        
        assert isinstance(diff_abs, float)
        assert isinstance(diff_rel, float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
