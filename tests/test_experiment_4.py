"""
Tests for Experiment 4: Breach-Regime Association
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from src.experiment_4 import (
    map_breaches_to_hurst_regime,
    compute_breach_regime_statistics,
    compute_unconditional_regime_probability,
    run_binomial_test
)
from src.utils import forward_fill_monthly_to_daily


def test_map_breaches_to_hurst_regime():
    """Test mapping of breach events to Hurst regime."""
    # Create monthly Hurst data
    month_ends = pd.date_range('2015-01-31', periods=12, freq='M')
    hurst_df = pd.DataFrame({
        'H': [0.55, 0.48, 0.52, 0.60, 0.45, 0.53, 0.49, 0.58, 0.51, 0.47, 0.54, 0.56]
    }, index=month_ends)
    
    # Create breach events
    breach_dates = pd.date_range('2015-03-15', periods=3, freq='2M')
    breach_events = pd.DataFrame({
        'breach_date': breach_dates,
        'D_t0': [1.20, 1.18, 1.22]
    })
    
    # Create daily price index for alignment
    daily_dates = pd.date_range('2015-01-01', '2015-12-31', freq='B')
    prices = pd.Series(100.0, index=daily_dates)
    
    # Map breaches to regime
    breach_mapping = map_breaches_to_hurst_regime(breach_events, hurst_df, prices)
    
    assert len(breach_mapping) == 3, "Should map all breaches"
    assert 'H_at_breach' in breach_mapping.columns
    assert 'is_trending' in breach_mapping.columns
    assert 'is_valid' in breach_mapping.columns


def test_forward_fill_monthly_to_daily():
    """Test forward-filling of monthly H to daily frequency."""
    # Monthly data
    month_ends = pd.date_range('2015-01-31', periods=3, freq='M')
    monthly_series = pd.Series([0.55, 0.48, 0.52], index=month_ends)
    
    # Daily dates
    daily_dates = pd.date_range('2015-01-01', '2015-03-31', freq='B')
    
    # Forward fill
    daily_series = forward_fill_monthly_to_daily(monthly_series, daily_dates)
    
    # Check that values are forward-filled correctly
    assert len(daily_series) == len(daily_dates)
    
    # Days before first month-end should be NaN
    assert pd.isna(daily_series.iloc[0])
    
    # Days after first month-end should have first value
    feb_dates = daily_dates[(daily_dates > '2015-01-31') & (daily_dates <= '2015-02-28')]
    assert all(daily_series[feb_dates] == 0.55)


def test_compute_breach_regime_statistics():
    """Test computation of breach-regime statistics."""
    breach_mapping = pd.DataFrame({
        'breach_date': pd.date_range('2015-01-01', periods=10, freq='M'),
        'D_t0': [1.2] * 10,
        'H_at_breach': [0.55, 0.48, 0.52, 0.60, 0.45, 0.53, 0.49, 0.58, 0.51, 0.56],
        'is_trending': [True, False, True, True, False, True, False, True, True, True],
        'is_valid': [True] * 10
    })
    
    stats = compute_breach_regime_statistics(breach_mapping, "TEST")
    
    assert stats is not None
    assert stats['N_total'] == 10
    assert stats['N_trending'] == 7  # 7 out of 10 are trending
    assert stats['p_breach_trending'] == 0.7


def test_compute_unconditional_regime_probability():
    """Test computation of unconditional P(H>0.5)."""
    # Create monthly Hurst data
    month_ends = pd.date_range('2015-01-31', periods=12, freq='M')
    hurst_df = pd.DataFrame({
        'H': [0.55, 0.48, 0.52, 0.60, 0.45, 0.53, 0.49, 0.58, 0.51, 0.47, 0.54, 0.56]
    }, index=month_ends)
    # 8 out of 12 are > 0.5
    
    # Daily prices for alignment
    daily_dates = pd.date_range('2015-01-01', '2015-12-31', freq='B')
    prices = pd.Series(100.0, index=daily_dates)
    
    p_uncond = compute_unconditional_regime_probability(hurst_df, prices)
    
    # Should be approximately 8/12 = 0.667
    assert 0.6 < p_uncond < 0.7, f"Expected ~0.667, got {p_uncond}"


def test_run_binomial_test():
    """Test binomial test for regime association."""
    # 9 out of 10 breaches in trending regime
    # Unconditional probability is 0.5
    
    result = run_binomial_test(
        N_trending=9,
        N_total=10,
        p_unconditional=0.5,
        index_name="TEST"
    )
    
    assert 'p_value' in result
    assert 'significant' in result
    
    # 9/10 vs 0.5 should be significant
    assert result['significant'] == True, "Should be significantly different from 0.5"
    assert result['p_value'] < 0.05


def test_run_binomial_test_no_difference():
    """Test binomial test when no difference from unconditional."""
    # 5 out of 10 breaches in trending regime
    # Unconditional probability is 0.5
    
    result = run_binomial_test(
        N_trending=5,
        N_total=10,
        p_unconditional=0.5,
        index_name="TEST"
    )
    
    # Should not be significant
    assert result['p_value'] > 0.05


def test_breach_regime_concentration():
    """Test that breach-regime association detects concentration."""
    np.random.seed(42)
    
    # Simulate scenario where 95% of breaches occur in trending regime
    # but only 60% of days are trending
    
    breach_mapping = pd.DataFrame({
        'breach_date': pd.date_range('2015-01-01', periods=100, freq='M'),
        'is_trending': [True] * 95 + [False] * 5,  # 95% trending
        'is_valid': [True] * 100
    })
    
    stats = compute_breach_regime_statistics(breach_mapping, "TEST")
    
    assert stats['p_breach_trending'] == 0.95
    
    # Binomial test against 0.6 unconditional
    result = run_binomial_test(
        N_trending=95,
        N_total=100,
        p_unconditional=0.6,
        index_name="TEST"
    )
    
    # Should be highly significant
    assert result['significant'] == True
    assert result['p_value'] < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
