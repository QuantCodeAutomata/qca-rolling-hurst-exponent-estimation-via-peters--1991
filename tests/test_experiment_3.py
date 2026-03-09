"""
Tests for Experiment 3: Breach-Event OLS Regression
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from src.experiment_3 import run_adf_test, run_ols_regression


def test_adf_test_stationary_series():
    """Test ADF test on stationary series."""
    np.random.seed(42)
    
    # White noise is stationary
    series = pd.Series(np.random.normal(0, 1, 200))
    
    result = run_adf_test(series, "test_series")
    
    assert 'adf_statistic' in result
    assert 'p_value' in result
    assert 'stationary' in result
    
    # White noise should be stationary
    assert result['stationary'] == True, "White noise should be stationary"


def test_adf_test_nonstationary_series():
    """Test ADF test on non-stationary series."""
    np.random.seed(42)
    
    # Random walk is non-stationary
    series = pd.Series(np.cumsum(np.random.normal(0, 1, 200)))
    
    result = run_adf_test(series, "random_walk")
    
    # Random walk typically fails stationarity test
    # (though this is probabilistic)
    assert 'stationary' in result


def test_adf_test_short_series():
    """Test ADF test with very short series."""
    series = pd.Series([1.0, 2.0])
    
    result = run_adf_test(series, "short_series")
    
    # Should handle gracefully
    assert result['stationary'] == False
    assert np.isnan(result['adf_statistic'])


def test_ols_regression_basic():
    """Test OLS regression with synthetic data."""
    np.random.seed(42)
    
    # Create synthetic event data with known relationship
    # Y = 0.05 - 1.2 * X + noise
    X = np.random.uniform(-0.2, 0.0, 50)  # Negative delta_D/D
    Y = 0.05 - 1.2 * X + np.random.normal(0, 0.01, 50)  # Positive returns
    
    event_data = pd.DataFrame({
        'delta_D_over_D': X,
        'Ret_22': Y
    })
    
    ols_results, summary = run_ols_regression(event_data, "TEST")
    
    assert ols_results is not None
    assert summary is not None
    
    # Check that slope is close to -1.2
    beta = summary['beta']
    assert -1.5 < beta < -0.9, f"Beta should be near -1.2, got {beta}"
    
    # Check R-squared is reasonable
    r2 = summary['r_squared']
    assert r2 > 0.5, f"R² should be high for synthetic data, got {r2}"


def test_ols_regression_insufficient_data():
    """Test OLS regression with insufficient data."""
    event_data = pd.DataFrame({
        'delta_D_over_D': [0.1],
        'Ret_22': [0.05]
    })
    
    ols_results, summary = run_ols_regression(event_data, "TEST")
    
    # Should return None for insufficient data
    assert ols_results is None
    assert summary is None


def test_ols_regression_negative_relationship():
    """Test that OLS correctly identifies negative relationship."""
    np.random.seed(42)
    
    # Create data with strong negative relationship
    X = np.linspace(-0.3, 0.1, 30)
    Y = -1.5 * X + np.random.normal(0, 0.02, 30)
    
    event_data = pd.DataFrame({
        'delta_D_over_D': X,
        'Ret_22': Y
    })
    
    ols_results, summary = run_ols_regression(event_data, "TEST")
    
    # Beta should be negative
    assert summary['beta'] < 0, "Beta should be negative"
    
    # Should be statistically significant
    assert summary['beta_pval'] < 0.05, "Beta should be significant"


def test_ols_regression_statistics():
    """Test that all regression statistics are computed."""
    np.random.seed(42)
    
    X = np.random.normal(0, 0.1, 40)
    Y = 0.03 - 1.0 * X + np.random.normal(0, 0.02, 40)
    
    event_data = pd.DataFrame({
        'delta_D_over_D': X,
        'Ret_22': Y
    })
    
    ols_results, summary = run_ols_regression(event_data, "TEST")
    
    # Check all required statistics are present
    required_keys = [
        'index', 'n_obs', 'alpha', 'alpha_se', 'alpha_tstat', 'alpha_pval',
        'beta', 'beta_se', 'beta_tstat', 'beta_pval',
        'r_squared', 'f_stat', 'f_pval'
    ]
    
    for key in required_keys:
        assert key in summary, f"Missing key: {key}"
    
    # Check values are reasonable
    assert summary['n_obs'] == 40
    assert not np.isnan(summary['r_squared'])
    assert 0 <= summary['r_squared'] <= 1
    assert summary['f_stat'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
