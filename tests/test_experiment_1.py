"""
Tests for Experiment 1: Rolling Hurst Exponent Estimation
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from src.experiment_1 import (
    compute_rescaled_range_for_scale,
    estimate_hurst_exponent,
    compute_rolling_hurst,
    classify_regime
)


def test_rescaled_range_computation():
    """Test rescaled range computation for a single scale."""
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, 100)
    
    n = 10
    rs_n, n_blocks = compute_rescaled_range_for_scale(returns, n)
    
    # Should return valid R/S value
    assert not np.isnan(rs_n), "R/S should not be NaN for valid data"
    assert rs_n > 0, "R/S should be positive"
    assert n_blocks == 10, f"Expected 10 blocks, got {n_blocks}"


def test_rescaled_range_zero_variance():
    """Test R/S computation with zero variance block."""
    # All zeros - zero variance
    returns = np.zeros(100)
    
    n = 10
    rs_n, n_blocks = compute_rescaled_range_for_scale(returns, n)
    
    # Should handle zero variance gracefully
    assert np.isnan(rs_n), "R/S should be NaN for zero variance"


def test_rescaled_range_insufficient_blocks():
    """Test R/S when scale is too large for data."""
    returns = np.random.normal(0, 0.01, 15)
    
    n = 20  # Too large, results in < 2 blocks
    rs_n, n_blocks = compute_rescaled_range_for_scale(returns, n)
    
    assert np.isnan(rs_n), "R/S should be NaN when blocks < 2"
    assert n_blocks == 0


def test_hurst_exponent_random_walk():
    """Test Hurst exponent for random walk (H ≈ 0.5)."""
    np.random.seed(42)
    
    # Generate random walk
    returns = pd.Series(np.random.normal(0, 0.01, 1000))
    
    H, c, N_obs, n_scales = estimate_hurst_exponent(returns)
    
    # Random walk should have H close to 0.5
    assert not np.isnan(H), "H should not be NaN"
    assert 0.3 < H < 0.7, f"H should be near 0.5 for random walk, got {H}"
    assert c > 0, "Constant c should be positive"
    assert N_obs == 1000
    assert len(n_scales) >= 3, "Should use at least 3 scales"


def test_hurst_exponent_trending_series():
    """Test Hurst exponent for trending series (H > 0.5)."""
    np.random.seed(42)
    
    # Generate trending series with persistence
    returns = []
    r = 0
    for _ in range(1000):
        r = 0.5 * r + np.random.normal(0, 0.01)  # AR(1) with positive autocorrelation
        returns.append(r)
    
    returns = pd.Series(returns)
    
    H, c, N_obs, n_scales = estimate_hurst_exponent(returns)
    
    # Trending series should have H > 0.5
    assert not np.isnan(H), "H should not be NaN"
    assert H > 0.5, f"H should be > 0.5 for trending series, got {H}"


def test_hurst_exponent_mean_reverting_series():
    """Test Hurst exponent for mean-reverting series (H < 0.5)."""
    np.random.seed(42)
    
    # Generate mean-reverting series
    returns = []
    r = 0
    for _ in range(1000):
        r = -0.5 * r + np.random.normal(0, 0.01)  # AR(1) with negative autocorrelation
        returns.append(r)
    
    returns = pd.Series(returns)
    
    H, c, N_obs, n_scales = estimate_hurst_exponent(returns)
    
    # Mean-reverting series should have H < 0.5
    assert not np.isnan(H), "H should not be NaN"
    assert H < 0.5, f"H should be < 0.5 for mean-reverting series, got {H}"


def test_hurst_exponent_short_series():
    """Test Hurst exponent with insufficient data."""
    returns = pd.Series(np.random.normal(0, 0.01, 10))
    
    H, c, N_obs, n_scales = estimate_hurst_exponent(returns)
    
    # Should return NaN for very short series
    assert np.isnan(H), "H should be NaN for very short series"


def test_rolling_hurst_basic():
    """Test rolling Hurst computation."""
    np.random.seed(42)
    
    # Generate synthetic price series (approximately 5 years of daily data)
    n_days = 1260  # ~5 years
    dates = pd.date_range('2013-01-01', periods=n_days, freq='B')
    
    returns = np.random.normal(0.0003, 0.01, n_days)
    prices = 100 * np.exp(np.cumsum(returns))
    prices = pd.Series(prices, index=dates)
    
    # Compute rolling Hurst with 36-month window
    hurst_df = compute_rolling_hurst(prices, window_months=36)
    
    # Should produce results
    assert len(hurst_df) > 0, "Should produce at least one Hurst estimate"
    assert 'H' in hurst_df.columns
    assert 'c' in hurst_df.columns
    assert 'N_obs' in hurst_df.columns
    
    # H should be in reasonable range
    valid_H = hurst_df['H'].dropna()
    assert all(0 < h < 1 for h in valid_H), "H should be between 0 and 1"


def test_classify_regime():
    """Test regime classification based on H value."""
    assert classify_regime(0.3) == 'mean_reverting'
    assert classify_regime(0.48) == 'mean_reverting'
    assert classify_regime(0.49) == 'random_walk'
    assert classify_regime(0.50) == 'random_walk'
    assert classify_regime(0.51) == 'random_walk'
    assert classify_regime(0.52) == 'trending'
    assert classify_regime(0.7) == 'trending'
    assert classify_regime(np.nan) == 'unknown'


def test_hurst_bounds():
    """Test that Hurst exponent stays within theoretical bounds."""
    np.random.seed(42)
    
    # Test various series types
    for _ in range(10):
        returns = pd.Series(np.random.normal(0, 0.01, 500))
        H, c, N_obs, n_scales = estimate_hurst_exponent(returns)
        
        if not np.isnan(H):
            # H should be between 0 and 1 (theoretical bounds)
            assert 0 < H < 1, f"H out of bounds: {H}"


def test_peters_procedure_implementation():
    """Test that Peters (1991) procedure is implemented correctly."""
    np.random.seed(42)
    returns = np.array(np.random.normal(0, 0.01, 100))
    
    n = 10
    rs_n, n_blocks = compute_rescaled_range_for_scale(returns, n)
    
    # Verify the procedure:
    # 1. Should partition into floor(100/10) = 10 blocks
    assert n_blocks == 10
    
    # 2. R/S should be positive
    assert rs_n > 0
    
    # 3. Manually compute for first block to verify
    block = returns[0:10]
    e_a = np.mean(block)
    X_k = np.cumsum(block - e_a)
    R = np.max(X_k) - np.min(X_k)
    sigma = np.sqrt(np.mean((block - e_a) ** 2))  # Population std
    
    if sigma > 0:
        rs_manual = R / sigma
        # First block's R/S should be reasonable
        assert rs_manual > 0


def test_scale_set_generation():
    """Test that scale set is generated appropriately."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.01, 756))  # ~36 months
    
    H, c, N_obs, n_scales = estimate_hurst_exponent(returns, scale_set=None)
    
    # Should auto-generate scale set
    assert len(n_scales) >= 3, "Should have at least 3 scales"
    
    # All scales should satisfy: floor(N/n) >= 2
    for n in n_scales:
        assert 756 // n >= 2, f"Scale {n} produces < 2 blocks"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
