"""
Tests for Experiment 2: Daily Fractal Dimension Estimation
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from src.experiment_2 import (
    compute_fractal_dimension,
    identify_breach_events,
    compute_event_windows
)
from src.utils import compute_log_returns


def test_fractal_dimension_basic():
    """Test basic fractal dimension computation."""
    np.random.seed(42)
    
    dates = pd.date_range('2015-01-01', periods=500, freq='B')
    returns_vals = np.random.normal(0.0003, 0.01, 500)
    prices_vals = 100 * np.exp(np.cumsum(returns_vals))
    
    prices = pd.Series(prices_vals, index=dates)
    returns = compute_log_returns(prices).dropna()
    
    D = compute_fractal_dimension(returns, prices, n=22)
    
    # Should produce valid D values
    valid_D = D.dropna()
    assert len(valid_D) > 0, "Should produce valid D values"
    
    # D should generally be between 1 and 2
    assert all(1.0 <= d <= 2.0 for d in valid_D), "D should be between 1 and 2"


def test_fractal_dimension_flat_returns():
    """Test fractal dimension with flat returns (R_n = 0)."""
    dates = pd.date_range('2015-01-01', periods=100, freq='B')
    
    # Create flat price series
    prices = pd.Series(100.0, index=dates)
    returns = compute_log_returns(prices).dropna()
    
    D = compute_fractal_dimension(returns, prices, n=22)
    
    # All D values should be NaN for flat series
    assert D.dropna().empty, "D should be NaN for flat returns"


def test_fractal_dimension_different_n():
    """Test fractal dimension with different scaling factors."""
    np.random.seed(42)
    
    dates = pd.date_range('2015-01-01', periods=300, freq='B')
    returns_vals = np.random.normal(0.0003, 0.01, 300)
    prices_vals = 100 * np.exp(np.cumsum(returns_vals))
    
    prices = pd.Series(prices_vals, index=dates)
    returns = compute_log_returns(prices).dropna()
    
    # Test different n values
    for n in [5, 10, 20, 22]:
        D = compute_fractal_dimension(returns, prices, n=n)
        valid_D = D.dropna()
        
        assert len(valid_D) > 0, f"Should produce valid D for n={n}"
        assert all(1.0 <= d <= 2.0 for d in valid_D), f"D out of range for n={n}"


def test_identify_breach_events():
    """Test breach event identification."""
    dates = pd.date_range('2015-01-01', periods=100, freq='B')
    
    # Create D series with artificial breach
    D_vals = np.ones(100) * 1.5  # Start above threshold
    D_vals[50:60] = 1.2  # Breach period
    D_vals[70:75] = 1.1  # Another breach
    
    D = pd.Series(D_vals, index=dates)
    
    breach_events = identify_breach_events(D, threshold=1.25)
    
    # Should identify breaches at positions 50 and 70 (crossings from above)
    assert len(breach_events) == 2, f"Expected 2 breaches, got {len(breach_events)}"


def test_identify_breach_no_crossings():
    """Test breach identification with no crossings."""
    dates = pd.date_range('2015-01-01', periods=100, freq='B')
    
    # D always above threshold
    D = pd.Series(np.ones(100) * 1.5, index=dates)
    
    breach_events = identify_breach_events(D, threshold=1.25)
    
    assert len(breach_events) == 0, "Should find no breaches when always above threshold"


def test_identify_breach_crossing_from_below():
    """Test that crossing from below is NOT identified as breach."""
    dates = pd.date_range('2015-01-01', periods=100, freq='B')
    
    # Start below, cross above (should NOT be breach)
    D_vals = np.ones(100) * 1.2
    D_vals[50:] = 1.5
    
    D = pd.Series(D_vals, index=dates)
    
    breach_events = identify_breach_events(D, threshold=1.25)
    
    assert len(breach_events) == 0, "Should not identify crossing from below as breach"


def test_compute_event_windows_basic():
    """Test event window computation."""
    np.random.seed(42)
    
    dates = pd.date_range('2015-01-01', periods=200, freq='B')
    
    # Create D series
    D_vals = np.random.uniform(1.2, 1.8, 200)
    D = pd.Series(D_vals, index=dates)
    
    # Create price series
    prices = pd.Series(100 + np.cumsum(np.random.normal(0, 1, 200)), index=dates)
    
    # Create breach events
    breach_events = pd.DataFrame({
        'breach_date': [dates[100]],
        'D_t0': [1.2]
    })
    
    event_data = compute_event_windows(breach_events, D, prices,
                                       pre_window=5, post_window=22)
    
    # Should produce valid event data
    assert len(event_data) > 0, "Should produce event data"
    assert 'delta_D_over_D' in event_data.columns
    assert 'Ret_22' in event_data.columns


def test_compute_event_windows_edge_cases():
    """Test event window computation with edge cases."""
    dates = pd.date_range('2015-01-01', periods=50, freq='B')
    
    D = pd.Series(np.random.uniform(1.2, 1.8, 50), index=dates)
    prices = pd.Series(100 + np.arange(50), index=dates)
    
    # Breach too close to start (no pre-window)
    breach_events = pd.DataFrame({
        'breach_date': [dates[2]],
        'D_t0': [1.2]
    })
    
    event_data = compute_event_windows(breach_events, D, prices,
                                       pre_window=5, post_window=22)
    
    # Should exclude this breach
    assert len(event_data) == 0, "Should exclude breach too close to start"
    
    # Breach too close to end (no post-window)
    breach_events = pd.DataFrame({
        'breach_date': [dates[45]],
        'D_t0': [1.2]
    })
    
    event_data = compute_event_windows(breach_events, D, prices,
                                       pre_window=5, post_window=22)
    
    # Should exclude this breach
    assert len(event_data) == 0, "Should exclude breach too close to end"


def test_joshi_formula_implementation():
    """Test that Joshi formula is implemented correctly."""
    np.random.seed(42)
    
    dates = pd.date_range('2015-01-01', periods=100, freq='B')
    returns_vals = np.random.normal(0.0003, 0.01, 100)
    prices_vals = 100 * np.exp(np.cumsum(returns_vals))
    
    prices = pd.Series(prices_vals, index=dates)
    returns = pd.Series(returns_vals, index=dates)
    
    n = 10
    D = compute_fractal_dimension(returns, prices, n=n)
    
    # Manually compute for one point to verify formula
    idx = 50
    
    # Get returns window
    returns_window = returns.iloc[idx-n+1:idx+1].values
    
    # Compute R_in (n-day log return)
    R_in = np.sum(returns_window)
    
    # Compute sum of absolute returns
    sum_abs = np.sum(np.abs(returns_window))
    
    if np.abs(R_in) > 1e-10:
        # N_in = (n * sum_abs) / |R_in|
        N_in_manual = (n * sum_abs) / np.abs(R_in)
        
        # D = ln(N_in) / ln(n)
        D_manual = np.log(N_in_manual) / np.log(n)
        
        # Compare with computed value
        D_computed = D.iloc[idx]
        
        if not np.isnan(D_computed):
            assert np.abs(D_computed - D_manual) < 1e-6, \
                f"D computation mismatch: {D_computed} vs {D_manual}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
