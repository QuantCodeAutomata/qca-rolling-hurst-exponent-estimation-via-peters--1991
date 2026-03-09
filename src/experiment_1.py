"""
Experiment 1: Rolling Hurst Exponent Estimation via Peters (1991) Rescaled-Range Method

Estimates time-varying Hurst exponents H(t) for equity indices using 36-month rolling windows
of daily data, stepped forward by one month.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import os

from .utils import (
    compute_log_returns,
    validate_price_data,
    get_month_end_dates,
    get_rolling_window_data,
    create_results_directory
)


def compute_rescaled_range_for_scale(
    returns: np.ndarray,
    n: int
) -> Tuple[float, int]:
    """
    Compute average rescaled range (R/S) for a given scale n using Peters (1991) procedure.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of log returns
    n : int
        Block length (scale)
        
    Returns
    -------
    tuple
        (average R/S value, number of blocks used)
    """
    N = len(returns)
    A = N // n  # Number of complete blocks
    
    if A < 2:
        return np.nan, 0
    
    rs_values = []
    
    # Process each block
    for a in range(A):
        # Extract block
        block_start = a * n
        block_end = (a + 1) * n
        block_returns = returns[block_start:block_end]
        
        # Compute block mean
        e_a = np.mean(block_returns)
        
        # Compute cumulative mean-adjusted deviations
        X_k = np.cumsum(block_returns - e_a)
        
        # Compute range
        R_la = np.max(X_k) - np.min(X_k)
        
        # Compute standard deviation (population std: divide by n, not n-1)
        sigma_la = np.sqrt(np.mean((block_returns - e_a) ** 2))
        
        # Handle zero standard deviation
        if sigma_la == 0:
            continue
        
        # Compute rescaled range for this block
        rs_a = R_la / sigma_la
        rs_values.append(rs_a)
    
    if len(rs_values) == 0:
        return np.nan, 0
    
    # Average across blocks
    rs_n = np.mean(rs_values)
    
    return rs_n, len(rs_values)


def estimate_hurst_exponent(
    returns: pd.Series,
    scale_set: List[int] = None
) -> Tuple[float, float, int, List[int]]:
    """
    Estimate Hurst exponent H via OLS regression on log(R/S) vs log(n).
    
    Implements Peters (1991) rescaled-range analysis.
    
    Parameters
    ----------
    returns : pd.Series
        Log returns series
    scale_set : list of int, optional
        Set of scales to use. If None, uses default set.
        
    Returns
    -------
    tuple
        (H: Hurst exponent, c: exp(intercept), N_obs: number of observations,
         n_scales_used: list of valid scales)
    """
    N = len(returns)
    
    if N < 20:
        return np.nan, np.nan, N, []
    
    # Default scale set if not provided
    if scale_set is None:
        # Use scales that give at least 2 blocks and cover range from ~10 to N/2
        scale_set = []
        for n in [10, 15, 20, 30, 42, 63, 84, 126, 189, 252, 378, 504]:
            if n <= N // 2 and N // n >= 2:
                scale_set.append(n)
        
        if len(scale_set) < 3:
            # Fallback for very short windows
            scale_set = [n for n in range(10, N // 2) if N // n >= 2]
            if len(scale_set) > 10:
                # Sample evenly
                indices = np.linspace(0, len(scale_set) - 1, 10, dtype=int)
                scale_set = [scale_set[i] for i in indices]
    
    # Compute R/S for each scale
    ln_n_values = []
    ln_rs_values = []
    valid_scales = []
    
    returns_array = returns.values
    
    for n in scale_set:
        rs_n, n_blocks = compute_rescaled_range_for_scale(returns_array, n)
        
        if not np.isnan(rs_n) and rs_n > 0:
            ln_n_values.append(np.log(n))
            ln_rs_values.append(np.log(rs_n))
            valid_scales.append(n)
    
    # Need at least 3 points for meaningful regression
    if len(ln_n_values) < 3:
        return np.nan, np.nan, N, []
    
    # OLS regression: ln(R/S) = ln(c) + H * ln(n)
    slope, intercept, r_value, p_value, std_err = stats.linregress(ln_n_values, ln_rs_values)
    
    H = slope
    c = np.exp(intercept)
    
    return H, c, N, valid_scales


def compute_rolling_hurst(
    prices: pd.Series,
    window_months: int = 36,
    scale_set: List[int] = None
) -> pd.DataFrame:
    """
    Compute rolling Hurst exponent with monthly step.
    
    Parameters
    ----------
    prices : pd.Series
        Daily closing prices indexed by date
    window_months : int, default=36
        Rolling window size in calendar months
    scale_set : list of int, optional
        Set of scales for R/S analysis
        
    Returns
    -------
    pd.DataFrame
        Rolling H(t) estimates with columns: date, H, c, N_obs, n_scales
    """
    # Validate data
    validate_price_data(prices)
    
    # Compute log returns
    returns = compute_log_returns(prices).dropna()
    
    # Get month-end dates
    month_ends = get_month_end_dates(prices)
    
    # Filter to dates where we have enough history
    start_date = prices.index.min() + pd.DateOffset(months=window_months)
    valid_month_ends = month_ends[month_ends >= start_date]
    
    results = []
    
    print(f"Computing rolling Hurst exponent for {len(valid_month_ends)} windows...")
    
    for i, anchor_date in enumerate(valid_month_ends):
        if (i + 1) % 12 == 0:
            print(f"  Processed {i + 1}/{len(valid_month_ends)} windows...")
        
        # Get rolling window data
        window_returns = get_rolling_window_data(returns, anchor_date, window_months)
        
        if len(window_returns) < 20:
            continue
        
        # Estimate Hurst exponent
        H, c, N_obs, n_scales = estimate_hurst_exponent(window_returns, scale_set)
        
        results.append({
            'date': anchor_date,
            'H': H,
            'c': c,
            'N_obs': N_obs,
            'n_scales': len(n_scales)
        })
    
    df = pd.DataFrame(results)
    df = df.set_index('date')
    
    return df


def classify_regime(H: float) -> str:
    """
    Classify market regime based on Hurst exponent.
    
    Parameters
    ----------
    H : float
        Hurst exponent value
        
    Returns
    -------
    str
        Regime classification: 'mean_reverting', 'random_walk', or 'trending'
    """
    if np.isnan(H):
        return 'unknown'
    elif H < 0.49:
        return 'mean_reverting'
    elif H <= 0.51:
        return 'random_walk'
    else:
        return 'trending'


def compute_regime_statistics(hurst_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics on regime classifications.
    
    Parameters
    ----------
    hurst_df : pd.DataFrame
        Rolling Hurst exponent results
        
    Returns
    -------
    pd.DataFrame
        Regime statistics
    """
    hurst_df = hurst_df.copy()
    hurst_df['regime'] = hurst_df['H'].apply(classify_regime)
    
    regime_counts = hurst_df['regime'].value_counts()
    total = len(hurst_df)
    
    stats_data = []
    for regime in ['mean_reverting', 'random_walk', 'trending', 'unknown']:
        count = regime_counts.get(regime, 0)
        fraction = count / total if total > 0 else 0
        stats_data.append({
            'regime': regime,
            'count': count,
            'fraction': fraction
        })
    
    return pd.DataFrame(stats_data)


def plot_hurst_timeseries(
    hurst_df: pd.DataFrame,
    index_name: str,
    save_path: str = None
) -> None:
    """
    Plot Hurst exponent time series.
    
    Parameters
    ----------
    hurst_df : pd.DataFrame
        Rolling Hurst exponent results
    index_name : str
        Name of the index for plot title
    save_path : str, optional
        Path to save plot
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(hurst_df.index, hurst_df['H'], linewidth=1.5, color='navy', label='H(t)')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='H=0.5 (Random Walk)')
    ax.axhline(y=0.49, color='orange', linestyle=':', linewidth=0.8, alpha=0.7)
    ax.axhline(y=0.51, color='orange', linestyle=':', linewidth=0.8, alpha=0.7)
    
    # Shade crisis period
    crisis_start = pd.Timestamp('2008-09-01')
    crisis_end = pd.Timestamp('2009-03-31')
    ax.axvspan(crisis_start, crisis_end, alpha=0.2, color='gray', label='2008-09 Crisis')
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Hurst Exponent H', fontsize=11)
    ax.set_title(f'Rolling Hurst Exponent: {index_name}\n36-Month Window, Monthly Step',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def validate_specific_window(
    prices: pd.Series,
    window_start: str = "2006-03-01",
    window_end: str = "2009-03-31",
    scale_set: List[int] = None
) -> Tuple[float, float]:
    """
    Validate specific window mentioned in paper (Mar 2006 - Mar 2009 for S&P 500).
    
    Expected: H ≈ 0.509, c ≈ 1.009
    
    Parameters
    ----------
    prices : pd.Series
        Price series
    window_start : str
        Window start date
    window_end : str
        Window end date
    scale_set : list of int, optional
        Set of scales
        
    Returns
    -------
    tuple
        (H, c) for validation
    """
    returns = compute_log_returns(prices).dropna()
    
    # Extract window
    mask = (returns.index >= window_start) & (returns.index <= window_end)
    window_returns = returns[mask]
    
    H, c, N_obs, n_scales = estimate_hurst_exponent(window_returns, scale_set)
    
    print(f"\nValidation Window: {window_start} to {window_end}")
    print(f"  N observations: {N_obs}")
    print(f"  H = {H:.4f} (expected: ~0.509)")
    print(f"  c = {c:.4f} (expected: ~1.009)")
    print(f"  Scales used: {len(n_scales)}")
    
    return H, c


def run_experiment_1(
    index_data: Dict[str, pd.Series],
    save_results: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Run Experiment 1 for all indices.
    
    Parameters
    ----------
    index_data : dict
        Dictionary mapping index names to price series
    save_results : bool, default=True
        Whether to save results to files
        
    Returns
    -------
    dict
        Dictionary mapping index names to Hurst exponent DataFrames
    """
    results_dir = create_results_directory()
    hurst_results = {}
    regime_stats_all = {}
    
    for index_name, prices in index_data.items():
        print(f"\n{'='*60}")
        print(f"Processing {index_name}")
        print(f"{'='*60}")
        
        # Compute rolling Hurst exponent
        hurst_df = compute_rolling_hurst(prices)
        hurst_results[index_name] = hurst_df
        
        # Compute regime statistics
        regime_stats = compute_regime_statistics(hurst_df)
        regime_stats_all[index_name] = regime_stats
        
        print(f"\nHurst Exponent Summary for {index_name}:")
        print(f"  Mean H: {hurst_df['H'].mean():.4f}")
        print(f"  Std H: {hurst_df['H'].std():.4f}")
        print(f"  Min H: {hurst_df['H'].min():.4f}")
        print(f"  Max H: {hurst_df['H'].max():.4f}")
        print(f"\nRegime Distribution:")
        print(regime_stats.to_string(index=False))
        
        if save_results:
            # Save CSV
            csv_path = os.path.join(results_dir, f'rolling_hurst_{index_name}.csv')
            hurst_df.to_csv(csv_path)
            print(f"\nSaved results to {csv_path}")
            
            # Save plot
            plot_path = os.path.join(results_dir, f'hurst_timeseries_{index_name}.png')
            plot_hurst_timeseries(hurst_df, index_name, plot_path)
        
        # Validation for S&P 500
        if index_name == 'SP500':
            print("\n" + "="*60)
            print("VALIDATION: S&P 500 Mar 2006 - Mar 2009 Window")
            print("="*60)
            validate_specific_window(prices)
    
    # Save regime summary
    if save_results:
        regime_summary_path = os.path.join(results_dir, 'hurst_regime_summary.csv')
        
        summary_data = []
        for index_name, stats in regime_stats_all.items():
            for _, row in stats.iterrows():
                summary_data.append({
                    'index': index_name,
                    'regime': row['regime'],
                    'count': row['count'],
                    'fraction': row['fraction']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(regime_summary_path, index=False)
        print(f"\nSaved regime summary to {regime_summary_path}")
    
    return hurst_results


if __name__ == "__main__":
    from .data_download import get_closing_prices
    
    # Load data for all indices
    indices = ['SP500', 'FTSE100', 'JSE']
    index_data = {}
    
    for index_name in indices:
        prices = get_closing_prices(index_name)
        index_data[index_name] = prices
    
    # Run experiment
    results = run_experiment_1(index_data)
