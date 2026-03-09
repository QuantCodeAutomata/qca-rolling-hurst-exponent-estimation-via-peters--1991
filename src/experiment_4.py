"""
Experiment 4: Breach-Regime Association

Tests whether D≤1.25 threshold breaches occur predominantly when the market is in
a trending (H>0.5) regime. Target: ~95% of JSE breaches occur during H>0.5 periods.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2_contingency
try:
    from scipy.stats import binomtest
except ImportError:
    from scipy.stats import binom_test as binomtest

from .utils import (
    create_results_directory,
    forward_fill_monthly_to_daily
)
from .experiment_1 import compute_rolling_hurst
from .experiment_2 import (
    compute_fractal_dimension,
    identify_breach_events
)


def map_breaches_to_hurst_regime(
    breach_events: pd.DataFrame,
    hurst_df: pd.DataFrame,
    prices: pd.Series
) -> pd.DataFrame:
    """
    Map each breach event to its corresponding Hurst regime.
    
    Uses forward-filled monthly H to avoid look-ahead bias.
    
    Parameters
    ----------
    breach_events : pd.DataFrame
        Breach events with breach_date column
    hurst_df : pd.DataFrame
        Monthly Hurst exponent results
    prices : pd.Series
        Daily price series for date alignment
        
    Returns
    -------
    pd.DataFrame
        Breach events with H regime mapping
    """
    # Forward-fill monthly H to daily frequency
    daily_H = forward_fill_monthly_to_daily(hurst_df['H'], prices.index)
    
    # Map each breach to its H value
    breach_mapping = []
    
    for _, row in breach_events.iterrows():
        breach_date = row['breach_date']
        
        # Get H at breach date (no look-ahead)
        if breach_date in daily_H.index:
            H_at_breach = daily_H[breach_date]
            
            is_valid = not np.isnan(H_at_breach)
            is_trending = H_at_breach > 0.5 if is_valid else False
            
            breach_mapping.append({
                'breach_date': breach_date,
                'D_t0': row['D_t0'],
                'H_at_breach': H_at_breach,
                'is_trending': is_trending,
                'is_valid': is_valid
            })
    
    return pd.DataFrame(breach_mapping)


def compute_breach_regime_statistics(
    breach_mapping: pd.DataFrame,
    index_name: str
) -> Dict:
    """
    Compute statistics on breach-regime association.
    
    Parameters
    ----------
    breach_mapping : pd.DataFrame
        Breach events with H regime mapping
    index_name : str
        Index name for reporting
        
    Returns
    -------
    dict
        Breach-regime statistics
    """
    # Filter to valid breaches (where H is available)
    valid_breaches = breach_mapping[breach_mapping['is_valid']]
    
    N_total = len(valid_breaches)
    N_trending = valid_breaches['is_trending'].sum()
    
    if N_total == 0:
        print(f"\nNo valid breaches for {index_name}")
        return None
    
    p_breach_trending = N_trending / N_total
    
    print(f"\n{'='*60}")
    print(f"Breach-Regime Statistics: {index_name}")
    print(f"{'='*60}")
    print(f"Total valid breaches: {N_total}")
    print(f"Breaches during H>0.5 (trending): {N_trending}")
    print(f"Fraction in trending regime: {p_breach_trending:.4f} ({p_breach_trending*100:.2f}%)")
    
    return {
        'index': index_name,
        'N_total': N_total,
        'N_trending': N_trending,
        'p_breach_trending': p_breach_trending
    }


def compute_unconditional_regime_probability(
    hurst_df: pd.DataFrame,
    prices: pd.Series
) -> float:
    """
    Compute unconditional probability of being in a trending regime.
    
    Parameters
    ----------
    hurst_df : pd.DataFrame
        Monthly Hurst exponent results
    prices : pd.Series
        Daily price series for date alignment
        
    Returns
    -------
    float
        Unconditional P(H>0.5)
    """
    # Forward-fill monthly H to daily
    daily_H = forward_fill_monthly_to_daily(hurst_df['H'], prices.index)
    
    # Remove NaN values
    daily_H_valid = daily_H.dropna()
    
    if len(daily_H_valid) == 0:
        return np.nan
    
    # Compute fraction of days with H>0.5
    n_trending = (daily_H_valid > 0.5).sum()
    p_unconditional = n_trending / len(daily_H_valid)
    
    return p_unconditional


def run_binomial_test(
    N_trending: int,
    N_total: int,
    p_unconditional: float,
    index_name: str
) -> Dict:
    """
    Test whether breach-conditional trending probability exceeds unconditional rate.
    
    Parameters
    ----------
    N_trending : int
        Number of breaches in trending regime
    N_total : int
        Total number of valid breaches
    p_unconditional : float
        Unconditional probability of trending regime
    index_name : str
        Index name for reporting
        
    Returns
    -------
    dict
        Test results
    """
    # One-sided binomial test: H1: p_breach > p_unconditional
    try:
        # Use binomtest (scipy >= 1.7)
        result = binomtest(N_trending, N_total, p_unconditional, alternative='greater')
        p_value = result.pvalue
    except:
        # Fallback to manual computation
        from scipy.stats import binom
        p_value = 1 - binom.cdf(N_trending - 1, N_total, p_unconditional)
    
    print(f"\n{'='*60}")
    print(f"Binomial Test: {index_name}")
    print(f"{'='*60}")
    print(f"H0: P(H>0.5 | breach) = {p_unconditional:.4f} (unconditional rate)")
    print(f"H1: P(H>0.5 | breach) > {p_unconditional:.4f}")
    print(f"Observed: {N_trending}/{N_total} = {N_trending/N_total:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Result: {'REJECT H0' if p_value < 0.05 else 'FAIL TO REJECT H0'} (α=0.05)")
    
    return {
        'index': index_name,
        'N_trending': N_trending,
        'N_total': N_total,
        'p_unconditional': p_unconditional,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def run_chi_square_test(
    breach_mapping: pd.DataFrame,
    hurst_df: pd.DataFrame,
    prices: pd.Series,
    index_name: str
) -> Dict:
    """
    Chi-square test for independence between breach events and H regime.
    
    Parameters
    ----------
    breach_mapping : pd.DataFrame
        Breach events with regime mapping
    hurst_df : pd.DataFrame
        Monthly Hurst results
    prices : pd.Series
        Daily prices
    index_name : str
        Index name
        
    Returns
    -------
    dict
        Chi-square test results
    """
    # Forward-fill H to daily
    daily_H = forward_fill_monthly_to_daily(hurst_df['H'], prices.index)
    daily_H_valid = daily_H.dropna()
    
    # Count days by regime
    n_days_trending = (daily_H_valid > 0.5).sum()
    n_days_nontrending = len(daily_H_valid) - n_days_trending
    
    # Count breach days by regime
    valid_breaches = breach_mapping[breach_mapping['is_valid']]
    n_breach_trending = valid_breaches['is_trending'].sum()
    n_breach_nontrending = len(valid_breaches) - n_breach_trending
    
    # Construct contingency table
    # Rows: breach vs non-breach
    # Cols: trending vs non-trending
    contingency_table = np.array([
        [n_breach_trending, n_breach_nontrending],
        [n_days_trending - n_breach_trending, n_days_nontrending - n_breach_nontrending]
    ])
    
    # Run chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    print(f"\n{'='*60}")
    print(f"Chi-Square Test: {index_name}")
    print(f"{'='*60}")
    print("Contingency Table:")
    print(f"                  H>0.5    H<=0.5")
    print(f"Breach days:      {n_breach_trending:6d}    {n_breach_nontrending:6d}")
    print(f"Non-breach days:  {n_days_trending - n_breach_trending:6d}    {n_days_nontrending - n_breach_nontrending:6d}")
    print(f"\nChi-square statistic: {chi2:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"p-value: {p_value:.6f}")
    print(f"Result: {'REJECT independence' if p_value < 0.05 else 'FAIL TO REJECT independence'} (α=0.05)")
    
    return {
        'index': index_name,
        'chi2_stat': chi2,
        'dof': dof,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def plot_breach_regime_barchart(
    results: Dict,
    save_path: str = None
) -> None:
    """
    Plot bar chart comparing breach-conditional vs unconditional trending probabilities.
    
    Parameters
    ----------
    results : dict
        Results dictionary with breach and unconditional stats
    save_path : str, optional
        Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    indices = list(results.keys())
    x = np.arange(len(indices))
    width = 0.35
    
    breach_probs = [results[idx]['breach_stats']['p_breach_trending'] for idx in indices]
    uncond_probs = [results[idx]['p_unconditional'] for idx in indices]
    
    bars1 = ax.bar(x - width/2, breach_probs, width, label='P(H>0.5 | Breach)', color='darkred', alpha=0.8)
    bars2 = ax.bar(x + width/2, uncond_probs, width, label='P(H>0.5) Unconditional', color='steelblue', alpha=0.8)
    
    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Breach-Conditional vs Unconditional Trending Regime Probability',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(indices)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved bar chart to {save_path}")
    
    plt.close()


def plot_hurst_with_breaches(
    hurst_df: pd.DataFrame,
    breach_mapping: pd.DataFrame,
    index_name: str,
    save_path: str = None
) -> None:
    """
    Plot H(t) time series with breach events marked and color-coded by regime.
    
    Parameters
    ----------
    hurst_df : pd.DataFrame
        Monthly Hurst results
    breach_mapping : pd.DataFrame
        Breach events with regime mapping
    index_name : str
        Index name
    save_path : str, optional
        Path to save plot
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot H(t)
    ax.plot(hurst_df.index, hurst_df['H'], linewidth=1.5, color='navy', label='H(t)')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='H=0.5')
    
    # Mark breaches by regime
    valid_breaches = breach_mapping[breach_mapping['is_valid']]
    
    trending_breaches = valid_breaches[valid_breaches['is_trending']]
    nontrending_breaches = valid_breaches[~valid_breaches['is_trending']]
    
    for _, breach in trending_breaches.iterrows():
        ax.axvline(x=breach['breach_date'], color='green', alpha=0.4, linewidth=1)
    
    for _, breach in nontrending_breaches.iterrows():
        ax.axvline(x=breach['breach_date'], color='orange', alpha=0.4, linewidth=1)
    
    # Create custom legend entries
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='navy', linewidth=1.5, label='H(t)'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='H=0.5'),
        Line2D([0], [0], color='green', alpha=0.6, linewidth=2, label=f'Breach (H>0.5): {len(trending_breaches)}'),
        Line2D([0], [0], color='orange', alpha=0.6, linewidth=2, label=f'Breach (H≤0.5): {len(nontrending_breaches)}')
    ]
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Hurst Exponent H', fontsize=11)
    ax.set_title(f'Hurst Exponent with Breach Events: {index_name}',
                 fontsize=13, fontweight='bold')
    ax.legend(handles=legend_elements, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved H with breaches plot to {save_path}")
    
    plt.close()


def run_experiment_4(
    index_data: Dict[str, pd.Series],
    n: int = 22,
    save_results: bool = True
) -> Dict:
    """
    Run Experiment 4 for all indices.
    
    Parameters
    ----------
    index_data : dict
        Dictionary mapping index names to price series
    n : int, default=22
        Scaling factor for fractal dimension
    save_results : bool, default=True
        Whether to save results
        
    Returns
    -------
    dict
        Results for each index
    """
    from .utils import compute_log_returns
    
    results_dir = create_results_directory()
    all_results = {}
    binomial_test_results = []
    chi_square_test_results = []
    
    for index_name, prices in index_data.items():
        print(f"\n{'='*60}")
        print(f"Processing {index_name}")
        print(f"{'='*60}")
        
        # Compute rolling Hurst exponent
        print("\nComputing rolling Hurst exponent...")
        hurst_df = compute_rolling_hurst(prices)
        
        # Compute fractal dimension and breaches
        print("\nComputing fractal dimension and identifying breaches...")
        returns = compute_log_returns(prices).dropna()
        D = compute_fractal_dimension(returns, prices, n)
        breach_events = identify_breach_events(D)
        
        print(f"Total breaches: {len(breach_events)}")
        
        # Map breaches to Hurst regime
        print("\nMapping breaches to Hurst regime...")
        breach_mapping = map_breaches_to_hurst_regime(breach_events, hurst_df, prices)
        
        # Compute breach-regime statistics
        breach_stats = compute_breach_regime_statistics(breach_mapping, index_name)
        
        # Compute unconditional regime probability
        p_unconditional = compute_unconditional_regime_probability(hurst_df, prices)
        print(f"\nUnconditional P(H>0.5): {p_unconditional:.4f} ({p_unconditional*100:.2f}%)")
        
        if breach_stats is not None:
            # Run binomial test
            binom_result = run_binomial_test(
                breach_stats['N_trending'],
                breach_stats['N_total'],
                p_unconditional,
                index_name
            )
            binomial_test_results.append(binom_result)
            
            # Run chi-square test
            chi2_result = run_chi_square_test(breach_mapping, hurst_df, prices, index_name)
            chi_square_test_results.append(chi2_result)
            
            all_results[index_name] = {
                'breach_mapping': breach_mapping,
                'breach_stats': breach_stats,
                'p_unconditional': p_unconditional,
                'binom_test': binom_result,
                'chi2_test': chi2_result,
                'hurst_df': hurst_df
            }
            
            if save_results:
                # Save breach-regime mapping
                mapping_path = os.path.join(results_dir, f'breach_regime_mapping_{index_name}.csv')
                breach_mapping.to_csv(mapping_path, index=False)
                print(f"\nSaved breach-regime mapping to {mapping_path}")
                
                # Save plot
                plot_path = os.path.join(results_dir, f'H_timeseries_with_breaches_{index_name}.png')
                plot_hurst_with_breaches(hurst_df, breach_mapping, index_name, plot_path)
    
    # Save summary results
    if save_results and len(all_results) > 0:
        # Binomial test summary
        binom_df = pd.DataFrame(binomial_test_results)
        binom_path = os.path.join(results_dir, 'binomial_test_results.csv')
        binom_df.to_csv(binom_path, index=False)
        print(f"\nSaved binomial test results to {binom_path}")
        
        # Chi-square test summary
        chi2_df = pd.DataFrame(chi_square_test_results)
        chi2_path = os.path.join(results_dir, 'chi_square_test_results.csv')
        chi2_df.to_csv(chi2_path, index=False)
        print(f"Saved chi-square test results to {chi2_path}")
        
        # Bar chart
        chart_path = os.path.join(results_dir, 'breach_H_regime_barchart.png')
        plot_breach_regime_barchart(all_results, chart_path)
    
    return all_results


if __name__ == "__main__":
    from .data_download import get_closing_prices
    
    # Load data for all indices
    indices = ['SP500', 'FTSE100', 'JSE']
    index_data = {}
    
    for index_name in indices:
        prices = get_closing_prices(index_name)
        index_data[index_name] = prices
    
    # Run experiment
    results = run_experiment_4(index_data)
