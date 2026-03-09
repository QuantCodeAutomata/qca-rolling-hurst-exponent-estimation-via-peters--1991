"""
Experiment 2: Daily Fractal Dimension D(t) Estimation and Threshold Breach Detection

Computes daily fractal dimension using Joshi (2014) formula and identifies
threshold breaches where D crosses below 1.25 from above.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import os

from .utils import (
    compute_log_returns,
    validate_price_data,
    create_results_directory
)


def compute_fractal_dimension(
    returns: pd.Series,
    prices: pd.Series,
    n: int = 22
) -> pd.Series:
    """
    Compute daily fractal dimension D(t) using Joshi formula.
    
    D_{i,n} = ln(N_{i,n}) / ln(n)
    where N_{i,n} = (n * sum|r_j|) / |R_{i,n}|
    and R_{i,n} = ln(P_i / P_{i-n})
    
    Parameters
    ----------
    returns : pd.Series
        Daily log returns
    prices : pd.Series
        Daily closing prices
    n : int, default=22
        Scaling factor (number of trading days)
        
    Returns
    -------
    pd.Series
        Daily fractal dimension series
    """
    # Compute n-day log return: R_{i,n} = sum of n daily returns
    R_in = returns.rolling(window=n).sum()
    
    # Compute sum of absolute returns over n days
    sum_abs_r = returns.abs().rolling(window=n).sum()
    
    # Compute N_{i,n} = (n * sum|r|) / |R_{i,n}|
    # Handle division by zero
    R_in_abs = R_in.abs()
    
    # Set threshold for near-zero values
    epsilon = 1e-10
    
    # Compute N_in
    N_in = pd.Series(index=returns.index, dtype=float)
    N_in = (n * sum_abs_r) / R_in_abs
    
    # Set to NaN where R_in is too close to zero
    N_in[R_in_abs < epsilon] = np.nan
    
    # Compute fractal dimension: D = ln(N) / ln(n)
    D = np.log(N_in) / np.log(n)
    
    # Replace inf and -inf with NaN
    D = D.replace([np.inf, -np.inf], np.nan)
    
    return D


def identify_breach_events(
    D: pd.Series,
    threshold: float = 1.25
) -> pd.DataFrame:
    """
    Identify threshold breach events where D crosses below threshold from above.
    
    Breach event: D_{t0} <= threshold AND D_{t0-1} > threshold
    
    Parameters
    ----------
    D : pd.Series
        Daily fractal dimension series
    threshold : float, default=1.25
        Breach threshold
        
    Returns
    -------
    pd.DataFrame
        DataFrame of breach events with columns: breach_date, D_t0
    """
    # Identify crossings from above
    D_prev = D.shift(1)
    
    # Breach condition: was above threshold yesterday, at or below today
    breach_mask = (D_prev > threshold) & (D <= threshold)
    
    # Get breach dates
    breach_dates = D.index[breach_mask]
    
    # Create breach events dataframe
    breach_events = pd.DataFrame({
        'breach_date': breach_dates,
        'D_t0': D[breach_dates].values
    })
    
    return breach_events


def compute_event_windows(
    breach_events: pd.DataFrame,
    D: pd.Series,
    prices: pd.Series,
    pre_window: int = 5,
    post_window: int = 22
) -> pd.DataFrame:
    """
    Compute pre-breach D change and post-breach return for each breach event.
    
    Parameters
    ----------
    breach_events : pd.DataFrame
        Breach events with breach_date column
    D : pd.Series
        Daily fractal dimension series
    prices : pd.Series
        Daily closing prices
    pre_window : int, default=5
        Number of trading days before breach for D change
    post_window : int, default=22
        Number of trading days after breach for return
        
    Returns
    -------
    pd.DataFrame
        Event dataset with pre/post metrics
    """
    events = []
    
    # Convert to position-based indexing for window calculations
    D_values = D.values
    price_values = prices.values
    dates = D.index
    
    for _, row in breach_events.iterrows():
        breach_date = row['breach_date']
        
        # Find position in series
        try:
            t0_pos = dates.get_loc(breach_date)
        except KeyError:
            continue
        
        # Check if we have enough pre-window data
        if t0_pos < pre_window:
            continue
        
        # Check if we have enough post-window data
        if t0_pos + post_window >= len(dates):
            continue
        
        # Get D at breach and pre-window
        D_t0 = D_values[t0_pos]
        D_t0m5 = D_values[t0_pos - pre_window]
        
        # Check for NaN
        if np.isnan(D_t0) or np.isnan(D_t0m5):
            continue
        
        # Compute pre-breach D change
        if D_t0m5 == 0:
            continue
        
        delta_D_over_D = (D_t0 - D_t0m5) / D_t0m5
        
        # Get prices
        P_t0 = price_values[t0_pos]
        P_t0p22 = price_values[t0_pos + post_window]
        
        # Compute post-breach simple return
        Ret_22 = (P_t0p22 - P_t0) / P_t0
        
        events.append({
            'breach_date': breach_date,
            'D_t0': D_t0,
            'D_t0m5': D_t0m5,
            'delta_D_over_D': delta_D_over_D,
            'P_t0': P_t0,
            'P_t0p22': P_t0p22,
            'Ret_22': Ret_22
        })
    
    return pd.DataFrame(events)


def run_sensitivity_analysis(
    returns: pd.Series,
    prices: pd.Series,
    n_values: List[int] = [5, 10, 15, 20, 22]
) -> Dict[int, Dict]:
    """
    Run sensitivity analysis across different scaling factors n.
    
    Parameters
    ----------
    returns : pd.Series
        Daily log returns
    prices : pd.Series
        Daily closing prices
    n_values : list of int
        Scaling factors to test
        
    Returns
    -------
    dict
        Results for each n value
    """
    results = {}
    
    for n in n_values:
        print(f"\nTesting n={n}...")
        
        # Compute fractal dimension
        D = compute_fractal_dimension(returns, prices, n)
        
        # Identify breaches
        breach_events = identify_breach_events(D)
        
        # Compute event windows
        event_data = compute_event_windows(breach_events, D, prices)
        
        n_breaches = len(breach_events)
        n_valid_events = len(event_data)
        
        print(f"  Total breaches: {n_breaches}")
        print(f"  Valid events (with pre/post windows): {n_valid_events}")
        
        results[n] = {
            'D': D,
            'breach_events': breach_events,
            'event_data': event_data,
            'n_breaches': n_breaches,
            'n_valid_events': n_valid_events
        }
    
    return results


def plot_fractal_dimension(
    D: pd.Series,
    breach_events: pd.DataFrame,
    index_name: str,
    n: int,
    save_path: str = None
) -> None:
    """
    Plot daily fractal dimension with breach markers.
    
    Parameters
    ----------
    D : pd.Series
        Daily fractal dimension series
    breach_events : pd.DataFrame
        Breach events
    index_name : str
        Index name for plot title
    n : int
        Scaling factor used
    save_path : str, optional
        Path to save plot
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot D(t)
    ax.plot(D.index, D.values, linewidth=0.8, color='darkblue', alpha=0.7, label='D(t)')
    
    # Add threshold line
    ax.axhline(y=1.25, color='red', linestyle='--', linewidth=1.5, label='D=1.25 Threshold')
    
    # Mark breach events
    if len(breach_events) > 0:
        for _, breach in breach_events.iterrows():
            ax.axvline(x=breach['breach_date'], color='orange', alpha=0.3, linewidth=0.5)
    
    # Shade crisis period
    crisis_start = pd.Timestamp('2008-09-01')
    crisis_end = pd.Timestamp('2009-03-31')
    ax.axvspan(crisis_start, crisis_end, alpha=0.1, color='gray', label='2008-09 Crisis')
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Fractal Dimension D', fontsize=11)
    ax.set_title(f'Daily Fractal Dimension: {index_name} (n={n})\n' +
                 f'Breach Events: {len(breach_events)}',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([1.0, 2.0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def run_experiment_2(
    index_data: Dict[str, pd.Series],
    n_values: List[int] = [5, 10, 15, 20, 22],
    primary_n: int = 22,
    save_results: bool = True
) -> Dict[str, Dict]:
    """
    Run Experiment 2 for all indices.
    
    Parameters
    ----------
    index_data : dict
        Dictionary mapping index names to price series
    n_values : list of int
        Scaling factors for sensitivity analysis
    primary_n : int
        Primary scaling factor to use for detailed results
    save_results : bool, default=True
        Whether to save results to files
        
    Returns
    -------
    dict
        Results for each index
    """
    results_dir = create_results_directory()
    all_results = {}
    sensitivity_summary = []
    
    for index_name, prices in index_data.items():
        print(f"\n{'='*60}")
        print(f"Processing {index_name}")
        print(f"{'='*60}")
        
        # Validate data
        validate_price_data(prices)
        
        # Compute log returns
        returns = compute_log_returns(prices).dropna()
        
        # Run sensitivity analysis
        print(f"\nRunning sensitivity analysis for {index_name}...")
        sensitivity_results = run_sensitivity_analysis(returns, prices, n_values)
        
        all_results[index_name] = sensitivity_results
        
        # Record sensitivity metrics
        for n, res in sensitivity_results.items():
            sensitivity_summary.append({
                'index': index_name,
                'n': n,
                'n_breaches': res['n_breaches'],
                'n_valid_events': res['n_valid_events']
            })
        
        # Use primary n for detailed output
        primary_results = sensitivity_results[primary_n]
        D = primary_results['D']
        breach_events = primary_results['breach_events']
        event_data = primary_results['event_data']
        
        print(f"\nPrimary Results (n={primary_n}):")
        print(f"  D(t) mean: {D.mean():.4f}")
        print(f"  D(t) std: {D.std():.4f}")
        print(f"  D(t) min: {D.min():.4f}")
        print(f"  D(t) max: {D.max():.4f}")
        print(f"  Total breaches: {len(breach_events)}")
        print(f"  Valid events: {len(event_data)}")
        
        if save_results:
            # Save daily D series
            D_df = pd.DataFrame({
                'date': D.index,
                'D': D.values,
                'price': prices[D.index].values
            })
            d_path = os.path.join(results_dir, f'daily_D_{index_name}_n{primary_n}.csv')
            D_df.to_csv(d_path, index=False)
            print(f"\nSaved D series to {d_path}")
            
            # Save breach events
            if len(event_data) > 0:
                event_path = os.path.join(results_dir, f'breach_events_{index_name}_n{primary_n}.csv')
                event_data.to_csv(event_path, index=False)
                print(f"Saved breach events to {event_path}")
            
            # Save plot
            plot_path = os.path.join(results_dir, f'D_timeseries_{index_name}_n{primary_n}.png')
            plot_fractal_dimension(D, breach_events, index_name, primary_n, plot_path)
    
    # Save sensitivity summary
    if save_results:
        sensitivity_df = pd.DataFrame(sensitivity_summary)
        sens_path = os.path.join(results_dir, 'sensitivity_summary.csv')
        sensitivity_df.to_csv(sens_path, index=False)
        print(f"\nSaved sensitivity summary to {sens_path}")
    
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
    results = run_experiment_2(index_data)
