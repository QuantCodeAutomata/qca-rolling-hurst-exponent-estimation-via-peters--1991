"""
Experiment 3: Breach-Event OLS Regression

Regresses 22-day post-breach returns on 5-day pre-breach fractal dimension changes.
Includes ADF stationarity tests.
Target: R² ≈ 0.85, slope ≈ -1.2
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import os

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy import stats

from .utils import create_results_directory
from .experiment_2 import (
    compute_fractal_dimension,
    identify_breach_events,
    compute_event_windows
)


def run_adf_test(series: pd.Series, series_name: str) -> Dict:
    """
    Run Augmented Dickey-Fuller test for stationarity.
    
    Parameters
    ----------
    series : pd.Series
        Time series to test
    series_name : str
        Name of series for reporting
        
    Returns
    -------
    dict
        ADF test results
    """
    # Drop NaN values
    series_clean = series.dropna()
    
    if len(series_clean) < 3:
        return {
            'series_name': series_name,
            'adf_statistic': np.nan,
            'p_value': np.nan,
            'n_lags': np.nan,
            'n_obs': len(series_clean),
            'stationary': False
        }
    
    # Run ADF test
    result = adfuller(series_clean, autolag='AIC')
    
    adf_stat, p_value, n_lags, n_obs = result[0], result[1], result[2], result[3]
    
    print(f"\nADF Test: {series_name}")
    print(f"  ADF Statistic: {adf_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Lags used: {n_lags}")
    print(f"  Observations: {n_obs}")
    print(f"  Result: {'STATIONARY' if p_value < 0.05 else 'NON-STATIONARY'} (α=0.05)")
    
    return {
        'series_name': series_name,
        'adf_statistic': adf_stat,
        'p_value': p_value,
        'n_lags': n_lags,
        'n_obs': n_obs,
        'stationary': p_value < 0.05
    }


def run_ols_regression(
    event_data: pd.DataFrame,
    index_name: str
) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, Dict]:
    """
    Run OLS regression of post-breach return on pre-breach D change.
    
    Regression: Ret_22 = alpha + beta * delta_D_over_D + epsilon
    
    Parameters
    ----------
    event_data : pd.DataFrame
        Event dataset with delta_D_over_D and Ret_22 columns
    index_name : str
        Index name for reporting
        
    Returns
    -------
    tuple
        (OLS results object, summary dict)
    """
    if len(event_data) < 3:
        print(f"\nInsufficient events for {index_name}: {len(event_data)}")
        return None, None
    
    # Prepare data
    X = event_data['delta_D_over_D'].values
    Y = event_data['Ret_22'].values
    
    # Add constant for intercept
    X_with_const = sm.add_constant(X)
    
    # Run OLS
    model = sm.OLS(Y, X_with_const)
    results = model.fit()
    
    # Extract statistics
    alpha = results.params[0]
    beta = results.params[1]
    alpha_se = results.bse[0]
    beta_se = results.bse[1]
    alpha_tstat = results.tvalues[0]
    beta_tstat = results.tvalues[1]
    alpha_pval = results.pvalues[0]
    beta_pval = results.pvalues[1]
    r_squared = results.rsquared
    f_stat = results.fvalue
    f_pval = results.f_pvalue
    n_obs = len(event_data)
    
    print(f"\n{'='*60}")
    print(f"OLS Regression Results: {index_name}")
    print(f"{'='*60}")
    print(f"\nModel: Ret_22 = alpha + beta * delta_D_over_D + epsilon")
    print(f"\nNumber of observations: {n_obs}")
    print(f"\nCoefficients:")
    print(f"  Intercept (alpha): {alpha:.6f} (SE: {alpha_se:.6f}, t: {alpha_tstat:.4f}, p: {alpha_pval:.4f})")
    print(f"  Slope (beta):      {beta:.6f} (SE: {beta_se:.6f}, t: {beta_tstat:.4f}, p: {beta_pval:.4f})")
    print(f"\nModel Statistics:")
    print(f"  R-squared: {r_squared:.4f} (Target: ~0.85)")
    print(f"  F-statistic: {f_stat:.4f} (p-value: {f_pval:.6f})")
    print(f"\nTarget Comparison:")
    print(f"  Beta ≈ -1.2? Current: {beta:.4f}, Difference: {abs(beta + 1.2):.4f}")
    print(f"  R² ≈ 0.85? Current: {r_squared:.4f}, Difference: {abs(r_squared - 0.85):.4f}")
    
    summary_dict = {
        'index': index_name,
        'n_obs': n_obs,
        'alpha': alpha,
        'alpha_se': alpha_se,
        'alpha_tstat': alpha_tstat,
        'alpha_pval': alpha_pval,
        'beta': beta,
        'beta_se': beta_se,
        'beta_tstat': beta_tstat,
        'beta_pval': beta_pval,
        'r_squared': r_squared,
        'f_stat': f_stat,
        'f_pval': f_pval
    }
    
    return results, summary_dict


def plot_regression_scatter(
    event_data: pd.DataFrame,
    ols_results: sm.regression.linear_model.RegressionResultsWrapper,
    index_name: str,
    save_path: str = None
) -> None:
    """
    Plot scatter plot with regression line (Fig. 11 reproduction).
    
    Parameters
    ----------
    event_data : pd.DataFrame
        Event dataset
    ols_results : RegressionResultsWrapper
        OLS results
    index_name : str
        Index name
    save_path : str, optional
        Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    X = event_data['delta_D_over_D'].values
    Y = event_data['Ret_22'].values
    
    # Scatter plot
    ax.scatter(X, Y, alpha=0.6, s=50, color='navy', edgecolors='black', linewidth=0.5)
    
    # Fitted line
    x_range = np.linspace(X.min(), X.max(), 100)
    y_pred = ols_results.params[0] + ols_results.params[1] * x_range
    ax.plot(x_range, y_pred, color='red', linewidth=2, label='OLS Fit')
    
    # Add regression statistics as text
    r2 = ols_results.rsquared
    beta = ols_results.params[1]
    n_obs = len(event_data)
    
    text_str = f'N = {n_obs}\n'
    text_str += f'R² = {r2:.4f}\n'
    text_str += f'β = {beta:.4f}\n'
    text_str += f'y = {ols_results.params[0]:.4f} + {beta:.4f}x'
    
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('5-Day Pre-Breach ΔD/D', fontsize=12)
    ax.set_ylabel('22-Day Post-Breach Return', fontsize=12)
    ax.set_title(f'Breach-Event Regression: {index_name}\n' +
                 f'Post-Breach Return vs Pre-Breach D Change',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved regression plot to {save_path}")
    
    plt.close()


def plot_residual_diagnostics(
    ols_results: sm.regression.linear_model.RegressionResultsWrapper,
    index_name: str,
    save_path: str = None
) -> None:
    """
    Plot residual diagnostics.
    
    Parameters
    ----------
    ols_results : RegressionResultsWrapper
        OLS results
    index_name : str
        Index name
    save_path : str, optional
        Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    residuals = ols_results.resid
    fitted = ols_results.fittedvalues
    
    # Residuals vs Fitted
    axes[0].scatter(fitted, residuals, alpha=0.6, s=40, color='navy', edgecolors='black', linewidth=0.5)
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Fitted Values', fontsize=11)
    axes[0].set_ylabel('Residuals', fontsize=11)
    axes[0].set_title('Residuals vs Fitted', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Normal Q-Q Plot', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(f'Residual Diagnostics: {index_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved residual diagnostics to {save_path}")
    
    plt.close()


def run_experiment_3(
    index_data: Dict[str, pd.Series],
    n: int = 22,
    save_results: bool = True
) -> Dict:
    """
    Run Experiment 3 for all indices.
    
    Parameters
    ----------
    index_data : dict
        Dictionary mapping index names to price series
    n : int, default=22
        Scaling factor for fractal dimension
    save_results : bool, default=True
        Whether to save results to files
        
    Returns
    -------
    dict
        Regression results for each index
    """
    from .utils import compute_log_returns
    
    results_dir = create_results_directory()
    all_regression_results = {}
    regression_summary = []
    adf_test_results = []
    
    for index_name, prices in index_data.items():
        print(f"\n{'='*60}")
        print(f"Processing {index_name}")
        print(f"{'='*60}")
        
        # Compute returns and fractal dimension
        returns = compute_log_returns(prices).dropna()
        D = compute_fractal_dimension(returns, prices, n)
        
        # Identify breaches and compute event windows
        breach_events = identify_breach_events(D)
        event_data = compute_event_windows(breach_events, D, prices)
        
        if len(event_data) < 3:
            print(f"\nSkipping {index_name}: insufficient events ({len(event_data)})")
            continue
        
        print(f"\nEvent Data Summary:")
        print(f"  Valid events: {len(event_data)}")
        print(f"  Delta D/D range: [{event_data['delta_D_over_D'].min():.4f}, {event_data['delta_D_over_D'].max():.4f}]")
        print(f"  Ret_22 range: [{event_data['Ret_22'].min():.4f}, {event_data['Ret_22'].max():.4f}]")
        
        # ADF tests on event-level series
        print(f"\n{'='*60}")
        print("STATIONARITY TESTS")
        print(f"{'='*60}")
        
        adf_delta_D = run_adf_test(event_data['delta_D_over_D'], f"{index_name}: delta_D_over_D")
        adf_delta_D['index'] = index_name
        adf_test_results.append(adf_delta_D)
        
        adf_ret = run_adf_test(event_data['Ret_22'], f"{index_name}: Ret_22")
        adf_ret['index'] = index_name
        adf_test_results.append(adf_ret)
        
        # ADF test on daily D series
        adf_D_daily = run_adf_test(D, f"{index_name}: Daily D(t)")
        adf_D_daily['index'] = index_name
        adf_test_results.append(adf_D_daily)
        
        # OLS Regression
        ols_results, summary_dict = run_ols_regression(event_data, index_name)
        
        if ols_results is not None:
            all_regression_results[index_name] = {
                'ols_results': ols_results,
                'event_data': event_data,
                'summary': summary_dict
            }
            regression_summary.append(summary_dict)
            
            if save_results:
                # Save scatter plot
                scatter_path = os.path.join(results_dir, f'scatter_regression_{index_name}.png')
                plot_regression_scatter(event_data, ols_results, index_name, scatter_path)
                
                # Save residual diagnostics
                resid_path = os.path.join(results_dir, f'residual_diagnostics_{index_name}.png')
                plot_residual_diagnostics(ols_results, index_name, resid_path)
                
                # Save regression summary text
                summary_path = os.path.join(results_dir, f'regression_results_{index_name}.txt')
                with open(summary_path, 'w') as f:
                    f.write(f"OLS Regression Results: {index_name}\n")
                    f.write("="*60 + "\n\n")
                    f.write(str(ols_results.summary()))
                print(f"\nSaved regression summary to {summary_path}")
    
    # Save summary tables
    if save_results and len(regression_summary) > 0:
        # Regression summary
        reg_summary_df = pd.DataFrame(regression_summary)
        reg_path = os.path.join(results_dir, 'regression_summary_table.csv')
        reg_summary_df.to_csv(reg_path, index=False)
        print(f"\nSaved regression summary table to {reg_path}")
        
        # ADF test results
        adf_df = pd.DataFrame(adf_test_results)
        adf_path = os.path.join(results_dir, 'adf_test_results.csv')
        adf_df.to_csv(adf_path, index=False)
        print(f"Saved ADF test results to {adf_path}")
    
    return all_regression_results


if __name__ == "__main__":
    from .data_download import get_closing_prices
    
    # Load data for all indices
    indices = ['SP500', 'FTSE100', 'JSE']
    index_data = {}
    
    for index_name in indices:
        prices = get_closing_prices(index_name)
        index_data[index_name] = prices
    
    # Run experiment
    results = run_experiment_3(index_data)
