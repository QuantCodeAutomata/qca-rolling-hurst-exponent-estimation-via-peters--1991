"""
Utility functions for financial data processing and analysis.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import os


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute daily log returns from price series.
    
    Parameters
    ----------
    prices : pd.Series
        Time series of prices indexed by date
        
    Returns
    -------
    pd.Series
        Log returns: ln(P_t / P_{t-1})
    """
    return np.log(prices / prices.shift(1))


def validate_price_data(prices: pd.Series) -> None:
    """
    Validate price data for common issues.
    
    Parameters
    ----------
    prices : pd.Series
        Price series to validate
        
    Raises
    ------
    ValueError
        If data contains non-positive prices or duplicates
    """
    if (prices <= 0).any():
        raise ValueError("Price data contains non-positive values")
    
    if prices.index.duplicated().any():
        raise ValueError("Price data contains duplicate dates")


def get_month_end_dates(prices: pd.Series, start_date: Optional[str] = None) -> pd.DatetimeIndex:
    """
    Extract month-end trading dates from price series.
    
    Parameters
    ----------
    prices : pd.Series
        Price series with DatetimeIndex
    start_date : str, optional
        Start date for filtering month-ends
        
    Returns
    -------
    pd.DatetimeIndex
        Month-end trading dates
    """
    month_ends = prices.resample('M').last().index
    
    if start_date is not None:
        month_ends = month_ends[month_ends >= pd.Timestamp(start_date)]
    
    return month_ends


def get_rolling_window_data(
    returns: pd.Series,
    anchor_date: pd.Timestamp,
    window_months: int = 36
) -> pd.Series:
    """
    Extract rolling window of returns for a given anchor date.
    
    Parameters
    ----------
    returns : pd.Series
        Daily log returns series
    anchor_date : pd.Timestamp
        End date of the window (month-end)
    window_months : int, default=36
        Number of calendar months in the window
        
    Returns
    -------
    pd.Series
        Returns within the rolling window
    """
    # Calculate start date as window_months before anchor_date
    start_date = anchor_date - pd.DateOffset(months=window_months)
    
    # Extract returns in the window (inclusive of anchor_date)
    window_returns = returns[(returns.index > start_date) & (returns.index <= anchor_date)]
    
    return window_returns


def create_results_directory() -> str:
    """
    Create results directory if it doesn't exist.
    
    Returns
    -------
    str
        Path to results directory
    """
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def forward_fill_monthly_to_daily(
    monthly_series: pd.Series,
    daily_dates: pd.DatetimeIndex
) -> pd.Series:
    """
    Forward-fill monthly values to daily frequency without look-ahead bias.
    
    For each trading day t, assign the value from the most recent month-end <= t.
    
    Parameters
    ----------
    monthly_series : pd.Series
        Monthly series indexed by month-end dates
    daily_dates : pd.DatetimeIndex
        Daily trading dates to fill
        
    Returns
    -------
    pd.Series
        Daily series with forward-filled values
    """
    # Reindex to daily frequency and forward fill
    daily_series = monthly_series.reindex(daily_dates, method='ffill')
    
    return daily_series


def compute_simple_return(
    prices: pd.Series,
    start_idx: int,
    end_idx: int
) -> float:
    """
    Compute simple arithmetic return between two indices.
    
    Parameters
    ----------
    prices : pd.Series
        Price series
    start_idx : int
        Starting position index
    end_idx : int
        Ending position index
        
    Returns
    -------
    float
        Simple return: (P_end - P_start) / P_start
    """
    p_start = prices.iloc[start_idx]
    p_end = prices.iloc[end_idx]
    
    return (p_end - p_start) / p_start


def save_results_summary(
    results_dict: dict,
    filename: str = "RESULTS.md"
) -> None:
    """
    Save experimental results to markdown file.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary containing results to save
    filename : str, default="RESULTS.md"
        Output filename
    """
    results_dir = create_results_directory()
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write("# Experimental Results Summary\n\n")
        f.write("## Rolling Hurst Exponent and Fractal Dimension Analysis\n\n")
        
        for key, value in results_dict.items():
            f.write(f"### {key}\n\n")
            if isinstance(value, pd.DataFrame):
                f.write(value.to_markdown())
                f.write("\n\n")
            elif isinstance(value, str):
                f.write(value)
                f.write("\n\n")
            else:
                f.write(f"{value}\n\n")
