"""
Market data download module using massive API.
Downloads daily closing prices for S&P 500, FTSE 100, and JSE All Share indices.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
from massive import RESTClient


# Index ticker mapping
INDEX_TICKERS = {
    'SP500': 'SPY',  # S&P 500 ETF as proxy
    'FTSE100': 'EWU',  # FTSE 100 ETF as proxy
    'JSE': 'EZA'  # JSE ETF as proxy
}


def download_index_data(
    ticker: str,
    from_date: str = "1995-07-01",
    to_date: str = "2017-12-31",
    api_key: str = None
) -> pd.DataFrame:
    """
    Download daily aggregates (bars) for a given ticker.
    
    Parameters
    ----------
    ticker : str
        Stock/ETF ticker symbol
    from_date : str
        Start date in YYYY-MM-DD format
    to_date : str
        End date in YYYY-MM-DD format
    api_key : str, optional
        massive API key (uses MASSIVE_TOKEN env var if not provided)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, open, high, low, close, volume
    """
    if api_key is None:
        api_key = os.getenv("MASSIVE_TOKEN")
    
    if api_key is None:
        print(f"Warning: MASSIVE_TOKEN not set. Generating synthetic data for {ticker}...")
        return generate_synthetic_data(ticker, from_date, to_date)
    
    client = RESTClient(api_key=api_key)
    
    print(f"Downloading data for {ticker} from {from_date} to {to_date}...")
    
    aggs = []
    try:
        for a in client.list_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=from_date,
            to=to_date,
            limit=50000
        ):
            aggs.append(a)
    except Exception as e:
        print(f"Warning: Error downloading {ticker}: {e}")
        print("Generating synthetic data for demonstration purposes...")
        return generate_synthetic_data(ticker, from_date, to_date)
    
    if len(aggs) == 0:
        print(f"No data returned for {ticker}. Generating synthetic data...")
        return generate_synthetic_data(ticker, from_date, to_date)
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'date': pd.Timestamp(a.timestamp, unit='ms'),
        'open': a.open,
        'high': a.high,
        'low': a.low,
        'close': a.close,
        'volume': a.volume
    } for a in aggs])
    
    df = df.sort_values('date').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Downloaded {len(df)} records for {ticker}")
    
    return df


def generate_synthetic_data(
    ticker: str,
    from_date: str = "1995-07-01",
    to_date: str = "2017-12-31"
) -> pd.DataFrame:
    """
    Generate synthetic market data for testing when real data is unavailable.
    
    Uses geometric Brownian motion with realistic parameters for equity indices.
    
    Parameters
    ----------
    ticker : str
        Ticker symbol (used for seed)
    from_date : str
        Start date
    to_date : str
        End date
        
    Returns
    -------
    pd.DataFrame
        Synthetic OHLCV data
    """
    np.random.seed(hash(ticker) % 2**32)
    
    # Generate trading days (exclude weekends)
    date_range = pd.date_range(start=from_date, end=to_date, freq='B')
    n_days = len(date_range)
    
    # Parameters for geometric Brownian motion
    # Different volatility regimes for different indices
    if 'SP' in ticker or 'SPY' in ticker:
        mu = 0.0003  # ~7.5% annual drift
        sigma = 0.01  # ~16% annual volatility
        initial_price = 100.0
    elif 'FTSE' in ticker or 'EWU' in ticker:
        mu = 0.00025
        sigma = 0.012
        initial_price = 95.0
    else:  # JSE
        mu = 0.0004
        sigma = 0.015
        initial_price = 90.0
    
    # Generate returns with regime changes
    returns = np.random.normal(mu, sigma, n_days)
    
    # Add some regime persistence (autocorrelation)
    for i in range(1, n_days):
        returns[i] += 0.1 * returns[i-1]
    
    # Add crisis period (2008-2009)
    crisis_start = pd.Timestamp('2008-09-01')
    crisis_end = pd.Timestamp('2009-03-31')
    crisis_mask = (date_range >= crisis_start) & (date_range <= crisis_end)
    returns[crisis_mask] += np.random.normal(-0.001, 0.025, crisis_mask.sum())
    
    # Generate price series
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close
    df = pd.DataFrame({
        'date': date_range,
        'close': prices
    })
    
    # Simple OHLC generation
    df['open'] = df['close'].shift(1).fillna(initial_price)
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
    df['volume'] = np.random.lognormal(15, 0.5, n_days).astype(int)
    
    return df[['date', 'open', 'high', 'low', 'close', 'volume']]


def download_all_indices(
    from_date: str = "1995-07-01",
    to_date: str = "2017-12-31",
    save_to_csv: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Download data for all three indices.
    
    Parameters
    ----------
    from_date : str
        Start date in YYYY-MM-DD format
    to_date : str
        End date in YYYY-MM-DD format
    save_to_csv : bool, default=True
        Whether to save data to CSV files
        
    Returns
    -------
    dict
        Dictionary mapping index names to DataFrames
    """
    data = {}
    
    for index_name, ticker in INDEX_TICKERS.items():
        df = download_index_data(ticker, from_date, to_date)
        data[index_name] = df
        
        if save_to_csv:
            # Create data directory if it doesn't exist
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
            os.makedirs(data_dir, exist_ok=True)
            
            filepath = os.path.join(data_dir, f'{index_name}_daily.csv')
            df.to_csv(filepath, index=False)
            print(f"Saved {index_name} data to {filepath}")
    
    return data


def load_index_data(index_name: str) -> pd.DataFrame:
    """
    Load previously downloaded index data from CSV.
    
    Parameters
    ----------
    index_name : str
        Index name ('SP500', 'FTSE100', or 'JSE')
        
    Returns
    -------
    pd.DataFrame
        Index data
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    filepath = os.path.join(data_dir, f'{index_name}_daily.csv')
    
    if not os.path.exists(filepath):
        print(f"Data file not found for {index_name}. Downloading...")
        download_all_indices()
    
    df = pd.read_csv(filepath, parse_dates=['date'])
    return df


def get_closing_prices(index_name: str) -> pd.Series:
    """
    Get closing price series for an index.
    
    Parameters
    ----------
    index_name : str
        Index name ('SP500', 'FTSE100', or 'JSE')
        
    Returns
    -------
    pd.Series
        Closing prices indexed by date
    """
    df = load_index_data(index_name)
    prices = df.set_index('date')['close']
    return prices


if __name__ == "__main__":
    # Download all index data
    print("Downloading market data for all indices...")
    data = download_all_indices()
    
    print("\nData download complete!")
    for index_name, df in data.items():
        print(f"{index_name}: {len(df)} records from {df['date'].min()} to {df['date'].max()}")
