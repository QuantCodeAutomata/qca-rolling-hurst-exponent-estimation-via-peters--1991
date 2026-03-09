# Rolling Hurst Exponent Estimation via Peters (1991) Rescaled-Range Method

This repository implements a comprehensive quantitative finance research project analyzing time-varying Hurst exponents and fractal dimensions for major equity indices.

## Research Overview

The project reproduces and extends the Peters (1991) rescaled-range (R/S) methodology to:

1. **Estimate time-varying Hurst exponents H(t)** for S&P 500, FTSE 100, and JSE All Share indices using 36-month rolling windows
2. **Compute daily fractal dimensions D(t)** using the Joshi (2014) formula
3. **Identify regime-specific market signals** by detecting threshold breaches in fractal dimension
4. **Predict post-breach returns** using pre-breach fractal dimension changes

## Project Structure

```
.
├── src/
│   ├── data_download.py          # Market data retrieval via massive API
│   ├── experiment_1.py            # Rolling Hurst exponent estimation
│   ├── experiment_2.py            # Daily fractal dimension computation
│   ├── experiment_3.py            # Breach-event OLS regression
│   ├── experiment_4.py            # Breach-regime association analysis
│   └── utils.py                   # Shared utility functions
├── tests/
│   ├── test_experiment_1.py       # Experiment 1 validation tests
│   ├── test_experiment_2.py       # Experiment 2 validation tests
│   ├── test_experiment_3.py       # Experiment 3 validation tests
│   └── test_experiment_4.py       # Experiment 4 validation tests
├── results/
│   ├── RESULTS.md                 # Summary of all experimental results
│   └── (plots and CSV outputs)
├── data/
│   └── (downloaded market data)
├── run_experiments.py             # Main pipeline execution script
└── requirements.txt               # Python dependencies

```

## Key Experiments

### Experiment 1: Rolling Hurst Exponent Estimation
- Implements Peters (1991) R/S procedure with 36-month rolling windows
- Produces monthly H(t) estimates for regime classification
- Identifies mean-reverting (H<0.5), random walk (H≈0.5), and trending (H>0.5) regimes

### Experiment 2: Daily Fractal Dimension Estimation
- Computes daily fractal dimension D(t) using Joshi formula
- Detects threshold breaches where D crosses below 1.25
- Sensitivity analysis across multiple scaling factors (n ∈ {5,10,15,20,22})

### Experiment 3: Breach-Event OLS Regression
- Regresses 22-day post-breach returns on 5-day pre-breach D changes
- Target: R² ≈ 0.85, slope ≈ -1.2 (paper validation)
- Includes ADF stationarity tests

### Experiment 4: Breach-Regime Association
- Tests whether D≤1.25 breaches occur predominantly in H>0.5 trending regimes
- Target: ~95% of JSE breaches in trending regime
- Binomial and chi-square statistical tests

## Installation

```bash
# Clone the repository
git clone https://github.com/QuantCodeAutomata/qca-rolling-hurst-exponent-estimation-via-peters--1991.git
cd qca-rolling-hurst-exponent-estimation-via-peters--1991

# Install dependencies
pip install -r requirements.txt

# Set up environment variable for massive API
export MASSIVE_TOKEN="your_api_key_here"
```

## Usage

### Run All Experiments
```bash
python run_experiments.py
```

### Run Individual Experiments
```bash
python -m src.experiment_1  # Rolling Hurst exponent
python -m src.experiment_2  # Fractal dimension
python -m src.experiment_3  # Regression analysis
python -m src.experiment_4  # Regime association
```

### Run Tests
```bash
pytest tests/ -v
```

## Data Requirements

- **Indices**: S&P 500, FTSE 100, JSE All Share
- **Period**: 1995-07-01 to 2017-12-31
- **Frequency**: Daily (trading calendar)
- **Source**: massive API

## Methodology

### Peters (1991) R/S Procedure
1. Partition N-length return series into non-overlapping blocks of length n
2. For each block: compute mean-adjusted cumulative deviations
3. Calculate range R and standard deviation σ for each block
4. Compute rescaled range (R/S) and average across blocks
5. OLS regression of ln(R/S) vs ln(n) yields Hurst exponent H

### Joshi Fractal Dimension Formula
D(t) = ln(N(t)) / ln(n)

where N(t) = (n × Σ|r_j|) / |R_n|

## Expected Results

1. **Developed markets** (S&P 500, FTSE 100): H(t) fluctuates around 0.5
2. **Emerging market** (JSE): More pronounced regime departures from H=0.5
3. **Crisis periods**: Notable H(t) deviations around 2008-2009
4. **Fractal dimension breaches**: Predominantly occur during H>0.5 trending regimes
5. **Predictive relationship**: Pre-breach D drop predicts post-breach return (R²≈0.85)

## References

- Peters, E.E. (1991). *Chaos and Order in the Capital Markets*. John Wiley & Sons.
- Joshi, R.M. (2014a,b). Fractal dimension analysis in financial markets.

## License

This project is for academic and research purposes.

## Author

QCA Agent - QuantCodeAutomata
