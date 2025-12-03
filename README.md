# Trading Analysis & CGT Calculator

A suite of Python tools designed for Australian (ASX) investors to analyze portfolio performance and calculate Capital Gains Tax (CGT) obligations with advanced tax minimization strategies.

## Features

### 1. Performance Analysis (`analyze_trades.py`)
- Calculates portfolio performance metrics:
  - Total Return
  - Annualized Volatility
  - Sharpe Ratio (using `AAA.AX` as risk-free rate)
- Benchmarks performance against an index/ETF (default: `GHHF.AX`).
- Generates a performance comparison chart (`portfolio_performance.png`).

### 2. CGT Calculator (`cgt_calculator.py`)
- **Tax Minimization Logic:** Automatically selects which share lots to sell to minimize tax liability:
  1.  Realized Losses (to offset gains)
  2.  Long-Term Gains (held > 12 months, eligible for 50% discount) - Lowest gain first.
  3.  Short-Term Gains (held < 12 months) - Lowest gain first.
- **Advanced Mode (`--advanced`):**
  - Handles AMIT cost base adjustments (increases and decreases) from dividend statements.
  - Caches user inputs for adjustments in `cgt_cache.json` to avoid repetitive data entry.
- **Reporting:** Generates a detailed Markdown report (`cgt_report.md`) including:
  - Realized Gains/Losses per lot.
  - Estimated Taxable Gain.
  - Remaining Holdings (Unrealized).
  - Log of Cost Base Adjustments.

## Setup

1. **Prerequisites:** Python 3.9+
2. **Install Dependencies:**
   ```bash
   pip install pandas yfinance matplotlib numpy
   ```

## Usage

### Performance Analysis
Run the analysis script to see how your trading strategy compares to the market.
```bash
python analyze_trades.py
```

### CGT Calculation
Run the calculator to generate a tax report.
```bash
python cgt_calculator.py
```

**Advanced Mode (Cost Base Adjustments):**
Use this flag if you want to process dividends and apply AMIT cost base adjustments.
```bash
python cgt_calculator.py --advanced
```
*You will be prompted to enter adjustment percentages for dividends found in `transfers.txt`. Enter positive values for decreases (tax deferred) and negative values for increases (shortfall).*

## Input File Formats

The tools expect two tab-separated text files in the working directory.

### `trades.txt`
Contains trade history.
**Format:** `Date` | `Ticker` | `Action` | `Quantity` | `Price` | `Amount`
```text
2025-12-03	WTC.AX	Buy	100	70.30	-7033.00
2025-12-03	WTC.AX	Sell	100	70.80	7077.00
```
*Note: The script handles reverse chronological order (newest first) automatically.*

### `transfers.txt`
Contains cash movements and dividends.
**Format:** `Date` | `Description` | `Amount` | `Category`
```text
2025-07-16	GHHF Dividend	150.00	Dividend
2025-07-01	Deposit	5000.00	Transfer
```

## Output

- **`cgt_report.md`**: A comprehensive table of all realized events and current holdings.
- **`portfolio_performance.png`**: A visual chart comparing your cumulative returns vs. the benchmark.
