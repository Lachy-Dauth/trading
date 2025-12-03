# Trading Analysis & CGT Calculator

A suite of Python tools designed for Australian (ASX) investors to analyze portfolio performance and calculate Capital Gains Tax (CGT) obligations with advanced tax minimization strategies. A web-based version of the CGT calculator can be found [here](https://lachy-dauth.github.io/trading/web_calculator/). 
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

### Web Calculator
For a graphical interface with drag-and-drop and copy-paste support:
1. Open [This](https://lachy-dauth.github.io/trading/web_calculator/) in your web browser.
2. Use the tabs to either upload files or paste data directly.

## Input File Formats

Directly copy transaction history from the Stake Aus dashboard.

**Trades Input:**
Copy from the "Trades" section. The parser expects blocks of 3 lines per trade:
1. `Date` `Ticker` (e.g., `2025-12-03 WTC`)
2. `Action` (e.g., `Buy` or `Sell`)
3. `Details` (e.g., `100 shares @ $70.30`)

**Transfers Input:**
Copy from "Funds & Balances" > "Transactions". The parser expects blocks of 3 lines per transfer:
1. `Date` `Description` (e.g., `2 Dec 2025 Accumulate distribution`)
2. `Status` (e.g., `Complete`)
3. `Amount` (e.g., `+A$19.72`)

## Output

- **`cgt_report.md`**: A comprehensive table of all realized events and current holdings.
- **`portfolio_performance.png`**: A visual chart comparing your cumulative returns vs. the benchmark.
