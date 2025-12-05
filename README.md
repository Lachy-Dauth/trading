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

### 2. Tax-Adjusted Performance Analysis (`analyze_trades_2.py`)
- **Tax-Aware Analysis:** Incorporates tax implications into portfolio performance analysis:
  - Calculates tax liability on dividends and capital gains using a user-specified tax rate.
  - Treats tax on income events as virtual deposits/withdrawals that adjust performance metrics.
  - Shows final portfolio value if all positions were sold today (including unrealized CGT).
  - Uses the same tax minimization logic as `cgt_calculator.py` (50% discount for holdings > 12 months).
- **Benchmark Comparison:** For the benchmark ETF, actually sells shares to pay tax (simulating real tax impact).
- **Enhanced Metrics:**
  - Portfolio value before and after tax adjustments.
  - Tax liability breakdown by event type (dividends, CGT).
  - Comparison against "no return" scenario (cash only, with dividend tax).
- **Generates:** `portfolio_performance_tax.png` chart showing tax-adjusted performance.

### 3. CGT Calculator (`cgt_calculator.py`)
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

**Optional:** Specify a benchmark ETF:
```bash
python analyze_trades.py --benchmark GHHF.AX
```

### Tax-Adjusted Performance Analysis
Run the tax-adjusted analysis script to see performance after accounting for tax implications.
```bash
python analyze_trades_2.py --tax-rate 0.325 --benchmark GHHF.AX
```
*Replace `0.325` with your marginal tax rate as a decimal (e.g., 0.37 for 37%).*

**Key Features:**
- Virtual tax deposits/withdrawals adjust the "no return" baseline
- Benchmark actually sells shares to pay tax, showing true comparison
- Shows final value if sold today, including unrealized CGT

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

- **`cgt_report.md`**: A comprehensive table of all realized events and current holdings (from `cgt_calculator.py`).
- **`portfolio_performance.png`**: A visual chart comparing your cumulative returns vs. the benchmark (from `analyze_trades.py`).
- **`portfolio_performance_tax.png`**: A tax-adjusted performance chart (from `analyze_trades_2.py`).
