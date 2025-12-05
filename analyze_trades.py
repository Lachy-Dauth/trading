import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import re
from datetime import datetime, timedelta
import numpy as np

import argparse

def parse_trades(file_path):
    trades = []
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    for i in range(0, len(lines), 3):
        if i + 2 >= len(lines):
            break
            
        line1 = lines[i]
        line2 = lines[i+1]
        line3 = lines[i+2]
        
        # Parse Line 1: Date and Ticker
        parts1 = line1.split('\t')
        date_str = parts1[0].strip()
        ticker = parts1[1].strip()
        
        # Parse Line 2: Action
        action = line2.strip()
        
        # Parse Line 3: Quantity, Price, Amount
        parts3 = line3.split('\t')
        details = parts3[0]
        amount_str = parts3[1] if len(parts3) > 1 else "0"
        
        # Extract Quantity
        qty_match = re.search(r'(\d+)\s+shares', details)
        quantity = int(qty_match.group(1)) if qty_match else 0
        
        # Extract Amount
        amount_clean = amount_str.replace('A$', '').replace(',', '').replace('+', '').replace(' ', '')
        try:
            amount = float(amount_clean)
        except ValueError:
            amount = 0.0
        
        # Extract Price
        price_match = re.search(r'\$(\d+\.\d+)', details)
        if price_match:
            price = float(price_match.group(1))
        else:
            price = abs(amount / quantity) if quantity else 0
            
        trades.append({
            'Date': date_str,
            'Ticker': ticker + '.AX', # Assuming ASX
            'Action': action,
            'Quantity': quantity,
            'Price': price,
            'Amount': amount
        })
        
    df = pd.DataFrame(trades)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date')

def get_market_data(tickers, start_date, end_date):
    data = {}
    dividends = {}
    
    print(f"Fetching data for {tickers} from {start_date} to {end_date}")
    
    for ticker in tickers:
        t = yf.Ticker(ticker)
        # Fetch history
        hist = t.history(start=start_date, end=end_date)
        if not hist.empty:
            # Remove timezone from index
            hist.index = hist.index.tz_localize(None)
            data[ticker] = hist['Close']
            
            divs = t.dividends
            if not divs.empty:
                divs.index = divs.index.tz_localize(None)
                dividends[ticker] = divs[(divs.index >= start_date) & (divs.index <= end_date)]
            else:
                dividends[ticker] = pd.Series(dtype=float)
        else:
            print(f"Warning: No data found for {ticker}")
            
    return pd.DataFrame(data), dividends

def calculate_portfolio(trades_df, transfers_df, price_data):
    # Create a date range from first activity to last available price date
    if price_data.empty:
        print("No price data available.")
        return pd.DataFrame()
        
    start_date = min(trades_df['Date'].min(), transfers_df['Date'].min())
    end_date = price_data.index.max()
    
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Optimized approach:
    # 1. Reindex price data to all days (ffill)
    price_data_filled = price_data.reindex(all_dates).ffill()
    
    # 2. Create a dataframe of holdings over time
    holdings_df = pd.DataFrame(0, index=all_dates, columns=trades_df['Ticker'].unique())
    cash_series = pd.Series(0.0, index=all_dates)
    
    # Process Trades (Stock changes and Cash changes from trades)
    trades_by_date = trades_df.groupby('Date')
    for date, day_trades in trades_by_date:
        if date in all_dates:
            for _, trade in day_trades.iterrows():
                ticker = trade['Ticker']
                qty = trade['Quantity']
                amount = trade['Amount']
                action = trade['Action']
                
                if action == 'Buy':
                    holdings_df.loc[date:, ticker] += qty
                    cash_series.loc[date:] += amount
                elif action == 'Sell':
                    holdings_df.loc[date:, ticker] -= qty
                    cash_series.loc[date:] += amount

    # Process Transfers (Deposits, Withdrawals, Dividends)
    transfers_by_date = transfers_df.groupby('Date')
    for date, day_transfers in transfers_by_date:
        if date in all_dates:
            for _, transfer in day_transfers.iterrows():
                amount = transfer['Amount']
                # All transfers affect cash directly
                cash_series.loc[date:] += amount
    
    # Portfolio Value = Cash + Stock Value
    stock_value = (holdings_df * price_data_filled).sum(axis=1)
    total_value = cash_series + stock_value
    
    return pd.DataFrame({
        'Total Value': total_value,
        'Cash': cash_series,
        'Stock Value': stock_value
    })

def parse_transfers(file_path):
    transfers = []
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    for i in range(0, len(lines), 3):
        if i + 2 >= len(lines):
            break
            
        line1 = lines[i]
        line2 = lines[i+1]
        line3 = lines[i+2]
        
        # Parse Line 1: Date and Description
        parts1 = line1.split('\t')
        date_str = parts1[0].strip()
        description = parts1[1].strip() if len(parts1) > 1 else ""
        
        # Parse Line 3: Amount
        amount_str = line3
        amount_clean = amount_str.replace('A$', '').replace(',', '').replace(' ', '')
        try:
            amount = float(amount_clean)
        except ValueError:
            amount = 0.0
            
        # Categorize
        category = 'Transfer'
        if 'Trade settlement' in description:
            category = 'Ignore'
        elif 'Dividend' in description:
            category = 'Dividend'
        
        if category != 'Ignore':
            transfers.append({
                'Date': date_str,
                'Description': description,
                'Amount': amount,
                'Category': category
            })
            
    df = pd.DataFrame(transfers)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date')

def calculate_benchmark(transfers_df, benchmark_ticker, price_series, dividend_series):
    if price_series.empty:
        return pd.DataFrame()
        
    start_date = transfers_df['Date'].min()
    end_date = price_series.index.max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Reindex price data
    price_filled = price_series.reindex(all_dates).ffill()
    
    shares_held = pd.Series(0.0, index=all_dates)
    cash_held = pd.Series(0.0, index=all_dates) # From dividends
    
    # Process Transfers (Deposits/Withdrawals only)
    transfers_by_date = transfers_df[transfers_df['Category'] == 'Transfer'].groupby('Date')
    
    for date, day_transfers in transfers_by_date:
        if date in all_dates:
            net_transfer = day_transfers['Amount'].sum()
            
            # Buy/Sell benchmark at that day's price
            if date in price_filled.index and not pd.isna(price_filled[date]):
                price = price_filled[date]
                shares_bought = net_transfer / price
                shares_held.loc[date:] += shares_bought
            else:
                # If no price, maybe keep as cash? Or assume bought at next available price?
                # For simplicity, let's assume ffill gave us a price, or if it's before first price, we can't buy.
                pass

    # Process Dividends
    if not dividend_series.empty:
        for div_date, div_amount in dividend_series.items():
            if div_date in shares_held.index:
                # Check shares held on previous day
                prev_day = div_date - timedelta(days=1)
                if prev_day in shares_held.index:
                    qty = shares_held[prev_day]
                    total_div = qty * div_amount
                    cash_held.loc[div_date:] += total_div
                    
    total_value = (shares_held * price_filled) + cash_held
    
    return pd.DataFrame({
        'Total Value': total_value,
        'Cash': cash_held,
        'Stock Value': (shares_held * price_filled)
    })

def print_summary(portfolio, benchmark=None, benchmark_name=None, metrics=None, no_return=None):
    print("\n" + "="*40)
    print("PERFORMANCE SUMMARY")
    print("="*40)
    
    # Calculate metrics
    final_date = portfolio.index[-1]
    final_val = portfolio['Total Value'].iloc[-1]
    
    print(f"{'Metric':<20} | {'Portfolio':<15}")
    print("-" * 40)
    print(f"{'Final Value':<20} | ${final_val:,.2f}")
    
    if no_return is not None and not no_return.empty:
        invested_val = no_return['Total Value'].iloc[-1]
        print(f"{'Net Invested':<20} | ${invested_val:,.2f}")
        
        total_return_abs = final_val - invested_val
        print(f"{'Total Return $':<20} | ${total_return_abs:,.2f}")
        
        if invested_val != 0:
            total_return_pct = (total_return_abs / invested_val) * 100
            print(f"{'Total Return %':<20} | {total_return_pct:.2f}%")

    if metrics:
        if metrics.get('Return') is not None:
            print(f"{'Ann. Return':<20} | {metrics['Return']*100:.2f}%")
        if metrics.get('Volatility') is not None:
            print(f"{'Ann. Volatility':<20} | {metrics['Volatility']*100:.2f}%")
        if metrics.get('Sharpe') is not None:
            print(f"{'Sharpe Ratio':<20} | {metrics['Sharpe']:.2f}")
    
    if benchmark is not None and not benchmark.empty:
        print("-" * 40)
        bench_final_val = benchmark['Total Value'].iloc[-1]
        print(f"{benchmark_name + ' Value':<20} | ${bench_final_val:,.2f}")
        
        diff = final_val - bench_final_val
        print(f"{'Difference':<20} | ${diff:,.2f}")
        
        if bench_final_val != 0:
            outperformance = (final_val - bench_final_val) / bench_final_val * 100
            print(f"{'Outperformance %':<20} | {outperformance:+.2f}%")
            
    print("="*40 + "\n")


def calculate_metrics(portfolio_df, transfers_df, rf_price_series, rf_dividend_series):
    metrics = {
        'Sharpe': None,
        'Volatility': None,
        'Return': None
    }
    
    if portfolio_df.empty:
        return metrics
        
    # 1. Calculate Portfolio Daily Returns
    # We need to adjust for Net Flows (Deposits/Withdrawals)
    # R_t = (V_t - V_{t-1} - Flow_t) / V_{t-1}
    # Assuming Flow_t happens at the end of the day or doesn't contribute to return?
    # If we use: R_t = (V_t) / (V_{t-1} + Flow_t) - 1 (Modified Dietz / TWR approximation for daily)
    
    # Get daily net flows
    daily_flows = pd.Series(0.0, index=portfolio_df.index)
    
    # Only consider 'Transfer' category as external flows (Deposits/Withdrawals)
    external_transfers = transfers_df[transfers_df['Category'] == 'Transfer']
    transfers_by_date = external_transfers.groupby('Date')['Amount'].sum()
    
    # Reindex to match portfolio dates
    daily_flows = transfers_by_date.reindex(portfolio_df.index, fill_value=0.0)
    
    # Calculate Portfolio Returns
    # Let's use the "Flow at Start" assumption
    prev_values = portfolio_df['Total Value'].shift(1)
    
    # Denominator: Previous Value + Flow today
    denom = prev_values + daily_flows
    
    # Avoid division by zero
    denom = denom.replace(0, np.nan)
    
    portfolio_returns = (portfolio_df['Total Value'] / denom) - 1
    
    # Calculate Volatility and Return
    valid_returns = portfolio_returns.dropna()
    if len(valid_returns) > 1:
        metrics['Volatility'] = valid_returns.std() * np.sqrt(252)
        metrics['Return'] = valid_returns.mean() * 252
    
    # 2. Calculate Risk-Free Returns (AAA.AX)
    if rf_price_series is not None and not rf_price_series.empty:
        rf_price = rf_price_series.reindex(portfolio_df.index).ffill()
        rf_divs = rf_dividend_series.reindex(portfolio_df.index, fill_value=0.0)
        
        rf_prev = rf_price.shift(1)
        rf_returns = (rf_price + rf_divs - rf_prev) / rf_prev
        
        # 3. Calculate Excess Returns
        excess_returns = portfolio_returns - rf_returns
        
        # Drop NaNs (first day etc)
        excess_returns = excess_returns.dropna()
        
        if len(excess_returns) >= 2:
            # 4. Calculate Sharpe Ratio
            # Annualized
            mean_excess = excess_returns.mean()
            std_excess = excess_returns.std()
            
            if std_excess != 0:
                metrics['Sharpe'] = (mean_excess / std_excess) * np.sqrt(252)
                
    return metrics

def calculate_no_return(transfers_df, start_date, end_date):
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    cash_invested = pd.Series(0.0, index=all_dates)
    
    # Process Transfers (Deposits/Withdrawals only)
    transfers_by_date = transfers_df[transfers_df['Category'] == 'Transfer'].groupby('Date')
    
    for date, day_transfers in transfers_by_date:
        if date in all_dates:
            net_transfer = day_transfers['Amount'].sum()
            cash_invested.loc[date:] += net_transfer
            
    return pd.DataFrame({
        'Total Value': cash_invested
    })

def main():
    parser = argparse.ArgumentParser(description='Analyze trading portfolio.')
    parser.add_argument('--benchmark', type=str, help='Ticker for benchmark ETF (e.g., GHHF.AX)')
    args = parser.parse_args()

    file_path = 'trades.txt'
    transfers_path = 'transfers.txt'
    
    print("Parsing trades...")
    trades_df = parse_trades(file_path)
    
    print("Parsing transfers...")
    transfers_df = parse_transfers(transfers_path)
    
    tickers = list(trades_df['Ticker'].unique())
    if args.benchmark and args.benchmark not in tickers:
        tickers.append(args.benchmark)
        
    # Add AAA.AX for risk-free rate
    rf_ticker = 'AAA.AX'
    if rf_ticker not in tickers:
        tickers.append(rf_ticker)
        
    start_date = min(trades_df['Date'].min(), transfers_df['Date'].min())
    end_date = datetime.now() # Use current real time
    
    # If trades are in the future relative to now, we can't fetch data.
    if start_date > end_date:
        print(f"Warning: First activity date {start_date} is in the future relative to system time {end_date}.")
        print("Cannot fetch historical data for future dates.")
        return

    print("Fetching market data...")
    price_data, dividend_data = get_market_data(tickers, start_date, end_date)
        
    print("Calculating portfolio...")
    portfolio = calculate_portfolio(trades_df, transfers_df, price_data)
    
    benchmark_portfolio = pd.DataFrame()
    if args.benchmark:
        print(f"Calculating benchmark ({args.benchmark})...")
        if args.benchmark in price_data:
            benchmark_portfolio = calculate_benchmark(
                transfers_df, 
                args.benchmark, 
                price_data[args.benchmark], 
                dividend_data.get(args.benchmark, pd.Series(dtype=float))
            )
            
    # Calculate Metrics (Sharpe, Volatility, Return)
    metrics = None
    rf_ticker = 'AAA.AX'
    
    rf_price = None
    rf_divs = None
    
    if rf_ticker in price_data:
        print(f"Using {rf_ticker} for risk-free rate...")
        rf_price = price_data[rf_ticker]
        rf_divs = dividend_data.get(rf_ticker, pd.Series(dtype=float))
        
    metrics = calculate_metrics(
        portfolio, 
        transfers_df, 
        rf_price, 
        rf_divs
    )
    
    no_return = calculate_no_return(transfers_df, start_date, end_date)
    
    print_summary(portfolio, benchmark_portfolio, args.benchmark, metrics, no_return)
    
    if not portfolio.empty:
        print("Plotting...")
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio.index, portfolio['Total Value'], label='Total Portfolio Value')
        # plt.plot(portfolio.index, portfolio['Cash'], label='Cash Position', linestyle='--')
        # plt.plot(portfolio.index, portfolio['Stock Value'], label='Stock Holdings', linestyle=':')
        
        if not no_return.empty:
            plt.plot(no_return.index, no_return['Total Value'], label='Net Invested Capital (0% Return)', linestyle=':', color='gray')

        if not benchmark_portfolio.empty:
            plt.plot(benchmark_portfolio.index, benchmark_portfolio['Total Value'], label=f'Benchmark ({args.benchmark})', linestyle='--')
            
        plt.title('Portfolio Performance Over Time')
        plt.xlabel('Date')
        plt.ylabel('Value (AUD)')
        plt.legend()
        plt.grid(True)
        plt.savefig('portfolio_performance.png')
        print("Graph saved to portfolio_performance.png")
    else:
        print("Portfolio is empty or could not be calculated.")

if __name__ == "__main__":
    main()
