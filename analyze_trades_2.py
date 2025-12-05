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

class Lot:
    def __init__(self, date, ticker, quantity, price, cost_base):
        self.date = date
        self.ticker = ticker
        self.quantity = quantity
        self.price = price
        self.cost_base = cost_base

    @property
    def unit_cost_base(self):
        return self.cost_base / self.quantity if self.quantity > 0 else 0

def calculate_portfolio_with_tax(trades_df, transfers_df, price_data, dividend_data, tax_rate):
    """
    Calculate portfolio value with tax adjustments.
    - When dividends are paid, treat tax owed as a deposit (to offset future withdrawal)
    - When CGT events happen, make withdrawals for losses
    """
    if price_data.empty:
        print("No price data available.")
        return pd.DataFrame(), []
        
    start_date = min(trades_df['Date'].min(), transfers_df['Date'].min())
    end_date = price_data.index.max()
    
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Reindex price data
    price_data_filled = price_data.reindex(all_dates).ffill()
    
    # Track holdings over time
    holdings_df = pd.DataFrame(0, index=all_dates, columns=trades_df['Ticker'].unique())
    cash_series = pd.Series(0.0, index=all_dates)
    tax_adjustments = pd.Series(0.0, index=all_dates)  # Track tax adjustments
    
    # Track lots for CGT calculation (using same logic as cgt_calculator.py)
    holdings = {}  # Ticker -> List of Lots
    cgt_events = []  # Track CGT events for reporting
    
    # Process Trades
    trades_by_date = trades_df.groupby('Date')
    for date, day_trades in trades_by_date:
        if date in all_dates:
            for _, trade in day_trades.iterrows():
                ticker = trade['Ticker']
                qty = trade['Quantity']
                amount = trade['Amount']
                action = trade['Action']
                
                if ticker not in holdings:
                    holdings[ticker] = []
                
                if action == 'Buy':
                    holdings_df.loc[date:, ticker] += qty
                    cash_series.loc[date:] += amount
                    
                    # Add lot
                    cost_base = abs(amount)
                    new_lot = Lot(date, ticker, qty, trade['Price'], cost_base)
                    holdings[ticker].append(new_lot)
                    
                elif action == 'Sell':
                    holdings_df.loc[date:, ticker] -= qty
                    cash_series.loc[date:] += amount
                    
                    # Calculate CGT using tax minimization logic
                    remaining_to_sell = qty
                    total_proceeds = amount
                    unit_price = total_proceeds / qty
                    
                    def get_lot_priority(lot):
                        unit_gain = unit_price - lot.unit_cost_base
                        held_duration = date - lot.date
                        is_long_term = held_duration > timedelta(days=365)
                        
                        if unit_gain < 0:
                            return (1, unit_gain)  # Losses first
                        else:
                            if is_long_term:
                                return (2, unit_gain)  # LT gains next
                            else:
                                return (3, unit_gain)  # ST gains last
                    
                    available_lots = holdings[ticker]
                    available_lots.sort(key=get_lot_priority)
                    
                    lots_to_remove = []
                    
                    for lot in available_lots:
                        if remaining_to_sell <= 0:
                            break
                            
                        if lot.quantity <= remaining_to_sell:
                            sold_qty = lot.quantity
                            cost_base_portion = lot.cost_base
                            proceeds_portion = unit_price * sold_qty
                            
                            gain = proceeds_portion - cost_base_portion
                            held_duration = date - lot.date
                            discount_eligible = held_duration > timedelta(days=365)
                            
                            # Calculate tax on this gain/loss
                            if gain > 0:
                                if discount_eligible:
                                    taxable_gain = gain * 0.5  # 50% discount
                                else:
                                    taxable_gain = gain
                                tax_owed = taxable_gain * tax_rate
                                # Tax owed is a withdrawal (negative adjustment)
                                tax_adjustments.loc[date:] -= tax_owed
                            else:
                                # Loss - acts as a withdrawal to offset future gains
                                # We treat this as a "benefit" that reduces future tax
                                # So we add it back as a positive adjustment
                                loss_benefit = abs(gain) * tax_rate
                                tax_adjustments.loc[date:] += loss_benefit
                            
                            cgt_events.append({
                                'Date': date,
                                'Ticker': ticker,
                                'Quantity': sold_qty,
                                'Gain': gain,
                                'DiscountEligible': discount_eligible,
                                'TaxImpact': tax_owed if gain > 0 else -loss_benefit
                            })
                            
                            remaining_to_sell -= sold_qty
                            lots_to_remove.append(lot)
                            
                        else:
                            sold_qty = remaining_to_sell
                            cost_base_portion = (sold_qty / lot.quantity) * lot.cost_base
                            proceeds_portion = unit_price * sold_qty
                            
                            gain = proceeds_portion - cost_base_portion
                            held_duration = date - lot.date
                            discount_eligible = held_duration > timedelta(days=365)
                            
                            if gain > 0:
                                if discount_eligible:
                                    taxable_gain = gain * 0.5
                                else:
                                    taxable_gain = gain
                                tax_owed = taxable_gain * tax_rate
                                tax_adjustments.loc[date:] -= tax_owed
                            else:
                                loss_benefit = abs(gain) * tax_rate
                                tax_adjustments.loc[date:] += loss_benefit
                            
                            cgt_events.append({
                                'Date': date,
                                'Ticker': ticker,
                                'Quantity': sold_qty,
                                'Gain': gain,
                                'DiscountEligible': discount_eligible,
                                'TaxImpact': tax_owed if gain > 0 else -loss_benefit
                            })
                            
                            lot.cost_base -= cost_base_portion
                            lot.quantity -= sold_qty
                            remaining_to_sell = 0
                    
                    for lot in lots_to_remove:
                        holdings[ticker].remove(lot)

    # Process Transfers
    transfers_by_date = transfers_df.groupby('Date')
    for date, day_transfers in transfers_by_date:
        if date in all_dates:
            for _, transfer in day_transfers.iterrows():
                amount = transfer['Amount']
                category = transfer['Category']
                
                if category == 'Dividend':
                    # Dividend income is taxed at full marginal rate
                    tax_owed = amount * tax_rate
                    # Treat tax as a deposit (offset future withdrawal when tax is paid)
                    tax_adjustments.loc[date:] -= tax_owed
                
                # All transfers affect cash
                cash_series.loc[date:] += amount
    
    # Calculate current portfolio value if sold today
    current_price_data = price_data_filled.iloc[-1]
    unrealized_gain = 0.0
    unrealized_tax = 0.0
    
    for ticker, lots in holdings.items():
        if ticker in current_price_data and not pd.isna(current_price_data[ticker]):
            current_price = current_price_data[ticker]
            for lot in lots:
                gain = (current_price * lot.quantity) - lot.cost_base
                held_duration = end_date - lot.date
                discount_eligible = held_duration > timedelta(days=365)
                
                unrealized_gain += gain
                
                if gain > 0:
                    if discount_eligible:
                        taxable_gain = gain * 0.5
                    else:
                        taxable_gain = gain
                    unrealized_tax += taxable_gain * tax_rate
    
    # Portfolio Value = Cash + Stock Value + Tax Adjustments
    stock_value = (holdings_df * price_data_filled).sum(axis=1)
    total_value = cash_series + stock_value + tax_adjustments
    total_value_if_sold = total_value.iloc[-1] - unrealized_tax
    
    return pd.DataFrame({
        'Total Value': total_value,
        'Cash': cash_series,
        'Stock Value': stock_value,
        'Tax Adjustments': tax_adjustments
    }), cgt_events, unrealized_gain, unrealized_tax, total_value_if_sold

def calculate_benchmark_with_tax(transfers_df, benchmark_ticker, price_series, dividend_series, tax_rate):
    """
    Calculate benchmark with tax adjustments.
    For benchmark, actually sell shares to withdraw the tax amount when events happen.
    """
    if price_series.empty:
        return pd.DataFrame()
        
    start_date = transfers_df['Date'].min()
    end_date = price_series.index.max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    price_filled = price_series.reindex(all_dates).ffill()
    
    shares_held = pd.Series(0.0, index=all_dates)
    cash_held = pd.Series(0.0, index=all_dates)
    
    # Track cost base for CGT on sales
    lots = []  # List of (date, quantity, cost_base_total)
    
    # Process Transfers
    transfers_by_date = transfers_df[transfers_df['Category'] == 'Transfer'].groupby('Date')
    
    for date, day_transfers in transfers_by_date:
        if date in all_dates:
            net_transfer = day_transfers['Amount'].sum()
            
            if date in price_filled.index and not pd.isna(price_filled[date]):
                price = price_filled[date]
                shares_bought = net_transfer / price
                shares_held.loc[date:] += shares_bought
                
                # Track as a lot for CGT
                if shares_bought > 0:
                    lots.append({
                        'date': date,
                        'quantity': shares_bought,
                        'cost_base': abs(net_transfer)
                    })
                elif shares_bought < 0:
                    # Selling shares - need to calculate CGT
                    # For simplicity, use FIFO
                    shares_to_sell = abs(shares_bought)
                    proceeds = abs(net_transfer)
                    
                    remaining = shares_to_sell
                    total_cost_base = 0.0
                    
                    lots_copy = []
                    for lot in lots:
                        if remaining <= 0:
                            lots_copy.append(lot)
                            continue
                        
                        if lot['quantity'] <= remaining:
                            total_cost_base += lot['cost_base']
                            remaining -= lot['quantity']
                        else:
                            portion = remaining / lot['quantity']
                            total_cost_base += lot['cost_base'] * portion
                            lot['quantity'] -= remaining
                            lot['cost_base'] -= lot['cost_base'] * portion
                            lots_copy.append(lot)
                            remaining = 0
                    
                    lots = lots_copy

    # Process Dividends
    if not dividend_series.empty:
        for div_date, div_amount in dividend_series.items():
            if div_date in shares_held.index:
                prev_day = div_date - timedelta(days=1)
                if prev_day in shares_held.index:
                    qty = shares_held[prev_day]
                    total_div = qty * div_amount
                    
                    # Tax on dividend
                    tax_owed = total_div * tax_rate
                    
                    # Sell shares to pay tax
                    if qty > 0 and div_date in price_filled.index and not pd.isna(price_filled[div_date]):
                        current_price = price_filled[div_date]
                        shares_to_sell = tax_owed / current_price
                        
                        # Can't sell more than we have
                        shares_to_sell = min(shares_to_sell, qty)
                        
                        # Calculate CGT on the sale
                        remaining = shares_to_sell
                        total_cost_base = 0.0
                        
                        lots_copy = []
                        for lot in lots:
                            if remaining <= 0:
                                lots_copy.append(lot)
                                continue
                            
                            if lot['quantity'] <= remaining:
                                total_cost_base += lot['cost_base']
                                remaining -= lot['quantity']
                            else:
                                portion = remaining / lot['quantity']
                                total_cost_base += lot['cost_base'] * portion
                                lot['quantity'] -= remaining
                                lot['cost_base'] -= lot['cost_base'] * portion
                                lots_copy.append(lot)
                                remaining = 0
                        
                        lots = lots_copy
                        
                        sale_proceeds = shares_to_sell * current_price
                        gain = sale_proceeds - total_cost_base
                        
                        # Apply CGT discount rules similar to portfolio calculation
                        # For simplicity in benchmark, we'll assume FIFO and check first lot
                        if gain > 0 and lots:
                            held_duration = div_date - lots[0]['date']
                            discount_eligible = held_duration > timedelta(days=365)
                            
                            if discount_eligible:
                                taxable_gain = gain * 0.5
                            else:
                                taxable_gain = gain
                            
                            additional_tax = taxable_gain * tax_rate
                            tax_owed += additional_tax
                        elif gain < 0:
                            # Loss reduces tax owed
                            loss_benefit = abs(gain) * tax_rate
                            tax_owed = max(0, tax_owed - loss_benefit)
                        
                        shares_held.loc[div_date:] -= shares_to_sell
                        
                    # Add dividend to cash
                    cash_held.loc[div_date:] += total_div
                    # Subtract tax paid
                    cash_held.loc[div_date:] -= tax_owed
                    
    total_value = (shares_held * price_filled) + cash_held
    
    return pd.DataFrame({
        'Total Value': total_value,
        'Cash': cash_held,
        'Stock Value': (shares_held * price_filled)
    })

def calculate_no_return_with_tax(transfers_df, start_date, end_date):
    """
    Calculate no-return scenario (just holding cash).
    Tax adjustments don't apply here since there's no income or gains.
    """
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    cash_invested = pd.Series(0.0, index=all_dates)
    
    transfers_by_date = transfers_df[transfers_df['Category'] == 'Transfer'].groupby('Date')
    
    for date, day_transfers in transfers_by_date:
        if date in all_dates:
            net_transfer = day_transfers['Amount'].sum()
            cash_invested.loc[date:] += net_transfer
            
    return pd.DataFrame({
        'Total Value': cash_invested
    })

def print_summary(portfolio, benchmark=None, benchmark_name=None, metrics=None, no_return=None, 
                  cgt_events=None, unrealized_gain=None, unrealized_tax=None, total_value_if_sold=None, tax_rate=None):
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY (WITH TAX ADJUSTMENTS)")
    print("="*60)
    
    if tax_rate is not None:
        print(f"Tax Rate: {tax_rate*100:.1f}%")
        print("-" * 60)
    
    # Calculate metrics
    final_date = portfolio.index[-1]
    final_val = portfolio['Total Value'].iloc[-1]
    
    print(f"{'Metric':<30} | {'Portfolio':<20}")
    print("-" * 60)
    print(f"{'Final Value (with tax adj.)':<30} | ${final_val:,.2f}")
    
    if total_value_if_sold is not None:
        print(f"{'Final Value (if sold today)':<30} | ${total_value_if_sold:,.2f}")
        if unrealized_tax is not None:
            print(f"{'  - Unrealized Tax Liability':<30} | ${unrealized_tax:,.2f}")
    
    if no_return is not None and not no_return.empty:
        invested_val = no_return['Total Value'].iloc[-1]
        print(f"{'Net Invested':<30} | ${invested_val:,.2f}")
        
        total_return_abs = final_val - invested_val
        print(f"{'Total Return $ (with tax adj.)':<30} | ${total_return_abs:,.2f}")
        
        if invested_val != 0:
            total_return_pct = (total_return_abs / invested_val) * 100
            print(f"{'Total Return % (with tax adj.)':<30} | {total_return_pct:.2f}%")

    if metrics:
        if metrics.get('Return') is not None:
            print(f"{'Ann. Return':<30} | {metrics['Return']*100:.2f}%")
        if metrics.get('Volatility') is not None:
            print(f"{'Ann. Volatility':<30} | {metrics['Volatility']*100:.2f}%")
        if metrics.get('Sharpe') is not None:
            print(f"{'Sharpe Ratio':<30} | {metrics['Sharpe']:.2f}")
    
    if benchmark is not None and not benchmark.empty:
        print("-" * 60)
        bench_final_val = benchmark['Total Value'].iloc[-1]
        print(f"{benchmark_name + ' Value (with tax)':<30} | ${bench_final_val:,.2f}")
        
        diff = final_val - bench_final_val
        print(f"{'Difference':<30} | ${diff:,.2f}")
        
        if bench_final_val != 0:
            outperformance = (final_val - bench_final_val) / bench_final_val * 100
            print(f"{'Outperformance %':<30} | {outperformance:+.2f}%")
    
    # Print CGT summary
    if cgt_events:
        print("-" * 60)
        print(f"{'CGT Events':<30} | {'Count: ' + str(len(cgt_events)):<20}")
        total_gains = sum(e['Gain'] for e in cgt_events if e['Gain'] > 0)
        total_losses = sum(abs(e['Gain']) for e in cgt_events if e['Gain'] < 0)
        total_tax_impact = sum(e['TaxImpact'] for e in cgt_events)
        print(f"{'  - Total Capital Gains':<30} | ${total_gains:,.2f}")
        print(f"{'  - Total Capital Losses':<30} | ${total_losses:,.2f}")
        print(f"{'  - Net Tax Impact':<30} | ${total_tax_impact:,.2f}")
    
    if unrealized_gain is not None:
        print(f"{'Unrealized Gain':<30} | ${unrealized_gain:,.2f}")
            
    print("="*60 + "\n")

def calculate_metrics(portfolio_df, transfers_df, rf_price_series, rf_dividend_series):
    metrics = {
        'Sharpe': None,
        'Volatility': None,
        'Return': None
    }
    
    if portfolio_df.empty:
        return metrics
        
    daily_flows = pd.Series(0.0, index=portfolio_df.index)
    external_transfers = transfers_df[transfers_df['Category'] == 'Transfer']
    transfers_by_date = external_transfers.groupby('Date')['Amount'].sum()
    daily_flows = transfers_by_date.reindex(portfolio_df.index, fill_value=0.0)
    
    prev_values = portfolio_df['Total Value'].shift(1)
    denom = prev_values + daily_flows
    denom = denom.replace(0, np.nan)
    
    portfolio_returns = (portfolio_df['Total Value'] / denom) - 1
    
    valid_returns = portfolio_returns.dropna()
    if len(valid_returns) > 1:
        metrics['Volatility'] = valid_returns.std() * np.sqrt(252)
        metrics['Return'] = valid_returns.mean() * 252
    
    if rf_price_series is not None and not rf_price_series.empty:
        rf_price = rf_price_series.reindex(portfolio_df.index).ffill()
        rf_divs = rf_dividend_series.reindex(portfolio_df.index, fill_value=0.0)
        
        rf_prev = rf_price.shift(1)
        rf_returns = (rf_price + rf_divs - rf_prev) / rf_prev
        
        excess_returns = portfolio_returns - rf_returns
        excess_returns = excess_returns.dropna()
        
        if len(excess_returns) >= 2:
            mean_excess = excess_returns.mean()
            std_excess = excess_returns.std()
            
            if std_excess != 0:
                metrics['Sharpe'] = (mean_excess / std_excess) * np.sqrt(252)
                
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Analyze trading portfolio with tax adjustments.')
    parser.add_argument('--tax-rate', type=float, required=True, 
                        help='Tax rate as a decimal (e.g., 0.37 for 37%%)')
    parser.add_argument('--benchmark', type=str, 
                        help='Ticker for benchmark ETF (e.g., GHHF.AX)')
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
        
    rf_ticker = 'AAA.AX'
    if rf_ticker not in tickers:
        tickers.append(rf_ticker)
        
    start_date = min(trades_df['Date'].min(), transfers_df['Date'].min())
    end_date = datetime.now()
    
    if start_date > end_date:
        print(f"Warning: First activity date {start_date} is in the future relative to system time {end_date}.")
        print("Cannot fetch historical data for future dates.")
        return

    print("Fetching market data...")
    price_data, dividend_data = get_market_data(tickers, start_date, end_date)
        
    print("Calculating portfolio with tax adjustments...")
    portfolio, cgt_events, unrealized_gain, unrealized_tax, total_value_if_sold = calculate_portfolio_with_tax(
        trades_df, transfers_df, price_data, dividend_data, args.tax_rate
    )
    
    benchmark_portfolio = pd.DataFrame()
    if args.benchmark:
        print(f"Calculating benchmark ({args.benchmark}) with tax adjustments...")
        if args.benchmark in price_data:
            benchmark_portfolio = calculate_benchmark_with_tax(
                transfers_df, 
                args.benchmark, 
                price_data[args.benchmark], 
                dividend_data.get(args.benchmark, pd.Series(dtype=float)),
                args.tax_rate
            )
            
    metrics = None
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
    
    no_return = calculate_no_return_with_tax(transfers_df, start_date, end_date)
    
    print_summary(
        portfolio, benchmark_portfolio, args.benchmark, metrics, no_return,
        cgt_events, unrealized_gain, unrealized_tax, total_value_if_sold, args.tax_rate
    )
    
    if not portfolio.empty:
        print("Plotting...")
        plt.figure(figsize=(14, 8))
        plt.plot(portfolio.index, portfolio['Total Value'], label='Total Portfolio Value (with tax adj.)', linewidth=2)
        
        if not no_return.empty:
            plt.plot(no_return.index, no_return['Total Value'], 
                    label='Net Invested Capital (0% Return)', linestyle=':', color='gray')

        if not benchmark_portfolio.empty:
            plt.plot(benchmark_portfolio.index, benchmark_portfolio['Total Value'], 
                    label=f'Benchmark ({args.benchmark}) (with tax)', linestyle='--')
            
        plt.title(f'Portfolio Performance Over Time (Tax Rate: {args.tax_rate*100:.1f}%)')
        plt.xlabel('Date')
        plt.ylabel('Value (AUD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('portfolio_performance_with_tax.png', dpi=150, bbox_inches='tight')
        print("Graph saved to portfolio_performance_with_tax.png")
    else:
        print("Portfolio is empty or could not be calculated.")

if __name__ == "__main__":
    main()
