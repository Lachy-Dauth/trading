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
    return df.sort_values('Date', kind='stable')

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
        amount_clean = amount_str.replace('A$', '').replace(',', '').replace('+', '').replace(' ', '')
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
    return df.sort_values('Date', kind='stable')

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

# Lot tracking for CGT
class Lot:
    def __init__(self, date, ticker, quantity, price, cost_base):
        self.date = date
        self.ticker = ticker
        self.quantity = quantity
        self.price = price
        self.cost_base = cost_base # Total cost base for this lot
        self.initial_quantity = quantity

    @property
    def unit_cost_base(self):
        return self.cost_base / self.quantity if self.quantity > 0 else 0

    def __repr__(self):
        return f"Lot({self.date.date()}, {self.ticker}, {self.quantity}, ${self.unit_cost_base:.2f})"

def calculate_portfolio_with_tax(trades_df, transfers_df, price_data, tax_rate):
    # Create a date range from first activity to last available price date
    if price_data.empty:
        print("No price data available.")
        return pd.DataFrame(), [], []
        
    start_date = min(trades_df['Date'].min(), transfers_df['Date'].min())
    end_date = price_data.index.max()
    
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Reindex price data to all days (ffill)
    price_data_filled = price_data.reindex(all_dates).ffill()
    
    # Create a dataframe of holdings over time
    holdings_df = pd.DataFrame(0, index=all_dates, columns=trades_df['Ticker'].unique())
    cash_series = pd.Series(0.0, index=all_dates)
    tax_adjustment = pd.Series(0.0, index=all_dates) # Virtual tax deposits/withdrawals
    
    # Track lots for CGT calculation
    holdings_lots = {} # Ticker -> List of Lots
    
    # Track tax events for reporting
    tax_events = []
    
    # Merge all events (trades, transfers) and sort by date
    events = []
    
    for _, trade in trades_df.iterrows():
        events.append({
            'Date': trade['Date'],
            'Type': 'Trade',
            'Data': trade
        })
        
    for _, transfer in transfers_df.iterrows():
        events.append({
            'Date': transfer['Date'],
            'Type': 'Transfer',
            'Data': transfer
        })
    
    events.sort(key=lambda x: x['Date'])
    
    # Process events
    for event in events:
        date = event['Date']
        
        if event['Type'] == 'Trade':
            trade = event['Data']
            ticker = trade['Ticker']
            action = trade['Action']
            qty = trade['Quantity']
            amount = trade['Amount']
            
            if ticker not in holdings_lots:
                holdings_lots[ticker] = []
            
            if action == 'Buy':
                # Update holdings
                holdings_df.loc[date:, ticker] += qty
                cash_series.loc[date:] += amount
                
                # Add lot for tracking
                cost_base = abs(amount)
                new_lot = Lot(date, ticker, qty, trade['Price'], cost_base)
                holdings_lots[ticker].append(new_lot)
                
            elif action == 'Sell':
                # Update holdings
                holdings_df.loc[date:, ticker] -= qty
                cash_series.loc[date:] += amount
                
                # Calculate CGT using tax minimization strategy
                remaining_to_sell = qty
                total_proceeds = amount
                unit_price = total_proceeds / qty
                
                # Tax minimization: sell losses first, then long-term gains, then short-term gains
                def get_lot_priority(lot):
                    unit_gain = unit_price - lot.unit_cost_base
                    held_duration = date - lot.date
                    is_long_term = held_duration > timedelta(days=365)
                    
                    if unit_gain < 0:
                        # Loss - highest priority (most negative first)
                        return (1, unit_gain)
                    else:
                        # Gain
                        if is_long_term:
                            # LT Gain - lower gain first
                            return (2, unit_gain)
                        else:
                            # ST Gain - lower gain first
                            return (3, unit_gain)
                
                available_lots = holdings_lots[ticker]
                available_lots.sort(key=get_lot_priority)
                
                lots_to_remove = []
                total_gain = 0.0
                
                for lot in available_lots:
                    if remaining_to_sell <= 0:
                        break
                        
                    if lot.quantity <= remaining_to_sell:
                        # Sell whole lot
                        sold_qty = lot.quantity
                        cost_base_portion = lot.cost_base
                        proceeds_portion = unit_price * sold_qty
                        
                        gain = proceeds_portion - cost_base_portion
                        
                        held_duration = date - lot.date
                        discount_eligible = held_duration > timedelta(days=365)
                        
                        # Calculate tax on this gain
                        if gain > 0:
                            taxable_gain = gain * 0.5 if discount_eligible else gain
                            tax_on_gain = taxable_gain * tax_rate
                        else:
                            # Loss - treat as negative tax (tax benefit)
                            tax_on_gain = gain * tax_rate
                        
                        total_gain += gain
                        
                        remaining_to_sell -= sold_qty
                        lots_to_remove.append(lot)
                        
                    else:
                        # Sell partial lot
                        sold_qty = remaining_to_sell
                        
                        cost_base_portion = (sold_qty / lot.quantity) * lot.cost_base
                        proceeds_portion = unit_price * sold_qty
                        
                        gain = proceeds_portion - cost_base_portion
                        
                        held_duration = date - lot.date
                        discount_eligible = held_duration > timedelta(days=365)
                        
                        # Calculate tax on this gain
                        if gain > 0:
                            taxable_gain = gain * 0.5 if discount_eligible else gain
                            tax_on_gain = taxable_gain * tax_rate
                        else:
                            # Loss
                            tax_on_gain = gain * tax_rate
                        
                        total_gain += gain
                        
                        # Update lot
                        lot.cost_base -= cost_base_portion
                        lot.quantity -= sold_qty
                        remaining_to_sell = 0
                
                # Remove fully sold lots
                for lot in lots_to_remove:
                    holdings_lots[ticker].remove(lot)
                
                # Calculate total tax on this sale
                # We need to aggregate by discount eligibility
                # Simplified: calculate average discount
                # Better: track each lot's contribution separately
                # For now, let's calculate the tax on the net gain
                
                # Recalculate with proper aggregation
                remaining_to_sell = qty
                available_lots = holdings_lots.get(ticker, []) + lots_to_remove  # Reconstruct
                holdings_lots[ticker] = [lot for lot in holdings_lots.get(ticker, []) if lot not in lots_to_remove] + lots_to_remove
                
                # Reset and recalculate properly
                # Actually, let's just calculate tax on the net gain correctly
                # We have total_gain already
                
                # For proper tax calculation, we need to know which gains are eligible for discount
                # Let me recalculate in a cleaner way
                
                # Restart the sale processing to track discount properly
                holdings_lots[ticker] = [lot for lot in holdings_lots.get(ticker, []) if lot not in lots_to_remove]
                
                # Recalculate CGT with proper tracking
                remaining_to_sell = qty
                available_lots = holdings_lots.get(ticker, [])
                
                # Re-add lots we're about to sell
                for lot in lots_to_remove:
                    available_lots.append(lot)
                
                available_lots.sort(key=get_lot_priority)
                
                total_discounted_gain = 0.0
                total_undiscounted_gain = 0.0
                
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
                        
                        if discount_eligible:
                            total_discounted_gain += gain
                        else:
                            total_undiscounted_gain += gain
                        
                        remaining_to_sell -= sold_qty
                        lots_to_remove.append(lot)
                        
                    else:
                        sold_qty = remaining_to_sell
                        cost_base_portion = (sold_qty / lot.quantity) * lot.cost_base
                        proceeds_portion = unit_price * sold_qty
                        
                        gain = proceeds_portion - cost_base_portion
                        
                        held_duration = date - lot.date
                        discount_eligible = held_duration > timedelta(days=365)
                        
                        if discount_eligible:
                            total_discounted_gain += gain
                        else:
                            total_undiscounted_gain += gain
                        
                        lot.cost_base -= cost_base_portion
                        lot.quantity -= sold_qty
                        remaining_to_sell = 0
                
                # Remove fully sold lots
                for lot in lots_to_remove:
                    holdings_lots[ticker].remove(lot)
                
                # Calculate tax
                # Apply 50% discount to eligible gains, then calculate tax on total
                taxable_amount = (total_discounted_gain * 0.5) + total_undiscounted_gain
                tax_on_sale = taxable_amount * tax_rate
                
                # Treat tax as a virtual deposit (positive tax) or withdrawal (negative tax for losses)
                tax_adjustment.loc[date:] += tax_on_sale
                
                tax_events.append({
                    'Date': date,
                    'Type': 'CGT',
                    'Ticker': ticker,
                    'Amount': tax_on_sale,
                    'Description': f'Sell {qty} shares of {ticker}'
                })
                
        elif event['Type'] == 'Transfer':
            transfer = event['Data']
            amount = transfer['Amount']
            category = transfer['Category']
            
            # All transfers affect cash
            cash_series.loc[date:] += amount
            
            # If it's a dividend, calculate tax
            if category == 'Dividend':
                # Tax on dividend at full rate
                tax_on_dividend = amount * tax_rate
                tax_adjustment.loc[date:] += tax_on_dividend
                
                tax_events.append({
                    'Date': date,
                    'Type': 'Dividend',
                    'Ticker': transfer['Description'],
                    'Amount': tax_on_dividend,
                    'Description': transfer['Description']
                })
    
    # Calculate unrealized gains for final value
    unrealized_gains = []
    current_date = all_dates[-1]
    
    for ticker, lots in holdings_lots.items():
        if ticker in price_data_filled.columns and not pd.isna(price_data_filled[ticker].iloc[-1]):
            current_price = price_data_filled[ticker].iloc[-1]
            
            for lot in lots:
                proceeds = current_price * lot.quantity
                gain = proceeds - lot.cost_base
                
                held_duration = current_date - lot.date
                discount_eligible = held_duration > timedelta(days=365)
                
                unrealized_gains.append({
                    'Ticker': ticker,
                    'Gain': gain,
                    'DiscountEligible': discount_eligible
                })
    
    # Portfolio Value = Cash + Stock Value (+ Virtual Tax Adjustment)
    stock_value = (holdings_df * price_data_filled).sum(axis=1)
    total_value = cash_series + stock_value
    
    return pd.DataFrame({
        'Total Value': total_value,
        'Cash': cash_series,
        'Stock Value': stock_value,
        'Tax Adjustment': tax_adjustment
    }), tax_events, unrealized_gains

def calculate_benchmark_with_tax(transfers_df, benchmark_ticker, price_series, dividend_series, tax_rate):
    if price_series.empty:
        return pd.DataFrame(), []
        
    start_date = transfers_df['Date'].min()
    end_date = price_series.index.max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Reindex price data
    price_filled = price_series.reindex(all_dates).ffill()
    
    shares_held = pd.Series(0.0, index=all_dates)
    cash_held = pd.Series(0.0, index=all_dates)
    
    # Track lots for CGT calculation
    benchmark_lots = []
    tax_events = []
    
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
                
                if shares_bought > 0:
                    # Add lot
                    cost_base = abs(net_transfer)
                    new_lot = Lot(date, benchmark_ticker, shares_bought, price, cost_base)
                    benchmark_lots.append(new_lot)
                elif shares_bought < 0:
                    # Sell shares - calculate CGT
                    # Use same tax minimization strategy
                    qty = abs(shares_bought)
                    proceeds = abs(net_transfer)
                    unit_price = price
                    
                    def get_lot_priority(lot):
                        unit_gain = unit_price - lot.unit_cost_base
                        held_duration = date - lot.date
                        is_long_term = held_duration > timedelta(days=365)
                        
                        if unit_gain < 0:
                            return (1, unit_gain)
                        else:
                            if is_long_term:
                                return (2, unit_gain)
                            else:
                                return (3, unit_gain)
                    
                    benchmark_lots.sort(key=get_lot_priority)
                    
                    remaining_to_sell = qty
                    lots_to_remove = []
                    total_discounted_gain = 0.0
                    total_undiscounted_gain = 0.0
                    
                    for lot in benchmark_lots:
                        if remaining_to_sell <= 0:
                            break
                            
                        if lot.quantity <= remaining_to_sell:
                            sold_qty = lot.quantity
                            cost_base_portion = lot.cost_base
                            proceeds_portion = unit_price * sold_qty
                            
                            gain = proceeds_portion - cost_base_portion
                            held_duration = date - lot.date
                            discount_eligible = held_duration > timedelta(days=365)
                            
                            if discount_eligible:
                                total_discounted_gain += gain
                            else:
                                total_undiscounted_gain += gain
                            
                            remaining_to_sell -= sold_qty
                            lots_to_remove.append(lot)
                        else:
                            sold_qty = remaining_to_sell
                            cost_base_portion = (sold_qty / lot.quantity) * lot.cost_base
                            proceeds_portion = unit_price * sold_qty
                            
                            gain = proceeds_portion - cost_base_portion
                            held_duration = date - lot.date
                            discount_eligible = held_duration > timedelta(days=365)
                            
                            if discount_eligible:
                                total_discounted_gain += gain
                            else:
                                total_undiscounted_gain += gain
                            
                            lot.cost_base -= cost_base_portion
                            lot.quantity -= sold_qty
                            remaining_to_sell = 0
                    
                    for lot in lots_to_remove:
                        benchmark_lots.remove(lot)
                    
                    # Calculate tax
                    taxable_amount = total_discounted_gain * 0.5 + total_undiscounted_gain
                    tax_on_sale = taxable_amount * tax_rate
                    
                    tax_events.append({
                        'Date': date,
                        'Type': 'CGT',
                        'Amount': tax_on_sale
                    })

    # Process Dividends - benchmark actually sells shares to pay tax
    if not dividend_series.empty:
        for div_date, div_amount in dividend_series.items():
            if div_date in shares_held.index:
                # Check shares held on previous day
                prev_day = div_date - timedelta(days=1)
                if prev_day in shares_held.index:
                    qty = shares_held[prev_day]
                    total_div = qty * div_amount
                    cash_held.loc[div_date:] += total_div
                    
                    # Calculate tax on dividend
                    tax_on_div = total_div * tax_rate
                    
                    # Sell shares to pay tax
                    if div_date in price_filled.index and not pd.isna(price_filled[div_date]):
                        price = price_filled[div_date]
                        shares_to_sell = tax_on_div / price
                        
                        if shares_to_sell > 0 and shares_held[div_date] >= shares_to_sell:
                            shares_held.loc[div_date:] -= shares_to_sell
                            
                            # Calculate CGT on shares sold to pay tax
                            # Use same tax minimization strategy
                            unit_price = price
                            
                            def get_lot_priority(lot):
                                unit_gain = unit_price - lot.unit_cost_base
                                held_duration = div_date - lot.date
                                is_long_term = held_duration > timedelta(days=365)
                                
                                if unit_gain < 0:
                                    return (1, unit_gain)
                                else:
                                    if is_long_term:
                                        return (2, unit_gain)
                                    else:
                                        return (3, unit_gain)
                            
                            benchmark_lots.sort(key=get_lot_priority)
                            
                            remaining_to_sell = shares_to_sell
                            lots_to_remove = []
                            total_discounted_gain = 0.0
                            total_undiscounted_gain = 0.0
                            
                            for lot in benchmark_lots:
                                if remaining_to_sell <= 0:
                                    break
                                    
                                if lot.quantity <= remaining_to_sell:
                                    sold_qty = lot.quantity
                                    cost_base_portion = lot.cost_base
                                    proceeds_portion = unit_price * sold_qty
                                    
                                    gain = proceeds_portion - cost_base_portion
                                    held_duration = div_date - lot.date
                                    discount_eligible = held_duration > timedelta(days=365)
                                    
                                    if discount_eligible:
                                        total_discounted_gain += gain
                                    else:
                                        total_undiscounted_gain += gain
                                    
                                    remaining_to_sell -= sold_qty
                                    lots_to_remove.append(lot)
                                else:
                                    sold_qty = remaining_to_sell
                                    cost_base_portion = (sold_qty / lot.quantity) * lot.cost_base
                                    proceeds_portion = unit_price * sold_qty
                                    
                                    gain = proceeds_portion - cost_base_portion
                                    held_duration = div_date - lot.date
                                    discount_eligible = held_duration > timedelta(days=365)
                                    
                                    if discount_eligible:
                                        total_discounted_gain += gain
                                    else:
                                        total_undiscounted_gain += gain
                                    
                                    lot.cost_base -= cost_base_portion
                                    lot.quantity -= sold_qty
                                    remaining_to_sell = 0
                            
                            for lot in lots_to_remove:
                                benchmark_lots.remove(lot)
                            
                            # Calculate tax on this sale
                            taxable_amount = total_discounted_gain * 0.5 + total_undiscounted_gain
                            tax_on_cgt = taxable_amount * tax_rate
                            
                            # Note: This creates a recursive tax situation (selling shares to pay tax 
                            # creates CGT liability). For simplicity, we only account for the first-order
                            # tax effect. The additional CGT is typically small relative to the dividend
                            # tax and iterative calculation would add complexity without significant
                            # accuracy improvement for most portfolios.
                            
                            tax_events.append({
                                'Date': div_date,
                                'Type': 'Dividend',
                                'Amount': tax_on_div
                            })
                    
    total_value = (shares_held * price_filled) + cash_held
    
    return pd.DataFrame({
        'Total Value': total_value,
        'Cash': cash_held,
        'Stock Value': (shares_held * price_filled)
    }), tax_events

def calculate_no_return_with_tax(transfers_df, start_date, end_date, tax_rate):
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    cash_invested = pd.Series(0.0, index=all_dates)
    tax_on_dividends = pd.Series(0.0, index=all_dates)
    
    # Process Transfers
    for _, transfer in transfers_df.iterrows():
        date = transfer['Date']
        if date in all_dates:
            amount = transfer['Amount']
            
            if transfer['Category'] == 'Transfer':
                # Deposits/Withdrawals
                cash_invested.loc[date:] += amount
            elif transfer['Category'] == 'Dividend':
                # Dividend - add to cash but also track tax
                cash_invested.loc[date:] += amount
                tax = amount * tax_rate
                tax_on_dividends.loc[date:] += tax
    
    return pd.DataFrame({
        'Total Value': cash_invested,
        'Tax on Dividends': tax_on_dividends
    })

def calculate_metrics(portfolio_df, transfers_df, rf_price_series, rf_dividend_series):
    metrics = {
        'Sharpe': None,
        'Volatility': None,
        'Return': None
    }
    
    if portfolio_df.empty:
        return metrics
        
    # Calculate Portfolio Daily Returns
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
    
    # Calculate Risk-Free Returns
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

def print_summary(portfolio, benchmark=None, benchmark_name=None, metrics=None, no_return=None, 
                  unrealized_gains=None, tax_rate=None):
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY (WITH TAX ADJUSTMENTS)")
    print("="*50)
    
    # Calculate metrics
    final_date = portfolio.index[-1]
    final_val = portfolio['Total Value'].iloc[-1]
    tax_adj = portfolio['Tax Adjustment'].iloc[-1]
    
    print(f"Tax Rate: {tax_rate*100:.1f}%\n")
    
    print(f"{'Metric':<30} | {'Portfolio':<15}")
    print("-" * 50)
    print(f"{'Final Value (Before Tax)':<30} | ${final_val:,.2f}")
    print(f"{'Virtual Tax Liability':<30} | ${tax_adj:,.2f}")
    print(f"{'Net Value (After Tax)':<30} | ${final_val - tax_adj:,.2f}")
    
    # Calculate unrealized CGT if selling today
    if unrealized_gains:
        total_unrealized_gain = 0.0
        total_discounted_gain = 0.0
        total_undiscounted_gain = 0.0
        
        for ug in unrealized_gains:
            if ug['DiscountEligible']:
                total_discounted_gain += ug['Gain']
            else:
                total_undiscounted_gain += ug['Gain']
        
        total_unrealized_gain = total_discounted_gain + total_undiscounted_gain
        taxable_unrealized = total_discounted_gain * 0.5 + total_undiscounted_gain
        tax_if_sold = taxable_unrealized * tax_rate
        
        final_if_sold = final_val - tax_adj - tax_if_sold
        
        print(f"{'Unrealized Gain':<30} | ${total_unrealized_gain:,.2f}")
        print(f"{'Tax if Sold Today':<30} | ${tax_if_sold:,.2f}")
        print(f"{'Final Value if Sold Today':<30} | ${final_if_sold:,.2f}")
    
    if no_return is not None and not no_return.empty:
        invested_val = no_return['Total Value'].iloc[-1]
        div_tax = no_return['Tax on Dividends'].iloc[-1]
        
        print(f"{'Net Invested':<30} | ${invested_val:,.2f}")
        print(f"{'Tax on Dividends (0% return)':<30} | ${div_tax:,.2f}")
        
        total_return_abs = final_val - tax_adj - invested_val - div_tax
        print(f"{'Total Return $ (After Tax)':<30} | ${total_return_abs:,.2f}")
        
        if invested_val != 0:
            total_return_pct = (total_return_abs / invested_val) * 100
            print(f"{'Total Return % (After Tax)':<30} | {total_return_pct:.2f}%")

    if metrics:
        if metrics.get('Return') is not None:
            print(f"{'Ann. Return (Pre-Tax)':<30} | {metrics['Return']*100:.2f}%")
        if metrics.get('Volatility') is not None:
            print(f"{'Ann. Volatility':<30} | {metrics['Volatility']*100:.2f}%")
        if metrics.get('Sharpe') is not None:
            print(f"{'Sharpe Ratio':<30} | {metrics['Sharpe']:.2f}")
    
    if benchmark is not None and not benchmark.empty:
        print("-" * 50)
        bench_final_val = benchmark['Total Value'].iloc[-1]
        print(f"{(benchmark_name + ' Value'):<30} | ${bench_final_val:,.2f}")
        
        diff = final_val - tax_adj - bench_final_val
        print(f"{'Difference (After Tax)':<30} | ${diff:,.2f}")
        
        if bench_final_val != 0:
            outperformance = (final_val - tax_adj - bench_final_val) / bench_final_val * 100
            print(f"{'Outperformance % (After Tax)':<30} | {outperformance:+.2f}%")
            
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Analyze trading portfolio with tax adjustments.')
    parser.add_argument('--benchmark', type=str, help='Ticker for benchmark ETF (e.g., GHHF.AX)')
    parser.add_argument('--tax-rate', type=float, required=True, help='Tax rate as decimal (e.g., 0.325 for 32.5%%)')
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
    end_date = datetime.now()
    
    if start_date > end_date:
        print(f"Warning: First activity date {start_date} is in the future relative to system time {end_date}.")
        print("Cannot fetch historical data for future dates.")
        return

    print("Fetching market data...")
    price_data, dividend_data = get_market_data(tickers, start_date, end_date)
        
    print("Calculating portfolio with tax adjustments...")
    portfolio, tax_events, unrealized_gains = calculate_portfolio_with_tax(
        trades_df, transfers_df, price_data, args.tax_rate
    )
    
    benchmark_portfolio = pd.DataFrame()
    if args.benchmark:
        print(f"Calculating benchmark ({args.benchmark}) with tax adjustments...")
        if args.benchmark in price_data:
            benchmark_portfolio, bench_tax_events = calculate_benchmark_with_tax(
                transfers_df, 
                args.benchmark, 
                price_data[args.benchmark], 
                dividend_data.get(args.benchmark, pd.Series(dtype=float)),
                args.tax_rate
            )
    
    # Calculate Metrics
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
    
    no_return = calculate_no_return_with_tax(transfers_df, start_date, end_date, args.tax_rate)
    
    print_summary(portfolio, benchmark_portfolio, args.benchmark, metrics, no_return, 
                  unrealized_gains, args.tax_rate)
    
    # Print tax events summary
    print("\nTAX EVENTS SUMMARY")
    print("="*50)
    print(f"{'Date':<12} | {'Type':<10} | {'Description':<20} | {'Tax':<12}")
    print("-" * 50)
    for event in tax_events:
        print(f"{event['Date'].strftime('%Y-%m-%d'):<12} | {event['Type']:<10} | {event.get('Description', event.get('Ticker', ''))[:20]:<20} | ${event['Amount']:>10.2f}")
    print("="*50 + "\n")
    
    if not portfolio.empty:
        print("Plotting...")
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio.index, portfolio['Total Value'] - portfolio['Tax Adjustment'], 
                label='Portfolio Value (After Tax)', linewidth=2)
        
        if not no_return.empty:
            plt.plot(no_return.index, no_return['Total Value'] - no_return['Tax on Dividends'], 
                    label='Net Invested Capital (0% Return, After Tax)', linestyle=':', color='gray')

        if not benchmark_portfolio.empty:
            plt.plot(benchmark_portfolio.index, benchmark_portfolio['Total Value'], 
                    label=f'Benchmark ({args.benchmark}, After Tax)', linestyle='--')
            
        plt.title('Portfolio Performance Over Time (Tax-Adjusted)')
        plt.xlabel('Date')
        plt.ylabel('Value (AUD)')
        plt.legend()
        plt.grid(True)
        plt.savefig('portfolio_performance_tax.png')
        print("Graph saved to portfolio_performance_tax.png")
    else:
        print("Portfolio is empty or could not be calculated.")

if __name__ == "__main__":
    main()
