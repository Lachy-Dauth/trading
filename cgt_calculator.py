import pandas as pd
import json
import os
import re
import argparse
import copy
from datetime import datetime, timedelta

# --- Parsing Functions (Copied from analyze_trades.py) ---

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

# --- CGT Logic ---

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

def load_cache(cache_file='cgt_cache.json'):
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache, cache_file='cgt_cache.json'):
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=4)

def calculate_cgt(trades_df, transfers_df=None, advanced_mode=False):
    holdings = {} # Ticker -> List of Lots
    realized_gains = []
    adjustments = []
    holdings_snapshots = {} # (fy_start, fy_end) -> holdings_copy
    
    cache = load_cache()
    
    # Merge trades and transfers (for dividends) if advanced mode
    events = []
    
    for _, trade in trades_df.iterrows():
        events.append({
            'Date': trade['Date'],
            'Type': 'Trade',
            'Data': trade
        })
        
    if advanced_mode and transfers_df is not None:
        dividends = transfers_df[transfers_df['Category'] == 'Dividend']
        for _, div in dividends.iterrows():
            events.append({
                'Date': div['Date'],
                'Type': 'Dividend',
                'Data': div
            })
            
    # Sort events by date
    events.sort(key=lambda x: x['Date'])
    
    # Determine the first FY
    current_fy_end = None
    if events:
        first_date = events[0]['Date']
        if first_date.month >= 7:
            current_fy_end = datetime(first_date.year + 1, 6, 30)
        else:
            current_fy_end = datetime(first_date.year, 6, 30)
    
    for event in events:
        date = event['Date']
        
        # Check if we crossed a FY boundary
        while current_fy_end and date > current_fy_end:
            # Save snapshot for the FY that just ended
            fy_start = datetime(current_fy_end.year - 1, 7, 1)
            holdings_snapshots[(fy_start, current_fy_end)] = copy.deepcopy(holdings)
            
            # Move to next FY
            current_fy_end = datetime(current_fy_end.year + 1, 6, 30)
        
        if event['Type'] == 'Trade':
            trade = event['Data']
            ticker = trade['Ticker']
            action = trade['Action']
            qty = trade['Quantity']
            amount = trade['Amount'] # Negative for buy, Positive for sell
            
            if ticker not in holdings:
                holdings[ticker] = []
                
            if action == 'Buy':
                # Cost base is absolute value of amount (includes brokerage if in amount)
                cost_base = abs(amount)
                new_lot = Lot(date, ticker, qty, trade['Price'], cost_base)
                holdings[ticker].append(new_lot)
                
            elif action == 'Sell':
                # Tax Minimization Matching
                remaining_to_sell = qty
                total_proceeds = amount # Positive
                unit_price = total_proceeds / qty
                
                # Helper to calculate sort key for a lot
                def get_lot_priority(lot):
                    # Calculate potential gain per share
                    unit_gain = unit_price - lot.unit_cost_base
                    held_duration = date - lot.date
                    is_long_term = held_duration > timedelta(days=365)
                    
                    # Priority 1: Largest Short Term Capital Loss (Gain < 0, ST)
                    # Priority 2: Long Term Capital Loss (Gain < 0, LT) - Grouped with ST Loss for now
                    # Priority 3: Long Term Capital Gains with Lowest Gain (Gain >= 0, LT)
                    # Priority 4: Lowest Short Term Capital Gain (Gain >= 0, ST)
                    
                    if unit_gain < 0:
                        # Loss
                        # Sort by Gain ascending (Most negative first)
                        # Rank 1
                        return (1, unit_gain)
                    else:
                        # Gain
                        if is_long_term:
                            # LT Gain
                            # Sort by Gain ascending (Lowest gain first)
                            # Rank 2
                            return (2, unit_gain)
                        else:
                            # ST Gain
                            # Sort by Gain ascending (Lowest gain first)
                            # Rank 3
                            return (3, unit_gain)

                # Get available lots and sort them
                available_lots = holdings[ticker]
                # We need to sort a copy or indices, but we will modify the lots in place.
                # Let's sort the list itself? No, holdings[ticker] order matters for FIFO usually, 
                # but here we are re-ordering for this specific sale.
                # Actually, if we pick specific lots, the order in holdings[ticker] doesn't strictly matter 
                # unless we want to preserve FIFO for ties.
                
                # Sort lots by priority
                available_lots.sort(key=get_lot_priority)
                
                # Iterate through sorted lots
                # We need to be careful about modifying the list while iterating if we remove items.
                # Better to iterate a copy or use an index.
                
                lots_to_remove = []
                
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
                        
                        realized_gains.append({
                            'Date': date,
                            'Ticker': ticker,
                            'Quantity': sold_qty,
                            'Acquired': lot.date,
                            'Proceeds': proceeds_portion,
                            'CostBase': cost_base_portion,
                            'Gain': gain,
                            'DiscountEligible': discount_eligible
                        })
                        
                        remaining_to_sell -= sold_qty
                        lots_to_remove.append(lot)
                        
                    else:
                        # Sell partial lot
                        sold_qty = remaining_to_sell
                        
                        # Pro-rate cost base
                        cost_base_portion = (sold_qty / lot.quantity) * lot.cost_base
                        proceeds_portion = unit_price * sold_qty
                        
                        gain = proceeds_portion - cost_base_portion
                        
                        held_duration = date - lot.date
                        discount_eligible = held_duration > timedelta(days=365)
                        
                        realized_gains.append({
                            'Date': date,
                            'Ticker': ticker,
                            'Quantity': sold_qty,
                            'Acquired': lot.date,
                            'Proceeds': proceeds_portion,
                            'CostBase': cost_base_portion,
                            'Gain': gain,
                            'DiscountEligible': discount_eligible
                        })
                        
                        # Update lot
                        lot.cost_base -= cost_base_portion
                        lot.quantity -= sold_qty
                        remaining_to_sell = 0
                
                # Remove fully sold lots from holdings
                for lot in lots_to_remove:
                    holdings[ticker].remove(lot)
                        
        elif event['Type'] == 'Dividend':
            div = event['Data']
            # Identify ticker from description? 
            # Description format: "QPON Dividend", "GHHF Dividend", "A200 Dividend"
            desc = div['Description']
            ticker_match = None
            for t in holdings.keys():
                # Simple check if ticker base name is in description
                base_ticker = t.replace('.AX', '')
                if base_ticker in desc:
                    ticker_match = t
                    break
            
            if ticker_match and holdings[ticker_match]:
                # Ask for cost base adjustment
                # Check cache first
                # Cache key: Ticker + Date (or just Ticker if user wants constant %?)
                # User asked for "cost basis adjustment of dividends as a percentage"
                # Let's assume it varies by dividend event.
                
                cache_key = f"{ticker_match}_{date.strftime('%Y-%m-%d')}"
                adjustment_pct = 0.0
                
                if cache_key in cache:
                    adjustment_pct = cache[cache_key]
                else:
                    # Ask user
                    print(f"\nDividend detected: {desc} on {date.strftime('%Y-%m-%d')} Amount: ${div['Amount']:.2f}")
                    try:
                        inp = input(f"Enter cost base adjustment % (positive for decrease, negative for increase) for {ticker_match} (default 0): ")
                        if inp.strip():
                            adjustment_pct = float(inp)
                    except ValueError:
                        adjustment_pct = 0.0
                    
                    cache[cache_key] = adjustment_pct
                    save_cache(cache)
                
                if adjustment_pct != 0:
                    reduction_amount = div['Amount'] * (adjustment_pct / 100.0)
                    
                    # Reduce cost base of ALL shares held at this time?
                    # Usually AMIT reduces cost base of shares held on record date.
                    # We assume holdings[ticker_match] contains all shares held.
                    
                    total_shares = sum(lot.quantity for lot in holdings[ticker_match])
                    if total_shares > 0:
                        reduction_per_share = reduction_amount / total_shares
                        
                        adjustments.append({
                            'Date': date,
                            'Ticker': ticker_match,
                            'Description': desc,
                            'TotalAmount': reduction_amount,
                            'PerShare': reduction_per_share,
                            'Type': 'Decrease' if reduction_amount > 0 else 'Increase'
                        })
                        
                        for lot in holdings[ticker_match]:
                            lot_reduction = reduction_per_share * lot.quantity
                            lot.cost_base -= lot_reduction
                            # Cost base can't go below 0? Usually capital gain if it does.
                            if lot.cost_base < 0:
                                # Realize immediate gain?
                                # For simplicity, let's floor at 0 and warn?
                                # Or just let it be negative (technically not allowed, triggers CGT event E4)
                                # Let's just print warning.
                                pass

    # Capture the final state for the current/last FY
    if current_fy_end:
        fy_start = datetime(current_fy_end.year - 1, 7, 1)
        holdings_snapshots[(fy_start, current_fy_end)] = copy.deepcopy(holdings)

    return realized_gains, holdings, adjustments, holdings_snapshots

def generate_report(realized_gains, holdings, adjustments, fy_start, fy_end, filename='cgt_report.md'):
    
    # Filter for current FY
    fy_gains = [g for g in realized_gains if fy_start <= g['Date'] <= fy_end]
    fy_adjustments = [a for a in adjustments if fy_start <= a['Date'] <= fy_end]
    
    total_gain = 0.0
    total_discounted_gain = 0.0
    
    with open(filename, 'w') as f:
        f.write(f"# Capital Gains Tax Report\n")
        f.write(f"**Financial Year:** {fy_start.date()} to {fy_end.date()}\n\n")
        
        f.write("## Realized Gains/Losses\n")
        f.write("| Date | Ticker | Qty | Acquired | Proceeds | Cost Base | Gain/Loss | Discount Eligible |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        
        for g in fy_gains:
            date_str = g['Date'].strftime('%Y-%m-%d')
            acq_str = g['Acquired'].strftime('%Y-%m-%d')
            gain = g['Gain']
            disc = "Yes" if g['DiscountEligible'] else "No"
            
            f.write(f"| {date_str} | {g['Ticker']} | {g['Quantity']} | {acq_str} | ${g['Proceeds']:.2f} | ${g['CostBase']:.2f} | ${gain:.2f} | {disc} |\n")
            
            # Calculate Taxable Gain
            # Losses offset gains.
            # Discount applies to remaining gains.
            # This is a simple summary, not a full tax return calc.
            
            total_gain += gain
            
        f.write(f"\n**Total Net Gain/Loss:** ${total_gain:.2f}\n")
        
        # Simple Discount Calculation (Apply 50% to eligible gains if Net Gain is positive)
        # Note: Correct method is: Net Capital Gain = Total Gains - Total Losses.
        # Then apply discount to eligible components of Net Gain.
        # We need to separate eligible and non-eligible.
        
        eligible_gains = sum(g['Gain'] for g in fy_gains if g['DiscountEligible'] and g['Gain'] > 0)
        ineligible_gains = sum(g['Gain'] for g in fy_gains if not g['DiscountEligible'] and g['Gain'] > 0)
        total_losses = sum(abs(g['Gain']) for g in fy_gains if g['Gain'] < 0)
        
        f.write("\n### Tax Estimate (Simplified)\n")
        f.write(f"- Total Capital Gains: ${eligible_gains + ineligible_gains:.2f}\n")
        f.write(f"- Total Capital Losses: ${total_losses:.2f}\n")
        
        net_position = (eligible_gains + ineligible_gains) - total_losses
        
        if net_position > 0:
            # Apply losses to ineligible gains first (standard strategy to maximize discount)
            remaining_losses = total_losses
            
            # Offset ineligible
            if remaining_losses > ineligible_gains:
                remaining_losses -= ineligible_gains
                net_ineligible = 0
            else:
                net_ineligible = ineligible_gains - remaining_losses
                remaining_losses = 0
                
            # Offset eligible
            net_eligible = max(0, eligible_gains - remaining_losses)
            
            # Apply Discount
            taxable_eligible = net_eligible * 0.5
            total_taxable = taxable_eligible + net_ineligible
            
            f.write(f"- Net Capital Gain: ${net_position:.2f}\n")
            f.write(f"- **Estimated Taxable Gain (after 50% discount):** ${total_taxable:.2f}\n")
        else:
            f.write(f"- **Net Capital Loss (Carry Forward):** ${abs(net_position):.2f}\n")

        f.write("\n## Remaining Holdings (Unrealized)\n")
        f.write("| Ticker | Date | Quantity | Cost Base | Unit Cost |\n")
        f.write("|---|---|---|---|---|\n")
        
        for ticker, lots in holdings.items():
            # Sort by date for reporting
            sorted_lots = sorted(lots, key=lambda x: x.date)
            for lot in sorted_lots:
                f.write(f"| {ticker} | {lot.date.strftime('%Y-%m-%d')} | {lot.quantity} | ${lot.cost_base:.2f} | ${lot.unit_cost_base:.2f} |\n")

        if fy_adjustments:
            f.write("\n## Cost Base Adjustments\n")
            f.write("| Date | Ticker | Description | Type | Total Amount | Per Share |\n")
            f.write("|---|---|---|---|---|---|\n")
            for adj in fy_adjustments:
                f.write(f"| {adj['Date'].strftime('%Y-%m-%d')} | {adj['Ticker']} | {adj['Description']} | {adj['Type']} | ${abs(adj['TotalAmount']):.2f} | ${abs(adj['PerShare']):.4f} |\n")


def main():
    parser = argparse.ArgumentParser(description='Calculate Capital Gains Tax.')
    parser.add_argument('--advanced', action='store_true', help='Enable advanced mode for cost base adjustments.')
    args = parser.parse_args()

    trades_file = 'trades.txt'
    transfers_file = 'transfers.txt'
    
    print("Parsing trades...")
    trades_df = parse_trades(trades_file)
    
    transfers_df = None
    if args.advanced:
        print("Parsing transfers for dividends...")
        transfers_df = parse_transfers(transfers_file)
        
    print("Calculating CGT...")
    realized_gains, holdings, adjustments, holdings_snapshots = calculate_cgt(trades_df, transfers_df, args.advanced)
    
    # Create reports directory
    reports_dir = 'reports'
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
        
    # Generate report for each FY found in snapshots
    if not holdings_snapshots:
        print("No data found to generate reports.")
        return

    for (fy_start, fy_end), fy_holdings in sorted(holdings_snapshots.items(), key=lambda x: x[0][0]):
        report_filename = os.path.join(reports_dir, f'cgt_report_{fy_end.year}.md')
        print(f"Generating report for FY {fy_start.year}-{fy_end.year} -> {report_filename}...")
        generate_report(realized_gains, fy_holdings, adjustments, fy_start, fy_end, filename=report_filename)
        
    print("All reports generated.")

if __name__ == "__main__":
    main()
