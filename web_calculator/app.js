class Lot {
    constructor(date, ticker, quantity, price, costBase) {
        this.date = date;
        this.ticker = ticker;
        this.quantity = quantity;
        this.price = price;
        this.costBase = costBase;
        this.initialQuantity = quantity;
    }

    get unitCostBase() {
        return this.quantity > 0 ? this.costBase / this.quantity : 0;
    }
}

// --- Parsing Logic ---

function parseDate(dateStr) {
    // Expecting YYYY-MM-DD
    return new Date(dateStr);
}

function parseTrades(content) {
    const lines = content.split('\n').map(l => l.trim()).filter(l => l);
    const trades = [];

    for (let i = 0; i < lines.length; i += 3) {
        if (i + 2 >= lines.length) break;

        const line1 = lines[i];
        const line2 = lines[i + 1];
        const line3 = lines[i + 2];

        // Try tab split first, then regex for space separated
        let dateStr, ticker;
        if (line1.includes('\t')) {
            const parts1 = line1.split('\t');
            dateStr = parts1[0].trim();
            ticker = parts1[1].trim();
        } else {
            // Regex for "YYYY-MM-DD Ticker"
            const match = line1.match(/^(\d{4}-\d{2}-\d{2})\s+(.*)$/);
            if (match) {
                dateStr = match[1];
                ticker = match[2].trim();
            } else {
                // Fallback or error? Skip
                continue;
            }
        }

        const action = line2.trim();

        // Line 3: "100 shares @ $70.30" [tab] "$7,033.00" OR just "$7,033.00" if pasted differently
        // Stake Aus copy paste might put amount on a new line? 
        // The user said "not tab separated".
        // Let's assume line3 contains the details string.
        
        let details = line3;
        let amountStr = "0";

        if (line3.includes('\t')) {
            const parts3 = line3.split('\t');
            details = parts3[0];
            amountStr = parts3.length > 1 ? parts3[1] : "0";
        } else {
            // If no tab, maybe the amount is at the end or it's just the details?
            // If it's Stake Aus, usually the amount is separate.
            // But based on the loop i+=3, we assume 3 lines per trade.
            // If the amount is missing from line 3, we might need to look at line 4?
            // But let's stick to the 3-line structure for now, assuming the amount is part of line 3 or we calculate it.
            
            // If line 3 is just "100 shares @ $70.30", we can calculate amount.
            // If line 3 has the amount at the end like "100 shares @ $70.30 $7,033.00", we can try to extract it.
            
            // Let's try to find the last dollar amount in the string
            const moneyMatches = line3.match(/-?\$[\d,]+\.\d{2}/g);
            if (moneyMatches && moneyMatches.length > 1) {
                // Last one is likely the total amount
                amountStr = moneyMatches[moneyMatches.length - 1];
            } else if (moneyMatches && moneyMatches.length === 1) {
                // Only one money value? Might be price or amount.
                // If "shares @ $Price", then it's price.
                // We'll calculate amount later.
            }
        }

        // Extract Quantity
        const qtyMatch = details.match(/(\d+)\s+shares/);
        const quantity = qtyMatch ? parseInt(qtyMatch[1]) : 0;

        // Extract Amount
        const amountClean = amountStr.replace('A$', '').replace(/,/g, '').replace('+', '').replace(/\s/g, '');
        let amount = parseFloat(amountClean);
        if (isNaN(amount)) amount = 0.0;

        // Extract Price
        const priceMatch = details.match(/@\s*\$(\d+\.\d+)/); // Look for @ $Price
        let price = 0;
        if (priceMatch) {
            price = parseFloat(priceMatch[1]);
        } else {
            // Fallback: try to find any price-like number if not found with @
            const simplePriceMatch = details.match(/\$(\d+\.\d+)/);
            if (simplePriceMatch) {
                price = parseFloat(simplePriceMatch[1]);
            } else {
                price = quantity ? Math.abs(amount / quantity) : 0;
            }
        }
        
        // If amount is 0 but we have qty and price, calculate it
        if (amount === 0 && quantity > 0 && price > 0) {
            amount = quantity * price;
            // Adjust sign based on action
            if (action === 'Buy') amount = -amount;
        }

        trades.push({
            date: parseDate(dateStr),
            ticker: ticker + '.AX',
            action: action,
            quantity: quantity,
            price: price,
            amount: amount
        });
    }

    // Sort by date ascending
    return trades.sort((a, b) => a.date.getTime() - b.date.getTime());
}

function parseTransfers(content) {
    const lines = content.split('\n').map(l => l.trim()).filter(l => l);
    const transfers = [];

    for (let i = 0; i < lines.length; i += 3) {
        if (i + 2 >= lines.length) break;

        const line1 = lines[i];
        const line3 = lines[i + 2];

        let dateStr, description;
        
        if (line1.includes('\t')) {
            const parts1 = line1.split('\t');
            dateStr = parts1[0].trim();
            description = parts1.length > 1 ? parts1[1].trim() : "";
        } else {
             // Regex for "YYYY-MM-DD Description"
             const match = line1.match(/^(\d{4}-\d{2}-\d{2})\s+(.*)$/);
             if (match) {
                 dateStr = match[1];
                 description = match[2].trim();
             } else {
                 continue;
             }
        }

        const amountStr = line3;
        const amountClean = amountStr.replace('A$', '').replace(/,/g, '').replace('+', '').replace(/\s/g, '');
        let amount = parseFloat(amountClean);
        if (isNaN(amount)) amount = 0.0;

        let category = 'Transfer';
        if (description.includes('Trade settlement')) {
            category = 'Ignore';
        } else if (description.includes('Dividend')) {
            category = 'Dividend';
        }

        if (category !== 'Ignore') {
            transfers.push({
                date: parseDate(dateStr),
                description: description,
                amount: amount,
                category: category
            });
        }
    }

    return transfers.sort((a, b) => a.date.getTime() - b.date.getTime());
}

// --- CGT Calculation ---

function calculateCGT(trades, transfers, advancedMode, dividendAdjustments = {}) {
    const holdings = {};
    const realizedGains = [];
    const adjustments = [];
    const holdingsSnapshots = {}; // key: "YYYY-YYYY"

    const events = trades.map(t => ({ date: t.date, type: 'Trade', data: t }));

    if (advancedMode) {
        const dividends = transfers.filter(t => t.category === 'Dividend');
        dividends.forEach(d => {
            events.push({ date: d.date, type: 'Dividend', data: d });
        });
    }

    events.sort((a, b) => a.date.getTime() - b.date.getTime());

    // Determine first FY end
    let currentFyEnd = null;
    if (events.length > 0) {
        const firstDate = events[0].date;
        const year = firstDate.getFullYear();
        if (firstDate.getMonth() >= 6) { // Month is 0-indexed. 6 is July.
            currentFyEnd = new Date(year + 1, 5, 30); // June 30
        } else {
            currentFyEnd = new Date(year, 5, 30);
        }
    }

    for (const event of events) {
        const date = event.date;

        // Snapshot logic
        while (currentFyEnd && date > currentFyEnd) {
            const fyStartYear = currentFyEnd.getFullYear() - 1;
            const fyEndYear = currentFyEnd.getFullYear();
            const key = `${fyStartYear}-${fyEndYear}`;
            
            // Deep copy holdings
            const snapshot = {};
            for (const t in holdings) {
                snapshot[t] = holdings[t].map(l => new Lot(l.date, l.ticker, l.quantity, l.price, l.costBase));
            }
            holdingsSnapshots[key] = snapshot;

            currentFyEnd = new Date(currentFyEnd.getFullYear() + 1, 5, 30);
        }

        if (event.type === 'Trade') {
            const trade = event.data;
            const ticker = trade.ticker;
            
            if (!holdings[ticker]) holdings[ticker] = [];

            if (trade.action === 'Buy') {
                const costBase = Math.abs(trade.amount);
                holdings[ticker].push(new Lot(date, ticker, trade.quantity, trade.price, costBase));
            } else if (trade.action === 'Sell') {
                let remainingToSell = trade.quantity;
                const totalProceeds = trade.amount;
                const unitPrice = totalProceeds / trade.quantity;

                const availableLots = holdings[ticker];

                // Sort by Tax Minimization Priority
                availableLots.sort((a, b) => {
                    const unitGainA = unitPrice - a.unitCostBase;
                    const unitGainB = unitPrice - b.unitCostBase;
                    
                    const heldDurationA = date.getTime() - a.date.getTime();
                    const isLongTermA = heldDurationA > (365 * 24 * 60 * 60 * 1000);

                    const heldDurationB = date.getTime() - b.date.getTime();
                    const isLongTermB = heldDurationB > (365 * 24 * 60 * 60 * 1000);

                    // Priority Logic
                    // 1. Loss (Gain < 0) -> Sort by Gain Ascending (Most negative first)
                    // 2. LT Gain -> Sort by Gain Ascending (Lowest gain first)
                    // 3. ST Gain -> Sort by Gain Ascending (Lowest gain first)

                    const getRank = (gain, isLT) => {
                        if (gain < 0) return 1;
                        if (isLT) return 2;
                        return 3;
                    };

                    const rankA = getRank(unitGainA, isLongTermA);
                    const rankB = getRank(unitGainB, isLongTermB);

                    if (rankA !== rankB) return rankA - rankB;
                    return unitGainA - unitGainB;
                });

                const lotsToRemove = [];

                for (const lot of availableLots) {
                    if (remainingToSell <= 0) break;

                    let soldQty = 0;
                    let costBasePortion = 0;
                    let proceedsPortion = 0;

                    if (lot.quantity <= remainingToSell) {
                        soldQty = lot.quantity;
                        costBasePortion = lot.costBase;
                        proceedsPortion = unitPrice * soldQty;
                        lotsToRemove.push(lot);
                    } else {
                        soldQty = remainingToSell;
                        costBasePortion = (soldQty / lot.quantity) * lot.costBase;
                        proceedsPortion = unitPrice * soldQty;
                        
                        lot.costBase -= costBasePortion;
                        lot.quantity -= soldQty;
                    }

                    const gain = proceedsPortion - costBasePortion;
                    const heldDuration = date.getTime() - lot.date.getTime();
                    const discountEligible = heldDuration > (365 * 24 * 60 * 60 * 1000);

                    realizedGains.push({
                        date: date,
                        ticker: ticker,
                        quantity: soldQty,
                        acquired: lot.date,
                        proceeds: proceedsPortion,
                        costBase: costBasePortion,
                        gain: gain,
                        discountEligible: discountEligible
                    });

                    remainingToSell -= soldQty;
                }

                // Remove fully sold lots
                holdings[ticker] = holdings[ticker].filter(l => !lotsToRemove.includes(l));
            }

        } else if (event.type === 'Dividend') {
            const div = event.data;
            const desc = div.description;
            
            let tickerMatch = null;
            for (const t in holdings) {
                const baseTicker = t.replace('.AX', '');
                if (desc.includes(baseTicker)) {
                    tickerMatch = t;
                    break;
                }
            }

            if (tickerMatch && holdings[tickerMatch] && holdings[tickerMatch].length > 0) {
                // Check for pre-supplied adjustment
                // Key format: Date_Amount_Description (Matches promptForDividends)
                const key = `${div.date.toISOString()}_${div.amount}_${div.description}`;
                let adjustmentPct = 0;
                
                if (dividendAdjustments[key] !== undefined) {
                    adjustmentPct = dividendAdjustments[key];
                }

                if (!isNaN(adjustmentPct) && adjustmentPct !== 0) {
                    const reductionAmount = div.amount * (adjustmentPct / 100.0);
                    const totalShares = holdings[tickerMatch].reduce((sum, l) => sum + l.quantity, 0);
                    
                    if (totalShares > 0) {
                        const reductionPerShare = reductionAmount / totalShares;
                        
                        adjustments.push({
                            date: date,
                            ticker: tickerMatch,
                            description: desc,
                            totalAmount: reductionAmount,
                            perShare: reductionPerShare,
                            type: reductionAmount > 0 ? 'Decrease' : 'Increase'
                        });

                        for (const lot of holdings[tickerMatch]) {
                            const lotReduction = reductionPerShare * lot.quantity;
                            lot.costBase -= lotReduction;
                        }
                    }
                }
            }
        }
    }

    // Final snapshot
    if (currentFyEnd) {
        const fyStartYear = currentFyEnd.getFullYear() - 1;
        const fyEndYear = currentFyEnd.getFullYear();
        const key = `${fyStartYear}-${fyEndYear}`;
        const snapshot = {};
        for (const t in holdings) {
            snapshot[t] = holdings[t].map(l => new Lot(l.date, l.ticker, l.quantity, l.price, l.costBase));
        }
        holdingsSnapshots[key] = snapshot;
    }

    return { realizedGains, holdings, adjustments, holdingsSnapshots };
}

// --- UI Logic ---

const calculateBtn = document.getElementById('calculateBtn');
const tradesInput = document.getElementById('tradesFile');
const tradesText = document.getElementById('tradesText');
const transfersInput = document.getElementById('transfersFile');
const transfersText = document.getElementById('transfersText');
const advancedCheck = document.getElementById('advancedMode');
const outputDiv = document.getElementById('output');

// Tab Logic
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const targetId = btn.getAttribute('data-target');
        const parent = btn.closest('.input-group');
        
        // Toggle active tab button
        parent.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Toggle active content
        parent.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        document.getElementById(targetId).classList.add('active');
    });
});

// Modals
const dividendModal = document.getElementById('dividendModal');
const dividendListDiv = document.getElementById('dividendList');
const submitDividendsBtn = document.getElementById('submitDividends');
const helpBtn = document.getElementById('helpBtn');
const helpModal = document.getElementById('helpModal');
const closeHelp = document.querySelector('.close-help');

// Help Modal Logic
helpBtn.onclick = () => helpModal.style.display = "block";
closeHelp.onclick = () => helpModal.style.display = "none";
window.onclick = (event) => {
    if (event.target == helpModal) helpModal.style.display = "none";
    if (event.target == dividendModal) {
        // Prevent closing dividend modal by clicking outside? 
        // Maybe allow it but it cancels calculation? Let's keep it strict for now.
    }
}

calculateBtn.addEventListener('click', async () => {
    let tradesContent = "";
    
    // Check active tab for trades
    const tradesFileTab = document.getElementById('trades-file');
    if (tradesFileTab.classList.contains('active')) {
        if (!tradesInput.files || tradesInput.files.length === 0) {
            alert("Please select a trades file.");
            return;
        }
        const tradesFile = tradesInput.files[0];
        tradesContent = await tradesFile.text();
    } else {
        tradesContent = tradesText.value;
        if (!tradesContent.trim()) {
            alert("Please paste trades content.");
            return;
        }
    }

    const trades = parseTrades(tradesContent);

    let transfersContent = "";
    // Check active tab for transfers
    const transfersFileTab = document.getElementById('transfers-file');
    if (transfersFileTab.classList.contains('active')) {
        if (transfersInput.files && transfersInput.files.length > 0) {
            const transfersFile = transfersInput.files[0];
            transfersContent = await transfersFile.text();
        }
    } else {
        transfersContent = transfersText.value;
    }

    let transfers = [];
    if (transfersContent.trim()) {
        transfers = parseTransfers(transfersContent);
    }

    const advanced = advancedCheck.checked;
    let dividendAdjustments = {};

    if (advanced && transfers.length > 0) {
        // Identify potential dividends
        // We need to know which tickers exist in trades to filter dividends
        const tickers = new Set(trades.map(t => t.ticker.replace('.AX', '')));
        
        const potentialDividends = transfers.filter(t => {
            if (t.category !== 'Dividend') return false;
            // Check if description contains any ticker
            for (let ticker of tickers) {
                if (t.description.includes(ticker)) return true;
            }
            return false;
        });

        if (potentialDividends.length > 0) {
            // Show Modal
            dividendAdjustments = await promptForDividends(potentialDividends);
        }
    }

    const result = calculateCGT(trades, transfers, advanced, dividendAdjustments);
    renderReport(result);
});

function promptForDividends(dividends) {
    return new Promise((resolve) => {
        dividendListDiv.innerHTML = '';
        const inputs = [];

        dividends.forEach((div, index) => {
            const divItem = document.createElement('div');
            divItem.className = 'div-item';
            
            // Generate a unique key for this dividend
            // Note: In real app, might need ID. Here using Date+Ticker+Amount
            // We need to extract ticker again for the key
            // Simple extraction for display
            const dateStr = div.date.toISOString().split('T')[0];
            
            divItem.innerHTML = `
                <div class="div-info">
                    <strong>${dateStr}</strong> - ${div.description}<br>
                    Amount: $${div.amount.toFixed(2)}
                </div>
                <input type="number" step="0.01" value="0" class="div-input" data-index="${index}">
                <span>%</span>
            `;
            dividendListDiv.appendChild(divItem);
        });

        dividendModal.style.display = "block";

        // Handle Submit
        // Remove old listener to avoid duplicates if run multiple times?
        // Better: clone node or use 'once' option, but we need to pass data.
        // Let's just assign onclick.
        submitDividendsBtn.onclick = () => {
            const adjustments = {};
            const inputElements = dividendListDiv.querySelectorAll('.div-input');
            
            inputElements.forEach(input => {
                const index = parseInt(input.getAttribute('data-index'));
                const val = parseFloat(input.value);
                if (val !== 0 && !isNaN(val)) {
                    const div = dividends[index];
                    // Re-derive ticker for key (same logic as in calculateCGT)
                    // This is a bit redundant but ensures key match
                    // We can just store the key on the element?
                    // Let's just iterate tickers again.
                    // Actually, calculateCGT does the matching. We need to match that logic.
                    // Let's pass the raw list index? No, calculateCGT iterates events.
                    
                    // We need to generate the SAME key as calculateCGT will.
                    // calculateCGT iterates holdings keys.
                    // Here we don't have holdings yet.
                    // But we know the description contains the ticker.
                    // Let's extract the ticker from description based on known tickers?
                    // We don't have the list of known tickers easily here without re-parsing.
                    
                    // Alternative: calculateCGT logic finds ticker by checking holdings keys.
                    // Here we can just use the description as part of the key?
                    // No, calculateCGT uses `tickerMatch`.
                    
                    // Let's try to extract the ticker here properly.
                    // We can pass the tickers set to this function?
                    // Or just use a simpler key: "Date_Amount_Description" ?
                    // But calculateCGT needs to look it up.
                    
                    // Let's update calculateCGT to look up by a key that doesn't depend on Ticker?
                    // No, Ticker is good.
                    
                    // Let's just extract the ticker here. We know it's in the description.
                    // We can assume the ticker is the word that matches a known ticker.
                    // But we don't have the known tickers list here easily unless passed.
                    
                    // Let's pass tickers to promptForDividends?
                    // Actually, let's just use the Date and Amount and Description as key.
                    // Update calculateCGT to use that key.
                }
            });
            
            // Wait, I need to fix the key generation.
            // Let's update calculateCGT to use a key based on the dividend object itself (Date + Amount + Description).
            // That is robust.
            
            const result = {};
            inputElements.forEach(input => {
                const index = parseInt(input.getAttribute('data-index'));
                const val = parseFloat(input.value);
                if (val !== 0 && !isNaN(val)) {
                    const div = dividends[index];
                    // Key: Date_Amount_Description
                    const key = `${div.date.toISOString()}_${div.amount}_${div.description}`;
                    result[key] = val;
                }
            });

            dividendModal.style.display = "none";
            resolve(result);
        };
    });
}

function renderReport(data) {
    outputDiv.innerHTML = '';

    // Render per FY
    const sortedKeys = Object.keys(data.holdingsSnapshots).sort();

    if (sortedKeys.length === 0) {
        outputDiv.innerHTML = '<p>No data to display.</p>';
        return;
    }

    sortedKeys.forEach(fyKey => {
        const [startYear, endYear] = fyKey.split('-').map(Number);
        const fyStart = new Date(startYear, 6, 1); // July 1
        const fyEnd = new Date(endYear, 5, 30); // June 30

        const fyGains = data.realizedGains.filter(g => g.date >= fyStart && g.date <= fyEnd);
        const fyAdjustments = data.adjustments.filter(a => a.date >= fyStart && a.date <= fyEnd);
        const fyHoldings = data.holdingsSnapshots[fyKey];

        const section = document.createElement('div');
        section.className = 'fy-section';
        section.innerHTML = `<h2>FY ${startYear}-${endYear}</h2>`;

        // Gains Table
        let gainsHtml = `
            <h3>Realized Gains/Losses</h3>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Ticker</th>
                        <th>Qty</th>
                        <th>Acquired</th>
                        <th>Proceeds</th>
                        <th>Cost Base</th>
                        <th>Gain/Loss</th>
                        <th>Discount?</th>
                    </tr>
                </thead>
                <tbody>
        `;

        let totalGain = 0;
        let eligibleGains = 0;
        let ineligibleGains = 0;
        let totalLosses = 0;

        fyGains.forEach(g => {
            totalGain += g.gain;
            if (g.gain > 0) {
                if (g.discountEligible) eligibleGains += g.gain;
                else ineligibleGains += g.gain;
            } else {
                totalLosses += Math.abs(g.gain);
            }

            gainsHtml += `
                <tr>
                    <td>${g.date.toLocaleDateString()}</td>
                    <td>${g.ticker}</td>
                    <td>${g.quantity}</td>
                    <td>${g.acquired.toLocaleDateString()}</td>
                    <td>$${g.proceeds.toFixed(2)}</td>
                    <td>$${g.costBase.toFixed(2)}</td>
                    <td class="${g.gain >= 0 ? 'positive' : 'negative'}">$${g.gain.toFixed(2)}</td>
                    <td>${g.discountEligible ? 'Yes' : 'No'}</td>
                </tr>
            `;
        });
        gainsHtml += `</tbody></table>`;

        // Tax Estimate
        const netPosition = (eligibleGains + ineligibleGains) - totalLosses;
        let taxHtml = `<div class="summary-box"><h3>Tax Estimate</h3>`;
        taxHtml += `<p>Total Gains: $${(eligibleGains + ineligibleGains).toFixed(2)}</p>`;
        taxHtml += `<p>Total Losses: $${totalLosses.toFixed(2)}</p>`;
        taxHtml += `<p>Net Position: <strong>$${netPosition.toFixed(2)}</strong></p>`;
        
        if (netPosition > 0) {
            // Simplified discount logic
            let remainingLosses = totalLosses;
            let netIneligible = 0;
            
            if (remainingLosses > ineligibleGains) {
                remainingLosses -= ineligibleGains;
                netIneligible = 0;
            } else {
                netIneligible = ineligibleGains - remainingLosses;
                remainingLosses = 0;
            }
            
            const netEligible = Math.max(0, eligibleGains - remainingLosses);
            const taxable = (netEligible * 0.5) + netIneligible;
            
            taxHtml += `<p>Estimated Taxable Gain (after discount): <strong>$${taxable.toFixed(2)}</strong></p>`;
        }
        taxHtml += `</div>`;

        // Holdings Table
        let holdingsHtml = `
            <h3>Holdings at End of FY</h3>
            <table>
                <thead>
                    <tr>
                        <th>Ticker</th>
                        <th>Date</th>
                        <th>Qty</th>
                        <th>Cost Base</th>
                        <th>Unit Cost</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        for (const ticker in fyHoldings) {
            const lots = fyHoldings[ticker].sort((a, b) => a.date.getTime() - b.date.getTime());
            lots.forEach(lot => {
                holdingsHtml += `
                    <tr>
                        <td>${lot.ticker}</td>
                        <td>${lot.date.toLocaleDateString()}</td>
                        <td>${lot.quantity}</td>
                        <td>$${lot.costBase.toFixed(2)}</td>
                        <td>$${lot.unitCostBase.toFixed(2)}</td>
                    </tr>
                `;
            });
        }
        holdingsHtml += `</tbody></table>`;

        // Adjustments Table
        let adjHtml = '';
        if (fyAdjustments.length > 0) {
            adjHtml = `
                <h3>Cost Base Adjustments</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Ticker</th>
                            <th>Description</th>
                            <th>Type</th>
                            <th>Total</th>
                            <th>Per Share</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            fyAdjustments.forEach(adj => {
                adjHtml += `
                    <tr>
                        <td>${adj.date.toLocaleDateString()}</td>
                        <td>${adj.ticker}</td>
                        <td>${adj.description}</td>
                        <td>${adj.type}</td>
                        <td>$${Math.abs(adj.totalAmount).toFixed(2)}</td>
                        <td>$${Math.abs(adj.perShare).toFixed(4)}</td>
                    </tr>
                `;
            });
            adjHtml += `</tbody></table>`;
        }

        section.innerHTML += gainsHtml + taxHtml + holdingsHtml + adjHtml;
        outputDiv.appendChild(section);
    });
}
