import sqlite3
import os
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf
import requests

DB_PATH = "db/wealth_management_ai_agent.db"
os.makedirs("db", exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Drop tables if they exist
cursor.execute("DROP TABLE IF EXISTS alerts")
cursor.execute("DROP TABLE IF EXISTS holdings")
cursor.execute("DROP TABLE IF EXISTS clients")
cursor.execute("DROP TABLE IF EXISTS portfolio_timeseries")

# Create tables
cursor.execute('''
CREATE TABLE clients (
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT,
    risk_profile TEXT,
    goals TEXT,
    notes TEXT
)
''')

cursor.execute('''
CREATE TABLE holdings (
    id INTEGER PRIMARY KEY,
    client_id INTEGER,
    asset_symbol TEXT,
    asset_type TEXT,
    sector TEXT,
    quantity REAL,
    value_usd REAL,
    cost_basis REAL,
    FOREIGN KEY(client_id) REFERENCES clients(id)
)
''')

cursor.execute('''
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY,
    client_id INTEGER,
    alert_text TEXT,
    source_url TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(client_id) REFERENCES clients(id),
    UNIQUE(client_id, source_url)
)
''')

cursor.execute("""
CREATE TABLE portfolio_timeseries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id INTEGER,
    month TEXT,
    portfolio_value REAL,
    monthly_return REAL,
    FOREIGN KEY(client_id) REFERENCES clients(id)
)
""")

# Insert sample clients
clients = [
    ("Alice Johnson", "alice@example.com", "balanced", "Retire in 15 years", "Prefers low-cost ETFs"),
    ("Bob Smith", "bob@example.com", "aggressive", "Maximize capital growth", "Interested in tech and crypto"),
    ("Carla Reyes", "carla@example.com", "conservative", "Preserve capital", "Avoids high-volatility sectors"),
]
cursor.executemany('''
INSERT INTO clients (name, email, risk_profile, goals, notes)
VALUES (?, ?, ?, ?, ?)
''', clients)

# --- Dummy historical performance using synthetic data ---
months = pd.date_range(start="2023-04-01", periods=12, freq="M")
client_ids = [1, 2, 3]

for client_id in client_ids:
    # Seeded randomness per client for repeatability
    np.random.seed(client_id * 10)

    # Simulate returns (e.g. average 1% monthly return, std dev 3%)
    monthly_returns = np.random.normal(loc=0.01, scale=0.03, size=len(months))
    portfolio_values = [100000]  # Start with $100k
    for r in monthly_returns:
        portfolio_values.append(portfolio_values[-1] * (1 + r))
    portfolio_values = portfolio_values[1:]

    # Insert into database
    for i, (month, value, r) in enumerate(zip(months, portfolio_values, monthly_returns)):
        cursor.execute("""
            INSERT INTO portfolio_timeseries (client_id, month, portfolio_value, monthly_return)
            VALUES (?, ?, ?, ?)
        """, (client_id, month.strftime("%Y-%m"), round(value, 2), round(r, 4)))


# Create diversified holdings based on ending portfolio value
client_holdings_template = {
    1: [("AAPL", "stock", "Tech", 0.25),
        ("VTI", "etf", "Broad Market", 0.25),
        ("ETH", "crypto", "DeFi", 0.20),
        ("USDC", "crypto", "Stablecoin", 0.10),
        ("BND", "bond", "Fixed Income", 0.20)],

    2: [("TSLA", "stock", "Tech", 0.30),
        ("BTC", "crypto", "Crypto", 0.35),
        ("ARKK", "etf", "Innovation", 0.15),
        ("SOL", "crypto", "L1s", 0.10),
        ("QQQ", "etf", "Tech", 0.10)],

    3: [("TLT", "bond", "Gov Bonds", 0.40),
        ("BND", "etf", "Fixed Income", 0.25),
        ("GLD", "etf", "Commodities", 0.15),
        ("SCHD", "etf", "Dividends", 0.10),
        ("XOM", "stock", "Energy", 0.10)],
}

def get_live_prices(symbols):
    coingecko_ids = {
        "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "USDC": "usd-coin"
    }
    coingecko_symbols = [s for s in symbols if s in coingecko_ids]
    yahoo_symbols = [s for s in symbols if s not in coingecko_ids]

    prices = {}

    # --- CoinGecko fetch ---
    if coingecko_symbols:
        ids = [coingecko_ids[sym] for sym in coingecko_symbols]
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={','.join(ids)}&vs_currencies=usd"
        response = requests.get(url).json()
        for sym in coingecko_symbols:
            coingecko_id = coingecko_ids[sym]
            if coingecko_id in response:
                prices[sym] = response[coingecko_id]["usd"]

    # --- Yahoo Finance fetch ---
    if yahoo_symbols:
        tickers = yf.Tickers(" ".join(yahoo_symbols))
        for sym in yahoo_symbols:
            ticker = tickers.tickers.get(sym)
            if ticker and ticker.info.get("regularMarketPrice"):
                prices[sym] = ticker.info["regularMarketPrice"]

    return prices

# Pull all symbols from the template
all_symbols = sorted({symbol for asset_list in client_holdings_template.values() for symbol, *_ in asset_list})
live_prices = get_live_prices(all_symbols)

for client_id, assets in client_holdings_template.items():
    cursor.execute("""
        SELECT portfolio_value FROM portfolio_timeseries
        WHERE client_id = ? ORDER BY month DESC LIMIT 1
    """, (client_id,))
    latest_value = cursor.fetchone()[0]

    for symbol, asset_type, sector, weight in assets:
        price = live_prices.get(symbol.upper())
        if not price:
            print(f"⚠️ Warning: No live price found for {symbol}. Skipping.")
            continue

        # Simulate a quantity directly, then derive value_usd
        quantity = round(np.random.uniform(10, 100), 4) if price < 100 else round(np.random.uniform(1, 10), 4)
        value_usd = round(quantity * price, 2)
        cost_basis = round(value_usd * np.random.uniform(0.8, 1.1), 2)
        cursor.execute("""
            INSERT INTO holdings (client_id, asset_symbol, asset_type, sector, quantity, value_usd, cost_basis)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (client_id, symbol, asset_type, sector, quantity, value_usd, cost_basis))
        
conn.commit()
conn.close()
print("✅ Database initialized with sample data.")
