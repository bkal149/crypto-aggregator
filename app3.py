import sqlite3
import requests
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from fpdf import FPDF
from datetime import datetime
import openai
import pandas as pd
import matplotlib.pyplot as plt
from functools import lru_cache
from matplotlib.backends.backend_pdf import PdfPages
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")  # safer absolute path

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "").strip()
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

DB_PATH = "db/wealth_management_ai_agent.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def get_clients():
    with get_connection() as conn:
        return conn.execute("SELECT * FROM clients").fetchall()

def get_holdings(client_id):
    with get_connection() as conn:
        return conn.execute(
            "SELECT asset_symbol, asset_type, sector, quantity, value_usd, cost_basis FROM holdings WHERE client_id=?",
            (client_id,)
        ).fetchall()

def get_alerts(client_id):
    with get_connection() as conn:
        return conn.execute(
            "SELECT alert_text, source_url, created_at FROM alerts WHERE client_id=? ORDER BY created_at DESC",
            (client_id,)
        ).fetchall()
        
CATEGORY_KEYWORDS = {
    "Analyst Reviews": ["analyst", "ratings", "forecast", "review"],
    "SEC Filings": ["sec", "10-k", "10-q", ".txt"],
    "ETF Trends": ["etf", "inflow", "sector rotation"],
    "Market News": ["inflation", "gdp", "interest rate", "stocks", "market"]
}

def classify_alert(alert_text, alert_url):
    """Return the category name (str) based on the alert text/url."""
    text_lower = (alert_text + " " + alert_url).lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return category
    return "Other News"

def insert_alert(client_id, alert_text, source_url):
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT 1 FROM alerts WHERE client_id = ? AND source_url = ?",
            (client_id, source_url)
        ).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO alerts (client_id, alert_text, source_url) VALUES (?, ?, ?)",
                (client_id, alert_text, source_url)
            )
            conn.commit()

def fetch_market_news(query, api_key):
    response = requests.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"query": query, "search_depth": "advanced"}
    )
    return response.json().get("results", [])

def match_news_to_client(news_item, holdings):
    title = news_item.get("title", "").lower()
    content = news_item.get("content", "").lower()
    for symbol, asset_type, sector, *_ in holdings:
        if any(kw in title or kw in content for kw in [symbol.lower(), asset_type.lower(), sector.lower()]):
            return True
    return False

from openai import OpenAI

# ✅ Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_client_insights(client_name, holdings, alerts, risk_profile, goals):
    holding_summary = ", ".join([f"{symbol} in {sector}" for symbol, asset_type, sector, quantity, value_usd, cost_basis in holdings])
    alert_summary = "\n".join([f"- {a[0]}" for a in alerts])

    prompt = f"""
    You are a financial analyst creating a short internal report for a client named {client_name}.
    This client has a {risk_profile} risk profile and the following goals: {goals}.
    Current holdings: {holding_summary}.
    Recent market news includes:
    {alert_summary}

    Please provide a concise internal report focusing on potential points of interest or concern,
    without giving explicit investment recommendations. Use a clear, factual tone.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating insights: {e}"


def sanitize_text(text):
    return text.encode('latin-1', 'replace').decode('latin-1')

def create_portfolio_charts(timeseries_df, allocation_df, client_name):
    chart_files = []

    # Line Chart: Portfolio Value Over Time
    fig1, ax1 = plt.subplots()
    timeseries_df["month"] = pd.to_datetime(timeseries_df["month"])
    ax1.plot(timeseries_df["month"], timeseries_df["portfolio_value"], marker='o')
    ax1.set_title("Portfolio Value Over Time")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Value ($)")
    line_chart_path = f"{client_name}_line_chart.png"
    fig1.savefig(line_chart_path)
    chart_files.append(line_chart_path)

    # Pie Chart: Asset Allocation
    fig2, ax2 = plt.subplots()
    ax2.pie(
        allocation_df["value_usd"],
        labels=allocation_df["type"],
        autopct="%1.1f%%",
        startangle=90
    )
    ax2.set_title("Asset Allocation")
    ax2.axis("equal")
    pie_chart_path = f"{client_name}_pie_chart.png"
    fig2.savefig(pie_chart_path)
    chart_files.append(pie_chart_path)

    plt.close('all')
    return chart_files

def generate_pdf_report(client_name, holdings, alerts, insights, avg_return, std_dev, sharpe, chart_files):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, f"Client Report: {client_name}", ln=True, align='C')

    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Holdings:", ln=True)
    for h in holdings:
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, f"- {h[0]} ({h[1]}), Sector: {h[2]}, Value: ${h[3]:,.2f}", ln=True)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Alerts:", ln=True)
    for alert in alerts:
        pdf.set_font("Arial", size=12)
        alert_text = sanitize_text(f"- {alert[0]} | Source: {alert[1]} | Date: {alert[2]}")
        pdf.multi_cell(0, 10, alert_text, align='L')
        pdf.cell(0, 5, '', ln=True)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Short Internal Report:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, sanitize_text(insights), align='L')

    # Add Risk Metrics (optional)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Portfolio Risk Metrics:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Avg Monthly Return: {avg_return:.2%}", ln=True)
    pdf.cell(200, 10, f"Standard Deviation: {std_dev:.2%}", ln=True)
    pdf.cell(200, 10, f"Sharpe Ratio: {sharpe:.2f}", ln=True)
    
    # Embed charts into PDF
    for chart_path in chart_files:
        pdf.add_page()
        pdf.image(chart_path, x=10, y=30, w=180)

    file_path = f"client_report_{client_name.replace(' ', '_')}.pdf"
    pdf.output(file_path)
    return file_path

def send_email_with_pdf(recipient, subject, body, pdf_path):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = recipient
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))
        with open(pdf_path, 'rb') as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(pdf_path))
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(pdf_path)}"'
            msg.attach(part)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

import yfinance as yf

def get_live_prices(symbols):
    try:
        # CoinGecko ID map
        coingecko_ids = {
            "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "USDC": "usd-coin",
        }
        coingecko_symbols = [s for s in symbols if s in coingecko_ids]
        yahoo_symbols = [s for s in symbols if s not in coingecko_ids]

        prices = {}

        # --- CoinGecko fetch ---
        if coingecko_symbols:
            ids = [coingecko_ids[sym] for sym in coingecko_symbols]
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={','.join(ids)}&vs_currencies=usd"
            response = requests.get(url).json()
            prices.update({
                sym: response[coingecko_ids[sym]]["usd"]
                for sym in coingecko_symbols if coingecko_ids[sym] in response
            })

        # --- Yahoo Finance fetch ---
        if yahoo_symbols:
            tickers = yf.Tickers(" ".join(yahoo_symbols))
            for sym in yahoo_symbols:
                ticker = tickers.tickers.get(sym)
                if ticker and hasattr(ticker, "fast_info") and ticker.fast_info.get("last_price"):
                    prices[sym] = ticker.info["regularMarketPrice"]

        return prices
    except Exception as e:
        st.warning(f"Error fetching real-time prices: {e}")
        return {}

# ----------------- Streamlit UI ----------------- #
import streamlit as st
st.set_page_config(
    page_title="AI Wealth Management Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
    
@st.cache_data(ttl=600)  # Cache for 10 minutes instead of 5
def get_live_prices_cached(symbol_str):
    symbol_list = symbol_str.split(",")
    coingecko_ids = {
        "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "USDC": "usd-coin",
    }
    coingecko_symbols = [s for s in symbol_list if s in coingecko_ids]
    yahoo_symbols = [s for s in symbol_list if s not in coingecko_ids]

    prices = {}

    # --- CoinGecko fetch with error handling ---
    if coingecko_symbols:
        try:
            ids = [coingecko_ids[sym] for sym in coingecko_symbols]
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={','.join(ids)}&vs_currencies=usd"
            response = requests.get(url, timeout=5)
            if response.status_code == 429:
                st.warning("CoinGecko rate limit exceeded. Crypto prices may not be up to date.")
            else:
                data = response.json()
                prices.update({
                    sym: data.get(coingecko_ids[sym], {}).get("usd")
                    for sym in coingecko_symbols
                })
        except Exception as e:
            st.warning(f"Error fetching CoinGecko prices: {e}")

    # --- Yahoo Finance fallback for non-crypto symbols ---
    if yahoo_symbols:
        import yfinance as yf
        tickers = yf.Tickers(" ".join(yahoo_symbols))
        for sym in yahoo_symbols:
            try:
                ticker = tickers.tickers.get(sym)
                if ticker and hasattr(ticker, "fast_info") and ticker.fast_info.get("last_price"):
                    prices[sym] = ticker.info["regularMarketPrice"]
            except Exception:
                continue

    return prices

# Minimal styling for a polished look
st.markdown(
    """
    <style>
    h1, h2, h3, h4 {
        color: #2c3e50;
    }
    body {
        background-color: #fafafa;
    }
    .css-12oz5g7 {
        font-size: 1.75rem;
        font-weight: 600;
    }
    .css-uf99v8 {
        font-size: 2.0rem;
        font-weight: 700;
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #3498db !important;
        color: #fff !important;
        border-radius: 8px !important;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2978a0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("AI Wealth Management Dashboard")
st.sidebar.header("Client Selector")

clients = get_clients()
if not clients:
    st.error("No clients found. Ensure the DB is initialized with sample data.")
    st.stop()

client_names = [c[1] for c in clients]
selected_name = st.sidebar.selectbox("Select Client", client_names)
selected_client = next(c for c in clients if c[1] == selected_name)

# Display client profile
st.subheader(f"Client Profile: {selected_name}")
st.markdown(f"**Email:** {selected_client[2]}")
st.markdown(f"**Risk Profile:** {selected_client[3]}")
st.markdown(f"**Goals:** {selected_client[4]}")

st.divider()

# Show holdings
st.subheader("📄 Holdings with Real-Time Pricing")
holdings = get_holdings(selected_client[0])
if holdings:
    holdings_df = pd.DataFrame(holdings, columns=["symbol", "type", "sector", "quantity", "value_usd", "cost_basis"])
    
    # Overwrite value_usd with real-time data
    symbols = holdings_df["symbol"].unique().tolist()
    prices = get_live_prices_cached(",".join(symbols))
    holdings_df["live_price"] = holdings_df["symbol"].map(prices)
    holdings_df["value_usd"] = holdings_df["live_price"] * holdings_df["quantity"]

    st.dataframe(
        holdings_df[["symbol", "type", "sector", "quantity", "live_price", "value_usd", "cost_basis"]]
        .sort_values(by="value_usd", ascending=False)
        .style.format({
            "quantity": "{:,.4f}",
            "live_price": "${:,.2f}",
            "value_usd": "${:,.2f}",
            "cost_basis": "${:,.2f}"
        })
    )
else:
    st.info("No holdings found for this client.")


st.divider()

# Subheader for the entire "Recent Alerts" section
st.subheader("\U0001F4E2 Recent Alerts")

alerts = get_alerts(selected_client[0])
if not alerts:
    st.info("No alerts found for this client.")
else:
    # 1. Create a dictionary of categories -> list of alerts
    grouped_alerts = {}

    # Initialize each known category in grouped_alerts for a stable order
    for cat in CATEGORY_KEYWORDS.keys():
        grouped_alerts[cat] = []
    grouped_alerts["Other News"] = []

    # 2. Classify each alert
    for alert_text, alert_url, created_at in alerts:
        cat = classify_alert(alert_text, alert_url)
        grouped_alerts[cat].append((alert_text, alert_url, created_at))

    # 3. Display them under subheadings
    for category in grouped_alerts:
        if grouped_alerts[category]:
            st.markdown(f"**{category}**")  # Subheading for the category
            for alert_text, alert_url, created_at in grouped_alerts[category]:
                st.markdown(f"- [{alert_text}]({alert_url}) ({created_at})")

st.divider()

# -----------------
# Market Intelligence Agent
# -----------------
st.subheader("Run Market Intelligence Agent")
query_type = st.selectbox(
    "Select Data Type to Fetch",
    [
        "Market News",
        "Analyst Reports",
        "SEC Filings (10-K, 10-Q)",
        "ETF Trends",
        "Macroeconomic Insights"
    ]
)

# --- Load time series data ---
def get_portfolio_timeseries(client_id):
    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT month, portfolio_value, monthly_return
            FROM portfolio_timeseries
            WHERE client_id = ? ORDER BY month
            """,
            conn,
            params=(client_id,)
        )

print("🔍 Debug: TAVILY_API_KEY =", TAVILY_API_KEY)
if st.button("Fetch Intelligence"):
    if not TAVILY_API_KEY.strip():
        st.error("Tavily API Key not found. Set TAVILY_API_KEY in your .env file.")
    else:
        if query_type == "Market News":
            query = "interest rates, inflation, ETFs, stocks"
        elif query_type == "Analyst Reports":
            query = f"{', '.join([h[0] for h in holdings])} analyst report"
        elif query_type == "SEC Filings (10-K, 10-Q)":
            query = f"{', '.join([h[0] for h in holdings])} 10-K OR 10-Q site:sec.gov"
        elif query_type == "ETF Trends":
            query = "ETF inflows, sector rotation, top performing ETFs"
        elif query_type == "Macroeconomic Insights":
            query = "US inflation, GDP forecast, Fed interest rate outlook"

        news_results = fetch_market_news(query, TAVILY_API_KEY)
        match_count = 0
        for res in news_results:
            if match_news_to_client(res, holdings):
                insert_alert(selected_client[0], res['title'], res['url'])
                match_count += 1

        st.success(f"Fetched {len(news_results)} items. {match_count} matched the client's holdings.")

        # Refresh alerts after fetch
        alerts = get_alerts(selected_client[0])

        # Regroup alerts
        grouped_alerts = {cat: [] for cat in CATEGORY_KEYWORDS}
        grouped_alerts["Other News"] = []

        for alert_text, alert_url, created_at in alerts:
            cat = classify_alert(alert_text, alert_url)
            grouped_alerts[cat].append((alert_text, alert_url, created_at))

        # Show categorized alerts
        st.markdown("### Updated Alerts")
        for category in grouped_alerts:
            if grouped_alerts[category]:
                st.markdown(f"**{category}**")
                for alert_text, alert_url, created_at in grouped_alerts[category]:
                    st.markdown(f"- [{alert_text}]({alert_url}) ({created_at})")

        # Internal report (GPT-generated)
        internal_report = generate_client_insights(
            selected_client[1],
            holdings,
            alerts,
            selected_client[3],
            selected_client[4]
        )
        st.markdown("### 🧠 Short Internal Report")
        st.write(internal_report)

        # --- Portfolio performance stats (needed for PDF) ---
        timeseries_df = get_portfolio_timeseries(selected_client[0])
        timeseries_df["month"] = pd.to_datetime(timeseries_df["month"])
        avg_return = timeseries_df["monthly_return"].mean()
        std_dev = timeseries_df["monthly_return"].std()
        sharpe = avg_return / std_dev if std_dev > 0 else 0

        # --- Create charts ---
        allocation_df = holdings_df.groupby("type")["value_usd"].sum().reset_index()
        chart_files = create_portfolio_charts(timeseries_df, allocation_df, selected_client[1])

        # --- Generate PDF with insights and visuals ---
        report_path = generate_pdf_report(
            selected_client[1],
            holdings,
            alerts,
            internal_report,
            avg_return,
            std_dev,
            sharpe,
            chart_files
        )
        st.write(f"📄 PDF saved at: `{report_path}`")

        with open(report_path, "rb") as file:
            st.download_button(
                label="📥 Download PDF Report",
                data=file,
                file_name=os.path.basename(report_path),
                mime="application/pdf"
            )

        if send_email_with_pdf(
            "brett.kaliner149@gmail.com",
            f"Weekly Internal Report for {selected_client[1]}",
            "Attached is the latest internal client report.",
            report_path
        ):
            st.success("✅ PDF report emailed successfully.")

# --- Enhancements to be added into app3.py ---
# Add after the "Holdings" section or wherever you'd like the analytics to appear

st.divider()
st.subheader("📊 Performance & Risk Overview")

timeseries_df = get_portfolio_timeseries(selected_client[0])

if timeseries_df.empty:
    st.info("No performance data available for this client.")
else:
    timeseries_df["month"] = pd.to_datetime(timeseries_df["month"])
    st.line_chart(timeseries_df.set_index("month")["portfolio_value"], use_container_width=True)

    avg_return = timeseries_df["monthly_return"].mean()
    std_dev = timeseries_df["monthly_return"].std()
    sharpe = avg_return / std_dev if std_dev > 0 else None

    st.markdown("#### Risk Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Monthly Return", f"{avg_return:.2%}")
    col2.metric("Std Dev", f"{std_dev:.2%}")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}" if sharpe is not None else "N/A")

    st.markdown("_Sharpe ratio indicates return per unit of risk. A value > 1 is considered good. Compare with client's risk profile:_")
    risk_map = {"aggressive": 1.0, "balanced": 0.6, "conservative": 0.3}
    threshold = risk_map.get(selected_client[3].lower(), 0.5)
    if sharpe is not None:
        if sharpe < threshold:
            st.warning(f"This portfolio's Sharpe ratio ({sharpe:.2f}) may not align with the client's {selected_client[3]} profile.")
        else:
            st.success("Sharpe ratio aligns well with the client's risk profile.")

    with st.expander("📈 View Monthly Returns Table"):
        st.dataframe(timeseries_df[["month", "monthly_return"]].set_index("month").style.format({"monthly_return": "{:.2%}"}))

# --- Diversification by Asset Type ---
st.subheader("🌐 Diversification by Asset Class")
holdings_df = pd.DataFrame(holdings, columns=["symbol", "type", "sector", "quantity", "value_usd", "cost_basis"])

# 🔁 Updated: Get real-time prices safely and handle CoinGecko limits
symbols = holdings_df["symbol"].dropna().unique().tolist()

# ✅ Fetch prices
prices = get_live_prices_cached(",".join(symbols))

# 🧮 Apply prices to DataFrame
holdings_df["live_price"] = holdings_df["symbol"].map(prices)
holdings_df["value_usd"] = holdings_df["live_price"] * holdings_df["quantity"]

if holdings_df.empty:
    st.info("No holdings data for diversification analysis.")
else:
    asset_type_breakdown = holdings_df.groupby("type")["value_usd"].sum().reset_index()
    st.bar_chart(asset_type_breakdown.set_index("type"))
    fig, ax = plt.subplots()
    ax.pie(
        asset_type_breakdown["value_usd"],
        labels=asset_type_breakdown["type"],
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title("Portfolio Allocation by Asset Class")
    ax.axis("equal")  # Equal aspect ratio ensures pie is a circle.
    st.pyplot(fig)

# --- Placeholder: Real-Time Price Integration (CoinGecko / Yahoo) ---
# --- Real-Time Price Calculation ---
st.subheader("💹 Real-Time Portfolio Valuation")

symbols = holdings_df["symbol"].unique().tolist()
prices = get_live_prices_cached(",".join(symbols))

if not prices:
    st.info("No real-time prices found for these assets.")
else:
    holdings_df["live_price"] = holdings_df["symbol"].map(prices)
    holdings_df["real_time_value"] = holdings_df["live_price"] * holdings_df["quantity"]
    total_live_value = holdings_df["real_time_value"].sum()

    st.metric("Current Market Value (Real-Time)", f"${total_live_value:,.2f}")

    with st.expander("📊 Asset-Level Real-Time Valuation"):
        st.dataframe(
            holdings_df[["symbol", "quantity", "live_price", "real_time_value"]]
            .sort_values(by="real_time_value", ascending=False)
            .style.format({
                "quantity": "{:,.4f}",
                "live_price": "${:,.2f}",
                "real_time_value": "${:,.2f}"
            })
        )

# --- Tax Estimator ---
# st.write("Holdings columns:", holdings_df.columns.tolist())
st.subheader("💰 Unrealized Gains Tax Estimator")
TAX_RATE = st.slider("Select Estimated Tax Rate (%)", 0, 40, 20)

if "cost_basis" in holdings_df.columns:
    holdings_df["unrealized_gain"] = holdings_df["value_usd"] - holdings_df["cost_basis"]
    est_tax = holdings_df["unrealized_gain"].sum() * TAX_RATE / 100
    st.metric("Estimated Tax if Liquidated Today", f"${est_tax:,.2f}")
    
    with st.expander("📄 Asset-Level Unrealized Gains"):
        st.dataframe(
            holdings_df[["symbol", "value_usd", "cost_basis", "unrealized_gain"]]
            .sort_values(by="unrealized_gain", ascending=False)
            .style.format({
                "value_usd": "${:,.2f}",
                "cost_basis": "${:,.2f}",
                "unrealized_gain": "${:,.2f}"
            })
        )
else:
    st.info("Tax estimation requires cost basis data, which was not found in the holdings.")

# Footer / Disclaimer
st.markdown("---")
st.markdown(
    "_Disclaimer: This dashboard is for informational purposes only and does not constitute "
    "investment advice. Please consult a licensed professional for specific guidance._"
)
