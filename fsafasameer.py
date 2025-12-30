import streamlit as st

# --- 1. SAFE IMPORTS (CRASH PREVENTION) ---
# This block checks if you have the libraries. If not, it stops gracefully.
try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud, STOPWORDS
    import numpy as np
    from pypdf import PdfReader
    import re
    import difflib
    from collections import Counter
    # These are specific libraries that often cause errors if missing
    from textstat import textstat
    from textblob import TextBlob
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    st.error(f"❌ MISSING LIBRARY ERROR: {e}")
    st.warning("You must install the required libraries. Run this command in your terminal:")
    st.code("pip install streamlit pandas plotly matplotlib wordcloud textstat textblob scikit-learn pypdf openpyxl statsmodels")
    st.stop()

# --- 2. PAGE CONFIG (MUST BE FIRST) ---
st.set_page_config(page_title="Forensic & Credit Risk Suite", page_icon="⚖️", layout="wide")

# --- 3. CSS STYLING ---
st.markdown("""
    <style>
    .metric-card { 
        background-color: #f8f9fa; 
        padding: 15px; 
        border-radius: 8px; 
        border-left: 5px solid #4e8cff; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
    }
    .risk-high { background-color: #ffe6e6; border-left: 5px solid #ff4b4b; padding: 15px; border-radius: 5px; }
    .risk-med { background-color: #fff8e6; border-left: 5px solid #ffa500; padding: 15px; border-radius: 5px; }
    .risk-low { background-color: #e6fffa; border-left: 5px solid #00cc96; padding: 15px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# MODULE A: FINANCIAL ENGINE (CREDIT RISK)
# ==========================================
class FinancialEngine:
    def __init__(self):
        # Simulated Data (In production, replace with real CSV)
        self.years = ['2019', '2020', '2021', '2022', '2023']
        self.financials = pd.DataFrame({
            "Year": self.years,
            "Revenue": [1000, 1100, 1050, 1300, 1450],
            "EBITDA": [200, 210, 180, 240, 250],
            "Net_Income": [80, 85, 40, 100, 110],
            "CFO": [180, 190, 120, 200, 150],
            "Total_Debt": [400, 450, 500, 600, 750],
            "Cash": [50, 60, 40, 50, 40],
            "Interest_Expense": [30, 32, 35, 45, 60],
            "Capex": [50, 55, 40, 80, 100],
            "Equity": [500, 550, 580, 650, 700]
        })
        self.financials['Net_Debt'] = self.financials['Total_Debt'] - self.financials['Cash']
        self.financials['Net_Leverage'] = self.financials['Net_Debt'] / self.financials['EBITDA']
        self.financials['ICR'] = self.financials['EBITDA'] / self.financials['Interest_Expense']

    def run_stress_test(self, rev_shock, margin_shock, rate_shock):
        base = self.financials.iloc[-1]
        new_rev = base['Revenue'] * (1 + rev_shock)
        new_ebitda = new_rev * ((base['EBITDA']/base['Revenue']) + margin_shock)
        new_int = base['Interest_Expense'] + (base['Total_Debt'] * rate_shock)
        
        new_lev = base['Net_Debt'] / new_ebitda if new_ebitda > 0 else 50.0
        new_icr = new_ebitda / new_int if new_int > 0 else 0.0
        
        return {"EBITDA": new_ebitda, "Leverage": new_lev, "ICR": new_icr}

    def detect_red_flags(self):
        flags = []
        latest = self.financials.iloc[-1]
        prev = self.financials.iloc[-2]
        
        if latest['Revenue'] > prev['Revenue'] and latest['CFO'] < prev['CFO']:
            flags.append("Earnings Quality: Revenue up, Cash Flow down.")
        if latest['Net_Leverage'] > prev['Net_Leverage'] and latest['EBITDA'] <= prev['EBITDA']:
            flags.append("Capital Structure: Debt funding operations, not growth.")
        return flags

# ==========================================
# MODULE B: QUALITATIVE FORENSICS (NLP)
# ==========================================
class QualitativeForensics:
    def __init__(self, text):
        self.text = text.lower() if text else ""

    def analyze_manipulation_risk(self):
        hedges = ['approximately', 'estimated', 'possibly', 'might', 'suggests']
        one_timers = ['exceptional', 'one-time', 'non-recurring', 'special item']
        
        h_count = sum(self.text.count(w) for w in hedges)
        o_count = sum(self.text.count(w) for w in one_timers)
        
        score = 0
        reasons = []
        if h_count > 10: score += 1; reasons.append("Excessive Hedging (Vagueness)")
        if o_count > 3: score += 1; reasons.append("Repeated 'One-Time' Items")
        
        rating = "HIGH" if score >= 2 else "MODERATE" if score == 1 else "LOW"
        color = "#ff4b4b" if score >= 2 else "#ffa500" if score == 1 else "#00cc96"
        return {"Rating": rating, "Color": color, "Drivers": reasons}

    def detect_boilerplate(self, prev_text):
        if not prev_text: return "N/A"
        # Calculate similarity ratio
        ratio = difflib.SequenceMatcher(None, self.text[:5000], prev_text[:5000].lower()).ratio()
        if ratio > 0.95: return "CRITICAL: Lazy Disclosure (Copy-Paste)"
        elif ratio > 0.85: return "HIGH: Boilerplate Repetition"
        return "LOW: Fresh Language"

    def analyze_justification(self):
        triggers = ['because', 'due to', 'owing to', 'despite', 'attributed to']
        words = self.text.split()
        total = len(words) if words else 1
        count = sum(1 for w in words if w in triggers)
        density = (count / total) * 1000
        
        level = "HIGH" if density > 15 else "MEDIUM" if density > 8 else "LOW"
        return {"Level": level, "Score": density}

# ==========================================
# MODULE C: HELPER FUNCTIONS
# ==========================================
@st.cache_data
def load_dictionary():
    # Built-in fallback to prevent crash if file missing
    data = {
        "Word": ["contingent", "litigation", "uncertainty", "material", "adverse", "going concern", "restatement"],
        "Category": ["Uncertainty", "Legal", "Uncertainty", "Materiality", "Risk", "Viability", "Accounting"]
    }
    return pd.DataFrame(data)

@st.cache_data
def extract_pdf(file):
    text = ""
    try:
        reader = PdfReader(file)
        # Scan first 50 pages max to save time
        for i in range(min(50, len(reader.pages))):
            text += reader.pages[i].extract_text() + "\n"
        return text
    except: return ""

def calculate_metrics(text):
    if not text: return None
    try: fog = textstat.gunning_fog(text)
    except: fog = 0
    try: sent = TextBlob(text).sentiment.polarity
    except: sent = 0
    return {"fog": fog, "sentiment": sent}

# ==========================================
# MAIN APP UI
# ==========================================
fin_engine = FinancialEngine()

# Sidebar
with st.sidebar:
    st.title("📂 Inputs")
    file_curr = st.file_uploader("Current Year Report (PDF)", type="pdf")
    file_prev = st.file_uploader("Previous Year Report (PDF)", type="pdf")
    
    st.markdown("---")
    st.header("Financial Check")
    net_income = st.number_input("Net Income (Cr)", value=0.0)
    cash_flow = st.number_input("Cash Flow (Cr)", value=0.0)

if file_curr:
    text_curr = extract_pdf(file_curr)
    metrics_curr = calculate_metrics(text_curr)
    
    text_prev = extract_pdf(file_prev) if file_prev else None
    metrics_prev = calculate_metrics(text_prev) if text_prev else None
    
    st.title("Forensic & Credit Risk Dashboard")

    if net_income > (cash_flow * 1.5) and cash_flow > 0:
        st.warning(f"⚠️ **ACCRUALS ALERT:** Net Income ({net_income}) is significantly higher than Cash Flow ({cash_flow}).")

    # TABS
    tab_credit, tab_forensic, tab_qual = st.tabs(["💰 Credit Risk", "📊 Forensic Metrics", "🧠 Qualitative Intent"])

    # --- TAB 1: CREDIT RISK ---
    with tab_credit:
        latest = fin_engine.financials.iloc[-1]
        
        # Risk Snapshot
        score = 0
        if latest['Net_Leverage'] > 4: score += 1
        if latest['ICR'] < 3: score += 1
        
        lvl = "HIGH" if score >= 2 else "MEDIUM" if score == 1 else "LOW"
        color = "#ff4b4b" if lvl == "HIGH" else "#ffa500" if lvl == "MEDIUM" else "#00cc96"
        
        st.markdown(f"""
        <div style="background-color:{color}20; padding:20px; border-left:5px solid {color}; border-radius:10px;">
            <h3 style="margin:0">Credit Risk Level</h3>
            <h1 style="color:{color}; margin:0">{lvl}</h1>
            <p>Leverage: {latest['Net_Leverage']:.2f}x | ICR: {latest['ICR']:.2f}x</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("⚡ Stress Test")
        c1, c2, c3 = st.columns(3)
        s_rev = c1.slider("Revenue Shock (%)", -30, 0, 0, step=5)
        s_mar = c2.slider("Margin Shock (bps)", -500, 0, 0, step=50)
        s_int = c3.slider("Rate Hike (bps)", 0, 500, 0, step=50)
        
        res = fin_engine.run_stress_test(s_rev/100, s_mar/10000, s_int/10000)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Stressed EBITDA", f"{int(res['EBITDA'])}")
        c2.metric("Stressed Leverage", f"{res['Leverage']:.2f}x")
        c3.metric("Stressed ICR", f"{res['ICR']:.2f}x")
        
        flags = fin_engine.detect_red_flags()
        if flags:
            st.error(f"🚩 **Red Flags:** {', '.join(flags)}")

    # --- TAB 2: FORENSIC METRICS ---
    with tab_forensic:
        c1, c2 = st.columns(2)
        c1.metric("Fog Index (Complexity)", f"{metrics_curr['fog']:.1f}")
        c2.metric("Sentiment Score", f"{metrics_curr['sentiment']:.2f}")
        
        if metrics_prev:
            st.info(f"Previous Year Fog: {metrics_prev['fog']:.1f} | Previous Sentiment: {metrics_prev['sentiment']:.2f}")

        # Cloud
        wc = WordCloud(background_color="white", width=800, height=300).generate(text_curr)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear'); ax.axis("off")
        st.pyplot(fig)

    # --- TAB 3: QUALITATIVE INTENT ---
    with tab_qual:
        ql = QualitativeForensics(text_curr)
        risk = ql.analyze_manipulation_risk()
        
        st.markdown(f"""
        <div style="background-color:{risk['Color']}20; padding:15px; border-left:5px solid {risk['Color']};">
            <h3>Manipulation Risk: {risk['Rating']}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if risk['Drivers']:
            st.write("**Drivers:**")
            for d in risk['Drivers']: st.markdown(f"- {d}")
            
        jd = ql.analyze_justification()
        st.metric("Justification Density", f"{jd['Score']:.1f}", jd['Level'])
        
        if text_prev:
            bp = ql.detect_boilerplate(text_prev)
            st.write(f"**Boilerplate Status:** {bp}")

else:
    st.info("Upload a PDF to start.")
