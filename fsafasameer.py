import streamlit as st

# --- 1. SAFE IMPORTS & SETUP ---
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
    from textstat import textstat
    from textblob import TextBlob
    from sklearn.linear_model import LinearRegression
except ImportError as e:
    st.error(f"❌ LIBRARY ERROR: {e}")
    st.warning("Please install required libraries: pip install streamlit pandas plotly matplotlib wordcloud textstat textblob scikit-learn pypdf openpyxl")
    st.stop()

st.set_page_config(page_title="Forensic & Credit Risk Master", page_icon="⚖️", layout="wide")

# --- CSS STYLING ---
st.markdown("""
    <style>
    .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #4e8cff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .risk-high { background-color: #ffe6e6; border-left: 5px solid #ff4b4b; padding: 15px; border-radius: 5px; }
    .risk-med { background-color: #fff8e6; border-left: 5px solid #ffa500; padding: 15px; border-radius: 5px; }
    .risk-low { background-color: #e6fffa; border-left: 5px solid #00cc96; padding: 15px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# MODULES & LOGIC
# ==========================================

class FinancialEngine:
    """Handles Credit Risk & Stress Testing"""
    def __init__(self):
        # Simulated Data (Replace with real extraction if needed)
        self.years = ['2019', '2020', '2021', '2022', '2023']
        self.financials = pd.DataFrame({
            "Year": self.years,
            "Revenue": [1000, 1100, 1050, 1300, 1450],
            "EBITDA": [200, 210, 180, 240, 250],
            "Net_Income": [80, 85, 40, 100, 110],
            "CFO": [180, 190, 120, 200, 150],
            "Total_Debt": [400, 450, 500, 600, 750],
            "Cash": [50, 60, 40, 50, 40],
            "Interest_Expense": [30, 32, 35, 45, 60]
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

class QualitativeForensics:
    """Handles NLP & Intent Detection"""
    def __init__(self, text):
        self.text = text.lower() if text else ""

    def analyze_manipulation_risk(self):
        hedges = ['approximately', 'estimated', 'possibly', 'might', 'suggests', 'assumed']
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

    def analyze_justification(self):
        triggers = ['because', 'due to', 'owing to', 'despite', 'attributed to', 'driven by']
        words = self.text.split()
        count = sum(1 for w in words if w in triggers)
        density = (count / (len(words) or 1)) * 1000
        level = "HIGH" if density > 15 else "MEDIUM" if density > 8 else "LOW"
        return {"Level": level, "Score": density}

    def detect_boilerplate(self, prev_text):
        if not prev_text: return "N/A"
        ratio = difflib.SequenceMatcher(None, self.text[:5000], prev_text[:5000].lower()).ratio()
        if ratio > 0.95: return "CRITICAL: Lazy Disclosure (Copy-Paste)"
        elif ratio > 0.85: return "HIGH: Boilerplate Repetition"
        return "LOW: Fresh Language"

@st.cache_data
def load_red_flag_dictionary(uploaded_file=None):
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            if 'Word' in df.columns:
                df['Word'] = df['Word'].astype(str).str.lower().str.strip()
                return df
        except: pass
    
    # Fallback Data
    data = {
        "Word": ["contingent", "estimate", "fluctuate", "litigation", "claim", "uncertainty", "material", "adverse", "going concern", "restatement", "write-off", "impairment"],
        "Category": ["Uncertainty", "Uncertainty", "Volatility", "Legal", "Legal", "Uncertainty", "Materiality", "Negative", "Viability", "Accounting", "Loss", "Loss"]
    }
    return pd.DataFrame(data)

@st.cache_data
def extract_text_fast(file):
    text = ""
    try:
        reader = PdfReader(file)
        # Scan first 50 pages to prevent timeout
        for i in range(min(50, len(reader.pages))):
            text += reader.pages[i].extract_text() + "\n"
        return text
    except: return ""

def analyze_metrics(text, red_flag_df):
    if not text: return None
    words = re.findall(r'\w+', text.lower())
    total_words = len(words) if words else 1
    
    try: fog = textstat.gunning_fog(text)
    except: fog = 0
    try: sent = TextBlob(text).sentiment.polarity
    except: sent = 0
    
    # Complex Words (3+ syllables)
    complex_words = [w for w in words if textstat.syllable_count(w) >= 3]
    complex_pct = (len(complex_words) / total_words) * 100
    
    # Red Flag Matching
    red_set = set(red_flag_df['Word'].unique()) if not red_flag_df.empty else set()
    matched = [w for w in words if w in red_set]
    
    word_to_cat = pd.Series(red_flag_df.Category.values, index=red_flag_df.Word).to_dict() if not red_flag_df.empty else {}
    cats = [word_to_cat.get(w, "Unknown") for w in matched]
    
    return {
        "fog": fog, "sentiment": sent, "complex_words": complex_words,
        "complex_pct": complex_pct, "matched_words": matched, "matched_categories": cats,
        "total_words": total_words
    }

# ==========================================
# MAIN UI
# ==========================================
fin_engine = FinancialEngine()

with st.sidebar:
    st.title("📂 Inputs")
    file_curr = st.file_uploader("Current Year Report (PDF)", type="pdf")
    file_prev = st.file_uploader("Previous Year Report (PDF)", type="pdf")
    uploaded_dict = st.file_uploader("Custom Red Flags (CSV/Excel)", type=["csv", "xlsx"])
    
    st.markdown("---")
    st.header("Financial Smoke Test")
    net_income = st.number_input("Net Income (Cr)", value=0.0)
    cash_flow = st.number_input("Cash Flow (Cr)", value=0.0)

if file_curr:
    # 1. PROCESS DATA
    red_flags_df = load_red_flag_dictionary(uploaded_dict)
    text_curr = extract_text_fast(file_curr)
    metrics_curr = analyze_metrics(text_curr, red_flags_df)
    
    text_prev = extract_text_fast(file_prev) if file_prev else None
    
    st.title("Forensic & Credit Risk Master")
    if net_income > (cash_flow * 1.5) and cash_flow > 0:
        st.warning(f"⚠️ **ACCRUALS ALERT:** Net Income ({net_income}) > 1.5x Cash Flow. Check earnings quality.")

    # --- TABS FOR DIFFERENT PAGES ---
    tab1, tab2, tab3 = st.tabs(["1️⃣ Forensic Snapshot", "2️⃣ Regression Analysis", "3️⃣ Credit & Qualitative"])

    # ==========================================
    # PAGE 1: FORENSIC SNAPSHOT (Fog, Word Cloud)
    # ==========================================
    with tab1:
        st.subheader("📊 Document Complexity & Red Flags")
        
        # Row 1: Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Fog Index (Complexity)", f"{metrics_curr['fog']:.1f}", "Target: <18")
        c2.metric("Complex Word %", f"{metrics_curr['complex_pct']:.1f}%", "Syllables >= 3")
        c3.metric("Sentiment Score", f"{metrics_curr['sentiment']:.2f}", "-1 (Neg) to +1 (Pos)")
        
        st.markdown("---")
        
        # Row 2: Word Cloud & Pie Chart
        col_cloud, col_stats = st.columns([2, 1])
        
        with col_cloud:
            st.markdown("##### 🚩 Red Flag Word Cloud (Custom Dictionary)")
            if metrics_curr['matched_words']:
                # Generate Cloud from MATCHED words only
                wc = WordCloud(background_color="white", colormap="Reds", width=600, height=350, collocations=False).generate(" ".join(metrics_curr['matched_words']))
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear'); ax.axis("off")
                st.pyplot(fig)
            else:
                st.info("No words from your uploaded dictionary were found in the text.")

        with col_stats:
            st.markdown("##### 🚨 Anomaly Breakdown")
            if metrics_curr['matched_words']:
                # Dataframe
                counts = Counter(metrics_curr['matched_words']).most_common(10)
                st.dataframe(pd.DataFrame(counts, columns=["Word", "Freq"]), hide_index=True, use_container_width=True, height=150)
                
                # Pie Chart
                cat_counts = Counter(metrics_curr['matched_categories'])
                if cat_counts:
                    fig_pie = px.pie(names=list(cat_counts.keys()), values=list(cat_counts.values()), hole=0.4)
                    fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=150, showlegend=False)
                    st.plotly_chart(fig_pie, use_container_width=True)

    # ==========================================
    # PAGE 2: REGRESSION ANALYSIS
    # ==========================================
    with tab2:
        st.subheader("📈 Industry Benchmark Regression")
        c_desc, c_chart = st.columns([1, 2])
        
        with c_desc:
            st.markdown("""
            **How to read this:**
            We compare your document against a simulated industry baseline of 50 companies.
            
            * **X-Axis (Fog Index):** Reading difficulty.
            * **Y-Axis (Fraud Risk):** Probability of irregularities.
            * **Trend Line:** Normal industry behavior.
            * **Red X:** Your document.
            
            **Insight:** If the Red X is significantly *above* the line, the complexity is unjustified and suggests higher risk.
            """)
            
        with c_chart:
            # Generate Dummy Benchmark Data
            np.random.seed(42)
            bench_fog = np.random.normal(16, 3, 50)
            bench_risk = (bench_fog * 2.5) + np.random.normal(0, 8, 50)
            df_bench = pd.DataFrame({"Fog Index": bench_fog, "Risk Score": bench_risk})
            
            # Train Model
            model = LinearRegression()
            model.fit(df_bench[["Fog Index"]], df_bench["Risk Score"])
            pred = model.predict([[metrics_curr['fog']]])[0]
            
            # Plot
            fig_reg = px.scatter(df_bench, x="Fog Index", y="Risk Score", opacity=0.4, title="Complexity vs. Fraud Risk")
            line_x = np.linspace(df_bench["Fog Index"].min(), df_bench["Fog Index"].max(), 100).reshape(-1, 1)
            fig_reg.add_traces(go.Scatter(x=line_x.flatten(), y=model.predict(line_x), mode='lines', name='Industry Trend'))
            fig_reg.add_traces(go.Scatter(x=[metrics_curr['fog']], y=[pred], mode='markers+text', marker=dict(color='red', size=15, symbol='x'), name='Your File', text=["YOU"], textposition="top center"))
            st.plotly_chart(fig_reg, use_container_width=True)

    # ==========================================
    # PAGE 3: CREDIT & QUALITATIVE
    # ==========================================
    with tab3:
        st.subheader("💰 Credit Risk & Qualitative Intent")
        
        # 1. Credit Snapshot
        latest = fin_engine.financials.iloc[-1]
        risk_score = 0
        if latest['Net_Leverage'] > 4: risk_score += 1
        if latest['ICR'] < 3: risk_score += 1
        
        lvl = "HIGH" if risk_score >= 2 else "MEDIUM" if risk_score == 1 else "LOW"
        color = "#ff4b4b" if lvl == "HIGH" else "#ffa500" if lvl == "MEDIUM" else "#00cc96"
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f'<div style="background-color:{color}20; padding:15px; border-left:5px solid {color}; border-radius:5px;"><h3>Credit Risk</h3><h1 style="color:{color}">{lvl}</h1></div>', unsafe_allow_html=True)
        with c2:
            st.write(f"**Key Drivers:** Net Leverage: {latest['Net_Leverage']:.2f}x | ICR: {latest['ICR']:.2f}x")
            st.caption("Based on simulated financial data (replace with real extraction for production).")

        st.markdown("---")
        
        # 2. Qualitative Forensics
        ql = QualitativeForensics(text_curr)
        risk = ql.analyze_manipulation_risk()
        
        col_qual1, col_qual2 = st.columns(2)
        with col_qual1:
            st.markdown(f"**Manipulation Risk: {risk['Rating']}**")
            for d in risk["Drivers"]: st.markdown(f"🔴 {d}")
            
        with col_qual2:
            jd = ql.analyze_justification()
            st.metric("Justification Density", f"{jd['Score']:.1f}", jd['Level'])
            st.caption("High density = Defensive/Explanatory tone.")
            
        # 3. Tone Drift (if Prev Year exists)
        if text_prev:
            st.markdown("---")
            st.subheader("🔄 Year-over-Year Drift")
            bp = ql.detect_boilerplate(text_prev)
            st.write(f"**Boilerplate Status:** {bp}")

else:
    st.info("Please upload a PDF to begin analysis.")
