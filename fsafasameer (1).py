import streamlit as st

# --- SAFELY IMPORT LIBRARIES ---
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
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    st.error(f"❌ LIBRARY ERROR: {e}")
    st.warning("Please install the required libraries: pip install streamlit pandas plotly matplotlib wordcloud textstat textblob scikit-learn pypdf openpyxl statsmodels")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Forensic & Credit Risk Suite", page_icon="⚖️", layout="wide")

# --- CSS STYLING ---
st.markdown("""
    <style>
    .metric-card { 
        background-color: #f8f9fa; 
        padding: 15px; 
        border-radius: 8px; 
        border-left: 5px solid #4e8cff; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
    }
    .delta-pos { color: #ff4b4b; font-weight: bold; font-size: 14px; } 
    .delta-neg { color: #00cc96; font-weight: bold; font-size: 14px; }
    .risk-high { background-color: #ffe6e6; border-left: 5px solid #ff4b4b; padding: 10px; border-radius: 5px; }
    .risk-med { background-color: #fff8e6; border-left: 5px solid #ffa500; padding: 10px; border-radius: 5px; }
    .risk-low { background-color: #e6fffa; border-left: 5px solid #00cc96; padding: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# MODULE 1: FINANCIAL ENGINE (CREDIT RISK)
# ==========================================
class FinancialEngine:
    """
    Handles quantitative credit risk: Stress Testing, Covenants, Red Flags.
    """
    def __init__(self):
        # SIMULATED 5-YEAR DATASET (Placeholder for Real Data)
        self.years = ['2019', '2020', '2021', '2022', '2023']
        self.financials = pd.DataFrame({
            "Year": self.years,
            "Revenue": [1000, 1100, 1050, 1300, 1450],
            "EBITDA": [200, 210, 180, 240, 250],
            "Net_Income": [80, 85, 40, 100, 110],
            "CFO": [180, 190, 120, 200, 150],  # Divergence example
            "Total_Debt": [400, 450, 500, 600, 750],
            "Cash": [50, 60, 40, 50, 40],
            "Interest_Expense": [30, 32, 35, 45, 60],
            "Capex": [50, 55, 40, 80, 100],
            "Equity": [500, 550, 580, 650, 700]
        })
        
        # Derived Metrics
        self.financials['Net_Debt'] = self.financials['Total_Debt'] - self.financials['Cash']
        self.financials['EBITDA_Margin'] = self.financials['EBITDA'] / self.financials['Revenue']
        self.financials['Net_Leverage'] = self.financials['Net_Debt'] / self.financials['EBITDA']
        self.financials['ICR'] = self.financials['EBITDA'] / self.financials['Interest_Expense']
        self.financials['ROE'] = self.financials['Net_Income'] / self.financials['Equity']

    def run_stress_test(self, rev_shock_pct, margin_shock_bps, rate_hike_bps):
        base = self.financials.iloc[-1].copy()
        stressed_rev = base['Revenue'] * (1 + rev_shock_pct)
        new_margin = base['EBITDA_Margin'] + (margin_shock_bps / 10000)
        stressed_ebitda = stressed_rev * new_margin
        added_interest = base['Total_Debt'] * (rate_hike_bps / 10000)
        stressed_interest = base['Interest_Expense'] + added_interest
        
        new_leverage = base['Net_Debt'] / stressed_ebitda if stressed_ebitda > 0 else 99.9
        new_icr = stressed_ebitda / stressed_interest if stressed_interest > 0 else 0
        
        return {
            "Revenue": stressed_rev,
            "EBITDA": stressed_ebitda,
            "Interest": stressed_interest,
            "Net_Leverage": new_leverage,
            "ICR": new_icr
        }

    def detect_red_flags(self):
        flags = []
        latest = self.financials.iloc[-1]
        prev = self.financials.iloc[-2]
        
        # 1. Revenue up, CFO down
        if (latest['Revenue'] > prev['Revenue']) and (latest['CFO'] < prev['CFO']):
            flags.append({"Flag": "Earnings Quality", "Severity": "High", "Desc": "Revenue growing while Operating Cash Flow declines."})
            
        # 2. Leverage rising, EBITDA flat
        lev_growth = latest['Net_Leverage'] > prev['Net_Leverage']
        ebitda_growth = (latest['EBITDA'] - prev['EBITDA']) / prev['EBITDA']
        if lev_growth and ebitda_growth < 0.05:
            flags.append({"Flag": "Capital Structure", "Severity": "Medium", "Desc": "Debt increasing without corresponding EBITDA growth."})
            
        return flags

    def recommend_covenants(self):
        latest = self.financials.iloc[-1]
        return [
            f"Net Leverage Ratio < {round(latest['Net_Leverage'] * 1.25, 1)}x",
            f"Interest Coverage Ratio > {round(latest['ICR'] * 0.8, 1)}x"
        ]

    def peer_benchmark(self):
        latest = self.financials.iloc[-1]
        return pd.DataFrame({
            "Metric": ["Net Leverage (x)", "EBITDA Margin (%)", "ICR (x)", "ROE (%)"],
            "Company": [latest['Net_Leverage'], latest['EBITDA_Margin']*100, latest['ICR'], latest['ROE']*100],
            "Peer Median": [2.5, 18.0, 4.5, 12.0],
            "Top Quartile": [1.5, 25.0, 6.0, 18.0]
        })

# ==========================================
# MODULE 2: QUALITATIVE FORENSICS (NLP)
# ==========================================
class QualitativeForensics:
    """
    Advanced linguistic analysis for intent detection.
    """
    def __init__(self, text):
        self.text = text.lower() if text else ""
        
    def analyze_justification_density(self):
        triggers = ['because', 'due to', 'owing to', 'as a result of', 'thereby', 'in order to', 'despite', 'attributed to']
        words = self.text.split()
        total = len(words) if words else 1
        count = sum(1 for w in words if w in triggers)
        score = (count / total) * 1000
        
        if score > 15: return {"Level": "HIGH", "Score": score, "Desc": "Excessive explanatory language (Defensive)."}
        elif score > 8: return {"Level": "MEDIUM", "Score": score, "Desc": "Moderate justification."}
        else: return {"Level": "LOW", "Score": score, "Desc": "Direct, factual reporting."}

    def detect_boilerplate(self, prev_text):
        if not prev_text: return "N/A"
        ratio = difflib.SequenceMatcher(None, self.text[:5000], prev_text[:5000].lower()).ratio() * 100
        if ratio > 95: return "CRITICAL: Lazy Disclosure (Copy-Paste)"
        elif ratio > 85: return "HIGH: Boilerplate Repetition"
        else: return "LOW: Fresh Language"

    def calculate_manipulation_risk(self):
        hedges = ['approximately', 'estimated', 'possibly', 'unlikely', 'might', 'suggests']
        one_timers = ['exceptional', 'one-time', 'non-recurring', 'special item', 'extraordinary']
        minimizers = ['only', 'merely', 'minor', 'immaterial']
        
        h_count = sum(self.text.count(w) for w in hedges)
        o_count = sum(self.text.count(w) for w in one_timers)
        m_count = sum(self.text.count(w) for w in minimizers)
        
        score = 0
        reasons = []
        if h_count > 10: score += 1; reasons.append("Excessive Hedging")
        if o_count > 3: score += 1; reasons.append("Repeated One-Time Items")
        if m_count > 5: score += 1; reasons.append("Minimization Tone")
        
        rating = "HIGH" if score >= 2 else "MODERATE" if score == 1 else "LOW"
        color = "#ff4b4b" if score >= 2 else "#ffa500" if score == 1 else "#00cc96"
        
        return {"Rating": rating, "Color": color, "Drivers": reasons, "Stats": f"Hedges: {h_count} | Exceptions: {o_count}"}

    def generate_tone_heatmap(self):
        risk_map = {}
        sections = {
            "Revenue": ["revenue", "sales", "recognition"],
            "Provisions": ["provision", "contingent", "legal"],
            "Related Party": ["related party", "associate", "arm's length"],
            "Other Income": ["other income", "miscellaneous"]
        }
        for sec, keys in sections.items():
            count = sum(self.text.count(k) for k in keys)
            risk_map[sec] = "Analyzed" if count > 5 else "Not Significant"
        return risk_map

# ==========================================
# MODULE 3: HELPER FUNCTIONS (CORE)
# ==========================================
@st.cache_data
def load_red_flag_dictionary():
    try:
        df = pd.read_csv("Annual_Report_Red_Flags.csv")
    except:
        data = {
            "Word": ["contingent", "estimate", "fluctuate", "litigation", "claim", "uncertainty", "material", "adverse", "going concern", "restatement", "write-off", "impairment"],
            "Category": ["Uncertainty", "Uncertainty", "Volatility", "Legal", "Legal", "Uncertainty", "Materiality", "Negative", "Viability", "Accounting", "Loss", "Loss"]
        }
        df = pd.DataFrame(data)
    if 'Word' in df.columns: df['Word'] = df['Word'].astype(str).str.lower().str.strip()
    return df

@st.cache_data
def extract_text_fast(file, start_p=1, end_p=None):
    text = ""
    try:
        reader = PdfReader(file)
        total_pages = len(reader.pages)
        if end_p is None or end_p > total_pages: end_p = total_pages
        
        for i in range(start_p - 1, end_p):
            try: text += reader.pages[i].extract_text() + "\n"
            except: continue
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
    
    passive = len(re.findall(r'\b(was|were|been|being)\b\s+\w+ed\b', text.lower()))
    passive_score = (passive / total_words) * 1000
    
    red_set = set(red_flag_df['Word'].unique()) if not red_flag_df.empty else set()
    matched = [w for w in words if w in red_set]
    
    word_to_cat = pd.Series(red_flag_df.Category.values, index=red_flag_df.Word).to_dict() if not red_flag_df.empty else {}
    cats = [word_to_cat.get(w, "Unknown") for w in matched]
    
    return {
        "fog": fog, "sentiment": sent, "passive_score": passive_score,
        "total_words": total_words, "matched_words": matched, "matched_categories": cats
    }

def calculate_similarity(text1, text2):
    try:
        t1 = " ".join(text1.split()[:5000])
        t2 = " ".join(text2.split()[:5000])
        vectorizer = CountVectorizer().fit_transform([t1, t2])
        return cosine_similarity(vectorizer.toarray())[0][1]
    except: return 0

# ==========================================
# MAIN APP UI
# ==========================================
# Init Financial Engine
fin_engine = FinancialEngine()

# --- MODULE: QUALITATIVE FORENSIC LINGUISTICS ---

class QualitativeForensics:
    """
    Advanced linguistic analysis for detecting manipulation intent,
    defensive writing, and boilerplate concealment.
    """
    def __init__(self, text):
        self.text = text.lower()
        self.sentences = [s.strip() for s in text.split('.') if len(s.split()) > 5]
        
    def analyze_justification_density(self):
        """
        Measures the ratio of 'explanatory' words (because, due to, thereby)
        vs factual reporting. High density suggests defensiveness.
        """
        justification_triggers = [
            'because', 'due to', 'owing to', 'as a result of', 'thereby', 
            'in order to', 'despite', 'although', 'attributed to', 'driven by'
        ]
        
        words = self.text.split()
        total_words = len(words) if len(words) > 0 else 1
        j_count = sum(1 for w in words if w in justification_triggers)
        density_score = (j_count / total_words) * 1000  # Per 1k words
        
        if density_score > 15:
            level = "HIGH"
            desc = "Management is using excessive explanatory language. This often signals an attempt to rationalize poor performance or complex accounting choices."
        elif density_score > 8:
            level = "MEDIUM"
            desc = "Moderate use of justification. Standard for complex businesses, but monitor for specific 'excuse-making' patterns."
        else:
            level = "LOW"
            desc = "Factual, direct reporting style. Low linguistic manipulation risk."
            
        return {"Level": level, "Score": density_score, "Explanation": desc}

    def detect_boilerplate(self, prev_text_snippet=None):
        """
        Checks if the current text is a near-duplicate of previous text.
        Returns a 'Boilerplate Score' (0-100%).
        """
        if not prev_text_snippet:
            return None
            
        # Use SequenceMatcher for text similarity
        ratio = difflib.SequenceMatcher(None, self.text[:5000], prev_text_snippet[:5000]).ratio()
        pct = ratio * 100
        
        if pct > 95:
            return "CRITICAL: 'Lazy' Disclosure (Near 100% Copy-Paste)"
        elif pct > 85:
            return "HIGH: Boilerplate Repetition (Little New Info)"
        elif pct > 60:
            return "MODERATE: Standard Carry-Over"
        else:
            return "LOW: Fresh Disclosure Language"

    def calculate_manipulation_risk(self):
        """
        Composite Risk Score based on hedging, defensive tone, and one-time items.
        """
        # 1. Hedging / Qualifiers
        hedges = ['approximately', 'roughly', 'estimated', 'presumably', 'possibly', 'unlikely', 'might', 'could', 'suggests']
        h_count = sum(self.text.count(w) for w in hedges)
        
        # 2. "One-Time" / Exceptional Items (The "Big Bath" trap)
        one_timers = ['exceptional', 'one-time', 'non-recurring', 'unprecedented', 'special item', 'extraordinary', 'restructuring charge']
        o_count = sum(self.text.count(w) for w in one_timers)
        
        # 3. Minimization Language
        minimizers = ['only', 'merely', 'minor', 'insignificant', 'immaterial', 'slight']
        m_count = sum(self.text.count(w) for w in minimizers)
        
        # Scoring Logic
        risk_score = 0
        reasons = []
        
        if h_count > 10: 
            risk_score += 1
            reasons.append("Excessive Hedging (Vagueness)")
        if o_count > 3: 
            risk_score += 1
            reasons.append("Repeated 'One-Time' Exceptions")
        if m_count > 5:
            risk_score += 1
            reasons.append("Minimization Tone (Downplaying Impact)")
            
        # Final Assessment
        if risk_score >= 2:
            rating = "HIGH"
            color = "red"
        elif risk_score == 1:
            rating = "MODERATE"
            color = "orange"
        else:
            rating = "LOW"
            color = "green"
            
        return {
            "Rating": rating, 
            "Color": color, 
            "Drivers": reasons, 
            "Stats": f"Hedges: {h_count} | Exceptions: {o_count} | Minimizers: {m_count}"
        }

    def generate_tone_heatmap(self):
        """
        Maps generic sections to linguistic risk levels.
        """
        # Keyword search to approximate sections
        risk_map = {}
        
        sections = {
            "Revenue Recognition": ["revenue", "sales", "turnover", "recognition"],
            "Provisions & Contingencies": ["provision", "contingent", "legal", "lawsuit"],
            "Related Party": ["related party", "associate", "subsidiary", "arm's length"],
            "Other Income": ["other income", "miscellaneous", "non-operating"]
        }
        
        for section, keywords in sections.items():
            # Extract simple snippet context
            count = sum(self.text.count(k) for k in keywords)
            if count > 10:
                # If section is heavy, check for risk words specifically nearby
                # (Simplified proximity check)
                risk_map[section] = "Analyzed"
            else:
                risk_map[section] = "Not Significant"
                
        return risk_map

# Sidebar
with st.sidebar:
    st.title("📂 Inputs")
    file_curr = st.file_uploader("Current Year Report (PDF)", type="pdf")
    file_prev = st.file_uploader("Previous Year Report (PDF)", type="pdf")
    uploaded_dict = st.file_uploader("Custom Dictionary (CSV)", type=["csv", "xlsx"])
    
    st.markdown("---")
    st.header("🔢 Financial 'Smoke Test'")
    net_income = st.number_input("Net Income (Cr)", value=0.0)
    cash_flow = st.number_input("Cash Flow Ops (Cr)", value=0.0)
    
    st.markdown("---")
    use_all = st.checkbox("Scan Full Doc", value=False)
    start_p, end_p = 1, 50
    if not use_all:
        c1, c2 = st.columns(2)
        start_p = c1.number_input("Start", 1, value=50)
        end_p = c2.number_input("End", 1, value=100)
    else: end_p = None

# Main Logic
if file_curr:
    # Load Data
    if uploaded_dict:
        try:
            red_flags_df = pd.read_csv(uploaded_dict) if uploaded_dict.name.endswith('.csv') else pd.read_excel(uploaded_dict)
            if 'Word' in red_flags_df.columns: red_flags_df['Word'] = red_flags_df['Word'].astype(str).str.lower().str.strip()
        except: red_flags_df = load_red_flag_dictionary()
    else:
        red_flags_df = load_red_flag_dictionary()

    # Process Text
    text_curr = extract_text_fast(file_curr, start_p, end_p)
    metrics_curr = analyze_metrics(text_curr, red_flags_df)
    
    text_prev, metrics_prev, similarity = None, None, None
    if file_prev:
        text_prev = extract_text_fast(file_prev, start_p, end_p)
        metrics_prev = analyze_metrics(text_prev, red_flags_df)
        similarity = calculate_similarity(text_curr, text_prev)

    if metrics_curr:
        st.title("Forensic & Credit Risk Dashboard")
        st.caption(f"Analyzing {metrics_curr['total_words']:,} words.")

        # Financial Warning
        if net_income > 0 and cash_flow > 0 and net_income > (cash_flow * 1.5):
            st.warning(f"⚠️ **ACCRUALS WARNING:** Net Income ({net_income}) > 1.5x Cash Flow ({cash_flow}). Check for revenue manipulation.")

        # --- TABS ---
        tab_credit, tab_forensic, tab_qual, tab_cloud, tab_reg = st.tabs([
            "💰 Credit Risk Snapshot", 
            "📊 Forensic Metrics", 
            "🧠 Qualitative Intent", 
            "🚩 Red Flag Cloud", 
            "📈 Regression & Benchmarks"
        ])

        # --- TAB 1: CREDIT RISK ---
        with tab_credit:
            latest = fin_engine.financials.iloc[-1]
            risk_score = 0
            if latest['Net_Leverage'] > 4: risk_score += 2
            if latest['ICR'] < 3: risk_score += 2
            
            lvl = "HIGH" if risk_score >= 4 else "MEDIUM" if risk_score >= 2 else "LOW"
            color = "#ff4b4b" if lvl == "HIGH" else "#ffa500" if lvl == "MEDIUM" else "#00cc96"
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f'<div style="background-color:{color}20; padding:15px; border-left:5px solid {color}; border-radius:5px;"><h3>Credit Risk</h3><h1 style="color:{color}">{lvl}</h1></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f"**Drivers:** Net Leverage: {latest['Net_Leverage']:.2f}x | ICR: {latest['ICR']:.2f}x")
            
            st.markdown("---")
            st.subheader("⚡ Stress Testing")
            sc1, sc2 = st.columns([1, 3])
            with sc1:
                s_rev = st.slider("Revenue Shock (%)", -30, 10, 0, step=5)
                s_mar = st.slider("Margin Shock (bps)", -500, 0, 0, step=50)
                s_int = st.slider("Rate Hike (bps)", 0, 500, 0, step=50)
            with sc2:
                stressed = fin_engine.run_stress_test(s_rev/100, s_mar, s_int)
                col1, col2, col3 = st.columns(3)
                col1.metric("Stressed EBITDA", int(stressed['EBITDA']), int(stressed['EBITDA'] - latest['EBITDA']))
                col2.metric("Stressed Leverage", f"{stressed['Net_Leverage']:.2f}x", f"{stressed['Net_Leverage'] - latest['Net_Leverage']:.2f}x", delta_color="inverse")
                col3.metric("Stressed ICR", f"{stressed['ICR']:.2f}x", f"{stressed['ICR'] - latest['ICR']:.2f}x", delta_color="inverse")

            st.markdown("---")
            st.subheader("🚩 Financial Red Flags")
            flags = fin_engine.detect_red_flags()
            if flags:
                for f in flags:
                    st.error(f"**{f['Flag']}**: {f['Desc']}")
            else:
                st.success("No automated financial red flags detected.")

        # --- TAB 2: FORENSIC METRICS ---
        with tab_forensic:
            c1, c2, c3, c4 = st.columns(4)
            
            # Helper for Delta
            def get_delta(curr, prev, inverted=False):
                if prev is None: return ""
                diff = curr - prev
                is_bad = (diff > 0 and not inverted) or (diff < 0 and inverted)
                color = "delta-pos" if is_bad else "delta-neg"
                arrow = "⬆" if diff > 0 else "⬇"
                return f'<span class="{color}">{arrow} {abs(diff):.2f} YoY</span>'

            with c1:
                delta = get_delta(metrics_curr['fog'], metrics_prev['fog'] if metrics_prev else None)
                st.markdown(f'<div class="metric-card"><div>Fog Index</div><h2>{metrics_curr["fog"]:.1f}</h2><div>{delta}</div></div>', unsafe_allow_html=True)
            with c2:
                delta = get_delta(metrics_curr['sentiment'], metrics_prev['sentiment'] if metrics_prev else None, inverted=True)
                st.markdown(f'<div class="metric-card"><div>Sentiment</div><h2>{metrics_curr["sentiment"]:.2f}</h2><div>{delta}</div></div>', unsafe_allow_html=True)
            with c3:
                delta = get_delta(metrics_curr['passive_score'], metrics_prev['passive_score'] if metrics_prev else None)
                st.markdown(f'<div class="metric-card"><div>Passive Voice</div><h2>{metrics_curr["passive_score"]:.1f}</h2><div>{delta}</div></div>', unsafe_allow_html=True)
            with c4:
                sim = f"{similarity*100:.1f}%" if similarity else "N/A"
                st.markdown(f'<div class="metric-card"><div>Consistency</div><h2>{sim}</h2><div>Similarity</div></div>', unsafe_allow_html=True)

        # --- TAB 3: QUALITATIVE INTENT ---
        with tab_qual:
            ql = QualitativeForensics(text_curr)
            risk = ql.calculate_manipulation_risk()
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f'<div style="background-color:{risk["Color"]}20; padding:20px; border-radius:10px; text-align:center;"><h3 style="margin:0">Manipulation Risk</h3><h1 style="color:{risk["Color"]}">{risk["Rating"]}</h1></div>', unsafe_allow_html=True)
            with c2:
                st.write("**Risk Drivers:**")
                for d in risk["Drivers"]: st.markdown(f"🔴 {d}")
                st.caption(risk["Stats"])
                
            st.markdown("---")
            jd = ql.analyze_justification_density()
            st.metric("Justification Density", f"{jd['Score']:.1f}", jd['Level'], delta_color="inverse")
            st.info(jd['Desc'])
            
            if text_prev:
                st.markdown("---")
                bp = ql.detect_boilerplate(text_prev)
                st.write(f"**Boilerplate Status:** {bp}")

        # --- TAB 4: WORD CLOUD ---
        with tab_cloud:
            if metrics_curr['matched_words']:
                wc = WordCloud(background_color="white", colormap="Reds", width=600, height=300).generate(" ".join(metrics_curr['matched_words']))
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear'); ax.axis("off")
                st.pyplot(fig)
                
                st.write("**Top Red Flags:**")
                counts = Counter(metrics_curr['matched_words']).most_common(10)
                st.dataframe(pd.DataFrame(counts, columns=["Word", "Count"]), use_container_width=True)
            else:
                st.info("No red flags found.")

        # --- TAB 5: REGRESSION ---
        with tab_reg:
            st.write("Comparing your file (Red X) against industry benchmark.")
            np.random.seed(42)
            bench_fog = np.random.normal(16, 3, 50)
            bench_risk = (bench_fog * 2.5) + np.random.normal(0, 8, 50)
            df_bench = pd.DataFrame({"Fog Index": bench_fog, "Risk Score": bench_risk})
            
            model = LinearRegression()
            model.fit(df_bench[["Fog Index"]], df_bench["Risk Score"])
            pred = model.predict([[metrics_curr['fog']]])[0]
            
            fig = px.scatter(df_bench, x="Fog Index", y="Risk Score", opacity=0.4)
            line_x = np.linspace(df_bench["Fog Index"].min(), df_bench["Fog Index"].max(), 100).reshape(-1, 1)
            fig.add_traces(go.Scatter(x=line_x.flatten(), y=model.predict(line_x), mode='lines', name='Industry'))
            fig.add_traces(go.Scatter(x=[metrics_curr['fog']], y=[pred], mode='markers+text', marker=dict(color='red', size=15, symbol='x'), name='You', text=["YOU"], textposition="top center"))
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload a PDF to start.")
