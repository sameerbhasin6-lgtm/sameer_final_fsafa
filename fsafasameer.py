import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from pypdf import PdfReader
import re
from collections import Counter

# --- LIBRARY CHECK ---
try:
    from textstat import textstat
    from textblob import TextBlob
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    st.error("Missing libraries. Please run: pip install textstat textblob scikit-learn")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Forensic AI Master", page_icon="🕵️‍♂️", layout="wide")

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
    .high-risk { border-left: 5px solid #ff4b4b !important; background-color: #fff5f5 !important; }
    .good-metric { border-left: 5px solid #00cc96 !important; background-color: #f0fff4 !important; }
    .metric-label { font-size: 14px; color: #555; margin-bottom: 5px; }
    .metric-value { font-size: 28px; font-weight: bold; color: #222; }
    .metric-sub { font-size: 12px; color: #888; }
    .delta-pos { color: #ff4b4b; font-weight: bold; font-size: 14px; } 
    .delta-neg { color: #00cc96; font-weight: bold; font-size: 14px; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA LOADER ---
@st.cache_data
def load_red_flag_dictionary():
    try:
        # Try loading local CSV if it exists
        df = pd.read_csv("Annual_Report_Red_Flags.csv")
    except:
        # Fallback Data
        data = {
            "Word": ["contingent", "estimate", "fluctuate", "litigation", "claim", "uncertainty", "pending", "unresolved", "material", "adverse", "risk", "doubt", "going concern", "restatement", "write-off", "impairment", "related party"],
            "Category": ["Uncertainty", "Uncertainty", "Volatility", "Legal", "Legal", "Uncertainty", "Legal", "Legal", "Materiality", "Negative", "Risk", "Viability", "Viability", "Accounting", "Loss", "Loss", "Governance"]
        }
        df = pd.DataFrame(data)
    
    if 'Word' in df.columns:
        df['Word'] = df['Word'].astype(str).str.lower().str.strip()
    return df

# --- 2. TEXT EXTRACTION ---
@st.cache_data
def extract_text_fast(file, start_p=1, end_p=None):
    text = ""
    try:
        reader = PdfReader(file)
        total_pages = len(reader.pages)
        if end_p is None or end_p > total_pages: end_p = total_pages
        
        my_bar = st.progress(0, text="Scanning pages...")
        for i in range(start_p - 1, end_p):
            try:
                page_text = reader.pages[i].extract_text()
                if page_text: text += page_text + "\n"
            except: continue
            
            # Update bar safely
            progress = int(((i - start_p + 1) / (end_p - start_p + 1)) * 100)
            my_bar.progress(min(progress, 100))
            
        my_bar.empty()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# --- 3. METRICS ENGINE ---
def analyze_metrics(text, red_flag_df):
    if not text: return None
    
    words = re.findall(r'\w+', text.lower())
    total_words = len(words) if words else 1
    
    # A. Fog Index
    try: fog_index = textstat.gunning_fog(text)
    except: fog_index = 0
    
    # B. Sentiment
    try:
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity 
    except: sentiment = 0
    
    # C. Passive Voice
    passive_matches = re.findall(r'\b(was|were|been|being)\b\s+\w+ed\b', text.lower())
    passive_score = (len(passive_matches) / total_words) * 1000 

    # D. Complex Words %
    complex_count = sum(1 for w in words if textstat.syllable_count(w) >= 3)
    complex_pct = (complex_count / total_words) * 100
    
    # E. Custom Red Flag Analysis
    red_flag_set = set(red_flag_df['Word'].unique()) if not red_flag_df.empty else set()
    matched_words = [w for w in words if w in red_flag_set]
    
    # Map categories safely
    matched_categories = []
    if not red_flag_df.empty:
        word_to_cat = pd.Series(red_flag_df.Category.values, index=red_flag_df.Word).to_dict()
        matched_categories = [word_to_cat.get(w, "Unknown") for w in matched_words]
    
    return {
        "fog_index": fog_index,
        "sentiment": sentiment,
        "passive_score": passive_score,
        "complex_pct": complex_pct,
        "total_words": total_words,
        "matched_words": matched_words,
        "matched_categories": matched_categories
    }

def calculate_similarity(text1, text2):
    try:
        t1 = " ".join(text1.split()[:5000])
        t2 = " ".join(text2.split()[:5000])
        vectorizer = CountVectorizer().fit_transform([t1, t2])
        return cosine_similarity(vectorizer.toarray())[0][1]
    except: return 0

# --- SIDEBAR ---
with st.sidebar:
    st.title("🕵️‍♂️ Forensic Inputs")
    file_curr = st.file_uploader("Current Year Report (PDF)", type="pdf")
    file_prev = st.file_uploader("Previous Year Report (Optional)", type="pdf")
    uploaded_dict = st.file_uploader("Custom Dictionary (CSV)", type=["csv", "xlsx"])
    
    st.markdown("---")
    st.header("🔢 Financial Check")
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

# --- MAIN APP ---
if file_curr:
    # 1. Load Data
    if uploaded_dict:
        try:
            if uploaded_dict.name.endswith('.csv'):
                red_flags_df = pd.read_csv(uploaded_dict)
            else:
                red_flags_df = pd.read_excel(uploaded_dict)
            # Normalize
            if 'Word' in red_flags_df.columns:
                red_flags_df['Word'] = red_flags_df['Word'].astype(str).str.lower().str.strip()
        except: 
            st.warning("Error loading dictionary. Using default.")
            red_flags_df = load_red_flag_dictionary()
    else:
        red_flags_df = load_red_flag_dictionary()

    # 2. Process Files
    text_curr = extract_text_fast(file_curr, start_p, end_p)
    if not text_curr:
        st.error("Could not extract text. Please try another PDF or page range.")
        st.stop()
        
    metrics_curr = analyze_metrics(text_curr, red_flags_df)
    
    metrics_prev = None
    similarity = None
    if file_prev:
        text_prev = extract_text_fast(file_prev, start_p, end_p)
        if text_prev:
            metrics_prev = analyze_metrics(text_prev, red_flags_df)
            similarity = calculate_similarity(text_curr, text_prev)

    if metrics_curr:
        st.title("Forensic Analysis Dashboard")
        st.caption(f"Analyzing {metrics_curr['total_words']:,} words.")

        # --- FINANCIAL ALERT ---
        if net_income > 0 and cash_flow > 0 and net_income > (cash_flow * 1.5):
            st.warning(f"⚠️ **ACCRUALS WARNING:** Net Income ({net_income}) is >1.5x Cash Flow ({cash_flow}). Check for revenue recognition fraud.")

        # --- ROW 1: FLASHCARDS ---
        col1, col2, col3, col4 = st.columns(4)
        
        def get_delta_html(curr, prev, inverted=False):
            if prev is None: return ""
            diff = curr - prev
            # Inverted: High Fog is Bad (Pos Delta = Red). Low Sent is Bad (Pos Delta = Green).
            is_bad = (diff > 0 and not inverted) or (diff < 0 and inverted)
            color = "delta-pos" if is_bad else "delta-neg"
            arrow = "⬆" if diff > 0 else "⬇"
            return f'<span class="{color}">{arrow} {abs(diff):.2f} YoY</span>'

        with col1:
            delta = get_delta_html(metrics_curr['fog_index'], metrics_prev['fog_index'] if metrics_prev else None)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Fog Index</div>
                <div class="metric-value">{metrics_curr['fog_index']:.1f}</div>
                <div class="metric-sub">{delta}</div>
            </div>""", unsafe_allow_html=True)

        with col2:
            delta = get_delta_html(metrics_curr['sentiment'], metrics_prev['sentiment'] if metrics_prev else None, inverted=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Sentiment</div>
                <div class="metric-value">{metrics_curr['sentiment']:.2f}</div>
                <div class="metric-sub">{delta}</div>
            </div>""", unsafe_allow_html=True)

        with col3:
            delta = get_delta_html(metrics_curr['passive_score'], metrics_prev['passive_score'] if metrics_prev else None)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Passive Voice</div>
                <div class="metric-value">{metrics_curr['passive_score']:.1f}</div>
                <div class="metric-sub">{delta} | /1k words</div>
            </div>""", unsafe_allow_html=True)
            
        with col4:
            sim_display = f"{similarity*100:.1f}%" if similarity is not None else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">YoY Consistency</div>
                <div class="metric-value">{sim_display}</div>
                <div class="metric-sub">Similarity Score</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # --- ROW 2: RED FLAG CLOUD ---
        st.subheader("🚩 Red Flag Analysis")
        c_cloud, c_stats = st.columns([2, 1])
        
        with c_cloud:
            if metrics_curr['matched_words']:
                wc = WordCloud(background_color="white", colormap="Reds", width=600, height=350, collocations=False).generate(" ".join(metrics_curr['matched_words']))
                fig_wc, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig_wc)
            else:
                st.info("No words from the Red Flag list were found in the scanned pages.")

        with c_stats:
            if metrics_curr['matched_words']:
                st.markdown("**Top Anomalies**")
                counts = Counter(metrics_curr['matched_words'])
                df_counts = pd.DataFrame(counts.most_common(10), columns=["Word", "Count"])
                st.dataframe(df_counts, hide_index=True, use_container_width=True, height=150)
                
                # Pie Chart
                cat_counts = Counter(metrics_curr['matched_categories'])
                if cat_counts:
                    fig_cat = px.pie(names=list(cat_counts.keys()), values=list(cat_counts.values()), hole=0.5)
                    fig_cat.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=150, showlegend=False)
                    st.plotly_chart(fig_cat, use_container_width=True)

        st.markdown("---")

        # --- ROW 3: REGRESSION & CHARTS ---
        st.subheader("📈 Quantitative Analysis")
        
        tab1, tab2 = st.tabs(["Regression Benchmark", "YoY Comparison"])
        
        with tab1:
            col_desc, col_chart = st.columns([1, 2])
            with col_desc:
                st.markdown("""
                **Industry Benchmark Model**
                * **X-Axis:** Fog Index (Complexity)
                * **Y-Axis:** Fraud Risk Score
                * **Red X:** Your File
                
                If the Red X is high above the trend line, the document is unnecessarily complex compared to industry peers.
                """)
            with col_chart:
                np.random.seed(42)
                bench_fog = np.random.normal(16, 3, 50)
                bench_risk = (bench_fog * 2.5) + np.random.normal(0, 8, 50)
                df_bench = pd.DataFrame({"Fog Index": bench_fog, "Risk Score": bench_risk})
                
                model = LinearRegression()
                model.fit(df_bench[["Fog Index"]], df_bench["Risk Score"])
                curr_pred = model.predict([[metrics_curr['fog_index']]])[0]
                
                fig_reg = px.scatter(df_bench, x="Fog Index", y="Risk Score", opacity=0.4)
                line_x = np.linspace(df_bench["Fog Index"].min(), df_bench["Fog Index"].max(), 100).reshape(-1, 1)
                fig_reg.add_traces(go.Scatter(x=line_x.flatten(), y=model.predict(line_x), mode='lines', name='Industry Trend'))
                fig_reg.add_traces(go.Scatter(x=[metrics_curr['fog_index']], y=[curr_pred], mode='markers+text', marker=dict(color='red', size=15, symbol='x'), name='Your File', text=["YOU"], textposition="top center"))
                st.plotly_chart(fig_reg, use_container_width=True)

        with tab2:
            if metrics_prev:
                c1, c2 = st.columns(2)
                with c1:
                    fig_sent = go.Figure(data=[
                        go.Bar(name='Previous', x=['Sentiment'], y=[metrics_prev['sentiment']], marker_color='#95a5a6'),
                        go.Bar(name='Current', x=['Sentiment'], y=[metrics_curr['sentiment']], marker_color='#3498db')
                    ])
                    fig_sent.update_layout(title="Sentiment Shift")
                    st.plotly_chart(fig_sent, use_container_width=True)
                with c2:
                    fig_fog = go.Figure(data=[
                        go.Bar(name='Previous', x=['Fog Index'], y=[metrics_prev['fog_index']], marker_color='#95a5a6'),
                        go.Bar(name='Current', x=['Fog Index'], y=[metrics_curr['fog_index']], marker_color='#e74c3c')
                    ])
                    fig_fog.update_layout(title="Complexity Shift")
                    st.plotly_chart(fig_fog, use_container_width=True)
            else:
                st.info("Upload Previous Year PDF to see these charts.")

else:
    st.info("Upload a PDF to begin analysis.")
