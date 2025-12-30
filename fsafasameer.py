import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import difflib
from collections import Counter

from pypdf import PdfReader
from textstat import textstat
from textblob import TextBlob

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Creative Accounting Detector",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Driven Detection of Creative Accounting")
st.caption("Linguistic analysis of Notes-to-Accounts for accounting manipulation risk")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Annual Report (PDF)", type=["pdf"])

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    st.success("PDF successfully processed")

    # ---------------- BASIC TEXT METRICS ----------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Word Count", len(text.split()))
    with col2:
        st.metric("Sentence Count", textstat.sentence_count(text))
    with col3:
        st.metric("Readability", round(textstat.flesch_reading_ease(text), 2))
    with col4:
        st.metric("Avg Sentence Length", round(textstat.avg_sentence_length(text), 2))

    # ---------------- SENTIMENT ----------------
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity

    st.subheader("📈 Linguistic Tone Analysis")
    st.write(f"**Overall Sentiment Score:** `{round(sentiment,3)}`")

    # ---------------- LINGUISTIC MANIPULATION FLAGS ----------------
    st.subheader("🚩 Linguistic Red-Flag Indicators")

    hedge_words = [
        "approximately", "significant", "may", "could",
        "management believes", "subject to", "estimated",
        "expected", "primarily"
    ]

    hedge_count = sum(text.lower().count(word) for word in hedge_words)

    st.metric("Hedging Language Frequency", hedge_count)

    # ---------------- COSINE SIMILARITY ----------------
    st.subheader("🔍 Disclosure Repetition & Boilerplate Risk")

    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) > 5:
        vectorizer = CountVectorizer().fit_transform(sentences[:100])
        similarity_matrix = cosine_similarity(vectorizer)

        avg_similarity = np.mean(similarity_matrix)
        st.metric("Average Disclosure Similarity", round(avg_similarity, 3))
    else:
        st.warning("Not enough textual data for similarity analysis")

    # ---------------- INTERPRETATION ----------------
    st.subheader("🧠 Interpretation")

    if hedge_count > 50 or avg_similarity > 0.35:
        st.error("⚠️ Elevated linguistic manipulation risk detected")
    else:
        st.success("✅ No major linguistic manipulation indicators detected")

else:
    st.info("Upload a PDF annual report to begin analysis")
