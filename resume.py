# ======================================================
# IMPORTS
# ======================================================
import streamlit as st
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# ======================================================
# CUSTOM CSS (BACKGROUND + STYLE)
# ======================================================
st.markdown("""
<style>

/* BUTTON FULL FIX */
div.stButton > button {
    background-color: #00C853 !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 0.6rem 1.5rem !important;
    font-size: 16px !important;
}

div.stButton > button:hover {
    background-color: #00E676 !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)
# ======================================================
# FUNCTIONS
# ======================================================

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def create_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        title={'text': "Match Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#00FFAA"},
            'steps': [
                {'range': [0, 40], 'color': "#ff4d4d"},
                {'range': [40, 70], 'color': "#ffa64d"},
                {'range': [70, 100], 'color': "#00FFAA"}
            ],
        }))
    return fig

# ======================================================
# SAMPLE CORPUS (Improves accuracy)
# ======================================================
corpus = [
    "python machine learning data science deep learning nlp",
    "cyber security networking penetration testing ethical hacking",
    "web development html css javascript react node",
    "cloud computing aws docker kubernetes devops",
    "java spring boot backend microservices sql"
]

vectorizer = TfidfVectorizer(stop_words="english")
vectorizer.fit(corpus)

# ======================================================
# UI
# ======================================================
st.title("📄 AI Resume Analyzer")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Resume (PDF)")
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

with col2:
    st.subheader("Job Description")
    job_text = st.text_area("Paste Job Description")

# ======================================================
# ANALYSIS
# ======================================================
if st.button("Analyze Match"):

    if pdf_file and job_text:

        resume_text = extract_text_from_pdf(pdf_file)

        resume_clean = clean_text(resume_text)
        job_clean = clean_text(job_text)

        tfidf = vectorizer.transform([resume_clean, job_clean])

        similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

        resume_words = set(resume_clean.split())
        job_words = set(job_clean.split())

        keyword_score = len(resume_words & job_words) / len(job_words)

        final_score = (0.7 * similarity) + (0.3 * keyword_score)

        # ================= RESULTS =================
        st.header("📊 Results")

        c1, c2, c3 = st.columns(3)

        c1.metric("Semantic Similarity", f"{similarity*100:.2f}%")
        c2.metric("Keyword Match", f"{keyword_score*100:.2f}%")
        c3.metric("Final Score", f"{final_score*100:.2f}%")

        st.plotly_chart(create_gauge(final_score), use_container_width=True)

        # ================= MATCH STATUS =================
        if final_score > 0.65:
            st.success("✅ GOOD MATCH")
        elif final_score > 0.4:
            st.warning("⚠️ PARTIAL MATCH")
        else:
            st.error("❌ NOT A MATCH")

        # ================= MISSING KEYWORDS =================
        missing_skills = job_words - resume_words

        st.subheader("❗ Missing Keywords")

        if missing_skills:
            st.write(", ".join(list(missing_skills)[:20]))
        else:
            st.success("No major keywords missing 🎯")

        # ================= SKILL COMPARISON CHART =================
        st.subheader("📌 Top Skill Comparison")

        tfidf_temp = TfidfVectorizer(max_features=10, stop_words='english')
        X_temp = tfidf_temp.fit_transform([resume_clean, job_clean])

        df = pd.DataFrame({
            "Skill": tfidf_temp.get_feature_names_out(),
            "Resume": X_temp.toarray()[0],
            "Job": X_temp.toarray()[1]
        })

        df = df.melt(id_vars="Skill")

        fig = px.bar(df, x="Skill", y="value", color="variable", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

        # ================= DOWNLOAD REPORT =================
        report = f"""
MATCH SCORE: {final_score*100:.2f}%
SEMANTIC SIMILARITY: {similarity*100:.2f}%
KEYWORD MATCH: {keyword_score*100:.2f}%

MISSING KEYWORDS:
{", ".join(list(missing_skills)[:20])}
"""

        st.download_button("📄 Download Report", report, file_name="match_report.txt")

    else:
        st.warning("Upload resume PDF and enter job description.")
