# import streamlit as st
# import re
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # ------------------------------
# # Page Configuration
# # ------------------------------
# st.set_page_config(page_title="Resume-Job Matcher", layout="centered")

# # ------------------------------
# # Custom Styling
# # ------------------------------
# st.markdown("""
#     <style>
#         .main {
#             background-color: #0E1117;
#         }
#         h1 {
#             color: #4CAF50;
#         }
#         .result-box {
#             background-color: #1C1F26;
#             padding: 25px;
#             border-radius: 12px;
#             border: 1px solid #2E3440;
#         }
#         .metric-label {
#             font-size: 15px;
#             font-weight: 600;
#             color: #A0AEC0;
#         }
#         .metric-value {
#             font-size: 18px;
#             font-weight: bold;
#             color: #4CAF50;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # ------------------------------
# # Preprocessing Function
# # ------------------------------
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
#     return text

# # ------------------------------
# # Skill Extraction
# # ------------------------------
# def extract_skills(text, skill_list):
#     tokens = text.split()
#     matched = [skill for skill in skill_list if skill in tokens]
#     return matched

# # ------------------------------
# # Scoring Function
# # ------------------------------
# def compute_scores(resume, job_desc):

#     skill_ontology = {
#         "python": 0.2,
#         "machine": 0.1,
#         "learning": 0.1,
#         "nlp": 0.15,
#         "data": 0.1,
#         "analysis": 0.1,
#         "sql": 0.1,
#         "communication": 0.05,
#         "leadership": 0.1
#     }

#     resume = clean_text(resume)
#     job_desc = clean_text(job_desc)

#     # Semantic Similarity (TF-IDF)
#     vectorizer = TfidfVectorizer()
#     vectors = vectorizer.fit_transform([resume, job_desc])
#     semantic_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

#     # Skill Matching
#     job_skills = extract_skills(job_desc, skill_ontology.keys())
#     resume_skills = extract_skills(resume, skill_ontology.keys())

#     matched_skills = set(job_skills).intersection(set(resume_skills))

#     if len(job_skills) > 0:
#         skill_score = len(matched_skills) / len(job_skills)
#     else:
#         skill_score = 0

#     weighted_score = sum([skill_ontology[s] for s in matched_skills])

#     # ATS Score
#     job_keywords = job_desc.split()
#     resume_keywords = resume.split()

#     if len(set(job_keywords)) > 0:
#         ats_score = len(set(job_keywords).intersection(set(resume_keywords))) / len(set(job_keywords))
#     else:
#         ats_score = 0

#     # Final Hybrid Score
#     alpha = 0.4
#     beta = 0.3
#     gamma = 0.3

#     final_score = (alpha * semantic_score) + (beta * skill_score) + (gamma * ats_score)

#     return semantic_score, skill_score, ats_score, weighted_score, final_score

# # ------------------------------
# # UI Layout
# # ------------------------------
# st.title("Hybrid Resume–Job Matching System")
# st.write("Lightweight ML + NLP + ATS Based Matching Model")

# resume_input = st.text_area("Enter Resume Text")
# job_input = st.text_area("Enter Job Description Text")

# if st.button("Analyze Matching"):

#     if resume_input and job_input:

#         semantic, skill, ats, weighted, final = compute_scores(resume_input, job_input)

#         st.markdown('<div class="result-box">', unsafe_allow_html=True)

#         st.subheader("Matching Results")

#         col1, col2 = st.columns(2)

#         with col1:
#             st.markdown(f"""
#                 <div class="metric-label">Semantic Similarity</div>
#                 <div class="metric-value">{semantic * 100:.2f}%</div>
#             """, unsafe_allow_html=True)

#             st.markdown(f"""
#                 <div class="metric-label">Skill Match Score</div>
#                 <div class="metric-value">{skill * 100:.2f}%</div>
#             """, unsafe_allow_html=True)

#         with col2:
#             st.markdown(f"""
#                 <div class="metric-label">ATS Compatibility Score</div>
#                 <div class="metric-value">{ats * 100:.2f}%</div>
#             """, unsafe_allow_html=True)

#             st.markdown(f"""
#                 <div class="metric-label">Weighted Skill Contribution</div>
#                 <div class="metric-value">{weighted * 100:.2f}%</div>
#             """, unsafe_allow_html=True)

#         st.progress(final)

#         # Big Final Matching Score
#         st.markdown(f"""
#             <div style="
#                 text-align: center;
#                 font-size: 34px;
#                 font-weight: 900;
#                 background: linear-gradient(90deg, #1C1F26, #222831);
#                 padding: 18px;
#                 border-radius: 10px;
#                 border: 1px solid #4CAF50;
#                 color: #4CAF50;
#                 margin-top: 25px;">
#                 Final Matching Score: {final * 100:.2f}%
#             </div>
#         """, unsafe_allow_html=True)

#         if final > 0.6:
#             st.success("Candidate Highly Suitable")
#         elif final > 0.4:
#             st.warning("Candidate Moderately Suitable")
#         else:
#             st.error("Candidate Not Suitable")

#         st.markdown('</div>', unsafe_allow_html=True)

#     else:
#         st.warning("Please enter both Resume and Job Description.")
# ======================================================
# IMPORT LIBRARIES
# ======================================================
import streamlit as st
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import base64
from io import StringIO

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="AI Resume Analyzer", 
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# CUSTOM CSS FOR BETTER VISUAL APPEARANCE
# ======================================================
def local_css():
    st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Gradient background for headers */
    .gradient-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
    }
    
    /* Success/Error message styling */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid;
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: bold;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Info boxes */
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
    }
    
    /* Stats box */
    .stats-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    
    .stats-number {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .stats-label {
        color: #666;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ======================================================
# TEXT CLEANING FUNCTION
# ======================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ======================================================
# CREATE WORD CLOUD
# ======================================================
def create_wordcloud(text, title):
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=100,
        contour_width=1,
        contour_color='steelblue'
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, pad=20)
    return fig

# ======================================================
# CREATE GAUGE CHART
# ======================================================
def create_gauge_chart(probability, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 16}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': '#ff4444'},
                {'range': [33, 66], 'color': '#ffbb33'},
                {'range': [66, 100], 'color': '#00C851'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

# ======================================================
# CREATE SKILLS COMPARISON CHART
# ======================================================
def create_skills_chart(resume_text, job_text):
    # Extract top keywords from both texts
    vectorizer_temp = TfidfVectorizer(max_features=10, stop_words='english')
    
    try:
        combined = [resume_text, job_text]
        X_temp = vectorizer_temp.fit_transform(combined)
        feature_names = vectorizer_temp.get_feature_names_out()
        
        resume_scores = X_temp[0].toarray()[0]
        job_scores = X_temp[1].toarray()[0]
        
        # Create dataframe for visualization
        df_scores = pd.DataFrame({
            'Skill': feature_names,
            'Resume Score': resume_scores,
            'Job Score': job_scores
        })
        
        df_scores = df_scores.melt(id_vars=['Skill'], var_name='Source', value_name='Score')
        
        fig = px.bar(
            df_scores, 
            x='Skill', 
            y='Score', 
            color='Source',
            barmode='group',
            title='Top Skills Comparison',
            color_discrete_sequence=['#667eea', '#764ba2']
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=400,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    except:
        return None

# ======================================================
# SAMPLE TRAINING DATASET
# ======================================================
data = {
    "resume": [
        "Python machine learning SQL data analysis",
        "Java spring boot developer",
        "Deep learning tensorflow python",
        "Marketing sales communication",
        "Cloud computing AWS docker kubernetes",
        "React javascript frontend developer",
        "Data science python pandas numpy",
        "Project management agile scrum",
        "DevOps engineer CI/CD Jenkins",
        "UI/UX design figma sketch"
    ],
    "job": [
        "Looking for python machine learning engineer",
        "Hiring java backend developer",
        "AI engineer deep learning required",
        "Digital marketing executive",
        "DevOps engineer AWS kubernetes",
        "Frontend developer react javascript",
        "Data scientist machine learning",
        "Project manager agile scrum",
        "DevOps engineer CI/CD pipelines",
        "UI/UX designer with figma"
    ],
    "label": [1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)
df["combined"] = df["resume"] + " " + df["job"]

# ======================================================
# TF-IDF VECTORIZATION
# ======================================================
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df["combined"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======================================================
# TRAIN LOGISTIC REGRESSION MODEL
# ======================================================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# ======================================================
# MAIN APP LAYOUT
# ======================================================
local_css()

# Header with gradient
st.markdown("""
<div class="gradient-header">
    <h1>📄 AI Resume Analyzer with ML Model</h1>
    <p style="font-size: 1.2rem;">Advanced Resume Screening powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with model info
with st.sidebar:
    st.markdown("## 🤖 Model Information")
    
    st.markdown("""
    <div class="info-box">
        <h4>📊 Model Performance</h4>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{accuracy*100:.1f}%")
    with col2:
        st.metric("Samples", len(df))
    
    st.markdown("""
        <p><strong>Training Data Size:</strong> {} samples</p>
        <p><strong>Features:</strong> {} dimensions</p>
        <p><strong>Model Type:</strong> Logistic Regression</p>
        <p><strong>Vectorization:</strong> TF-IDF</p>
    </div>
    """.format(len(df), X.shape[1]), unsafe_allow_html=True)
    
    st.markdown("## 📋 Quick Tips")
    st.markdown("""
    - Paste a complete resume for best results
    - Include relevant keywords in job description
    - Longer text provides better analysis
    - Match probability above 70% is considered good
    """)
    
    st.markdown("## 🎯 Features")
    st.markdown("""
    - ✓ TF-IDF Vectorization
    - ✓ Cosine Similarity
    - ✓ Logistic Regression
    - ✓ Word Cloud Generation
    - ✓ Skills Comparison
    - ✓ Performance Metrics
    """)

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
        <h3>📝 Resume Input</h3>
    """, unsafe_allow_html=True)
    resume_input = st.text_area("Paste your resume here", height=200, 
                                placeholder="Paste the candidate's resume here...")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Resume stats if input provided
    if resume_input:
        st.markdown("""
        <div class="stats-box">
        """, unsafe_allow_html=True)
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            st.markdown(f"<p class='stats-number'>{len(resume_input.split())}</p>", unsafe_allow_html=True)
            st.markdown("<p class='stats-label'>Words</p>", unsafe_allow_html=True)
        with col1b:
            st.markdown(f"<p class='stats-number'>{len(set(resume_input.lower().split()))}</p>", unsafe_allow_html=True)
            st.markdown("<p class='stats-label'>Unique Words</p>", unsafe_allow_html=True)
        with col1c:
            st.markdown(f"<p class='stats-number'>{len(resume_input)}</p>", unsafe_allow_html=True)
            st.markdown("<p class='stats-label'>Characters</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h3>💼 Job Description</h3>
    """, unsafe_allow_html=True)
    job_input = st.text_area("Paste the job description here", height=200,
                             placeholder="Paste the job requirements here...")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Job stats if input provided
    if job_input:
        st.markdown("""
        <div class="stats-box">
        """, unsafe_allow_html=True)
        col2a, col2b, col2c = st.columns(3)
        with col2a:
            st.markdown(f"<p class='stats-number'>{len(job_input.split())}</p>", unsafe_allow_html=True)
            st.markdown("<p class='stats-label'>Words</p>", unsafe_allow_html=True)
        with col2b:
            st.markdown(f"<p class='stats-number'>{len(set(job_input.lower().split()))}</p>", unsafe_allow_html=True)
            st.markdown("<p class='stats-label'>Unique Words</p>", unsafe_allow_html=True)
        with col2c:
            st.markdown(f"<p class='stats-number'>{len(job_input)}</p>", unsafe_allow_html=True)
            st.markdown("<p class='stats-label'>Characters</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Analyze button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analyze_button = st.button("🚀 Analyze Match with ML Model")

# Results section
if analyze_button:
    if resume_input and job_input:
        
        with st.spinner("🤖 Analyzing with Machine Learning model..."):
            
            resume_clean = clean_text(resume_input)
            job_clean = clean_text(job_input)
            combined_input = resume_clean + " " + job_clean
            
            # ML Predictions
            input_vector = vectorizer.transform([combined_input])
            prediction = model.predict(input_vector)[0]
            probability = model.predict_proba(input_vector)[0][1]
            
            # Cosine similarity
            tfidf_input = vectorizer.transform([resume_clean, job_clean])
            similarity = cosine_similarity(tfidf_input[0:1], tfidf_input[1:2])[0][0]
            
            # Results header
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")
            
            # Metric cards
            col_res1, col_res2, col_res3, col_res4 = st.columns(4)
            
            with col_res1:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color: white; margin-bottom: 0;">Match Status</h3>
                """, unsafe_allow_html=True)
                if prediction == 1:
                    st.markdown("<h2 style='color: white;'>✅ GOOD</h2>", unsafe_allow_html=True)
                else:
                    st.markdown("<h2 style='color: white;'>❌ POOR</h2>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col_res2:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color: white; margin-bottom: 0;">Probability</h3>
                    <h2 style='color: white;'>{:.1f}%</h2>
                </div>
                """.format(probability*100), unsafe_allow_html=True)
            
            with col_res3:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color: white; margin-bottom: 0;">Similarity</h3>
                    <h2 style='color: white;'>{:.1f}%</h2>
                </div>
                """.format(similarity*100), unsafe_allow_html=True)
            
            with col_res4:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color: white; margin-bottom: 0;">Model Acc</h3>
                    <h2 style='color: white;'>{:.1f}%</h2>
                </div>
                """.format(accuracy*100), unsafe_allow_html=True)
            
            # Progress bar
            st.markdown("### Match Probability")
            st.progress(float(probability))
            
            # Visualizations in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Gauge Chart", "☁️ Word Clouds", "📊 Skills Comparison", "📋 Report"])
            
            with tab1:
                st.plotly_chart(create_gauge_chart(probability, "Match Probability"), use_container_width=True)
            
            with tab2:
                col_wc1, col_wc2 = st.columns(2)
                with col_wc1:
                    fig1 = create_wordcloud(resume_clean, "Resume Word Cloud")
                    st.pyplot(fig1)
                with col_wc2:
                    fig2 = create_wordcloud(job_clean, "Job Description Word Cloud")
                    st.pyplot(fig2)
            
            with tab3:
                fig3 = create_skills_chart(resume_clean, job_clean)
                if fig3:
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("Not enough data to generate skills comparison")
            
            with tab4:
                # Detailed report
                st.markdown("### 📄 Detailed Analysis Report")
                
                report_data = {
                    'Metric': ['Match Prediction', 'Match Probability', 'Cosine Similarity', 
                              'Model Accuracy', 'Resume Words', 'Job Words', 'Training Samples'],
                    'Value': [
                        'Good Match' if prediction == 1 else 'Not Good Match',
                        f'{probability*100:.2f}%',
                        f'{similarity*100:.2f}%',
                        f'{accuracy*100:.2f}%',
                        len(resume_input.split()),
                        len(job_input.split()),
                        len(df)
                    ]
                }
                
                report_df = pd.DataFrame(report_data)
                st.table(report_df)
                
                # Classification Report
                st.markdown("### 📊 Model Performance Metrics")
                report_dict = classification_report(y_test, predictions, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.format("{:.2f}"))
                
                # Download button
                report = f"""
                ========================================
                AI RESUME ANALYZER - MATCH REPORT
                ========================================
                
                Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                MATCH ANALYSIS
                --------------
                Prediction: {'Good Match' if prediction==1 else 'Not Good Match'}
                Match Probability: {probability*100:.2f}%
                Cosine Similarity: {similarity*100:.2f}%
                
                MODEL PERFORMANCE
                -----------------
                Model Accuracy: {accuracy*100:.2f}%
                Training Samples: {len(df)}
                
                TEXT STATISTICS
                ---------------
                Resume: {len(resume_input.split())} words, {len(resume_input)} characters
                Job Description: {len(job_input.split())} words, {len(job_input)} characters
                
                ========================================
                """
                
                st.download_button(
                    "📥 Download Full Report",
                    data=report,
                    file_name=f"resume_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    else:
        st.warning("⚠️ Please enter both Resume and Job Description to proceed.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Powered by Machine Learning | TF-IDF Vectorization | Logistic Regression</p>
    <p style="font-size: 0.8rem;">© 2024 AI Resume Analyzer. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)