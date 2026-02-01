import streamlit as st
import pandas as pd
import nltk, re
import PyPDF2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
from nltk.corpus import stopwords

st.set_page_config(page_title="AI Resume Screening", layout="centered")
st.title("🤖 AI Resume Screening & Ranking System")

# ---------- Load Dataset ----------
data = pd.read_csv("resume_data.csv")

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    words = [w for w in text.split() if w not in stopwords.words('english')]
    return " ".join(words)

data["Resume"] = data["Resume"].apply(clean_text)

# ---------- Train Classification Model ----------
vectorizer_cls = TfidfVectorizer()
X = vectorizer_cls.fit_transform(data["Resume"])
y = data["Category"]

model = MultinomialNB()
model.fit(X, y)

# ---------- PDF Reader ----------
def read_pdf(file):
    text = ""
    pdf = PyPDF2.PdfReader(file)
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# ---------- UI ----------
tab1, tab2 = st.tabs(["🔍 Resume Classification", "🏆 Resume Ranking"])

# ================= TAB 1 =================
with tab1:
    option = st.radio("Choose Input Type", ["Paste Resume Text", "Upload PDF Resume"])
    resume_text = ""

    if option == "Paste Resume Text":
        resume_text = st.text_area("📄 Paste Resume Here", height=200)
    else:
        uploaded_file = st.file_uploader("📤 Upload Resume (PDF)", type=["pdf"])
        if uploaded_file:
            resume_text = read_pdf(uploaded_file)
            st.success("PDF uploaded successfully")

    if st.button("Predict Job Role"):
        if resume_text.strip() == "":
            st.warning("Please enter resume text")
        else:
            cleaned = clean_text(resume_text)
            vec = vectorizer_cls.transform([cleaned])
            result = model.predict(vec)[0]
            st.success(f"🎯 Predicted Job Role: **{result}**")

# ================= TAB 2 =================
with tab2:
    st.subheader("🏆 Resume Ranking Based on Job Description")

    job_desc = st.text_area("🧾 Enter Job Description", height=150)

    uploaded_resumes = st.file_uploader(
        "📂 Upload Multiple Resume PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Rank Resumes"):
        if job_desc.strip() == "" or not uploaded_resumes:
            st.warning("Please provide JD and resumes")
        else:
            documents = [clean_text(job_desc)]
            resume_names = []

            for file in uploaded_resumes:
                text = read_pdf(file)
                documents.append(clean_text(text))
                resume_names.append(file.name)

            vectorizer_rank = TfidfVectorizer()
            vectors = vectorizer_rank.fit_transform(documents)

            similarities = cosine_similarity(vectors[0:1], vectors[1:])[0]

            ranking = sorted(
                zip(resume_names, similarities),
                key=lambda x: x[1],
                reverse=True
            )

            st.success("✅ Resume Ranking Completed")
            for i, (name, score) in enumerate(ranking, start=1):
                st.write(f"**{i}. {name}** — Match Score: **{round(score*100, 2)}%**")
