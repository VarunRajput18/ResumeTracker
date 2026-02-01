import streamlit as st
import pandas as pd
import nltk, re
import PyPDF2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')
from nltk.corpus import stopwords

st.set_page_config(page_title="AI Resume Screening", layout="centered")
st.title("🤖 AI Resume Screening System")

# ---------- Load Dataset ----------
data = pd.read_csv("resume_data.csv")

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    words = [w for w in text.split() if w not in stopwords.words('english')]
    return " ".join(words)

data["Resume"] = data["Resume"].apply(clean_text)

# ---------- Train Model ----------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["Resume"])
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
option = st.radio("Choose Input Type", ["Paste Resume Text", "Upload PDF Resume"])

resume_text = ""

if option == "Paste Resume Text":
    resume_text = st.text_area("📄 Paste Resume Here", height=200)

else:
    uploaded_file = st.file_uploader("📤 Upload Resume (PDF only)", type=["pdf"])
    if uploaded_file is not None:
        resume_text = read_pdf(uploaded_file)
        st.success("✅ PDF Uploaded Successfully")
        st.text_area("Extracted Text (Preview)", resume_text[:2000], height=200)

# ---------- Prediction ----------
if st.button("🔍 Predict Job Role"):
    if resume_text.strip() == "":
        st.warning("⚠️ Please provide resume text or upload PDF")
    else:
        resume_text = clean_text(resume_text)
        resume_vec = vectorizer.transform([resume_text])
        result = model.predict(resume_vec)[0]
        st.success(f"🎯 Predicted Job Role: **{result}**")
