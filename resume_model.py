import pandas as pd
import nltk, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')
from nltk.corpus import stopwords

data = pd.read_csv("dataset/resume_data.csv")

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    words = [w for w in text.split() if w not in stopwords.words('english')]
    return " ".join(words)

data["Resume"] = data["Resume"].apply(clean_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["Resume"])
y = data["Category"]

model = MultinomialNB()
model.fit(X, y)

def predict(resume_text):
    resume_text = clean_text(resume_text)
    vec = vectorizer.transform([resume_text])
    return model.predict(vec)[0]
