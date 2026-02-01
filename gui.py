import pandas as pd
import nltk
import re
import tkinter as tk
from tkinter import messagebox

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
data = pd.read_csv("resume_data.csv")

# Clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

data["Resume"] = data["Resume"].apply(clean_text)

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["Resume"])
y = data["Category"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# Prediction function
def predict_resume():
    resume_text = text_area.get("1.0", tk.END).strip()
    if resume_text == "":
        messagebox.showwarning("Warning", "Please enter resume text")
        return

    resume_text = clean_text(resume_text)
    resume_vec = vectorizer.transform([resume_text])
    result = model.predict(resume_vec)[0]

    result_label.config(text=f"Predicted Job Role: {result}")

# GUI Window
root = tk.Tk()
root.title("AI Resume Screening System")
root.geometry("500x400")

tk.Label(root, text="Paste Resume Text Below", font=("Arial", 12)).pack(pady=10)

text_area = tk.Text(root, height=10, width=55)
text_area.pack()

tk.Button(root, text="Predict Job Role", command=predict_resume,
          bg="blue", fg="white", font=("Arial", 11)).pack(pady=15)

result_label = tk.Label(root, text="", font=("Arial", 12, "bold"))
result_label.pack(pady=10)

root.mainloop()
