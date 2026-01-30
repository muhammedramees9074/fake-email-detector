import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Sample training data
data = {
    "text": [
        "Congratulations! You won a lottery",
        "Your bank account is locked",
        "Meeting scheduled tomorrow",
        "Project update attached",
        "Claim your free prize now",
        "Important invoice details"
    ],
    "label": [1, 1, 0, 0, 1, 0]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "fake_email_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model files created successfully!")
