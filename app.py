from flask import Flask, render_template, request
import os
import joblib

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "fake_email_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""

    if request.method == "POST":
        email_text = request.form["email"]

        transformed = vectorizer.transform([email_text])
        prediction = model.predict(transformed)[0]

        if prediction == 1:
            result = "⚠ Fake / Phishing Email"
        else:
            result = "✅ Real Email"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
