from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("fake_email_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

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
