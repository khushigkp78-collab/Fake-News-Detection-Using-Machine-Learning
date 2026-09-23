import time
from flask import Flask, render_template, request, send_file
import joblib
from reportlab.pdfgen import canvas

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
history = []

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    global history
    news = request.form.get("news")

    if not news:
        return render_template("index.html")

    start_time = time.time()

    news_vector = vectorizer.transform([news])

    prediction = model.predict(news_vector)[0]

    confidence = model.predict_proba(news_vector)[0]
    confidence = round(max(confidence) * 100, 2)

    end_time = time.time()
    prediction_time = round((end_time - start_time) * 1000, 2)

    if prediction == 1:
        result = "REAL NEWS ✅"
        color = "green"
    else:
        result = "FAKE NEWS ❌"
        color = "red"

    history.insert(0, {
        "news": news[:40] + ("..." if len(news) > 40 else ""),
        "result": result,
        "time": prediction_time
    })

    history = history[:5]

    return render_template(
        "index.html",
        prediction=result,
        color=color,
        prediction_time=prediction_time,
        confidence=confidence,
        history=history,
        news=news
    )
       
    
if __name__ == "__main__":
    app.run(debug=True)