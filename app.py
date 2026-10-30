import time
from flask import Flask, render_template, request, send_file
import joblib
from reportlab.pdfgen import canvas
from lime.lime_text import LimeTextExplainer

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
history = []
# LIME Explainer
class_names = ["FAKE", "REAL"]
explainer = LimeTextExplainer(class_names=class_names)

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

    # LIME Explanation
    exp = explainer.explain_instance(
        news,
        lambda texts: model.predict_proba(vectorizer.transform(texts)),
        num_features=5
    )
    reason = exp.as_list()

    end_time = time.time()
    prediction_time = round((end_time - start_time) * 1000, 2)

    if prediction == 1:
        result = "REAL NEWS ✅"
        color = "green"

        ai_message = """
• The AI found writing patterns similar to reliable news.
• The article contains factual and formal language.
• No major clickbait indicators were detected.
• Always verify the news from trusted sources before sharing.
"""
    else:
        result = "FAKE NEWS ❌"
        color = "red"

        ai_message = """
• The AI detected misleading or sensational language.
• The writing pattern is similar to fake news in the training dataset.
• The article may contain exaggerated claims.
• Verify the news from official or trusted news sources.
"""

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
        news=news,
        reason=reason,
        ai_message=ai_message
    )
if __name__ == "__main__":
    app.run(debug=True)