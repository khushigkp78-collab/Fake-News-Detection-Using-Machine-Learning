import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# Load Fake and Real datasets
# -----------------------------
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Keep only title + text
fake = fake[["title", "text"]]
true = true[["title", "text"]]

# Merge title and text
fake["text"] = fake["title"] + " " + fake["text"]
true["text"] = true["title"] + " " + true["text"]

# Add labels
fake["label"] = 0      # Fake
true["label"] = 1      # Real

# Keep required columns
fake = fake[["text", "label"]]
true = true[["text", "label"]]

# Merge datasets
data = pd.concat([fake, true], ignore_index=True)

# Shuffle dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset Shape:", data.shape)

# -----------------------------
# Split data
# -----------------------------
X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# -----------------------------
# Train Model
# -----------------------------
model = LogisticRegression(max_iter=1000)

model.fit(X_train_vector, y_train)

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test_vector)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", round(accuracy * 100, 2), "%")

# -----------------------------
# Save
# -----------------------------
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model Saved Successfully!")