import pickle
import string

# Load model
with open("models/fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Clean text
def clean_text(text):
    text = text.lower()
    return ''.join([c for c in text if c not in string.punctuation])

# Predict function
def predict_news(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return "REAL" if prediction[0] == 1 else "FAKE"

# Example
if __name__ == "__main__":
    sample = input("Enter a news headline or text:\n")
    result = predict_news(sample)
    print(f"🧠 Prediction: {result} news.")
