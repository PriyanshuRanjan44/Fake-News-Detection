import pickle
import string

# Load the saved model
with open("models/fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the TF-IDF vectorizer
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Function to clean input text
def clean_text(text):
    text = text.lower()
    return ''.join([c for c in text if c not in string.punctuation])

# Prediction function with confidence
def predict_news(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    # Check if model supports predict_proba
    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(vectorized)[0]
        conf_real = round(confidence[1] * 100, 2)
        conf_fake = round(confidence[0] * 100, 2)
    else:
        conf_real = conf_fake = "N/A"

    label = "REAL" if prediction == 1 else "FAKE"
    return label, conf_real, conf_fake

# CLI usage
if __name__ == "__main__":
    sample = input("📝 Enter a news headline or full article text:\n")
    label, conf_real, conf_fake = predict_news(sample)

    print("\n🧠 Prediction Result:")
    if label == "REAL":
        print("✅ This appears to be REAL news.")
    else:
        print("🚫 This appears to be FAKE news.")

    print(f"📊 Confidence → Real: {conf_real}% | Fake: {conf_fake}%")
