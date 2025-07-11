import streamlit as st
import pickle
import string

# Load model and vectorizer
with open("models/fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Preprocessing function
def clean_text(text):
    text = text.lower()
    return ''.join([c for c in text if c not in string.punctuation])

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.subheader("Enter a news headline or article to check if it's Real or Fake.")

user_input = st.text_area("News Text", placeholder="e.g. Breaking: Scientists discover water on Mars...")

if st.button("Check Now"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news text to analyze.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)

        if prediction[0] == 1:
            st.success("✅ This looks like **REAL news**.")
        else:
            st.error("🚫 This appears to be **FAKE news**.")
