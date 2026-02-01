import streamlit as st
import pickle
from preprocess import clean_text

# STEP 1 + STEP 2
# Load trained model and vectorizer (trained earlier using Kaggle)
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# STEP 3
# App UI
st.title("📰 Fake News Detection System")

news_input = st.text_area(
    "Enter news"
)

# STEP 4
# Prediction logic (works for ALL news)
if st.button("Predict"):
    if news_input.strip() == "":
        st.warning("Please enter some news text")
    else:
        cleaned = clean_text(news_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)

        if prediction[0] == 1:
            st.error("❌ Fake News")
        else:
            st.success("✅ Real News")
