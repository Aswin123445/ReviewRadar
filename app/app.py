import streamlit as st
import joblib

model = joblib.load('model/sentiment_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

st.title("🎬 Movie Review Sentiment Analyzer")

review = st.text_area("Enter your movie review here:")

if st.button("Analyze"):
    data = vectorizer.transform([review])
    prediction = model.predict(data)[0]
    st.success(f"Sentiment: {'Positive 😊' if prediction == 'pos' else 'Negative 😠'}")
