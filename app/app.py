import streamlit as st
import joblib
import os

model_path = os.path.join(os.path.dirname(__file__), 'model', 'sentiment_model.pkl')
model = joblib.load(model_path)
print(model_path)
vectorizer = joblib.load('model/vectorizer.pkl')

st.title("🎬 Movie Review Sentiment Analyzer")

review = st.text_area("Enter your movie review here:")

if st.button("Analyze"):
    data = vectorizer.transform([review])
    prediction = model.predict(data)[0]
    st.success(f"Sentiment: {'Positive 😊' if prediction == 'pos' else 'Negative 😠'}")
