import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load("clickbait_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit app title
st.title("Clickbait Detector")

# Input box for headline
headline = st.text_input("Enter a headline to check:")

# Predict button
if st.button("Check"):
    if headline.strip() == "":
        st.warning("Please enter a headline!")
    else:
        vec = vectorizer.transform([headline])
        result = model.predict(vec)[0]
        st.success("Prediction: Clickbait ✅" if result == 1 else "Prediction: Not Clickbait ❌")

