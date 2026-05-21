import streamlit as st
import tensorflow as tf
import pickle
import re

from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = tf.keras.models.load_model("model/imdb_rnn_model.h5")

# Load tokenizer
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 200

# Cleaning function
def clean_text(text):

    text = text.lower()

    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text

# Prediction function
def predict_sentiment(review):

    review = clean_text(review)

    sequence = tokenizer.texts_to_sequences([review])

    padded = pad_sequences(sequence, maxlen=max_len)

    prediction = model.predict(padded)

    return prediction[0][0]

# UI
st.title("IMDb Movie Review Sentiment Analysis")

review = st.text_area("Enter Movie Review")

if st.button("Predict"):

    score = predict_sentiment(review)

    if score > 0.5:
        st.success("Positive Review 😀")
    else:
        st.error("Negative Review 😡")

    st.write(f"Prediction Score: {score}")