import streamlit as st
import tensorflow as tf
import pickle
import re
import os

from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# =====================================
# LOAD MODEL + TOKENIZER
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# model path
model_path = os.path.join(
    BASE_DIR,
    "model",
    "imdb_rnn_model.h5"
)

# tokenizer path
tokenizer_path = os.path.join(
    BASE_DIR,
    "model",
    "tokenizer.pkl"
)

# load model
model = tf.keras.models.load_model(model_path)

# load tokenizer
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

# =====================================
# CLEANING FUNCTION
# =====================================

def clean_text(text):

    # lowercase
    text = text.lower()

    # remove html tags
    text = re.sub(r'<.*?>', '', text)

    # remove punctuation/numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text


# =====================================
# PREDICTION FUNCTION
# =====================================

max_len = 200

def predict_sentiment(review):

    # clean review
    review = clean_text(review)

    # text -> sequence
    sequence = tokenizer.texts_to_sequences([review])

    # padding
    padded_sequence = pad_sequences(
        sequence,
        maxlen=max_len
    )

    # prediction
    prediction = model.predict(padded_sequence)

    return prediction[0][0]


# =====================================
# UI
# =====================================

st.title("🎬 IMDb Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review below and the model will predict "
    "whether the sentiment is Positive or Negative."
)

review = st.text_area(
    "Enter Movie Review",
    height=200
)

# =====================================
# BUTTON
# =====================================

if st.button("Predict Sentiment"):

    if review.strip() == "":

        st.warning("Please enter a movie review.")

    else:

        score = predict_sentiment(review)

        # positive
        if score > 0.5:

            st.success("✅ Positive Review")

            st.write(
                f"Confidence Score: {score:.4f}"
            )

        # negative
        else:

            st.error("❌ Negative Review")

            st.write(
                f"Confidence Score: {1-score:.4f}"
            )

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.markdown(
    "Built with TensorFlow, SimpleRNN, and Streamlit"
)