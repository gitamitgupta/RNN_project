import streamlit as st
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SRC_PATH = BASE_DIR / "src"

sys.path.append(str(SRC_PATH))

from predict import predict_next_word

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🤖",
    layout="centered"
)

# =====================================
# UI
# =====================================

st.title("🤖 Next Word Prediction App")

st.write("Deep Learning based Next Word Predictor using LSTM")

text = st.text_input("Enter your sentence")

if st.button("Predict Next Word"):

    if text.strip() == "":

        st.warning("Please enter some text.")

    else:

        prediction = predict_next_word(text)

        st.success(f"Predicted Next Word: {prediction}")