import pickle
import numpy as np
from pathlib import Path

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================
# PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "next_word_lstm.keras"

TOKENIZER_PATH = BASE_DIR / "model" / "tokenizer.pkl"

# =====================================
# LOAD MODEL
# =====================================

model = load_model(MODEL_PATH)

# =====================================
# LOAD TOKENIZER
# =====================================

with open(TOKENIZER_PATH, 'rb') as f:

    tokenizer = pickle.load(f)

max_sequence_len = model.input_shape[1] + 1

# =====================================
# PREDICTION FUNCTION
# =====================================

def predict_next_word(text):

    token_list = tokenizer.texts_to_sequences([text])[0]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding='pre'
    )

    predicted = np.argmax(
        model.predict(token_list, verbose=0),
        axis=-1
    )

    output_word = ""

    for word, index in tokenizer.word_index.items():

        if index == predicted:

            output_word = word
            break

    return output_word