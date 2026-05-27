import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pickle
from pathlib import Path

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Bidirectional,
    Dropout
)

from preprocess import load_data
from visualize import plot_graphs

# =====================================
# PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "News_Category_Dataset_v3.json"

MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "next_word_lstm.keras"

TOKENIZER_PATH = MODEL_DIR / "tokenizer.pkl"

# =====================================
# LOAD DATASET
# =====================================

print("\nLoading dataset...")

headlines = load_data(
    DATA_PATH,
    limit=5000
)

print(f"Total Headlines Loaded: {len(headlines)}")

# =====================================
# TOKENIZATION
# =====================================

print("\nTokenizing text...")

tokenizer = Tokenizer()

tokenizer.fit_on_texts(headlines)

total_words = len(tokenizer.word_index) + 1

print(f"Vocabulary Size: {total_words}")

# =====================================
# CREATE INPUT SEQUENCES
# =====================================

print("\nCreating sequences...")

input_sequences = []

for line in headlines:

    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):

        n_gram_sequence = token_list[:i + 1]

        input_sequences.append(n_gram_sequence)

print(f"Total Sequences: {len(input_sequences)}")

# =====================================
# PADDING
# =====================================

print("\nPadding sequences...")

max_sequence_len = max(len(x) for x in input_sequences)

input_sequences = np.array(
    pad_sequences(
        input_sequences,
        maxlen=max_sequence_len,
        padding='pre'
    )
)

# =====================================
# SPLIT FEATURES & LABELS
# =====================================

X = input_sequences[:, :-1]

y = input_sequences[:, -1]

print(f"\nInput Shape: {X.shape}")

# =====================================
# BUILD MODEL
# =====================================

print("\nBuilding model...")

model = Sequential()

# Embedding Layer
model.add(
    Embedding(
        input_dim=total_words,
        output_dim=50,
        input_length=max_sequence_len - 1
    )
)

# BiLSTM Layer
model.add(
    Bidirectional(
        LSTM(
            64,
            return_sequences=False
        )
    )
)

# Dropout Layer
model.add(
    Dropout(0.2)
)

# Output Layer
model.add(
    Dense(
        total_words,
        activation='softmax'
    )
)

# =====================================
# COMPILE MODEL
# =====================================

print("\nCompiling model...")

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# =====================================
# TRAIN MODEL
# =====================================

print("\nTraining started...\n")

history = model.fit(
    X,
    y,
    epochs=10,
    batch_size=256,
    validation_split=0.2,
    verbose=1
)

# =====================================
# SAVE MODEL
# =====================================

print("\nSaving model...")

MODEL_DIR.mkdir(exist_ok=True)

model.save(MODEL_PATH)

# =====================================
# SAVE TOKENIZER
# =====================================

with open(TOKENIZER_PATH, 'wb') as f:

    pickle.dump(tokenizer, f)

print("\nTokenizer saved.")

# =====================================
# SAVE GRAPHS
# =====================================

plot_graphs(history)

print("\nTraining Completed Successfully!")