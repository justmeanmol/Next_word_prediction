import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(
    page_title="Next Word Prediction",
    layout="centered"
)

st.title("🔮 Next Word Prediction (LSTM)")
st.write("Type a sentence and let the model predict the next word.")

# ----------------------------
# Load Model & Vocabulary
# ----------------------------
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("next_word_model_best.h5")

    with open("word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)

    with open("idx2word.pkl", "rb") as f:
        idx2word = pickle.load(f)

    return model, word2idx, idx2word


model, word2idx, idx2word = load_assets()

MAX_SEQ_LEN = 20   # MUST be same as training

# ----------------------------
# Preprocessing
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9.,!? ]+", "", text)
    return text


def preprocess_input(text):
    text = clean_text(text)
    tokens = text.split()
    encoded = [word2idx.get(w, word2idx.get("<UNK>", 1)) for w in tokens]
    padded = pad_sequences([encoded], maxlen=MAX_SEQ_LEN, padding="pre")
    return padded


# ----------------------------
# Prediction
# ----------------------------
def predict_next_word(text, top_k=3):
    seq = preprocess_input(text)
    preds = model.predict(seq, verbose=0)[0]

    top_indices = np.argsort(preds)[-top_k:][::-1]
    results = [(idx2word[i], float(preds[i])) for i in top_indices]
    return results


# ----------------------------
# UI
# ----------------------------
user_input = st.text_input("✍️ Enter your text:")

top_k = st.slider("Number of suggestions", 1, 5, 3)

if st.button("Predict Next Word 🚀"):
    if len(user_input.strip()) == 0:
        st.warning("Please enter some text first.")
    else:
        predictions = predict_next_word(user_input, top_k)

        st.subheader("📌 Predicted Next Word(s)")
        for i, (word, prob) in enumerate(predictions, 1):
            st.write(f"**{i}. {word}** — probability: `{prob:.4f}`")

# ----------------------------
# Debug / Transparency (Optional)
# ----------------------------
with st.expander("🔍 Show processed input"):
    st.write(preprocess_input(user_input))
