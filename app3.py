import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, \
    retry_if_exception_type, wait_fixed, stop_after_delay
from requests.exceptions import ConnectionError, Timeout

st.set_page_config(page_title="Human-Likeness Score (HLS)", layout="wide")
st.title("Human-Likeness Score (HLS) Demo")

st.markdown("""
This demo computes the **Human-Likeness Score (HLS)** for your model's captions by comparing them to reference (human) captions using Sentence-BERT embeddings and cosine similarity.
""")

# --- Sample data: you can edit these lists ---
ref_captions = [
    "A dog is running through a field.",
    "Two children are playing with a ball.",
    "A man is riding a bicycle on the street.",
    "A group of people are sitting at a table.",
    "A woman is holding an umbrella in the rain."
]

model_captions = [
    "A dog runs in the grass.",
    "Kids play ball together.",
    "A person rides a bike outside.",
    "People sit around a table.",
    "A lady stands under an umbrella while it rains."
]

st.header("Sample Caption Pairs")
for i in range(len(ref_captions)):
    st.write(f"**Reference:** {ref_captions[i]}")
    st.write(f"**Model:** {model_captions[i]}")
    st.caption("---")

@retry(
    # stop=stop_after_attempt(5) + stop_after_delay(30),  # Max 5 attempts or 30 seconds
    wait=wait_exponential(multiplier=1, min=4, max=10), # Exponential backoff between 4 and 10 seconds
    retry=retry_if_exception_type((ConnectionError, Timeout)) # Retry on connection errors or timeouts
)
def load_sentence_transformer_model(model_name: str):
    return SentenceTransformer(model_name)

# --- Compute HLS ---
try:
    model = load_sentence_transformer_model('all-MiniLM-L6-v2')
except (ConnectionError, Timeout) as e:
    st.error(f"Failed to load the SentenceTransformer model due to a network error: {e}")
    st.stop()
ref_emb = model.encode(ref_captions)
model_emb = model.encode(model_captions)
sims = [cosine_similarity([r], [m])[0][0] for r, m in zip(ref_emb, model_emb)]
hls = np.mean(sims)

st.header("Human-Likeness Score (HLS)")
st.write(f"**HLS (average cosine similarity): {hls:.3f}**")
if hls >= 0.80:
    st.success("Result: This is a **high HLS**. Your model's captions are very similar to human captions.")
elif hls >= 0.60:
    st.info("Result: This is a **moderate HLS**. Your model's captions are somewhat similar to human captions.")
else:
    st.warning("Result: This is a **low HLS**. Your model's captions are less similar to human captions.")

st.caption("A higher HLS means your model's captions are more human-like in meaning and style.")
