import streamlit as st
from collections import Counter
import matplotlib.pyplot as plt

# --- Load cleaned captions ---
desc_file = 'desc.txt'
with open(desc_file, 'r') as f:
    lines = f.readlines()

clean_desc = {}
for line in lines:
    parts = line.strip().split('\t')
    if len(parts) == 2:
        key, caption = parts
        if key not in clean_desc:
            clean_desc[key] = []
        clean_desc[key].append(caption)

# --- Calculate CDI (Type-Token Ratio) ---
all_words = [word for captions in clean_desc.values() for caption in captions for word in caption.split()]
unique_words = set(all_words)
ttr = len(unique_words) / len(all_words) if all_words else 0

st.set_page_config(page_title="Caption Diversity Index", layout="wide")
st.title("Caption Diversity Index (CDI)")

st.header("CDI Metric")
st.write(f"**Type-Token Ratio (CDI):** {ttr:.3f}")
st.write(f"**Unique words:** {len(unique_words)}")
st.write(f"**Total words:** {len(all_words)}")

# --- Result interpretation ---
if ttr >= 0.10:
    st.success("Result: This is a **high CDI**. Your dataset has diverse and creative captions.")
else:
    st.warning("Result: This is a **low CDI**. Your dataset captions are less diverse and may be repetitive.")

st.caption("A higher CDI means more diverse and creative captions in your dataset.")

# --- Optional: Bar chart of top 15 most frequent words ---
st.header("Top 15 Most Frequent Words")
word_counts = Counter(all_words)
most_common = word_counts.most_common(15)
words, counts = zip(*most_common)
fig, ax = plt.subplots(figsize=(8, 3))
ax.bar(words, counts, color='skyblue')
ax.set_title('Top 15 Most Frequent Words')
ax.set_xlabel('Word')
ax.set_ylabel('Count')
plt.xticks(rotation=45)
st.pyplot(fig)
