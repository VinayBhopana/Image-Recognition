import time
import numpy as np
import nltk
from nltk.translate.meteor_score import meteor_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from pickle import load
from PIL import Image
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
from keras.utils import custom_object_scope
import keras

# --- Setup NLTK ---
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class NotEqual(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return keras.backend.cast(keras.backend.not_equal(inputs, 0), dtype='float32')

    def get_config(self):
        return super().get_config()

# --- Helper Functions ---
def extract_features(filename, model):
    image = Image.open(filename)
    image = image.resize((299, 299))
    image = np.array(image)
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text.split()[1:-1]  # remove startseq, endseq

def simple_accuracy(ref, pred):
    ref_words = set(ref[0].split())
    pred_words = set(pred.split())
    return len(ref_words & pred_words) / len(ref_words) if ref_words else 0

# --- Load Data ---
print("✅ Loading Tokenizer and Model...")
tokenizer = load(open("tokenizer.p", "rb"))
max_length = 34
with custom_object_scope({'NotEqual': NotEqual}):
    caption_model = load_model("C:/Users/Vinay/Desktop/ImageRecog/models2/model_9.h5")

xception_model = Xception(include_top=False, pooling="avg")

# Load test features and descriptions
test_features = load(open("features.p", "rb"))
test_descriptions = load(open("desc.txt", "rb"))

print(f"✅ Test Dataset size: {len(test_descriptions)}")

# --- Evaluation ---
meteor_scores = []
accuracies = []
latencies = []

print("\n🔍 Evaluating Model on Test Data...\n")

for img_id, refs in list(test_descriptions.items())[:50]:  # limit to 50 for speed
    photo = test_features[img_id].reshape((1, 2048))

    # Measure latency
    start = time.time()
    pred_caption_tokens = generate_caption(caption_model, tokenizer, photo, max_length)
    end = time.time()

    pred_caption = " ".join(pred_caption_tokens)
    latency = end - start

    # Metrics
    meteor = meteor_score(refs, pred_caption)
    acc = simple_accuracy(refs, pred_caption)

    meteor_scores.append(meteor)
    accuracies.append(acc)
    latencies.append(latency)

# --- Results ---
avg_meteor = np.mean(meteor_scores)
avg_acc = np.mean(accuracies)
avg_latency = np.mean(latencies)

print("✅ Evaluation Complete!\n")
print(f"Average Accuracy: {avg_acc*100:.2f}%")
print(f"Average METEOR Score: {avg_meteor:.4f}")
print(f"Average Latency per Caption: {avg_latency:.4f} seconds")
