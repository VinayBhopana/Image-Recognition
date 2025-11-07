# main.py — corrected version

import os
import string
from time import sleep
from pickle import dump, load
from tqdm.auto import tqdm

import numpy as np
from PIL import Image

import tensorflow as tf
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import img_to_array
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.utils.sequence_utils import pad_sequences
from keras.utils import to_categorical, get_file
from keras.layers import add
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Embedding, Dropout

# ---------- helper I/O ----------
def load_doc(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

def all_img_captions(filename):
    file = load_doc(filename)
    captions = file.split('\n')
    desc = {}
    for caption in captions:
        if not caption:
            continue
        parts = caption.split('\t')
        if len(parts) < 2:
            continue
        img, cap = parts[0], parts[1]
        key = img[:-2]  # as original code expected
        desc.setdefault(key, []).append(cap)
    return desc

def cleaning_text(captions):
    table = str.maketrans('', '', string.punctuation)
    for img, caps in captions.items():
        for i, img_caption in enumerate(caps):
            img_caption = img_caption.replace("-", " ")
            words = img_caption.split()
            words = [w.lower() for w in words]
            words = [w.translate(table) for w in words]
            words = [w for w in words if len(w) > 1 and w.isalpha()]
            captions[img][i] = ' '.join(words)
    return captions

def text_vocab(desc):
    vocab = set()
    for k in desc:
        [vocab.update(d.split()) for d in desc[k]]
    return vocab

def save_desc(desc, filename):
    lines = []
    for key, desc_list in desc.items():
        for d in desc_list:
            lines.append(key + '\t' + d)
    data = "\n".join(lines)
    with open(filename, "w", encoding='utf-8') as f:
        f.write(data)

# ---------- dataset paths ----------
dataset_text = "Flickr8k_text"
dataset_images = r"Flickr8k_Dataset\Flicker8k_Dataset"
token_file = os.path.join(dataset_text, "Flickr8k.token.txt")

# ---------- load & clean captions ----------
desc = all_img_captions(token_file)
print("Number of unique images in captions:", len(desc))
clean_desc = cleaning_text(desc)
vocab = text_vocab(clean_desc)
print("vocab size (unique words):", len(vocab))
save_desc(clean_desc, 'desc.txt')

# ---------- Xception feature model ----------
def download_with_retry(url, filename, max_retries=3):
    for attempt in range(max_retries):
        try:
            return get_file(filename, url)
        except Exception:
            if attempt == max_retries - 1:
                raise
            print("download attempt failed; retrying...")
            sleep(2)

weights_url = "https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_path = download_with_retry(weights_url, 'xception_weights.h5')
cnn_model = Xception(include_top=False, pooling="avg", weights=weights_path)

# ---------- extract or load features ----------
features_file = "features.p"

def extract_features(directory):
    features = {}
    valid_img_ext = {'.jpg', '.jpeg', '.png'}
    for img in tqdm(os.listdir(directory), desc="Extracting features"):
        ext = os.path.splitext(img)[1].lower()
        if ext not in valid_img_ext:
            continue
        path = os.path.join(directory, img)
        try:
            image = Image.open(path).convert('RGB').resize((299, 299))
        except Exception as e:
            print(f"skip {img}: {e}")
            continue
        arr = np.array(image).astype('float32')
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)  # use official preprocess
        feature = cnn_model.predict(arr, verbose=0)
        features[img] = feature.flatten()  # Ensure shape is (2048,)
    return features

if os.path.exists(features_file):
    print("Loading precomputed image features...")
    features = load(open(features_file, 'rb'))
else:
    print("No features.p found — extracting features (this may take time)...")
    features = extract_features(dataset_images)
    dump(features, open(features_file, 'wb'))
    print("Saved features to features.p")

# ---------- prepare training lists ----------
def load_photos(filename):
    file = load_doc(filename)
    photos = file.split("\n")[:-1]
    photos = [photo.strip() for photo in photos if len(photo.strip()) > 0]
    # Add ".jpg" extension if missing
    photos = [photo if photo.lower().endswith('.jpg') else photo + '.jpg' for photo in photos]
    photos_present = [photo for photo in photos if os.path.exists(os.path.join(dataset_images, photo))]
    print(f"Train images with features: {len(photos_present)} out of {len(photos)} requested")
    return photos_present


train_list_file = os.path.join(dataset_text, "Flickr_8k.trainImages.txt")
train_imgs = [os.path.basename(x.strip()) for x in open(train_list_file).readlines()]
train_descriptions = {}

# load cleaned desc (desc.txt) but only for train images
with open("desc.txt", 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        image = parts[0]
        caption = ' '.join(parts[1:])
        if image in train_imgs:
            train_descriptions.setdefault(image, []).append('<start> ' + caption + ' <end>')

# prepare features only for train imgs
train_features = {k: features[k] for k in train_imgs if k in features}
print(f"Train images with features: {len(train_features)} out of {len(train_imgs)} requested")

# ---------- tokenizer ----------
def dict_to_list(descriptions):
    return [cap for caps in descriptions.values() for cap in caps]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(dict_to_list(train_descriptions))
dump(tokenizer, open('tokenizer.p', 'wb'))
vocab_size = len(tokenizer.word_index) + 1
print("vocab_size (with reserved 0):", vocab_size)

max_length = max(len(c.split()) for c in dict_to_list(train_descriptions))
print("max caption length:", max_length)

# ---------- sequences creator ----------
def create_sequences(tokenizer, max_length, desc_list, img_feature):
    X1, X2, y = [], [], []
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(img_feature.reshape(-1))  # (2048,)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

# ---------- data generator that yields dict matching model inputs ----------
def data_generator(descriptions, features_dict, tokenizer, max_length):
    """Yields (inputs_dict, output_vector) repeatedly."""
    while True:
        for key, desc_list in descriptions.items():
            if key not in features_dict:
                continue
            img_feature = features_dict[key]  # shape (2048,)
            X1, X2, y = create_sequences(tokenizer, max_length, desc_list, img_feature)
            for i in range(len(X1)):
                inputs = {'input_1': X1[i].astype('float32'), 'input_2': X2[i].astype('int32')}
                output = y[i].astype('float32')
                yield inputs, output

# ---------- create tf dataset using output_signature ----------
dataset_signature = (
    {
        'input_1': tf.TensorSpec(shape=(2048,), dtype=tf.float32),
        'input_2': tf.TensorSpec(shape=(max_length,), dtype=tf.int32),
    },
    tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32),
)

train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_descriptions, train_features, tokenizer, max_length),
    output_signature=dataset_signature
).batch(32)

# test one batch (sanity)
for batch in train_dataset.take(1):
    inputs, targets = batch
    print("batch input_1 shape:", inputs['input_1'].shape)
    print("batch input_2 shape:", inputs['input_2'].shape)
    print("batch targets shape:", targets.shape)
    break

# ---------- model ----------
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,), name='input_1')
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,), name='input_2')
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    return model

model = define_model(vocab_size, max_length)

# ---------- training loop + ensure models directory exists ----------
models_dir = "models"
if not os.path.exists(models_dir):
    print("Creating models directory in:", os.getcwd())
    os.mkdir(models_dir)
else:
    print("Models directory already exists in:", os.getcwd())

epochs = 10  # reduce for quick test; change back to 10 as you wish
# steps_per_epoch = 100
def get_steps_per_epoch(train_descriptions):
    total_sequences = 0
    for img_captions in train_descriptions.values():
        for caption in img_captions:
            words = caption.split()
            total_sequences += len(words) - 1
    # Ensure at least 1 step, even if sequences < batch_size
    return max(1, total_sequences // 32)

# Update training loop
steps = get_steps_per_epoch(train_descriptions)

for epoch in range(epochs):
    print(f"\nStarting epoch {epoch+1}/{epochs}")
    model.fit(train_dataset, epochs=1, steps_per_epoch=steps, verbose=1)
    model_path = os.path.join(models_dir, f"modelnew_{epoch}.h5")
    model.save(model_path)
    print("Saved", model_path)

print("Training finished.")

















