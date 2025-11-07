
import streamlit as st
import os
import json
import datetime
import logging
import sqlite3
import random
from PIL import Image
import numpy as np

# Re-introducing Keras/TensorFlow imports for the actual model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model

# Re-introducing NLTK imports for metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import io # Import the io module

# --- 1. Configuration Management ---
config = {
    "log_file": "app.log",
    "db_file": "predictions.db",
    "temp_image_dir": "./temp_images", # Directory to store uploaded images temporarily
    "tokenizer_path": "tokenizer.p",
    "model_weights_path": 'models/modelnew_9.h5',
    "max_caption_length": 32, # Max length for captions
    "beam_width": 10, # Beam search width
    "flickr8k_text_path": "Flickr8k_text/Flickr8k.token.txt", # Path to Flickr8k captions file
    "references": ["a cat sits on a mat", "a cat is on a mat", "a cat is on a mat", "a cat is on a mat"], # Example ground truth references for metrics
    "img_path": "C:/Users/Vinay/Desktop/ImageRecog/Flickr8k_Dataset/Flicker8k_Dataset/3726629271_7639634703.jpg", # Default image path
    "known_good_example_path": "C:/Users/Vinay/Desktop/ImageRecog/Flickr8k_Dataset/Flickr8k_Dataset/1000268201_693b08cb0e.jpg",
    "known_good_example_caption": "A child in a pink dress is climbing up a set of stairs in an entry way ."
}

# Ensure temp image directory exists
os.makedirs(config["temp_image_dir"], exist_ok=True)

# --- 2. Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config["log_file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 3. Database Integration (SQLite) ---
def init_db():
    """Initializes the SQLite database and creates the predictions table."""
    try:
        conn = sqlite3.connect(config["db_file"])
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                caption TEXT NOT NULL,
                bleu1 REAL,
                meteor REAL,
                bleu2 REAL,
                bleu3 REAL,
                bleu4 REAL,
                timestamp TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Database initialization failed: {e}")

def log_caption_to_db(filename, caption, metrics):
    """Logs a caption and its metrics to the SQLite database."""
    try:
        conn = sqlite3.connect(config["db_file"])
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO predictions (filename, caption, bleu1, meteor, bleu2, bleu3, bleu4, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (filename, caption, metrics['bleu1'], metrics['meteor'], metrics['bleu2'], metrics['bleu3'], metrics['bleu4'], timestamp)
        )
        conn.commit()
        conn.close()
        logger.info(f"Caption logged to DB for {filename}: {caption} (BLEU-4: {metrics['bleu4']:.2f})")
    except sqlite3.Error as e:
        logger.error(f"Failed to log caption to DB: {e}")

def get_predictions_from_db():
    """Retrieves all predictions (captions) from the SQLite database."""
    try:
        conn = sqlite3.connect(config["db_file"])
        cursor = conn.cursor()
        cursor.execute("SELECT filename, caption, bleu1, meteor, bleu2, bleu3, bleu4, timestamp FROM predictions ORDER BY timestamp DESC")
        predictions = cursor.fetchall()
        conn.close()
        return predictions
    except sqlite3.Error as e:
        logger.error(f"Failed to retrieve predictions from DB: {e}")
        return []

# Initialize the database on app start
init_db()

# --- Model Loading and Core Captioning Functions (Modular Design) ---
@st.cache_resource
def load_image_tokenizer(path):
    """Loads the pre-trained tokenizer for caption generation."""
    try:
        return load(open(path, "rb"))
    except FileNotFoundError:
        logger.error(f"Tokenizer file not found at {path}.")
        st.error(f"Error: Tokenizer file not found at {path}. Please check `config.py`.")
        return None
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        st.error(f"Error loading tokenizer: {e}")
        return None

@st.cache_resource
def define_and_load_captioning_model(vocab_size, max_length, weights_path):
    """Defines the captioning model architecture and loads its pre-trained weights."""
    try:
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
        model.load_weights(weights_path)
        return model
    except FileNotFoundError:
        logger.error(f"Model weights file not found at {weights_path}.")
        st.error(f"Error: Model weights file not found at {weights_path}. Please check `config.py`.")
        return None
    except Exception as e:
        logger.error(f"Error defining or loading captioning model: {e}")
        st.error(f"Error defining or loading captioning model: {e}")
        return None

@st.cache_resource
def load_xception_feature_extractor():
    """Loads the Xception model for image feature extraction."""
    try:
        return Xception(include_top=False, pooling="avg")
    except Exception as e:
        logger.error(f"Error loading Xception model: {e}")
        st.error(f"Error loading Xception model: {e}")
        return None

# Load models and tokenizer
tokenizer = load_image_tokenizer(config["tokenizer_path"])
vocab_size = len(tokenizer.word_index) + 1 if tokenizer else 0
captioning_model = define_and_load_captioning_model(vocab_size, config["max_caption_length"], config["model_weights_path"])
xception_feature_extractor = load_xception_feature_extractor()

@st.cache_data
def extract_image_features(image_buffer, model):
    """Extracts features from an image buffer using the Xception model."""
    if image_buffer is None:
        logger.warning("No image buffer provided for feature extraction.")
        return None
    if model is None:
        logger.error("Xception feature extractor model is not loaded.")
        return None
    try:
        # Use io.BytesIO to make the buffer behave like a file
        image = Image.open(io.BytesIO(image_buffer))
        image = image.resize((299,299))
        image = np.array(image)
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image, verbose=0)
        return feature
    except Exception as e:
        logger.error(f"Error extracting features from image: {e}")
        return None

def word_for_id(integer, tokenizer_obj):
    """Returns the word for a given integer ID from the tokenizer."""
    if tokenizer_obj is None:
        return None
    for word, index in tokenizer_obj.word_index.items():
        if index == integer:
            return word
    return None

def beam_search_predictions(model, tokenizer_obj, photo_features, max_length, beam_width=10):
    """Generates a caption for image features using Beam Search decoding."""
    if model is None or tokenizer_obj is None or photo_features is None:
        logger.error("Missing model, tokenizer, or photo features for beam search.")
        return "Error: Model or features not loaded."
    
    start_token = tokenizer_obj.word_index.get('start')
    if start_token is None:
        logger.error("'start' token not found in tokenizer vocabulary.")
        return "Error: Tokenizer missing 'start' token."

    sequences = [[[start_token], 0.0]]

    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            padded_sequence = pad_sequences([seq], maxlen=max_length, padding='post')
            preds = model.predict([photo_features, padded_sequence], verbose=0)
            top_preds_indices = np.argsort(preds[0])[-beam_width:]

            for word_idx in top_preds_indices:
                word = word_for_id(word_idx, tokenizer_obj)
                if word is None:
                    continue
                new_seq = seq + [word_idx]
                new_score = score - np.log(preds[0][word_idx] + 1e-10) # Use log probability for score
                all_candidates.append([new_seq, new_score])

        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=False) # Lower score is better
        sequences = ordered[:beam_width]

    best_seq = sequences[0][0]
    final_caption_words = [word_for_id(i, tokenizer_obj) for i in best_seq]
    final_caption = ' '.join([w for w in final_caption_words if w not in ['start', 'end', None]])
    logger.info(f"Generated caption: {final_caption}")
    return final_caption

@st.cache_data
def calculate_caption_metrics(predicted_caption, ground_truth_references):
    """Calculates BLEU and METEOR scores for a predicted caption against references."""
    predicted_tokens = predicted_caption.split()
    # Ensure ground_truth_references are tokenized lists of lists
    reference_tokens = [ref.split() for ref in ground_truth_references]
    
    smooth = SmoothingFunction().method4
    metrics = {}
    metrics['bleu1'] = sentence_bleu(reference_tokens, predicted_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
    metrics['bleu2'] = sentence_bleu(reference_tokens, predicted_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    metrics['bleu3'] = sentence_bleu(reference_tokens, predicted_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    metrics['bleu4'] = sentence_bleu(reference_tokens, predicted_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    metrics['meteor'] = meteor_score(reference_tokens, predicted_tokens)
    
    logger.info(f"Calculated metrics: BLEU-4={metrics['bleu4']:.2f}, METEOR={metrics['meteor']:.2f}")
    return metrics

@st.cache_resource
def load_flickr_ground_truth_captions(filepath):
    """Loads and parses the Flickr8k ground truth captions from a file."""
    captions_map = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                # Each line is like: image.jpg#0	A caption describing the image.
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    image_id_part, caption = parts[0], parts[1]
                    # Extract filename without the #index (e.g., 'image.jpg#0' -> 'image.jpg')
                    filename = image_id_part.split('#')[0]
                    if filename not in captions_map:
                        captions_map[filename] = []
                    captions_map[filename].append(caption.strip())
        logger.info(f"Loaded {len(captions_map)} image entries from {filepath}.")
        return captions_map
    except FileNotFoundError:
        logger.error(f"Flickr8k captions file not found at {filepath}.")
        st.error(f"Error: Flickr8k captions file not found at {filepath}. Please check `config.py`.")
        return {}
    except Exception as e:
        logger.error(f"Error loading Flickr8k captions: {e}")
        st.error(f"Error loading Flickr8k captions: {e}")
        return {}

# Load Flickr8k captions for display and metrics
flickr_ground_truth_captions = load_flickr_ground_truth_captions(config["flickr8k_text_path"])


# --- 5. Unit Testing Function (Simulated) ---s
def run_dummy_unit_test_word_for_id():
    """Simulates a unit test for the word_for_id function."""
    st.subheader("🔬 Unit Test: `word_for_id` function")
    test_results = []
    
    # Create a dummy tokenizer for testing
    dummy_tokenizer = Tokenizer()
    dummy_tokenizer.word_index = {'start': 1, 'a': 2, 'cat': 3, 'sits': 4, 'on': 5, 'mat': 6, 'end': 7}

    # Test Case 1: Valid integer, expect a word
    word = word_for_id(3, dummy_tokenizer)
    if word == 'cat':
        test_results.append((True, f"Test 1 (Valid ID 3): Expected 'cat', got '{word}'."))
    else:
        test_results.append((False, f"Test 1 (Valid ID 3): Expected 'cat', got '{word}'."))

    # Test Case 2: Invalid integer, expect None
    word = word_for_id(99, dummy_tokenizer)
    if word is None:
        test_results.append((True, "Test 2 (Invalid ID 99): Handled gracefully (returns None)."))
    else:
        test_results.append((False, f"Test 2 (Invalid ID 99): Expected None, got '{word}'."))

    # Test Case 3: Empty tokenizer
    empty_tokenizer = Tokenizer()
    word = word_for_id(1, empty_tokenizer)
    if word is None:
        test_results.append((True, "Test 3 (Empty tokenizer): Handled gracefully (returns None)."))
    else:
        test_results.append((False, f"Test 3 (Empty tokenizer): Expected None, got '{word}'."))
        
    st.markdown("**Results:**")
    for success, msg in test_results:
        if success:
            st.success(f"✅ {msg}")
        else:
            st.error(f"❌ {msg}")
    
    logger.info("Dummy unit test for `word_for_id` completed.")

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Software Engineering Demo App 🚀")
st.title("Image Recognition System")
# st.markdown("This application demonstrates core software engineering concepts: Modular Design, Logging, Error Handling, Configuration Management, and Database Integration.")

# --- Session State Initialization ---
if 'initialized' not in st.session_state:
    st.session_state.run_prediction = False
    st.session_state.predicted_caption = ""
    st.session_state.caption_metrics = {}
    st.session_state.run_unit_test = False
    st.session_state.current_upload = None # Correctly initialize uploaded file object
    st.session_state.initialized = True # Mark session as initialized

# --- Sidebar for Controls and Information ---
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("Drag & Drop Image or Click to Upload", type=["jpg", "jpeg", "png"], key="sidebar_uploader")

if st.sidebar.button("Generate Caption & Log", key="generate_button"):
    st.session_state.run_prediction = True
    st.session_state.current_upload = uploaded_file

if st.sidebar.button("Run Unit Test", key="run_test_button"):
    st.session_state.run_unit_test = True

if st.sidebar.button("Clear History & Reset App", help="Clears all predictions from DB and resets app state.", key="reset_app_button"):
    try:
        conn = sqlite3.connect(config["db_file"])
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        conn.commit()
        conn.close()
        logger.info("All predictions cleared from database.")
    except sqlite3.Error as e:
        logger.error(f"Failed to clear predictions from DB: {e}")
    
    # Reset only specific session state variables, not all, to avoid issues with Streamlit's internal state
    st.session_state.run_prediction = False
    st.session_state.predicted_caption = ""
    st.session_state.caption_metrics = {}
    st.session_state.run_unit_test = False
    st.session_state.current_upload = None
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("ℹ️ Info")
st.sidebar.info("Upload an image, then click 'Generate Caption & Log' to generate a caption and log it to the database.")

# --- Main Content Tabs ---
tabs = st.tabs(["🖼️ Image Captioner", "📊 Prediction History", "🔬 Unit Tests", "📝 App Logs", "ℹ️ App Info & Example"])

# --- Tab 1: Image Captioner ---
with tabs[0]:
    st.header("Generate Image Caption")
    st.markdown("Upload an image and click 'Generate Caption & Log' in the sidebar to get a descriptive caption.")

    col1, col2 = st.columns([1, 1])

    image_buffer = None
    current_image_filename = "N/A"

    with col1:
        st.subheader("Input Image")
        if st.session_state.current_upload:
            image_buffer = st.session_state.current_upload.getvalue()
            current_image_filename = st.session_state.current_upload.name
            st.image(image_buffer, caption=f"Uploaded Image: {current_image_filename}", use_column_width=True)
        else:
            st.info("No image uploaded. Please drag & drop or upload an image from the sidebar.")
            
    with col2:
        st.subheader("Caption Result")
        if st.session_state.get('run_prediction', False):
            with st.spinner("Generating caption..."):
                if image_buffer is not None and xception_feature_extractor is not None and captioning_model is not None and tokenizer is not None:
                    # Extract features
                    st.write("Extracting image features...")
                    photo_features = extract_image_features(image_buffer, xception_feature_extractor)
                    
                    if photo_features is not None:
                        # Generate caption
                        st.write("Generating text caption using beam search...")
                        predicted_caption = beam_search_predictions(captioning_model, tokenizer, photo_features, config["max_caption_length"], config["beam_width"])
                        st.session_state.predicted_caption = predicted_caption

                        # Get ground truth references for the current image (if available)
                        image_references = config["references"] # Default references for metrics
                        if current_image_filename in flickr_ground_truth_captions:
                            image_references = flickr_ground_truth_captions[current_image_filename]
                            st.write(f"Found {len(image_references)} ground truth captions for this image.")
                        # else: # No need for a warning here, as it's already handled by the default references
                            # st.warning(f"No ground truth captions found for {current_image_filename} in Flickr8k data. Using default references for metrics.")

                        # Calculate metrics
                        st.write("Calculating evaluation metrics...")
                        current_metrics = calculate_caption_metrics(predicted_caption, image_references)
                        st.session_state.caption_metrics = current_metrics
                        
                        # Log to DB and file
                        log_caption_to_db(current_image_filename, predicted_caption, current_metrics)
                        st.success(f"Caption generated: **{predicted_caption}**")
                    else:
                        st.error("Failed to extract image features.")
                        logger.error("Image feature extraction failed.")
                else:
                    st.error("Model, tokenizer, or image data not loaded/available. Check logs for details.")
                    logger.error("Caption generation failed due to missing model, tokenizer, or image data.")
            st.session_state.run_prediction = False # Reset trigger
        elif st.session_state.get('predicted_caption'):
            st.info(f"Previous Caption: **{st.session_state.predicted_caption}**")
            st.subheader("Evaluation Metrics (Previous Caption)")
            if st.session_state.caption_metrics:
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric(label="BLEU-1", value=f"{st.session_state.caption_metrics.get('bleu1', 0.0) * 100:.2f}%")
                    st.metric(label="METEOR", value=f"{st.session_state.caption_metrics.get('meteor', 0.0) * 100:.2f}%")
                with metrics_col2:
                    st.metric(label="BLEU-2", value=f"{st.session_state.caption_metrics.get('bleu2', 0.0) * 100:.2f}%")
                with metrics_col3:
                    st.metric(label="BLEU-3", value=f"{st.session_state.caption_metrics.get('bleu3', 0.0) * 100:.2f}%")
                    st.metric(label="BLEU-4", value=f"{st.session_state.caption_metrics.get('bleu4', 0.0) * 100:.2f}%")

        else:
            st.write("No caption generated yet. Upload an image and click 'Generate Caption & Log'.")

    # New section for displaying actual captions
    st.markdown("---")
    st.subheader("📚 Actual Captions (Flickr8k Ground Truth)")
    if current_image_filename != "N/A" and current_image_filename in flickr_ground_truth_captions:
        st.markdown(f"**Captions for {current_image_filename}:**")
        for i, caption in enumerate(flickr_ground_truth_captions[current_image_filename]):
            st.write(f"- {caption}")
    elif current_image_filename != "N/A":
        st.info(f"No ground truth captions found for {current_image_filename} in the loaded Flickr8k dataset.")
    else:
        st.info("Upload an image to see its actual captions from the Flickr8k dataset.")

# --- Tab 2: Prediction History ---
with tabs[1]:
    st.header("📊 Prediction History")
    st.markdown("Review all image captions logged to the SQLite database.")
    
    predictions = get_predictions_from_db()
    if predictions:
        # Convert to DataFrame for better display
        import pandas as pd
        # Ensure column names match the DB schema for captions and metrics
        df = pd.DataFrame(predictions, columns=["Filename", "Caption", "BLEU-1", "METEOR", "BLEU-2", "BLEU-3", "BLEU-4", "Timestamp"])
        # Format metrics as percentages
        for col in ["BLEU-1", "METEOR", "BLEU-2", "BLEU-3", "BLEU-4"]:
            df[col] = df[col].apply(lambda x: f"{x * 100:.2f}%")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No captions in history yet. Generate some captions in the 'Image Captioner' tab!")

# --- Tab 3: Unit Tests ---
with tabs[2]:
    st.header("🔬 Unit Tests")
    st.markdown("Click the button below to run a dummy unit test for `word_for_id` function, demonstrating basic testing principles.")
    
    if st.session_state.get('run_unit_test', False):
        run_dummy_unit_test_word_for_id()
        st.session_state.run_unit_test = False # Reset trigger
    
# --- Tab 4: Application Logs ---
with tabs[3]:
    st.header("📝 Application Logs")
    st.markdown("View the application logs for insights and debugging.")
    
    if os.path.exists(config["log_file"]):
        with open(config["log_file"], "r") as f:
            log_content = f.read()
        st.code(log_content, language="log")
    else:
        st.info("Log file not found.")

# --- Tab 5: Application Info / Known Good Example ---
with tabs[4]:
    st.header("ℹ️ Application Information & Validation")
    st.markdown("""
    This section provides details about the application, the underlying models, and demonstrates core software engineering concepts utilized.
    """)
    st.subheader("About This Application")
    st.markdown("""
    This application demonstrates an image captioning system using a pre-trained Xception model for feature extraction 
    and an LSTM-based deep learning model for caption generation. The system predicts a descriptive caption for a given image 
    and evaluates its performance using various BLEU scores and the METEOR score.

    **Key Features:**
    - **Image Upload (Drag & Drop):** Easily upload images from your local machine.
    - **Deep Learning Models:** Leverages state-of-the-art models for accurate image understanding and text generation.
    - **Performance Metrics:** Provides industry-standard metrics to assess the quality of generated captions.
    - **Actual Captions Display:** View ground truth captions from the Flickr8k dataset for comparison.
    - **Streamlit UI:** An interactive and user-friendly interface built with Streamlit for ease of use.

    **Software Engineering Concepts Utilized:**
    - **Modularity:** Code is organized into functions for clarity and reusability.
    - **Efficiency (Caching):** Models and feature extraction results are cached to optimize performance.
    - **Robust Error Handling:** Comprehensive error handling for image loading and processing.
    - **User Experience:** Clear UI layout, informative sections, dynamic feedback, and state management for a better user interaction.
    - **Configuration Management:** Externalized configuration (e.g., model paths, ground truth data) into a dedicated `config.py` file.
    - **Observability:** Enhanced status messages provide better feedback during processing.
    - **Unit Testing:** A simulated unit test function demonstrates basic testing principles.
    """)

    st.markdown("---")
    st.subheader("✅ Known Good Example")
    st.markdown("This section provides a hardcoded example to conceptually demonstrate system validation against a known expected output.")

    known_col1, known_col2 = st.columns([1, 1])

    with known_col1:
        st.caption("Example Input Image")
        try:
            example_image = Image.open(config["known_good_example_path"])
            st.image(example_image, caption=f'Example Image from: {config["known_good_example_path"]}', use_column_width=True)
            st.write(f"Dimensions: {example_image.width}x{example_image.height} pixels")
        except Exception as e:
            st.error(f"Error loading example image from {config["known_good_example_path"]}: {e}")

    with known_col2:
        st.caption("Expected vs. Predicted Caption")
        st.markdown(f"**Expected Caption:** {config["known_good_example_caption"]}")
        
        # Generate prediction for the known good example
        if st.button("Generate Example Prediction", key="example_prediction_button"):
            with st.spinner("Generating prediction for known example..."):
                # Corrected: Read image to bytes before passing to extract_image_features
                with open(config["known_good_example_path"], "rb") as f:
                    example_image_bytes = f.read()
                example_photo = extract_image_features(example_image_bytes, xception_feature_extractor)
                if example_photo is not None:
                    example_predicted_caption = beam_search_predictions(captioning_model, tokenizer, example_photo, config["max_caption_length"], config["beam_width"])
                    st.session_state.example_predicted_caption = example_predicted_caption
                else:
                    st.session_state.example_predicted_caption = "Error: Could not process example image."

        if 'example_predicted_caption' in st.session_state and st.session_state.example_predicted_caption:
            st.success(f"**Predicted Caption:** {st.session_state.example_predicted_caption}")
        else:
            st.info("Click 'Generate Example Prediction' to see the prediction for the known good example.")
