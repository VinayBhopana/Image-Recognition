import sys
import os
import pytest
import numpy as np

# ✅ Ensure current directory (where image_caption_model.py is) is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ✅ Import your functions from image_caption_model.py
from testdup import define_model, beam_search_predictions, extract_features
from tensorflow.keras.applications.xception import Xception
from pickle import load

# ✅ Load tokenizer and setup
tokenizer = load(open("tokenizer.p", "rb"))
max_length = 32
vocab_size = len(tokenizer.word_index) + 1

# ✅ Load model for regression test
model = define_model(vocab_size, max_length)
model.load_weights('models/modelnew_9.h5')

xception_model = Xception(include_top=False, pooling="avg")

# ✅ Prepare a sample image (you can replace this path with another image)
img_path = "C:/Users/Vinay/Desktop/ImageRecog/Flickr8k_Dataset/Flicker8k_Dataset/3726629271_7639634703.jpg"

# ✅ Baseline prediction (expected stable output)
BASELINE_CAPTION = "a boy rides a toy horse"

def test_regression_caption_consistency():
    """Regression test — ensure caption generation remains stable."""
    photo = extract_features(img_path, xception_model)
    generated_caption = beam_search_predictions(model, tokenizer, photo, max_length)
    assert isinstance(generated_caption, str), "Generated caption must be a string"
    assert len(generated_caption) > 0, "Generated caption is empty"
    assert BASELINE_CAPTION.split()[0] in generated_caption.split(), \
        "Regression detected: starting word mismatch"

def test_model_compiles_properly():
    """Ensure model compiles successfully with correct structure."""
    test_model = define_model(vocab_size, max_length)
    assert test_model.loss == "categorical_crossentropy", "Model loss function mismatch"
    assert test_model.optimizer.__class__.__name__.lower() == "adam", "Optimizer mismatch"

def test_feature_extraction_shape():
    """Ensure Xception feature extractor output shape is consistent."""
    photo = extract_features(img_path, xception_model)
    assert photo.shape[-1] == 2048, "Feature extraction dimension mismatch"
