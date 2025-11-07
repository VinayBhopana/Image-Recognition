from keras.applications.xception import Xception
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
import numpy as np
import os
from tqdm import tqdm
from pickle import dump

# Paths
dataset_images = r"C:\Users\Vinay\Desktop\ImageRecog\Flickr8k_Dataset\Flicker8k_Dataset"
features_file = r"C:\Users\Vinay\Desktop\ImageRecog\features.p"

# Load Xception model (no top, pooling avg)
model = Xception(include_top=False, pooling='avg', weights='imagenet')

features = {}
valid_img = ['.jpg', '.jpeg', '.png']

for img_name in tqdm(os.listdir(dataset_images)):
    if os.path.splitext(img_name)[1].lower() not in valid_img:
        continue

    path = os.path.join(dataset_images, img_name)
    img = load_img(path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 127.5
    img = img - 1.0

    feature = model.predict(img)
    features[img_name] = feature[0]

# Save all features
dump(features, open(features_file, 'wb'))
print("Features saved to", features_file)
