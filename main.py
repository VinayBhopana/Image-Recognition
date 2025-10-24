from pyexpat import features
import string
import os
from PIL import Image
from time import time, sleep
import numpy as np
import matplotlib.pyplot as plt
from pickle import dump, load
import tensorflow as tf
from keras.applications.xception import Xception , preprocess_input
from keras.preprocessing.image import load_img , img_to_array
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.utils.sequence_utils import pad_sequences
from keras.utils import to_categorical,get_file
from keras.layers import add
from keras.models import Model,load_model
from keras.layers import Dense , Input , LSTM , Embedding , Dropout

from tqdm import tqdm
# tqdm().pandas()


def load_doc(filename):
    file = open(filename , 'r')
    text = file.read()
    return text


def all_img_captions(filename):
    file = load_doc(filename)
    captions  = file.split('\n')
    desc = {}
    for caption in captions[:-1]:
        img , caption = caption.split('\t')
        if img[:-2] not in desc:
            desc[img[:-2]] = [caption]
        else:
            desc[img[:-2]].append(caption)
    return desc



def cleaning_text(captions):
    table = str.maketrans('','',string.punctuation)
    for img , caps in captions.items():
        for i , img_caption in enumerate(caps):
            img_caption.replace("-","")
            desp = img_caption.split()
            desp = [word.lower() for word in desp]
            desp = [word.translate(table) for word in desp]
            desp = [word for word in desp if(len(word))>1]
            desp = [word for word in desp if(word.isalpha())]

            img_caption = ' '.join(desp)
            captions[img][i] = img_caption
    return captions


def text_vocab(desc):
    vocab = set()
    for key in desc.keys():
        [vocab.update(d.split()) for d in desc[key]]

    return vocab

def save_desc(desc,filename):
    lines = list()
    for key ,desc_list in desc.items():
        for d in desc_list:
            lines.append(key + '\t' + d)
    data = "\n".join(lines)
    file = open(filename , "w")
    file.write(data)
    file.close()    


dataset_text = "Flickr8k_text"
dataset_images = r"Flickr8k_Dataset\Flicker8k_Dataset"

filename = dataset_text + "/Flickr8k.token.txt"
desc = all_img_captions(filename)
print("Length of descriptions = " , len(desc))

clean_desc = cleaning_text(desc)
vocabulary = text_vocab(desc)
# print("Length of vocabulary = " , len(vocabulary))

# save_desc(clean_desc , "descriptions.txt")

clean_desc = cleaning_text(desc)
vocabulary = text_vocab(clean_desc)
print("length of vocabulary" , len(vocabulary))

save_desc(clean_desc , 'desc.txt')




def download_with_retry(url , filename , max_retries = 3):
    for attempt in range(max_retries):
        try:
            return get_file(filename , url)
        except Exception as e:
            if attempt == max_retries-1:
                raise e
            print(f"download attempt failed")
            sleep(3)


weights_url = "https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_path = download_with_retry(weights_url , 'xception_weights.h5')
model  = Xception(include_top=False,pooling = "avg" , weights = weights_path)



def extract_features(directory):
    features = {}
    valid_img = ['.jpg' , '.jpeg' , '.png']
    for img in tqdm(os.listdir(directory)):
        ext = os.path.splitext(img)[1].lower()
        if ext not in valid_img:
            continue
        filename = directory +'/' + img 
        image = Image.open(filename)
        image = image.resize((299 , 299))
        image = np.expand_dims(image , axis=0)
        image = image/127.5
        image = image - 1.0


        feature = model.predict(image)
        features[img] = feature
    
    return features


features = extract_features(dataset_images)
dump(features, open("features.p" , 'wb'))

features = load(open('features.p','rb'))

features_file = "features.p"

if os.path.exists(features_file):
    print("Loading precomputed image features...")
    features = load(open(features_file, 'rb'))
else:
    print("Extracting features from images... (this may take some time)")
    # features = extract_features(dataset_images)  # tqdm runs only here
    dump(features, open(features_file, 'wb'))

def load_photos(filename):
    file = load_doc(filename)
    photos = file.split("\n")[:-1]
    photos_present = [photo for photo in photos if os.path.exists(os.path.join(dataset_images, photo))]
    return photos_present



def load_clean_descriptions(filename, photos): 
    
    file = load_doc(filename)
    descriptions = {}
    for line in file.split("\n"):

        words = line.split()
        if len(words)<1 :
            continue

        image, image_caption = words[0], words[1:]

        if image in photos:
            if image not in descriptions:
                descriptions[image] = []
            desc = '<start> ' + " ".join(image_caption) + ' <end>'
            descriptions[image].append(desc)

    return descriptions

def load_features(photos):
    #loading all features
    all_features = load(open("features.p","rb"))
    #selecting only needed features
    features = {k:all_features[k] for k in photos}
    #print(features)
    return features

filename = dataset_text + "/" + "Flickr_8k.trainImages.txt"

#train = loading_data(filename)
train_imgs = load_photos(filename)
train_descriptions = load_clean_descriptions("desc.txt", train_imgs)
train_features = load_features(train_imgs)

#converting dictionary to clean list of descriptions
def dict_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

#creating tokenizer class 
#this will vectorise text corpus
#each integer will represent token in dictionary

def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer

# give each word an index, and store that into tokenizer.p pickle file
tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizer.p', 'wb'))


vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)





