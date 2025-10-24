import os
from pickle import dump, load



# Load features dictionary from features.p
features = load(open('features.p', 'rb'))

print(type(features))  # should print <class 'dict'>
print(len(features))   # number of images/features loaded
print(features[list(features.keys())[0]].shape)  # shape of feature vector of first image


