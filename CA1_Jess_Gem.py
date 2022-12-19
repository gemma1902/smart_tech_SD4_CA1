import glob
import os
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow.keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import Flatten
# from keras.layers.convolutional import Conv2D
# from keras.layers.convolutional import MaxPooling2D
from keras.utils.np_utils import to_categorical
# import random
# import requests
from PIL import Image
from numpy import asarray
import cv2
# import pickle
# import pandas as pd

def grayscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


# Brightens the Image
def equalize(image):
    # Only works on grayscaled images
    image = cv2.equalizeHist(image)
    return image


def preprocessing(image):
    image = grayscale(image)
    image = equalize(image)
    # flatten Image
    image = image / 255
    return image


# def leNet_model():
#     model = Sequential()
#     model.add(Conv2D(60, (5, 5), input_shape=(64, 64, 1), activation='relu'))  # 30 = number of filters 5x5 = kernal 32x32 === 28x28 output
#     model.add(MaxPooling2D(pool_size=(2, 2))) #14x14 = output
#     model.add(Conv2D(30, (3, 3), activation='relu')) #12x12 = output
#     model.add(MaxPooling2D(pool_size=(2, 2))) #6x6 = output
#     # flatten image before fully connected layer
#     model.add(Flatten())
#     model.add(Dense(500, activation='relu'))
#     model.add(Dropout(0.5))  # helps with overfitting & forces all nodes to work
#     model.add(Dense(num_classes, activation='softmax')) # softmax coz its a multiclass
#     model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy']) #adam optimizer
#     return model


DATA = "F:\College\Year 4\Smart Tech\JessicaSavage_GemmaRegan_SmartTech_CA1/tiny-imagenet-200"
VAL_ANNOTATIONS_PATH = "F:\College\Year 4\Smart Tech\JessicaSavage_GemmaRegan_SmartTech_CA1/tiny-imagenet-200/val/val_annotations.txt"
VAL_PATH = "F:\\College\\Year 4\\Smart Tech\\JessicaSavage_GemmaRegan_SmartTech_CA1\\tiny-imagenet-200\\val\\images"
TRAIN_PATH = "F:\\College\\Year 4\\Smart Tech\\JessicaSavage_GemmaRegan_SmartTech_CA1\\tiny-imagenet-200\\train"
TEST_PATH = "F:\\College\\Year 4\\Smart Tech\\JessicaSavage_GemmaRegan_SmartTech_CA1\\tiny-imagenet-200\\test\\images"
WNIDS_PATH = "F:\\College\\Year 4\\Smart Tech\\JessicaSavage_GemmaRegan_SmartTech_CA1\\tiny-imagenet-200\\wnids.txt"
WORDS_PATH = "F:\College\Year 4\Smart Tech\JessicaSavage_GemmaRegan_SmartTech_CA1/tiny-imagenet-200/words.txt"
num_classes=200



# ************************************
# Load wnids
with open(WNIDS_PATH, 'r') as f:
    wnids = [x.strip() for x in f]

# pair each wnid to an int
pair = {}
for i, wnid in enumerate(wnids):  # Loop through and assign label
    pair[wnid] = i

# print(type(WORDS_PATH))
# get name for each class
with open(WORDS_PATH, 'r') as f:
    wnid_to_words = dict(line.split('\t') for line in f)
    for wnid, words in wnid_to_words.items():
      wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
class_names = [wnid_to_words[wnid] for wnid in wnids]



# load in training data
X_train_data = []
y_train = []
image_index=[]

for file in os.listdir(TRAIN_PATH):
    h = os.path.join(TRAIN_PATH,file)
    for files in os.listdir(h):
        k = os.path.join(h, files)
        if os.path.isfile(k):
            with open (k, 'r') as f:
                filenames = [x.split('\t')[0] for x in f]
                num_images = len(filenames)


images = np.ndarray(shape=(64, (64*64*3)))
for file in os.listdir(TRAIN_PATH):
    y_train.append(file)
    # print(y_train)
    h = os.path.join(TRAIN_PATH,file)
    j = os.path.join(h,"images")
    for files in os.listdir(j):
        k = os.path.join(j, files)
        img = Image.open(k).convert('RGB')
        # print(np.array(img).shape)
        X_train_data.append(np.array(img))
        y_train.append(file)
        # if os.path.isfile(k):
        #     img = Image.open(k)
        #     X_train_numpy_data = np.asarray(img)
        #     # print(X_train_numpy_data)
        #     # X_train.append(X_train_numpy_data)
        #     # print(type(X_train))
        #     X_train = np.asarray([X_train_numpy_data])
            # print(X_train.shape)
            # print(" x train ====== ", type(X_train))
X_train = np.array(X_train_data)
#print("X_TRAIN SHAPE : ", X_train.shape)
#print("Y_TRAIN : ", y_train)

#load in validation data
X_val_data = []
y_val = []

for filename in os.listdir(VAL_PATH):
    f = os.path.join(VAL_PATH, filename)
    if os.path.isfile(f):
        img = Image.open(os.path.join(VAL_PATH,filename)).convert('RGB')
        X_val_numpy_data = np.asarray(img)
        X_val_data = np.asarray([X_val_numpy_data])
X_val = np.array(X_val_data)

with open(VAL_ANNOTATIONS_PATH, "r") as file:
    val_y = file.readlines()
    for line in val_y:
        words = line.split()
        val_label = words[1]
        y_val.append(val_label)

#print("X_VAL : ", X_val)
#print("Y_VAL : ", y_val)



#load in testing data
X_test_data = []
for file in os.listdir(TEST_PATH):
    h = os.path.join(TEST_PATH, file)
    if os.path.isfile(h):
        img = Image.open(h).convert('RGB')
        X_test_numpy_data = asarray(img)
        X_test_data.append(X_test_numpy_data)
X_test = np.array(X_test_data)
#print("X_TEST : ", X_test)

# DATA_PREPROCESSING
# Print number of Images
print("X_TRAIN SHAPE : ", X_train.shape)
print("X_VAL SHAPE : ", X_val.shape)
print("X_TEST SHAPE : ", X_test.shape)

# Is Number of labels == number of images
assert(X_train.shape[0] == y_train.shape[0]), "the number of training images is not equal to the number of training labels"
assert(X_val.shape[0] == y_val.shape[0]), "the number of validation images is not equal to the number of validation labels"
#assert(X_test.shape[0] == y_test.shape[0]), "the number of test images is not equal to the number of test labels"

# Check all images are 64 x 64
assert(X_train.shape[1:] == (64, 64, 3)), "the dimensions of the training images are not 64x64x3"
assert(X_val.shape[1:] == (64, 64, 3)), "the dimensions of the validation images are not 64x64x3"
assert(X_test.shape[1:] == (64, 64, 3)), "the dimensions of the test images are not 64x64x3"


# Apply preprocessing method to every image
# Map means you don't have to use a forLoop
X_train = np.array(list(map(preprocessing, X_train))) #avoid loops and use maps
X_val = np.array(list(map(preprocessing, X_val)))
X_test = np.array(list(map(preprocessing, X_test)))

#one hot encode labels
# y_train = to_categorical(y_train, num_classes)
# y_test = to_categorical(y_test, num_classes)
# y_val = to_categorical(y_val, num_classes)
