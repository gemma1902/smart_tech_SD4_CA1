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
# import csv
# import torchvision.datasets as datasets
# import torch.utils.data as data

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


DATA = "D:/smart_tech/tiny-imagenet-200"
VAL_ANNOTATIONS_PATH = "D:/smart_tech/tiny-imagenet-200/val/val_annotations.txt"
VAL_PATH = "D:/smart_tech/tiny-imagenet-200/val/images"
TRAIN_PATH = "D:\\smart_tech\\tiny-imagenet-200\\train\\"
TEST_PATH = "D:/smart_tech/tiny-imagenet-200/test/images"
WNIDS_PATH = "D:/smart_tech/tiny-imagenet-200/wnids.txt"
WORDS_PATH = "D:/smart_tech/tiny-imagenet-200/words.txt"
BOXES_PATH = "D:/smart_tech/tiny-imagenet-200/boxes.txt"
path = "D:/smart_tech/tiny-imagenet-200/"
num_classes= 200





# def leNet_model():
#     model = Sequential()
#     model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))  # 30 = number of filters 5x5 = kernal 32x32 === 28x28 output
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

# # LOAD WNIDS INTO ARRAY
# WNIDS = open(os.path.join(DATA, 'wnids.txt'), 'r')
# WNIDS_file = WNIDS.readlines()
# WNIDS_data = []
# for line in WNIDS_file:
#     words = line.split('\t')
#     WNIDS_data.append(words)
# WNIDS.close()
#
#
#
# # img = Image.open(k)
# #             X_train_numpy_data = np.asarray(img)
# #             # print(X_train_numpy_data)
# #             # X_train.append(X_train_numpy_data)
# #             # print(type(X_train))
# #             X_train = np.asarray([X_train_numpy_data])
# #             # print(X_train.shape)
# #             # print(" x train ====== ", type(X_train))
#
# # LOAD VALIDATION DATA
# X_val = []
# for filename in os.listdir(VAL_PATH):
#     f = os.path.join(VAL_PATH, filename)
#     if os.path.isfile(f):
#         img = Image.open(os.path.join(VAL_PATH,filename)).convert('RGB')
#         X_val_numpy_data = np.asarray(img)
#         X_val = np.asarray([X_val_numpy_data])
#
# print(X_val)
# print(X_val.shape)
#
# # LOAD TEST DATA
# X_test = []
# for file in os.listdir(TEST_PATH):
#     h = os.path.join(TEST_PATH, file)
#     if os.path.isfile(h):
#         img = Image.open(h)
#         X_test_numpy_data = asarray(img)
#         X_test.append(X_test_numpy_data)
#
# # print(test_data[0])
# # img = cv2.imread(test_data[4])
# # plt.imshow(img)
# # plt.show()
#
#
# # LOAD TRAINING DATA
# X_train = []
# y_train = []
# for file in os.listdir(TRAIN_PATH):
#     y_train.append(file)
#     # print(y_train)
#     h = os.path.join(TRAIN_PATH,file)
#     j = os.path.join(h,"images")
#     for files in os.listdir(j):
#         k = os.path.join(j, files)
#         if os.path.isfile(k):
#             img = Image.open(k)
#             X_train_numpy_data = np.asarray(img)
#             # print(X_train_numpy_data)
#             # X_train.append(X_train_numpy_data)
#             # print(type(X_train))
#             X_train = np.asarray([X_train_numpy_data])
#             # print(X_train.shape)
#             # print(" x train ====== ", type(X_train))
#
# print(X_train)
#
# # print(X_train[0])
# # img = cv2.imread(X_train[4])
# # plt.imshow(img)
# # plt.show()
#
#
# # LOAD IN DATA FROM WNIDS TEXT FILE
# with open(WNIDS_PATH, "r") as file:
#     wnids = file.read()
# # Directory to datasets
# TRAIN = os.path.join(DATA, 'train')
# VAL = os.path.join(DATA, 'val')
# TEST = os.path.join(DATA, 'test')
#
# # PAIR EACH WNIDS NUMBER TO AN INTERGER I.E. {n08966554 : 1, n0897856 : 2} etc.
# pair = {}
# for i, wnid in enumerate(wnids):  # Loop through and assign label
#     pair[wnid] = i
#
# y_val = []
# with open(VAL_ANNOTATIONS_PATH, "r") as file:
#     val_y = file.readlines()
#     for line in val_y:
#         words = line.split()
#         val_label = words[1]
#         y_val.append(val_label)

# print(label_data)

# Use words.txt to get names for each class  #not working fully
# with open(WORDS_PATH, 'r') as f:
#     wnid_to_words = dict(line.split('\t') for line in f)
#     for wnid, words in wnid_to_words.items():
#       wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
# class_names = [wnid_to_words[wnid] for wnid in wnids]



# plt.figure(figsize=(12, 4))
# plt.bar(range(0, num_classes), num_of_samples)
# plt.title("Distribution of training dataset")
# plt.xlabel("class number")
# plt.ylabel("NUmber of Imgs")
# plt.show()

# assert(X_train.shape[0] == y_train.shape[0]), "the number of training images is not equal to the number of training labels"
# assert(X_val.shape[0] == y_val.shape[0]), "the number of validation images is not equal to the number of validation labels"
# # assert(X_test.shape[0] == y_test.shape[0]), "the number of test images is not equal to the number of test labels"
# assert(X_train.shape[1:] == (64, 64, 3)), "the dimensions of the training images are not 32x32x3"
# assert(X_val.shape[1:] == (64, 64, 3)), "the dimensions of the validation images are not 32x32x3"
# assert(X_test.shape[1:] == (64, 64, 3)), "the dimensions of the test images are not 32x32x3"

# DATA_PREPROCESSING
# print(X_train.shape)
# print(X_val.shape)
# print(X_test.shape)



























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
X_train = []
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
        img = Image.open(k)
        # print(np.array(img).shape)
        X_train.append(np.array(img))
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

print(X_train)
print(y_train)

#load in validation data
X_val = []
y_val = []

for filename in os.listdir(VAL_PATH):
    f = os.path.join(VAL_PATH, filename)
    if os.path.isfile(f):
        img = Image.open(os.path.join(VAL_PATH,filename)).convert('RGB')
        X_val_numpy_data = np.asarray(img)
        X_val = np.asarray([X_val_numpy_data])

with open(VAL_ANNOTATIONS_PATH, "r") as file:
    val_y = file.readlines()
    for line in val_y:
        words = line.split()
        val_label = words[1]
        y_val.append(val_label)

print(X_val)
print(y_val)



#load in testing data
X_test = []
for file in os.listdir(TEST_PATH):
    h = os.path.join(TEST_PATH, file)
    if os.path.isfile(h):
        img = Image.open(h)
        X_test_numpy_data = asarray(img)
        X_test.append(X_test_numpy_data)

print(X_test)


# assert(X_train.shape[0] == y_train.shape[0]), "the number of training images is not equal to the number of training labels"
# assert(X_val.shape[0] == y_val.shape[0]), "the number of validation images is not equal to the number of validation labels"
# assert(X_test.shape[0] == y_test.shape[0]), "the number of test images is not equal to the number of test labels"
# assert(X_train.shape[1:] == (64, 64, 3)), "the dimensions of the training images are not 32x32x3"
# assert(X_val.shape[1:] == (64, 64, 3)), "the dimensions of the validation images are not 32x32x3"
# assert(X_test.shape[1:] == (64, 64, 3)), "the dimensions of the test images are not 32x32x3"

# X_train = np.array(list(map(preprocessing, X_train))) #avoid loops and use maps
# X_val = np.array(list(map(preprocessing, X_val)))
# X_test = np.array(list(map(preprocessing, X_test)))

#one hot encode labels
# y_train = to_categorical(y_train, num_classes)
# y_test = to_categorical(y_test, num_classes)
# y_val = to_categorical(y_val, num_classes)
