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
# from keras.utils.np_utils import to_categorical
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
    #flatten Image
    image = image/255
    return image

DATA = "D:/smart_tech/tiny-imagenet-200"
VAL_ANNOTATIONS_PATH = "D:/smart_tech/tiny-imagenet-200/val/val_annotations.txt"
VAL_PATH = "D:/smart_tech/tiny-imagenet-200/val/images"
TRAIN_PATH = "D:/smart_tech/tiny-imagenet-200/train"
TEST_PATH = "D:/smart_tech/tiny-imagenet-200/test/images"
WNIDS_PATH = "D:/smart_tech/tiny-imagenet-200/wnids.txt"
WORDS_PATH = "D:/smart_tech/tiny-imagenet-200/words.txt"
path = "D:/smart_tech/tiny-imagenet-200/"


# LOAD WNIDS INTO ARRAY
WNIDS = open(os.path.join(DATA, 'wnids.txt'), 'r')
WNIDS_file = WNIDS.readlines()
WNIDS_data = []
for line in WNIDS_file:
    words = line.split('\t')
    WNIDS_data.append(words)
WNIDS.close()

# LOAD VALIDATION DATA
X_val = []
for filename in os.listdir(VAL_PATH):
    f = os.path.join(VAL_PATH,filename)
    if os.path.isfile(f):
        img = Image.open(f)
        X_val_numpy_data = asarray(img)
        X_val.append(X_val_numpy_data)

# LOAD TEST DATA
X_test = []
for file in os.listdir(TEST_PATH):
    h = os.path.join(TEST_PATH,file)
    if os.path.isfile(h):
        img = Image.open(h)
        X_test_numpy_data = asarray(img)
        X_test.append(X_test_numpy_data)

# print(test_data[0])
# img = cv2.imread(test_data[4])
# plt.imshow(img)
# plt.show()



# LOAD TRAINING DATA
X_train = []
y_train = []
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
        if os.path.isfile(k):
            img = Image.open(k)
            X_train_numpy_data = asarray(img)
            X_train.append(X_train_numpy_data)
            # print(numpydata)



# print(X_train[0])
# img = cv2.imread(X_train[4])
# plt.imshow(img)
# plt.show()


#LOAD IN DATA FROM WNIDS TEXT FILE
with open(WNIDS_PATH, "r") as file:
    wnids = file.read()
# Directory to datasets
TRAIN = os.path.join(DATA, 'train')
VAL = os.path.join(DATA, 'val')
TEST = os.path.join(DATA, 'test')

#PAIR EACH WNIDS NUMBER TO AN INTERGER I.E. {n08966554 : 1, n0897856 : 2} etc.
pair = {}
for i, wnid in enumerate(wnids): #Loop through and assign label
    pair[wnid] = i


y_val = []
with open(VAL_ANNOTATIONS_PATH, "r") as file:
    val_y = file.readlines()
    for line in val_y:
        words = line.split()
        val_label = words[1]
        y_val.append(val_label)



#print(label_data)

#Use words.txt to get names for each class  #not working fully
# with open(WORDS_PATH, 'r') as f:
#     wnid_to_words = dict(line.split('\t') for line in f)
#     for wnid, words in wnid_to_words.items():
#       wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
# class_names = [wnid_to_words[wnid] for wnid in wnids]

# d1=zip(X_val, y_val)
# # print (d1)#Output:<zip object at 0x01149528>
# #Converting zip object to dict using dict() contructor.
# print (dict(d1))


# DATA_PREPROCESSING
print(X_train.shape)
# print(X_val.shape)
# print(X_test.shape)
