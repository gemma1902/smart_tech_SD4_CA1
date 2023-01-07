import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils.np_utils import to_categorical
import random
import requests
from PIL import Image
from numpy import asarray
import cv2
import pandas as pd

def grayscale(image):
    image = cv2.cvtColor(image.astype('float32'), cv2.COLOR_RGB2GRAY)
    return image


# Brightens the Image
def equalize(image):
    # Only works on grayscaled images
    image = cv2.equalizeHist(image.astype(np.uint8))
    return image


def preprocessing(image):
    image = grayscale(image)
    image = equalize(image)
    # flatten Image
    image = image / 255
    return image

def gaussianBlur(image):
    kernel_size = (5, 5)
    kernel = cv2.getGaussianKernel(kernel_size[0], 0)
    kernel = kernel * kernel.T
    image_blurred = cv2.filter2D(image, -1, kernel)
    return image_blurred


def leNet_model():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(64, 64, 1), activation='relu'))  # 30 = number of filters 5x5 = kernal 32x32 === 28x28 output
    model.add(MaxPooling2D(pool_size=(2, 2))) #14x14 = output
    model.add(Conv2D(30, (3, 3), activation='relu')) #12x12 = output
    model.add(MaxPooling2D(pool_size=(2, 2))) #6x6 = output
    # flatten image before fully connected layer
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))  # helps with overfitting & forces all nodes to work
    model.add(Dense(num_classes, activation='softmax')) # softmax coz its a multiclass
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy']) #adam optimizer
    return model

# def new_model():
#     model = Sequential()
#     model.add(Conv2D(60, (5, 5), input_shape=(64, 64, 1), activation='relu'))
#     model.add(Conv2D(60, (5, 5), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2))) #14x14 = output
#     model.add(Conv2D(30, (3, 3), activation='relu')) #12x12 = output
#     model.add(Conv2D(30, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2))) #6x6 = output
#     # model.add(Dropout(0.5))
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

class_labels = {}
with open(WORDS_PATH, 'r') as f:
    for line in f:
        label, name = line.strip().split('\t')
        class_labels[label] = name
#print(class_labels)

# load in training data
X_train_data = []
y_train_data = []
image_index=[]

for file in os.listdir(TRAIN_PATH):
    h = os.path.join(TRAIN_PATH,file)
    for files in os.listdir(h):
        k = os.path.join(h, files)
        if os.path.isfile(k):
            with open (k, 'r') as f:
                filenames = [x.split('\t')[0] for x in f]
                num_images = len(filenames)


for file in os.listdir(TRAIN_PATH):
    h = os.path.join(TRAIN_PATH,file)
    j = os.path.join(h,"images")
    for files in os.listdir(j):
        k = os.path.join(j, files)
        img = Image.open(k).convert('RGB')
        # print(np.array(img).shape)
        X_train_data.append(np.array(img))
        y_train_data.append(file)
X_train = np.array(X_train_data)
y_train = np.array(y_train_data)

class_name = class_labels[label]
#print("LINKING IMAGES", f'{file}: {class_name}')

#load in validation data
X_val_data = []
y_val_data = []

for filename in os.listdir(VAL_PATH):
    f = os.path.join(VAL_PATH, filename)
    img = Image.open(f).convert('RGB')
    # print(np.array(img).shape)
    X_val_data.append(np.array(img))
X_val = np.array(X_val_data)

with open(VAL_ANNOTATIONS_PATH, "r") as file:
    val_y = file.readlines()
    for line in val_y:
        words = line.split()
        val_label = words[1]
        y_val_data.append(val_label)
y_val = np.array(y_val_data)

#load in testing data
X_test_data = []
for file in os.listdir(TEST_PATH):
    h = os.path.join(TEST_PATH, file)
    if os.path.isfile(h):
        img = Image.open(h).convert('RGB')
        X_test_numpy_data = asarray(img)
        X_test_data.append(X_test_numpy_data)
X_test = np.array(X_test_data)

# DATA_PREPROCESSING
# Print number of Images
print("X_TRAIN SHAPE : ", X_train.shape)
print("X_VAL SHAPE : ", X_val.shape)
print("X_TEST SHAPE : ", X_test.shape)

#One hot encode labels 
# y_train = to_categorical(y_train, 10)
# y_test = to_categorical(y_test, 10)

#updated one hot encode

res = [sub[1:] for sub in y_train]
print(res)

new_y_train = [int(x) for x in res]

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(categories='auto')

new_y_train1 = np.array(new_y_train)
one_hot_encoded_y_train = encoder.fit_transform(new_y_train1.reshape(-1,1))

#Normalise data
X_train = X_train / 255
X_test = X_test / 255

print("normalise: ", X_train)

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

#plt.imshow(X_train[random.randint(0, len(X_train)-1)])
#plt.axis("off")
#plt.show()
#print(X_train.shape)

X_train = X_train.reshape(X_train.shape[0], 64, 64, 1)
X_test = X_test.reshape(X_test.shape[0], 64, 64, 1)
X_val = X_val.reshape(X_val.shape[0], 64, 64, 1)

model = leNet_model()
print("MODEL SUMMARY : ", model.summary())


history = model.fit(X_train, one_hot_encoded_y_train.toarray(), validation_split=0.1, epochs=20, batch_size=100, verbose=1, shuffle=1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.title('loss')
plt.xlabel('epochs')
plt.show()
