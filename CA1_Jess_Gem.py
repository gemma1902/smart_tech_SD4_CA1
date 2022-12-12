import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras.datasets import mnist
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
import cv2
import pickle
import pandas as pd
import csv


# Main Data Directory
DATA = "F:/College/Year 4/Smart Tech/JessicaSavage_GemmaRegan_CA1/tiny-imagenet-200"

LABELS = open(os.path.join(DATA, 'words.txt'), 'r')
labels_file = LABELS.readlines()
label_data = {}
for line in labels_file:
    words = line.split('\t')
    label_data[words[0]] = words[1]
LABELS.close()

# Directory to datasets
TRAIN = os.path.join(DATA, 'train')
VAL = os.path.join(DATA, 'val')
TEST = os.path.join(DATA, 'test')

val_annotations = open(os.path.join(VAL, 'val_annotations.txt'), 'r')
val_file = val_annotations.readlines()

val_data = {}
for line in val_file:
    words = line.split('\t')
    val_data[words[0]] = words[1]
val_annotations.close()

print(label_data)


