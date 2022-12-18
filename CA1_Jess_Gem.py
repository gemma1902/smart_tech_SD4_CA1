import glob
import os
from email.mime import image
from os import listdir
from shutil import move

# import mpimg as mpimg
import numpy as np

# load all images in a directory
from os import listdir
# from matplotlib import image
# import matplotlib.pyplot as plt
# import tensorflow.keras
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import Flatten
# from keras.layers.convolutional import Conv2D
# from keras.layers.convolutional import MaxPooling2D
# from keras.utils.np_utils import to_categorical
# import random
# import requests
# from PIL import Image
# import cv2
# import pickle
# import pandas as pd
# import csv
#
VAL_PATH = "D:/smart_tech/tiny-imagenet-200/val"
TRAIN_PATH = "D:/smart_tech/tiny-imagenet-200/train"
TEST_PATH = "D:/smart_tech/tiny-imagenet-200/test"
WNIDS_PATH = "D:/smart_tech/tiny-imagenet-200/wnids.txt"
WORDS_PATH = "D:/smart_tech/tiny-imagenet-200/words.txt"
num_classes=200


#LOAD IN DATA FROM WNIDS TEXT FILE
with open(WNIDS_PATH, "r") as file:
    wnids = file.read()

#PAIR EACH WNIDS NUMBER TO AN INTERGER I.E. {n08966554 : 1, n0897856 : 2} etc.
pair = {}
for i, wnid in enumerate(wnids): #Loop through and assign label
    pair[wnid] = i

#Use words.txt to get names for each class  #not working fully
# with open(WORDS_PATH, 'r') as f:
#     wnid_to_words = dict(line.split('\t') for line in f)
#     for wnid, words in wnid_to_words.items():
#       wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
# class_names = [wnid_to_words[wnid] for wnid in wnids]

