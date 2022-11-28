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

VAL_PATH = "D:/smart_tech/tiny-imagenet-200/tiny-imagenet-200/val/val_annotations.txt"


with open(VAL_PATH, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)



