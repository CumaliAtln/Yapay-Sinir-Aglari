# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 13:20:58 2022

@author: tr
"""
#%%
import pandas as pd
import numpy as np
import tensorflow as tf
import PIL
import cv2
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
#!pip install tensorflow-addons
import tensorflow_addons as tfadd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.applications import InceptionResNetV2
# !pip install fastai --upgrade
from fastai.vision.all import *
#%%
train_img_Path = 'C:\\Users\\tr\\Desktop\\input\\plant-pathology-2021-fgvc8\\train_images'

test_img_Path = 'C:\\Users\\tr\\Desktop\\input\\plant-pathology-2021-fgvc8\\test_images'

img_Path = 'C:\\Users\\tr\\Desktop\\input\\plant-pathology-2021-fgvc8\\train_images'

train = pd.read_csv(r'C:\\Users\\tr\\Desktop\\input\\plant-pathology-2021-fgvc8\\train.csv')

sample_submission = pd.read_csv(r'C:\\Users\\tr\\Desktop\\input\\plant-pathology-2021-fgvc8\\sample_submission.csv')

#%%
path = Path('C:\\Users\\tr\\Desktop\\input\\plant-pathology-2021-fgvc8')
i=0
#256, 384, 640
for sz in [512]:
    i+=1
    resize_images(path/'train_images', max_size=sz, dest=f'img_sz_{sz}')
    print(i)
    print(f'{sz} - Done!')
#%%
#%%
#%%
#%%
#%%