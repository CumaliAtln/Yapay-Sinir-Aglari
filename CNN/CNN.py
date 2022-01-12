# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 21:46:21 2022

@author: Cumali Atalan
"""

#%%

#gerekli kütüphaneler import edilir.
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tensorflow import keras
from keras.models import Sequential,load_model
from keras.layers import Dense,GlobalAveragePooling2D,Flatten,Conv2D,BatchNormalization,Dropout,MaxPooling2D,Activation
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator as Imgen

from PIL import Image
from sklearn.metrics import confusion_matrix,classification_report

import os

import warnings
warnings.filterwarnings("ignore")

#%%

#verilerin adresi verilir.
train_csv = pd.read_csv("C:\\Users\\tr\\Desktop\\input\\plant-pathology-2021-fgvc8\\train.csv")
train_path = "C:\\Users\\tr\\Desktop\\input\\plant-pathology-2021-fgvc8\\train_images"
test_path = "C:\\Users\\tr\\Desktop\\input\\plant-pathology-2021-fgvc8\\test_images"

#%%
#veriler hakkında bilgi alınır.
train_csv.head(3)
train_csv.info()

#%%

#verilerin içerisinde hangi etikette kaç veri olduğu konsola yazdırılır.
etiketler = train_csv["labels"].value_counts().index
veri = train_csv["labels"].value_counts().values

print(veri)
#%%
# veriler etiketlerine göre pasta grafiğinde gösterilir.

"""
explode = (0, 0.1, 0, 0,0,0,0,0,0,0,0,0)
pie, ax = plt.subplots(figsize=[10,6])
plt.pie(x=data, autopct="%.1f%%",explode=explode, labels=labels, pctdistance=0.5)
plt.title("Delivery Tips by type", fontsize=14);
pie.savefig("DeliveryPieChart.png")
"""

#%%
#her etiket bir class olarak ayrılır.
class_numarasi = len(train_csv["labels"].value_counts())

#%%

# verilerin ön işlemesi yapılır.
# xception'a göre ağırlıklar önceden verilir.
datagen = Imgen(preprocessing_function=keras.applications.xception.preprocess_input,
                 rotation_range=4,
                  shear_range=0.2,
                  zoom_range=0.15,
                  horizontal_flip=True,
                  validation_split=0.2
                 )

#%%
# datagende işlenen verilerin öğrenimi için gerekli veriler verilir.
train_ds = datagen.flow_from_dataframe(
    train_csv,
    directory = train_path, #nereden alacak veriyi
    x_col = 'image',
    y_col = 'labels',
    subset="training",
    color_mode="rgb",
    target_size = (224,224),
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=123
)
#%%
# datagende işlenen verilerin doğrulaması için gerekli veriler verilir.
val_ds = datagen.flow_from_dataframe(
    train_csv,
    directory = train_path,
    x_col = 'image',
    y_col = 'labels',
    subset="validation",
    color_mode="rgb",
    target_size = (224,224),
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=123
)

#%%
# np dizileri oluşturulur.
x,y = next(train_ds)
x.shape
#%%
from tensorflow.keras.applications.vgg16 import VGG16

# model olarak VGG16 kullanıyorum.
vgg = VGG16()
#%%
model = Sequential()
vgg_layers = vgg.layers # vgg modelinde katmanları tanımlıyorum.

for i in range(len(vgg_layers)-1):
    model.add(vgg_layers[i])
    #katmanlar modele ekleniyor.    
#%%
model.build(x.shape)
model.summary()
#%%

#katman özellikleri aktif ediliyor.

for l in vgg_layers:
    l.trainable = True

#%%

# aktivasyonu softmax olan dense katmanları ekleniyor.

model.add(Dense(class_numarasi, activation="softmax"))
model.summary()
#%%

# model tamamlanırken accuracy'e göre ölçüm yapması bekleniyor.

model.compile(loss = "categorical_crossentropy" , optimizer = "adam" , metrics=["accuracy"])
batch_size = 32

# epoch sayısı verilir. ve her epochta kaç veri işleneceği belirtilir. 
# 14900 görseli batch_size bölüp hep epochta öğrenimi işlenecek veri sayısı belirtilir.
# 3700 görseli batch_size bölüp hep epochta doğrulaması işlenecek veri sayısı belirtilir.

#öğrenme başlatılır.
hist = model.fit_generator(train_ds,
                           steps_per_epoch = 14900//batch_size,
                           epochs = 15,
                           validation_data = val_ds,
                           validation_steps = 3700//batch_size)

#%%

#öğrenme ve doğrulama oranları tablolaştırılır.
plt.plot(hist.history["accuracy"],label = "training accuracy")
plt.plot(hist.history["val_accuracy"],label = "validation accuracy")