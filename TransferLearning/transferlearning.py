#%%

import numpy as np 
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sns
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from keras import Sequential
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications import InceptionResNetV2, ResNet152V2, DenseNet169
from tensorflow.keras.layers import Dense

from keras.callbacks import EarlyStopping

#%%

train_img_Path = 'C:\\Users\\tr\\Desktop\\input\\plant-pathology-2021-fgvc8\\train_images'

test_img_Path = 'C:\\Users\\tr\\Desktop\\input\\plant-pathology-2021-fgvc8\\test_images'

img_Path = 'C:\\Users\\tr\\Desktop\\input\\plant-pathology-2021-fgvc8\\train_images'

train_csv = pd.read_csv(r'C:\\Users\\tr\\Desktop\\input\\plant-pathology-2021-fgvc8\\train.csv')

sample_submission = pd.read_csv(r'C:\\Users\\tr\\Desktop\\input\\plant-pathology-2021-fgvc8\\sample_submission.csv')


#%%

train_csv.head()
#%%

print(f'Number of pictures in the training dataset: {train_csv.shape[0]}\n')
print(f'Number of different labels: {len(train_csv.labels.unique())}\n')
print(f'Labels: {train_csv.labels.unique()}')
#%%

train_csv['labels'].value_counts()
#%%

plt.figure(figsize=(14,7))
a = sns.countplot(x='labels', data=train_csv, order=sorted(train_csv['labels'].unique()))
for item in a.get_xticklabels():
    item.set_rotation(90)
plt.title('Label Distribution', weight='bold')
plt.show()

#%%

plt.figure(figsize=(20,40))
i=1
for idx,s in train_csv.head(9).iterrows():
    img_path = os.path.join(img_Path,s['image'])
    img=cv2.imread(img_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    fig=plt.subplot(9,3,i)
    fig.imshow(img)
    fig.set_title(s['labels'])
    i+=1
#%%

CLASSES = train_csv['labels'].unique().tolist()
#%%

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range = 0.1,
                                   zoom_range = 0.1,
                                   horizontal_flip = True,
                                   validation_split=0.25)

train_data = train_datagen.flow_from_dataframe(train_csv,
                                              directory=img_Path,
                                              classes=CLASSES,
                                              x_col="image",
                                              y_col="labels",
                                              target_size=(150, 150),
                                              subset='training')

val_data = train_datagen.flow_from_dataframe(train_csv,
                                            directory=img_Path,
                                            classes=CLASSES,
                                            x_col="image",
                                            y_col="labels",
                                            target_size=(150, 150),
                                            subset='validation')
#%%

dict_classes = train_data.class_indices
dict_classes



#%%
base_Net = ResNet152V2(include_top = False, 
                         weights = 'C:\\Users\\tr\\Desktop\\input\\ResNet152V2_NoTop_ImageNet.h5', 
                         input_shape = train_data.image_shape, 
                         pooling='avg',
                         classes = CLASSES)

#%%

#Callbacks
EarlyStop_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
my_callback=[EarlyStop_callback]
#%%

#Adding the final layers to the above base models where the actual classification is done in the dense layers
model_Net = Sequential()
model_Net.add(base_Net)
model_Net.add(Dense(12, activation=('softmax')))
#%%
model_Net.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['AUC'])
#%%
model_Net.build(train_data.image_shape)
model_Net.summary()
#%%
# Training the CNN on the Train data and evaluating it on the val data
batch_size = 32
a = model_Net.fit(train_data, 
                  validation_data = val_data, 
                  steps_per_epoch = 14900//batch_size,
                  epochs = 10, 
                  callbacks=my_callback, 
                  validation_steps = 3700//batch_size)

#%% burada kaldık keke
base_InceptionResNetV2 = InceptionResNetV2(include_top = False, 
                         weights = 'C:\\Users\\tr\\Desktop\\input\\InceptionResNetV2_NoTop_ImageNet.h5', 
                         input_shape = train_data.image_shape, 
                         pooling='avg',
                         classes = CLASSES)

#%%

#Adding the final layers to the above base models where the actual classification is done in the dense layers
model_IResNet2 = Sequential()
model_IResNet2.add(base_InceptionResNetV2)
model_IResNet2.add(Dense(12, activation=('softmax')))

model_IResNet2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#%%
model_IResNet2.build(train_data.image_shape)
model_IResNet2.summary()

#%%
# Training the CNN on the Train data and evaluating it on the val data
batch_size = 32
c = model_IResNet2.fit(train_data, 
                       steps_per_epoch = 14900//batch_size,
                       validation_data = val_data, 
                       epochs = 10, 
                       callbacks=my_callback, 
                       validation_steps = 3700//batch_size)



#%%

base_DenseNet169 = DenseNet169(include_top = False, 
                         weights = 'C:\\Users\\tr\\Desktop\\input\\DenseNet169_NoTop_ImageNet.h5', 
                         input_shape = train_data.image_shape, 
                         pooling='avg',
                         classes = CLASSES)
#%%

#Adding the final layers to the above base models where the actual classification is done in the dense layers
model_dense = Sequential()
model_dense.add(base_DenseNet169)
model_dense.add(Dense(12, activation=('softmax')))

model_dense.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#%%
model_dense.build(train_data.image_shape)
model_dense.summary()

#%%
# Training the CNN on the Train data and evaluating it on the val data
batch_size=32
d = model_dense.fit(train_data, 
                    steps_per_epoch = 14900//batch_size,
                    validation_data = val_data, 
                    epochs = 10,
                    callbacks=my_callback, 
                    validation_steps = 3700//batch_size)

 
#%%

test_dir = 'C:\\Users\\tr\\Desktop\\input\\plant-pathology-2021-fgvc8\\test_images'
test_df = pd.DataFrame()
test_df['image'] = os.listdir(test_dir)

test_data = train_datagen.flow_from_dataframe(dataframe=test_df,
                                    directory=test_dir,
                                    x_col="image",
                                    y_col=None,
                                    batch_size=32,
                                    seed=42,
                                    shuffle=False,
                                    class_mode=None,
                                    target_size=(150, 150))
#%%

pred_net = model_Net.predict(test_data)
pred_iresnet2 = model_IResNet2.predict(test_data)
pred_dense = model_dense.predict(test_data)
#%%

pred = ((pred_net+pred_iresnet2+pred_dense)/3).tolist()
#%%

for i in range(len(pred)):
    pred[i] = np.argmax(pred[i])


def get_key(val):
    for key, value in dict_classes.items():
        if val == value:
            return key
        

for i in range(len(pred)):
    pred[i] = get_key(pred[i])
#%%

test_df['labels'] = pred
test_df.to_csv('submission.csv',index=False)


