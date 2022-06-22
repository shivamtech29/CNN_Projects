#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import matplotlib.pyplot as plt
import seaborn as sns


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import keras
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D
from keras.layers import Dropout
from sklearn.metrics import accuracy_score


# In[3]:


"""
train_glioma_dir = '../input/brain-tumor-classification-mri/Training/glioma_tumor'
train_menin_dir = '../input/brain-tumor-classification-mri/Training/meningioma_tumor'
train_no_dir = '../input/brain-tumor-classification-mri/Training/no_tumor'
train_pitu_dir = '../input/brain-tumor-classification-mri/Training/pituitary_tumor'"""


# In[5]:


"""train_glioma_fnames = os.listdir(train_glioma_dir)
print(train_glioma_fnames[:10])
"""


# In[4]:


"""test_glioma_dir = '../input/brain-tumor-classification-mri/Testing/glioma_tumor'
test_menin_dir = '../input/brain-tumor-classification-mri/Testing/meningioma_tumor'
test_no_dir = '../input/brain-tumor-classification-mri/Testing/no_tumor'
test_pitu_dir = '../input/brain-tumor-classification-mri/Testing/pituitary_tumor'"""


# In[5]:


"""print('total training glioma images:', len(os.listdir(train_glioma_dir)))
print('total training menin images:', len(os.listdir(train_menin_dir)))
print('total training no tumor images:', len(os.listdir(train_no_dir)))
print('total training pituitary images:', len(os.listdir(train_pitu_dir)))

print('total test glioma images:', len(os.listdir(test_glioma_dir)))
print('total test menin images:', len(os.listdir(test_menin_dir)))
print('total test no tumor images:', len(os.listdir(test_no_dir)))
print('total test pituitary images:', len(os.listdir(test_pitu_dir)))"""


# In[6]:


"""train_dir = '../input/brain-tumor-classification-mri/Training'
test_dir = '../input/brain-tumor-classification-mri/Testing'"""


# In[7]:


"""from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(256,256), 
                                                    batch_size=20,
                                                    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                        target_size=(256,256),
                                                        batch_size=20,
                                                        class_mode='categorical')"""


# In[28]:


import ipywidgets as widgets
import io
from PIL import Image
import tqdm
from sklearn.model_selection import train_test_split
import cv2


# In[33]:


from sklearn.utils import shuffle
import tensorflow as tf


# In[42]:


X_train = []
y_train = []
image_size = 150
labels = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']
for i in labels:
    folderPath = os.path.join('../input/brain-tumor-classification-mri/Training',i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size, image_size))
        X_train.append(img)
        y_train.append(i)
        
for i in labels:
    folderPath = os.path.join('../input/brain-tumor-classification-mri/Testing',i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        y_train.append(i)
        
X_train = np.array(X_train)
y_train = np.array(y_train)


# In[43]:


X_train,y_train=shuffle(X_train,y_train,random_state=101)
X_train.shape


# In[44]:


X_train,X_test,y_train,y_test=train_test_split(X_train,y_train,test_size=.1,random_state=101)
y_train_new = []
for i in y_train:
    y_train_new.append(labels.index(i))
y_train = y_train_new
y_train = tf.keras.utils.to_categorical(y_train)
y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))
y_test = y_test_new
y_test = tf.keras.utils.to_categorical(y_test)


# In[45]:


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150,150, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))


# In[46]:


model.summary()


# In[47]:


#from tensorflow.keras import optimizers
model.compile(loss='categorical_crossentropy', 
              optimizer='Adam',
              metrics=['accuracy'])


# In[48]:


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
callbacks = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
# autosave best Model
best_model_file = '/content/CNN_aug_best_weights.h5'
best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)


# In[49]:


history=model.fit(X_train,y_train,epochs=20,validation_split=.1,callbacks=[best_model])


# In[50]:


model.save('brain_tumour_dec.h5')

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc))

fig = plt.figure(figsize=(14,7))
plt.plot(epochs, acc, 'r', label="Training Accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc='lower right')
plt.show()


# In[51]:


fig2 = plt.figure(figsize=(14,7))
plt.plot(epochs, loss, 'r', label="Training Loss")
plt.plot(epochs, val_loss, 'b', label="Validation Loss")
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and validation loss')


# In[97]:


img = cv2.imread('../input/brain-tumor-classification-mri/Testing/pituitary_tumor/image(19).jpg')
img = cv2.resize(img,(150, 150))
img_array = np.array(img)

img_array.shape
img_array=img_array.reshape(1,150,150,3)


# In[99]:


from tensorflow.keras.preprocessing import image
img = image.load_img('../input/brain-tumor-classification-mri/Testing/pituitary_tumor/image(19).jpg', target_size=(256, 256))
plt.imshow(img, interpolation='nearest')
plt.show()


# In[100]:


a=model.predict(img_array)
indices = a.argmax()
if indices == 0:
    print("glioma")
elif indices == 1:
    print("No tumor")
elif indices == 2:
    print("Melign")
elif indices == 3:
    print("Pituitary")


# In[61]:


pred = model.predict(X_test)
pred = np.argmax(pred,axis=1)
y_test_new = np.argmax(y_test,axis=1)
accuracy = np.sum(pred==y_test_new)/len(pred)
print("Accuracy on testing dataset: {:.2f}%".format(accuracy*100))

