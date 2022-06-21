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


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D
from keras.layers import Dropout
from sklearn.metrics import accuracy_score


# In[ ]:


"""base_dir = '../input/dog-vs-cat/dogvscat/train'
train_dir = os.path.join(base_dir, 'train')"""
train_cats_dir = '../input/dog-vs-cat/dogvscat/train/0'
train_dogs_dir = '../input/dog-vs-cat/dogvscat/train/1'


# In[ ]:


train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)
print(train_cat_fnames[:10])
print(train_dog_fnames[:10])


# In[ ]:


test_cats_dir = '../input/dog-vs-cat/dogvscat/test/0'
test_dogs_dir = '../input/dog-vs-cat/dogvscat/test/1'


# In[ ]:


print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))

print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))


# In[ ]:


train_dir = '../input/dog-vs-cat/dogvscat/train'
test_dir = '../input/dog-vs-cat/dogvscat/test'


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(224,224), 
                                                    batch_size=20,
                                                    class_mode='binary')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                        target_size=(224, 224),
                                                        batch_size=20,
                                                        class_mode='binary')


# In[ ]:





# In[ ]:


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224,224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


model.summary()


# In[ ]:


from tensorflow.keras import optimizers
model.compile(loss='binary_crossentropy', 
              optimizer=optimizers.RMSprop(learning_rate=1e-3),
              metrics=['accuracy'])


# In[ ]:


history = model.fit_generator(train_generator,
                              steps_per_epoch=train_generator.samples//train_generator.batch_size,
                              epochs=10,
                              validation_data=test_generator,
                              validation_steps=test_generator.samples//test_generator.batch_size)


# In[ ]:


model.save('cats_and_dogs_small_1.h5')

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[ ]:


from tensorflow.keras.preprocessing import image
img = image.load_img('../input/dog-vs-cat/dogvscat/test/0/268.jpg', target_size=(224, 224))
plt.imshow(img, interpolation='nearest')
plt.show()


# In[ ]:


img_array = np.array(img)

img_array.shape
img_array=img_array.reshape(1,224,224,3)


# In[ ]:


a=model.predict(img_array)
if a==[[0]]:
    print("cat")
else:
    print("dog")

