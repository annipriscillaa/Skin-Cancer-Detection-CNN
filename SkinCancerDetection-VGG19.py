#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


# In[2]:


IMAGE_SIZE = [224, 224]
train_path = 'D:/college/SkinCancerDetectionCNN/images/images/training'
valid_path = 'D:/college/SkinCancerDetectionCNN/images/images/validation'


# In[4]:


vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[5]:


for layer in vgg.layers:
  layer.trainable = False


# In[6]:


folders = glob('D:/college/SkinCancerDetectionCNN/images/images/training/*')


# In[7]:


x = Flatten()(vgg.output)


# In[8]:


prediction = Dense(len(folders), activation='softmax')(x)


# In[9]:


model = Model(inputs=vgg.input, outputs=prediction)


# In[10]:


model.summary()


# In[11]:


model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[12]:


from keras.preprocessing.image import ImageDataGenerator


# In[13]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[14]:


training_set = train_datagen.flow_from_directory('D:/college/SkinCancerDetectionCNN/images/images/training',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('D:/college/SkinCancerDetectionCNN/images/images/validation',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[16]:


r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=30,
  steps_per_epoch=3,
  validation_steps=3
)


# In[17]:


plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')


# In[18]:


plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[ ]:




