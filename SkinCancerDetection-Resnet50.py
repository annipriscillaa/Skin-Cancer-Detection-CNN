#!/usr/bin/env python
# coding: utf-8

# In[3]:


from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


# In[4]:


IMAGE_SIZE = [224, 224]
train_path = 'D:/college/SkinCancerDetectionCNN/images/images/training'
valid_path = 'D:/college/SkinCancerDetectionCNN/images/images/validation'


# In[5]:


resnet50 = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[6]:


for layer in resnet50.layers:
  layer.trainable = False


# In[7]:


folders = glob('D:/college/SkinCancerDetectionCNN/images/images/training/*')


# In[9]:


x = Flatten()(resnet50.output)


# In[10]:


prediction = Dense(len(folders), activation='softmax')(x)


# In[12]:


model = Model(inputs=resnet50.input, outputs=prediction)


# In[13]:


model.summary()


# In[14]:


model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[15]:


from keras.preprocessing.image import ImageDataGenerator


# In[16]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[17]:


training_set = train_datagen.flow_from_directory('D:/college/SkinCancerDetectionCNN/images/images/training',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('D:/college/SkinCancerDetectionCNN/images/images/validation',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[18]:


r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=15,
  steps_per_epoch=3,
  validation_steps=3
)


# In[19]:


plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')


# In[20]:


plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[ ]:




