#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.preprocessing import image


# In[2]:


(train_x, train_y) , (test_x, test_y) = mnist.load_data()
train_x = train_x.reshape(60000,784)
test_x = test_x.reshape(10000,784)
train_y = keras.utils.to_categorical(train_y,10)
test_y = keras.utils.to_categorical(test_y,10)


# In[3]:


model = Sequential()
model.add(Dense(units=128,activation="relu",input_shape=(784,)))
model.add(Dense(units=128,activation="relu"))
model.add(Dense(units=128,activation="relu"))
model.add(Dense(units=10,activation="softmax"))
model.compile(optimizer=SGD(0.001),loss="categorical_crossentropy",metrics=["accuracy"])


# In[5]:


model.fit(train_x,train_y,batch_size=32,epochs=10,verbose=1)


# In[22]:


img = test_x[1356]


# In[23]:


test_img = img.reshape((1,784))


# In[24]:


img_class = model.predict_classes(test_img)


# In[25]:


prediction = img_class[0]


# In[26]:


classname = img_class[0]


# In[27]:


print("Class: ",classname)


# In[28]:


img = img.reshape((28,28))
plt.imshow(img)
plt.title(classname)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# 
# 
