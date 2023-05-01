#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


data = pd.read_csv("C:/Users/aydin/Desktop/data/data.csv")

data.head()
data.drop(["id","Unnamed: 32"],axis=1,inplace = True)


# In[3]:


data.tail()


# In[4]:


M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]


# In[5]:


plt.scatter(M.radius_mean,M.texture_mean,color="red",label="kotu",alpha=0.3)
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="ıyı",alpha=0.3)
plt.xlabel("radius_mean(tümör yarıçapı)")
plt.ylabel("texture_mean(tümör dokusu)")
plt.legend()
plt.show()


# In[6]:


data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
x_data = data.drop(["diagnosis"],axis=1)
y = data.diagnosis.values


# In[7]:


x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[8]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state =10)


# In[9]:


from sklearn.naive_bayes import GaussianNB
nb =GaussianNB()
nb.fit(x_train,y_train)


# In[10]:


print("accuracy of svm algorithm:",nb.score(x_test,y_test))

