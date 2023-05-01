#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler


# In[3]:


iris = load_iris()
iris_data = iris.data
iris_columns = iris.feature_names

iris_df = pd.DataFrame(data = iris_data, columns = iris_columns)


# In[4]:


print(iris_df.head(30))


# In[5]:


scaler = MinMaxScaler()
iris_df_scaled = scaler.fit_transform(iris_df)
iris_df_normalized= pd.DataFrame(data = iris_df_scaled, columns = iris_columns)
print(iris_df_normalized.head(30))


# In[6]:


import seaborn as sns 
sns.pairplot(iris_df_normalized)

