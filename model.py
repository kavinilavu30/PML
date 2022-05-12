#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as  np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[18]:


df = pd.read_csv('zoo1.csv')


# In[19]:


features = ['hair', 'feathers', 'eggs', 'milk', 'airborne',
       'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous',
       'fins', 'legs', 'tail', 'domestic', 'catsize']

X = df[features]
y = df['Class_Number']


# In[20]:


X_train,X_test,y_train,y_text = train_test_split(X,y , test_size = 0.25, random_state = 42)


# In[21]:


lor = LogisticRegression( )
lor.fit(X_train,y_train)
y_pred = lor.predict(X_test)


# In[22]:


pickle.dump(lor, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))


# In[ ]:




