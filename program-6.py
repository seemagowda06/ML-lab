#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from sklearn import tree, metrics, model_selection, preprocessing 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


# In[3]:


df = pd.read_csv('iris.csv')


# In[4]:


y = df.iloc[:,-1].values
x = df.iloc[:,0:4].values


# In[5]:


x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.2)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train,y_train)

y_predict = classifier.predict(x_test)

acuuracy = metrics.accuracy_score(y_test, y_predict)
acuuracy

confusion_matrix(y_test, y_predict)


# In[ ]:




