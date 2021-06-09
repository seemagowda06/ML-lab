#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np, pandas as pd, matplotlib.pyplot as plt 
from sklearn import tree, metrics, model_selection, preprocessing


# In[2]:


df = pd.read_csv('iris.csv')
df.head(5)

df['species_label'],i = pd.factorize(df['species'])
df['species'].unique()
df['species_label'].unique()


# In[3]:


y = df['species_label'] #Dependent feature
X = df[['sepal.length', 'sepal.width']] #Independent features (subset)


# In[4]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,test_size=0.3, random_state=0)


# In[5]:


dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
dtree.fit(X_train, y_train)


# In[6]:


y_pred = dtree.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))


# In[ ]:




