#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np


# In[2]:


data = pd.read_csv('play.csv')


# In[3]:


data.head()


# In[4]:


concept = np.array(data)[:,:-1] 
target = np.array(data)[:,-1]


# In[5]:


def train(tar,con): 
    for i,val in enumerate(tar): 
        if val =='yes': 
            specific = con[i].copy() 
        break
    for i,val in enumerate(con): 
        if tar[i] =='yes': 
            for x in range(len(specific)): 
                if val[x] != specific[x]: 
                    specific[x] = '?' 
                else: 
                    pass 
        print("specific [",(i+1),"]:",str(specific)) 
    return specific
print(train(target,concept))


# In[ ]:




