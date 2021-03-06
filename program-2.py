#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np


# In[2]:


data = pd.read_csv('play.csv') 
concept = np.array(data)[:,:-1] 
target = np.array(data)[:,-1]


# In[3]:


data.head()


# In[4]:


def train(con,tar): 
    specific = con[0].copy() 
    general=[['?' for x in range(len(specific))] for x in range(len(specific))] 
    for i,val in enumerate(con): 
        if tar[i] == 'yes': 
            for x in range(len(specific)): 
                if val[x] != specific[x]: 
                    specific[x] = '?' 
                    general[x][x] = '?' 
        else: 
            for x in range(len(specific)): 
                if val[x] != specific[x]: 
                    general[x][x] = specific[x] 
                else: 
                    general[x][x]='?' 
        print("Iteration["+ str(i+1) + "]") 
        print("Specific: "+str(specific)) 
        print("General: "+str(general)+"\n\n") 
    general =[general[i] for i,val in enumerate(general) if val != ['?' for x in range(len(specific))]] 
    return specific, general


# In[5]:


specific,general = train(concept,target) 
print("--------Final hypothesis-------") 
print("Specific hypothesis: " +str(specific)) 
print("General hypothses: "+ str(general))


# In[ ]:




