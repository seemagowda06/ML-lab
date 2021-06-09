#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[3]:


def sigmoid_derivation(x):
    return x*(x-1)


# In[4]:


input = np.array([[0,0,1],
                  [1,1,1],
                  [1,0,1],
                  [0,1,1]])


# In[5]:


a_output = np.array([[0,1,1,0]]).T


# In[6]:


np.random.seed(1)


# In[7]:


weights = 2* np.random.random((3,1)) - 1


# In[8]:


print('Initial weights are:')
print(weights)


# In[9]:


for i in range(10000):
    input_layer = input
    o_output = sigmoid(np.dot(input_layer, weights))


# In[10]:


loss = o_output - a_output


# In[11]:


adjustment = loss * sigmoid_derivation(o_output)


# In[12]:


weights = weights + np.dot(input_layer.T, adjustment)


# In[13]:


print('new weights are:')
print(weights)

print('obtained output values are')
print(o_output)


# In[ ]:




