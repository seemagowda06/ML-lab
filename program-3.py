#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('salary_data.csv')


# In[3]:


x=data.iloc[:,:-1]
y=data.iloc[:,1]
plt.scatter(x,y)


# In[4]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)


# In[5]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)


# In[6]:


reg.predict(x_test)


# In[7]:


viz_train=plt
viz_train.scatter(x_train,y_train)
viz_train.title('Salary vs Experience(train data)')
viz_train.plot(x_train, reg.predict(x_train),color='red')
viz_train.xlabel('Years of Experience')
viz_train.ylabel('Salary')


# In[ ]:




