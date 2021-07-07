#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('User_Data.csv')


# In[3]:


# input
x = dataset.iloc[:, [2, 3]].values


# In[4]:


# output
y = dataset.iloc[:, 4].values


# In[5]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size = 0.25, random_state = 0)


# In[6]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain) 
xtest = sc_x.transform(xtest)


# In[7]:


print (xtrain[0:10, :])


# In[8]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='lbfgs', max_iter=1000,random_state = 0)
classifier.fit(xtrain, ytrain)


# In[9]:


y_pred = classifier.predict(xtest)


# In[10]:



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, y_pred)


# In[11]:



print ("Confusion Matrix : \n", cm)


# In[12]:


from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(ytest, y_pred))


# In[22]:


from matplotlib.colors import ListedColormap
X_set, y_set = xtest, ytest
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
                               stop = X_set[:, 0].max() + 1,step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1,
                               stop = X_set[:, 1].max() + 1, 
                               step = 0.01))

plt.contourf(X1, X2, classifier.predict
			(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())



# In[21]:


for i, j in enumerate(np.unique(y_set)):
	plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
				c = ListedColormap(('red', 'green'))(i), label = j)
	
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[ ]:




