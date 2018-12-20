
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[15]:


X=np.array([0,2,4,6,8,10,12])
Y=np.array([1,3,5,6,4,3,9])


# In[16]:


def LR(x,y):
    
    x_m=np.mean(x)
    y_m=np.mean(y)
    
    m=((x*y).mean() - x.mean()*y.mean())/ ((x**2).mean() - x.mean()**2)
    b=y.mean() - m*x.mean()
    
    return m,b
    


# In[17]:


m,b=LR(X,Y)


# In[18]:


m,b


# In[25]:


x_t=np.array([0,1,3,5,6,7,8,9,10,11,12,13,14])


# In[26]:


y_pred=m*x_t+b


# In[27]:


plt.scatter(X,Y,color='b')
plt.plot(x_t,y_pred,color='r')


# In[28]:


from sklearn.linear_model import LinearRegression


# In[29]:


clf=LinearRegression()


# In[31]:


X.shape,Y.shape


# In[54]:


Y=Y.reshape(-1,1)
X=X.reshape(-1,1)


# In[55]:


X.shape,Y.shape


# In[56]:


clf.fit(X,Y)


# In[61]:


x_t=x_t.reshape(-1,1)


# In[62]:


x_t.shape


# In[63]:


ypred=clf.predict(x_t)


# In[64]:


plt.scatter(X,Y,color='b')
plt.plot(x_t,ypred,color='r')


# In[70]:


from sklearn.linear_model import Ridge


# In[89]:


clf=Ridge(alpha=10000)


# In[90]:


clf.fit(X,Y)


# In[91]:


ypred=clf.predict(x_t)


# In[92]:


plt.scatter(X,Y,color='b')
plt.plot(x_t,ypred,color='r')
plt.show()
