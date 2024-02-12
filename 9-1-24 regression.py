#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


# In[14]:


df=pd.read_csv("hprices.csv")
df


# In[15]:


plt.scatter(df.area,df.price)
plt.xlabel("area in sqrft")
plt.ylabel("prices($)")


# In[16]:


reg=linear_model.LinearRegression()


# In[17]:


reg.fit(df[["area"]],df.price)


# In[18]:


reg.predict([[3300]])


# In[19]:


reg.predict([[5000]])


# In[20]:


reg.predict([[2600]])


# In[21]:


reg.coef_


# In[22]:


reg.intercept_


# In[31]:


df1=pd.read_csv("salary_data.csv")
df1


# In[41]:


plt.scatter(df1.YearsExperience,df1.Salary)
plt.xlabel("Exp")
plt.ylabel("Salary")


# In[42]:


reg.fit(df1[["YearsExperience"]],df1.Salary)


# In[43]:


reg.predict([[5.2]])


# In[ ]:




