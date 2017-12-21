
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[2]:


ad_data = pd.read_csv('advertising.csv')


# In[3]:


ad_data.head()


# In[4]:


ad_data.info()


# In[5]:


ad_data.describe()


# In[6]:


ad_data['Age'].hist(bins =30)


# In[7]:


sns.jointplot('Age', 'Area Income', ad_data)


# In[8]:


sns.jointplot('Age', 'Daily Time Spent on Site', ad_data, kind= 'kde', color= 'red')


# In[9]:


sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,color='green')


# In[10]:


sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')


# In[11]:


X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]


# In[12]:


y= ad_data['Clicked on Ad']


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[15]:


from sklearn.linear_model import LogisticRegression


# In[16]:


model= LogisticRegression()
model.fit(X_train, y_train)


# In[17]:


predictions= model.predict(X_test)


# In[18]:


from sklearn.metrics import classification_report


# In[19]:


print(classification_report(y_test,predictions))

