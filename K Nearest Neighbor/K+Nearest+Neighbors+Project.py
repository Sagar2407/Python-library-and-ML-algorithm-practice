
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[2]:


data = pd.read_csv('KNN_Project_Data')


# In[4]:


data.head()


# In[3]:



data.describe()


# # EDA
# 
# Since this data is artificial, we'll just do a large pairplot with seaborn.
# 
# **Use seaborn on the dataframe to create a pairplot with the hue indicated by the TARGET CLASS column.**

# In[6]:


sns.pairplot(data, hue='TARGET CLASS')


# In[7]:


from sklearn.preprocessing import StandardScaler


# ** Create a StandardScaler() object called scaler.**

# In[8]:


scaler= StandardScaler()


# ** Fit scaler to the features.**

# In[10]:


scaler.fit(data.drop('TARGET CLASS',axis=1))


# In[11]:


data_scaled= scaler.transform(data.drop('TARGET CLASS',axis=1))


# **Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.**

# In[12]:


data_new = pd.DataFrame(data_scaled,columns=data.columns[:-1])
data_new.head()


# # Train Test Split
# 
# **Use train_test_split to split your data into a training set and a testing set.**

# In[13]:


from sklearn.model_selection import train_test_split


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(data_scaled,data['TARGET CLASS'],
                                                    test_size=0.30)


# In[16]:


from sklearn.neighbors import KNeighborsClassifier


# In[17]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[18]:


knn.fit(X_train,y_train)


# In[19]:


pred = knn.predict(X_test)


# ** Create a confusion matrix and classification report.**

# In[20]:


from sklearn.metrics import classification_report,confusion_matrix


# In[21]:


print(confusion_matrix(y_test,pred))


# In[22]:


print(classification_report(y_test,pred))


# In[23]:


error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[24]:


plt.figure(figsize=(10,8))
plt.plot(range(1,40),error_rate,color='black', linestyle='dashed', marker='x',
         markerfacecolor='red', markersize=5)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[27]:


knn = KNeighborsClassifier(n_neighbors=32)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

