
# coding: utf-8

# In[100]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[101]:


customers = pd.read_csv('Ecommerce Customers')


# In[102]:


customers.head()


# In[103]:


customers.describe()


# In[104]:


customers.info()


# In[105]:


sns.set_palette("GnBu_d")
sns.set_style('whitegrid')


# In[106]:




sns.jointplot('Time on Website' , 'Yearly Amount Spent' , customers)


# In[107]:


sns.jointplot('Time on App' , 'Yearly Amount Spent' , customers)


# In[108]:


sns.jointplot('Time on App' , 'Length of Membership' , customers , kind = 'hex')


# In[109]:


sns.pairplot(customers)


# In[110]:



sns.lmplot(x= 'Length of Membership', y= 'Yearly Amount Spent', data = customers)


# In[111]:


y = customers['Yearly Amount Spent']


# In[112]:


X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]


# In[113]:


from sklearn.model_selection import train_test_split


# In[114]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=101)


# In[115]:


from sklearn.linear_model import LinearRegression


# In[116]:


lm = LinearRegression()


# In[117]:


lm.fit(X_train ,y_train)


# In[118]:


lm.coef_


# In[119]:


predictions = lm.predict(X_test)


# In[120]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[121]:


from sklearn import metrics

print("MAE :", metrics.mean_absolute_error(y_test, predictions))
print("MSE :", metrics.mean_squared_error(y_test, predictions))
print("RMSE :", np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[122]:


lm.coef_
coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


# In[123]:



# Noramlizing the data to reduce the error

from sklearn import preprocessing
customers_new= pd.DataFrame(preprocessing.normalize(customers[['Avg. Session Length', 'Time on App','Time on Website','Yearly Amount Spent']])
                                                           ,columns= ['Avg. Session Length', 'Time on App','Time on Website','Yearly Amount Spent'])


# In[124]:


y = customers_normalized['Yearly Amount Spent']
X = customers_normalized[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=101)
lm.fit(X_train ,y_train)

predictions = lm.predict(X_test)
print("MAE :", metrics.mean_absolute_error(y_test, predictions))
print("MSE :", metrics.mean_squared_error(y_test, predictions))
print("RMSE :", np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[125]:


print(lm.coef_)


# In[126]:


sns.distplot((y_test-predictions),bins=30)


# In[127]:


coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients

