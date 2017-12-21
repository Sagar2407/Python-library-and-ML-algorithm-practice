
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # Random Forest Project 
# 
# For this project we will be exploring publicly available data from [LendingClub.com](www.lendingclub.com). Lending Club connects people who need money (borrowers) with people who have money (investors). Hopefully, as an investor you would want to invest in people who showed a profile of having a high probability of paying you back. We will try to create a model that will help predict this.
# 
# Lending club had a [very interesting year in 2016](https://en.wikipedia.org/wiki/Lending_Club#2016), so let's check out some of their data and keep the context in mind. This data is from before they even went public.
# 
# We will use lending data from 2007-2010 and be trying to classify and predict whether or not the borrower paid back their loan in full. You can download the data from [here](https://www.lendingclub.com/info/download-data.action) or just use the csv already provided. It's recommended you use the csv provided as it has been cleaned of NA values.
# 
# Here are what the columns represent:
# * credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
# * purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
# * int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
# * installment: The monthly installments owed by the borrower if the loan is funded.
# * log.annual.inc: The natural log of the self-reported annual income of the borrower.
# * dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
# * fico: The FICO credit score of the borrower.
# * days.with.cr.line: The number of days the borrower has had a credit line.
# * revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
# * revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
# * inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
# * delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
# * pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

# # Import Libraries
# 
# **Import the usual libraries for pandas and plotting. You can import sklearn later on.**

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[3]:


loan= pd.read_csv('loan_data.csv')


# In[4]:


loan.info()


# In[5]:


loan.head()


# In[7]:


loan.describe()


# # Exploratory Data Analysis
# 
# Let's do some data visualization! We'll use seaborn and pandas built-in plotting capabilities, but feel free to use whatever library you want. Don't worry about the colors matching, just worry about getting the main idea of the plot.
# 
# ** Create a histogram of two FICO distributions on top of each other, one for each credit.policy outcome.**
# 
# *Note: This is pretty tricky, feel free to reference the solutions. You'll probably need one line of code for each histogram, I also recommend just using pandas built in .hist()*

# In[8]:


plt.figure(figsize=(10,8))
loan[loan['credit.policy']==1]['fico'].hist(alpha=0.5,color='red',bins=30,label='Credit.Policy=1')
loan[loan['credit.policy']==0]['fico'].hist(alpha=0.5,color='blue',bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')


# In[10]:


plt.figure(figsize=(10,8))
loan[loan['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',bins=30,label='not.fully.paid=1')
loan[loan['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')


# ** Create a countplot using seaborn showing the counts of loans by purpose, with the color hue defined by not.fully.paid. **

# In[11]:


plt.figure(figsize=(10,8))
sns.countplot(x='purpose',hue='not.fully.paid',data=loan)


# ** Let's see the trend between FICO score and interest rate. Recreate the following jointplot.**

# In[16]:


sns.jointplot(x='fico',y='int.rate',data=loan,color='blue')


# In[17]:


plt.figure(figsize=(10,8))
sns.lmplot(y='int.rate',x='fico',data=loan,hue='credit.policy',col='not.fully.paid')


# In[19]:


loan_final = pd.get_dummies(loan,columns= ['purpose'],drop_first=True)


# In[20]:


loan_final.info()


# In[21]:


from sklearn.model_selection import train_test_split


# In[24]:


X = loan_final.drop('not.fully.paid',axis=1)
y = loan_final['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[28]:


from sklearn.tree import DecisionTreeClassifier


# In[29]:


model = DecisionTreeClassifier()


# In[30]:


model.fit(X_train,y_train)


# In[31]:


predictions = model.predict(X_test)


# In[32]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))


# In[33]:


print(confusion_matrix(y_test,predictions))


# In[34]:


from sklearn.ensemble import RandomForestClassifier


# In[35]:


rfc = RandomForestClassifier(n_estimators=600)


# In[36]:


rfc.fit(X_train,y_train)


# In[37]:


predictions = rfc.predict(X_test)


# In[38]:


from sklearn.metrics import classification_report,confusion_matrix


# In[39]:


print(classification_report(y_test,predictions))


# In[40]:


print(confusion_matrix(y_test,predictions))


# In[41]:


param_grid =  {'n_estimators': [100, 250, 500, 600, 1000]}


# In[50]:


from sklearn.grid_search import GridSearchCV

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)


# In[47]:


CV_rfc.fit(X, y)


# In[48]:


CV_rfc.best_params_


# In[49]:


rfc = RandomForestClassifier(n_estimators=250)
rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)
print(classification_report(y_test,predictions))

