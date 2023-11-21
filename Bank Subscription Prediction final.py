#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[2]:


train=pd.read_csv(r'C:\Users\asus 1\Desktop\Fliprobo\termdeposit_train.csv')
test=pd.read_csv(r'C:\Users\asus 1\Desktop\Fliprobo\termdeposit_test.csv')


# In[3]:


train.columns


# In[4]:


test.columns


# In[5]:


train.info()


# In[6]:


test.info()


# In[7]:


train.shape


# In[8]:


test.shape


# In[9]:


train.head()


# In[10]:


test.head()


# In[11]:


train.isnull().sum()


# In[12]:


test.isnull().sum()


# # Univariate Analysis

# In[13]:


train['subscribed'].value_counts()


# In[14]:


sns.countplot(data=train, x='subscribed')


# In[15]:


train['subscribed'].value_counts(normalize=True)


# In[16]:


train['job'].value_counts()


# In[17]:


train['job'].value_counts().plot(kind='bar', figsize=(10,6))


# In[18]:


sns.countplot(data=train, x='job')


# In[19]:


train['marital'].value_counts()


# In[20]:


sns.countplot(data=train, x='marital')


# In[21]:


sns.countplot(data=train, x='marital', hue='subscribed')


# In[22]:


sns.countplot(data=train, x='education', hue='subscribed')


# In[23]:


sns.countplot(data=train, x='housing', hue='subscribed')


# In[24]:


sns.countplot(data=train, x='loan', hue='subscribed')


# In[25]:


sns.distplot(train['age'])


# # Bivariate Analysis

# In[26]:


#job vs subscribed
print(pd.crosstab(train['job'],train['subscribed']))


# In[27]:


pd.crosstab(train['marital'], train['subscribed'])


# In[28]:


pd.crosstab(train['default'], train['subscribed'])


# In[29]:


# Converting the target variables into 0s and 1s
train['subscribed'].replace('no', 0,inplace=True)
train['subscribed'].replace('yes', 1,inplace=True)


# In[30]:


train['subscribed']


# In[31]:


fig, ax=plt.subplots()
fig.set_size_inches(10,5)
sns.heatmap(train.corr(), annot=True, cmap='YlGnBu')


# # Model Buiilding

# In[32]:


from sklearn.preprocessing import LabelEncoder
cols=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'subscribed']
le=LabelEncoder()
for col in cols:
    train[col]=le.fit_transform(train[col])


# In[33]:


train.head()


# In[34]:


train.info()


# In[35]:


pd.unique(train['subscribed'])


# In[36]:


train.drop(columns=['pdays'], inplace=True, axis=1)


# In[37]:


train.shape


# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


target = train['subscribed']
features = train.drop('subscribed', axis=1)


# In[40]:


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=12)


# # Logistic Regression

# In[41]:


from sklearn.linear_model import LogisticRegression


# In[42]:


lr=LogisticRegression()


# In[43]:


lr.fit(X_train, y_train)


# In[44]:


y_pred_lr = lr.predict(X_test)


# In[45]:


from sklearn.metrics import accuracy_score


# In[46]:


acc_score_lr=accuracy_score(y_test, y_pred_lr)
acc_score_lr


# # Decision Tree

# In[47]:


from sklearn.tree import DecisionTreeClassifier


# In[48]:


dt=DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
acc_score_dt=accuracy_score(y_test, y_pred_dt)
acc_score_dt


# In[49]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier


# In[50]:


rfc=RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred_rfc=rfc.predict(X_test)
acc_score_rfc=accuracy_score(y_test, y_pred_rfc)
acc_score_rfc


# In[51]:


gb=GradientBoostingClassifier()
gb.fit(X_train, y_train)
y_pred_gb=gb.predict(X_test)
acc_score_gb=accuracy_score(y_test, y_pred_gb)
acc_score_gb


# In[52]:


gnb=GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb=gnb.predict(X_test)
acc_score_gnb=accuracy_score(y_test, y_pred_gnb)
acc_score_gnb


# In[53]:


adb=AdaBoostClassifier()
adb.fit(X_train, y_train)
y_pred_adb=adb.predict(X_test)
acc_score_adb=accuracy_score(y_test, y_pred_adb)
acc_score_adb


# In[54]:


model_df=pd.DataFrame({'Models':['Logistic Regression', 'Decision Tree', 'Random Forest Classifier', 'Gradient Boosting Classifier', 'Gaussian Naive Bayes Classifier', 'AdaBoost Classifier'], 'Accuracy Score' : [acc_score_lr, acc_score_dt, acc_score_rfc, acc_score_gb, acc_score_gnb, acc_score_adb]})
round(model_df.sort_values(by='Accuracy Score', ascending=False), 3)


# In[55]:


# Random Forest model is the best model for prediction


# # Prediction on Test Data

# In[56]:


test.head()


# In[57]:


cols=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
le=LabelEncoder()
for col in cols:
    test[col]=le.fit_transform(test[col])
    
test.head()


# In[58]:


test.drop(columns=['pdays'], inplace=True, axis=1)
test.head()


# In[59]:


test.shape


# In[60]:


test_pred=rfc.predict(test)
test_pred


# In[61]:


result=pd.DataFrame()


# In[62]:


result['ID']=test['ID']
result['subscribed']=test_pred


# In[63]:


result['subscribed']


# In[64]:


result['subscribed'].replace(0, 'no', inplace=True )
result['subscribed'].replace(1, 'yes', inplace=True )
result['subscribed']


# In[66]:


result.to_csv('Bank_Prediction_Result.csv', header=True, index=False)


# In[ ]:




