#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


# In[2]:


student_data=pd.read_csv(r'C:\Users\asus 1\Desktop\Fliprobo\Grades.csv')


# In[3]:


student_data.head(20)


# In[4]:


student_data.shape


# In[5]:


student_data.info()


# In[6]:


student_data.isnull().sum()


# In[7]:


student_data.replace(np.nan, 0.0, inplace=True)
student_data.head()


# In[8]:


student_data.isnull().sum()


# In[9]:


pd.unique(student_data['PH-121'])


# In[10]:


student_data.replace({'A+':4.0, 'A':4.0, 'A-':3.7, 'B+':3.4 , 'B':3.0, 'B-':2.7, 'C+':2.4, 'C':2.0, 'C-':1.7, 'D+':1.4, 'D':1.0, 'WU':0.0, 'W':0.0, 'F':0.0, 'I':0.0}, inplace=True)
student_data.head()


# In[11]:


input=student_data.drop(columns=['Seat No.', 'CGPA'])
target=student_data['CGPA']


# In[12]:


input


# In[13]:


target


# In[14]:


df=pd.concat([input, target], axis=1)
df.head()


# In[15]:


df.hist(bins=30, figsize=(30,15))
plt.show()


# In[16]:


sns.heatmap(df.corr())


# In[17]:


df.head()


# In[18]:


df.columns


# In[19]:


df.info()


# # Model Building 

# ## Splitting data for model1(1st year), model2(1st & 2nd yr), model3(1st, 2nd & 3rd yr), model4(all 4 yrs)
# 

# In[40]:


model1_columns=[]
model2_columns=[]
model3_columns=[]
for item in [input]:
    for i in item:
        if i[3]=='1':
            model1_columns.append(i)
        elif i[3]=='2':
            model2_columns.append(i)
        elif i[3]=='3':
            model3_columns.append(i)
        pass

model2_columns=model1_columns+model2_columns
model3_columns=model2_columns+model3_columns


# In[41]:


model1_columns, model2_columns, model3_columns


# In[42]:


X1=input[list(model1_columns)].values
y1=target.values

X1, y1


# In[43]:


x_train_m1, x_test_m1, y_train_m1, y_test_m1 = train_test_split(X1, y1, test_size=0.3, random_state=2)
print('shape of training dataset: ',x_train_m1.shape)
print('shape of testing dataset: ',x_test_m1.shape)


# In[44]:


linreg1=LinearRegression()
linreg1.fit(x_train_m1, y_train_m1)
score1=linreg1.score(x_test_m1, y_test_m1)
score1


# In[45]:


rf_1 = RandomForestRegressor()
rf_1.fit(x_train_m1, y_train_m1)
y_pred1 = rf_1.predict(x_test_m1)
print('RF Mean Absolute Error:', mean_absolute_error(y_test_m1, y_pred1))
print('RF Mean Squared Error:', mean_squared_error(y_test_m1, y_pred1))
print('RF Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test_m1, y_pred1)))
print("The RF score of model for testing set",rf_1.score(x_test_m1, y_test_m1))


# In[46]:


dec_tree1=DecisionTreeRegressor()
dec_tree1.fit(x_train_m1, y_train_m1)
dec_tree_pred1 = dec_tree1.predict(x_test_m1)
print('DT Mean Absolute Error:', mean_absolute_error(y_test_m1, dec_tree_pred1))
print('DT Mean Squared Error:', mean_squared_error(y_test_m1, dec_tree_pred1))
print('DT Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test_m1, dec_tree_pred1)))
print("The DT score of model for testing set",dec_tree1.score(x_test_m1, y_test_m1))


# In[62]:


plt.scatter(y_test_m1, y_pred1,  color='blue')
plt.scatter(y_test_m1, dec_tree_pred1,  color='Orange')
plt.plot([student_data['CGPA'].min(), student_data['CGPA'].max()], [student_data['CGPA'].min(), student_data['CGPA'].max()], color='red')
plt.xlabel('CGPA')
plt.ylabel('Predicted CGPA')
plt.show()


# In[70]:


lr_metrics1=[{'Score':score1, 'MAE':mean_absolute_error(y_test_m1, dec_tree_pred1), 'MSE':mean_squared_error(y_test_m1, dec_tree_pred1), 'RMSE':np.sqrt(mean_squared_error(y_test_m1, dec_tree_pred1)), 'DT Score':dec_tree1.score(x_test_m1, y_test_m1)}]
lr_metrics1                                                                                                                            


# # Model2

# In[47]:


X2=input[list(model2_columns)].values
y2=target.values

X2,y2


# In[48]:


x_train_m2, x_test_m2, y_train_m2, y_test_m2 = train_test_split(X2, y2, test_size=0.3, random_state=2)
print('shape of training dataset: ',x_train_m2.shape)
print('shape of testing dataset: ',x_test_m2.shape)


# In[49]:


linreg2=LinearRegression()
linreg2.fit(x_train_m2, y_train_m2)
score2=linreg2.score(x_test_m2, y_test_m2)
score2


# In[50]:


rf_2 = RandomForestRegressor()
rf_2.fit(x_train_m2, y_train_m2)
rf_pred2 = rf_2.predict(x_test_m2)
print('RF2 Mean Absolute Error:', mean_absolute_error(y_test_m2, rf_pred2))
print('RF2 Mean Squared Error:', mean_squared_error(y_test_m2, rf_pred2))
print('RF2 Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test_m2, rf_pred2)))
print("The RF2 score of model for testing set",rf_2.score(x_test_m2, y_test_m2))


# In[52]:


dec_tree2=DecisionTreeRegressor()
dec_tree2.fit(x_train_m2, y_train_m2)
dec_tree_pred2 = dec_tree2.predict(x_test_m2)
print('DT2 Mean Absolute Error:', mean_absolute_error(y_test_m2, dec_tree_pred2))
print('DT2 Mean Squared Error:', mean_squared_error(y_test_m2, dec_tree_pred2))
print('DT2 Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test_m2, dec_tree_pred2)))
print("The DT2 score of model for testing set",dec_tree2.score(x_test_m2, y_test_m2))


# In[71]:


plt.scatter(y_test_m2, rf_pred2,  color='green')
plt.scatter(y_test_m2, dec_tree_pred2,  color='pink')
plt.plot([student_data['CGPA'].min(), student_data['CGPA'].max()], [student_data['CGPA'].min(), student_data['CGPA'].max()], color='red')
plt.xlabel('CGPA')
plt.ylabel('Predicted CGPA')
plt.show()


# In[72]:


lr_metrics2=[{'Score':score2, 'MAE':mean_absolute_error(y_test_m2, dec_tree_pred2), 'MSE':mean_squared_error(y_test_m2, dec_tree_pred2), 'RMSE':np.sqrt(mean_squared_error(y_test_m2, dec_tree_pred2)), 'DT Score':dec_tree2.score(x_test_m2, y_test_m2)}]
lr_metrics2   


# # Model 3

# In[53]:


X3=input[list(model3_columns)].values
y3=target.values

x_train_m3, x_test_m3, y_train_m3, y_test_m3 = train_test_split(X3, y3, test_size=0.3, random_state=2)
print('shape of training dataset: ',x_train_m3.shape)
print('shape of testing dataset: ',x_test_m3.shape)


# In[54]:


linreg3=LinearRegression()
linreg3.fit(x_train_m3, y_train_m3)
score3=linreg3.score(x_test_m3, y_test_m3)
score3


# In[55]:


rf_3 = RandomForestRegressor()
rf_3.fit(x_train_m3, y_train_m3)
rf_pred3 = rf_3.predict(x_test_m3)
print('RF3 Mean Absolute Error:', mean_absolute_error(y_test_m3, rf_pred3))
print('RF3 Mean Squared Error:', mean_squared_error(y_test_m3, rf_pred3))
print('RF3 Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test_m3, rf_pred3)))
print("The RF3 score of model for testing set",rf_3.score(x_test_m3, y_test_m3))


# In[56]:


dec_tree3=DecisionTreeRegressor()
dec_tree3.fit(x_train_m3, y_train_m3)
dec_tree_pred3 = dec_tree3.predict(x_test_m3)
print('DT3 Mean Absolute Error:', mean_absolute_error(y_test_m3, dec_tree_pred3))
print('DT3 Mean Squared Error:', mean_squared_error(y_test_m3, dec_tree_pred3))
print('DT3 Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test_m3, dec_tree_pred3)))
print("The DT3 score of model for testing set",dec_tree3.score(x_test_m3, y_test_m3))


# In[74]:


plt.scatter(y_test_m3, rf_pred3,  color='blue')
plt.scatter(y_test_m3, dec_tree_pred3,  color='Orange')
plt.plot([student_data['CGPA'].min(), student_data['CGPA'].max()], [student_data['CGPA'].min(), student_data['CGPA'].max()], color='red')
plt.xlabel('CGPA')
plt.ylabel('Predicted CGPA')
plt.show()


# In[73]:


lr_metrics3=[{'Score':score3, 'MAE':mean_absolute_error(y_test_m3, dec_tree_pred3), 'MSE':mean_squared_error(y_test_m3, dec_tree_pred3), 'RMSE':np.sqrt(mean_squared_error(y_test_m3, dec_tree_pred3)), 'DT Score':dec_tree3.score(x_test_m3, y_test_m3)}]
lr_metrics3   


# In[ ]:





# # Model 4

# In[57]:


X4=input.values
y4=target.values

x_train_m4, x_test_m4, y_train_m4, y_test_m4 = train_test_split(X4, y4, test_size=0.3, random_state=2)
print('shape of training dataset: ',x_train_m4.shape)
print('shape of testing dataset: ',x_test_m4.shape)


# In[58]:


linreg4=LinearRegression()
linreg4.fit(x_train_m4, y_train_m4)
score4=linreg4.score(x_test_m4, y_test_m4)
score4


# In[59]:


rf_4 = RandomForestRegressor()
rf_4.fit(x_train_m4, y_train_m4)
rf_pred4 = rf_4.predict(x_test_m4)
print('RF4 Mean Absolute Error:', mean_absolute_error(y_test_m4, rf_pred4))
print('RF4 Mean Squared Error:', mean_squared_error(y_test_m4, rf_pred4))
print('RF4 Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test_m4, rf_pred4)))
print("The RF4 score of model for testing set",rf_4.score(x_test_m4, y_test_m4))


# In[39]:


dec_tree4=DecisionTreeRegressor()
dec_tree4.fit(x_train_m4, y_train_m4)
dec_tree_pred4 = dec_tree4.predict(x_test_m4)
print('DT4 Mean Absolute Error:', mean_absolute_error(y_test_m4, dec_tree_pred4))
print('DT4 Mean Squared Error:', mean_squared_error(y_test_m4, dec_tree_pred4))
print('DT4 Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test_m4, dec_tree_pred4)))
print("The DT4 score of model for testing set",dec_tree4.score(x_test_m4, y_test_m4))


# In[79]:


plt.scatter(y_test_m4, rf_pred4,  color='yellow')
plt.scatter(y_test_m4, dec_tree_pred4,  color='red')
plt.plot([student_data['CGPA'].min(), student_data['CGPA'].max()], [student_data['CGPA'].min(), student_data['CGPA'].max()], color='blue')
plt.xlabel('CGPA')
plt.ylabel('Predicted CGPA')
plt.show()


# In[80]:


lr_metrics4=[{'Score':score4, 'MAE':mean_absolute_error(y_test_m4, dec_tree_pred4), 'MSE':mean_squared_error(y_test_m4, dec_tree_pred4), 'RMSE':np.sqrt(mean_squared_error(y_test_m4, dec_tree_pred4)), 'DT Score':dec_tree4.score(x_test_m4, y_test_m4)}]
lr_metrics4   


# In[ ]:




