#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


# In[2]:


df_train=pd.read_csv("train.csv")
df_test=pd.read_csv("test.csv")


# In[3]:



train_col=df_train.columns
print(train_col)


# In[4]:


import math
for column in train_col:
    if (column=='id' or column=='Label'):
        df_train[column]=df_train[column]
    else:
#         df_test[column]=df_test[column].astype(float)
#         df_train[column]=df_train[column].astype(float)
         df_train[column]=np.log2(df_train[column]+1)


# In[5]:


test_cols=df_test.columns


# In[6]:



for column in test_cols:
 if (column=='id'):
     df_test[column]=df_test[column]
 else:
     df_test[column]=np.log2(df_test[column]+1)


# In[7]:


var_train=df_train.loc[:,train_col[2:]].var()


# In[8]:


temp={}

for i in range(2,len(train_col)):
    temp[train_col[i]]=var_train[i-2]


# In[9]:


drop_col_after_variance=[]
for i in temp:
    if temp[i]<0.025:
        drop_col_after_variance.append(i)


# In[10]:


df_after_drop_train=df_train.drop(drop_col_after_variance,axis=1, inplace= True)
df_after_drop_test=df_test.drop(drop_col_after_variance,axis=1, inplace= True)


# In[11]:


print(len(df_train.columns))
train_newc=df_train.columns
test_newc=df_test.columns
print(test_newc)


# In[12]:


from sklearn import preprocessing 
min_max_scaler = preprocessing.MinMaxScaler(feature_range =(0, 1)) 
# scaled_df_train = min_max_scaler.fit_transform(df_train)
scaled_df_test = min_max_scaler.fit_transform(df_test)


# In[13]:


print ("\nAfter min max Scaling : \n", scaled_df_test[0])


# In[16]:


print ("\nAfter min max Scaling : \n", scaled_df_test)
x = df_train[train_newc[2:]].values
scaled_df_train = min_max_scaler.fit_transform(x)
print ("\nAfter min max Scaling : \n", scaled_df_train)


# In[19]:


X = df_train.iloc[:,2:23530]  #independent columns
y = df_train.iloc[:,1]    #target column i.e price range
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=100)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(100,'Score')) 


#  

# In[20]:


imp=featureScores.nlargest(100,'Score')
imp_feat=imp["Specs"]
imp_cols=[]
for i in range(len(imp_feat)):
    imp_cols.append(imp_feat.iloc[i])


# In[21]:


cols_drop1= list(set(train_newc).difference(set(imp_cols)))


# In[22]:


# print(cols_drop1)
cols_drop_final=[]
for i in cols_drop1:
    if i =="id" or i=="Label":
        print(i)
    else:
        cols_drop_final.append(i)


# In[24]:


df_after_drop_train1=df_train.drop(cols_drop_final,axis=1, inplace= True)
df_after_drop_test1=df_test.drop(cols_drop_final,axis=1, inplace= True)
        


# In[25]:


ftrain_col=df_train.columns
print(ftrain_col)


# In[26]:


independent_attri=[]
for i in range(2,len(ftrain_col)):
    independent_attri.append(ftrain_col[i])
print(independent_attri)


# In[33]:


ftest_col=df_test.columns

independent_attri_test=[]
for i in range(1,len(ftest_col)):
    independent_attri_test.append(ftest_col[i])
print(independent_attri_test)


# In[34]:


test_data=df_test[ftest_col]


# In[35]:


#applying PCA on training data
scalar = StandardScaler() 
x=df_train[independent_attri]
# fitting 
scalar.fit(x) 
scaled_df = scalar.transform(x) 

from sklearn.decomposition import PCA
p = PCA(n_components = 2) 
p.fit(scaled_df) 
x_axis = p.transform(scaled_df) 
x_std=p.fit_transform(scaled_df)
x_std.shape
print(len(x_std))


# In[36]:


#applying PCA on test data
scalar1 = StandardScaler() 
x_test=df_test[independent_attri_test]

# fitting 
scalar1.fit(x_test) 
scaled_df1 = scalar1.transform(x_test) 

from sklearn.decomposition import PCA
p1 = PCA(n_components = 2) 
p1.fit(scaled_df1) 
x_axis1 = p1.transform(scaled_df1) 
x_std1=p1.fit_transform(scaled_df1)
x_std1.shape
print(len(x_std1))


# In[84]:


y1=df_train.Label.values
x1=x_std
# x1=df_train[independent_attri]
xtrain, xtest, ytrain, ytest = train_test_split(x1,y1,test_size=0.2)


# In[88]:


from sklearn.tree import DecisionTreeClassifier
descision_clf=DecisionTreeClassifier(criterion = "entropy", random_state = 1, max_depth = 5)
descision_clf=descision_clf.fit(xtrain,ytrain)
y_pred_dt = descision_clf.predict(x_std1)
print("Lables after prediction are: ", y_pred_dt)


# In[86]:


list1=df_test["id"]
output_list=[]
test_c=["id","Label"]
output_list.append(test_c)
for i in range(len(list1)):
    t=[]
    t.append(list1[i])
    t.append(y_pred_mlp[i])
    output_list.append(t)
    


# In[87]:


print(output_list)
import csv
with open('output_dt1.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(output_list)
  


# In[ ]:




