#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
path="IIIT/Winter '20/BDMH/BDMH Project/Dataset"
training_data=path+'/'+'ann-train.data'
testing_data=path+'/'+'ann-test.data'
file=open(training_data,'r')
file_2=open(testing_data,'r')
text=file.read()
training_dataset=[]
text=text.strip()
text=text.split("\n")
for line in text:
    line=line.strip()
    words=line.split(" ")
    training_dataset.append(words)
print(training_dataset[0])   

text_2=file_2.read()
testing_dataset=[]
text_2=text_2.strip()
text_2=text_2.split("\n")
for line in text_2:
    line=line.strip()
    words=line.split(" ")
    testing_dataset.append(words)


# In[2]:



df = pd.DataFrame(training_dataset,columns = ['Age','Sex','On_thyroxine','Query_on_thyroxine','On_antithyroid_medication','Sick','Pregnant','Thyroid_surgery','I131_treatment','Query_hypothyroid','Query_hyperthyroid','Lithium','Goitre','Tumor'
                                  ,'Hypopituitary','Psych','TSH','T3','TT4','T4U','FTI','Class'])

df_test = pd.DataFrame(testing_dataset,columns = ['Age','Sex','On_thyroxine','Query_on_thyroxine','On_antithyroid_medication','Sick','Pregnant','Thyroid_surgery','I131_treatment','Query_hypothyroid','Query_hyperthyroid','Lithium','Goitre','Tumor'
                                  ,'Hypopituitary','Psych','TSH','T3','TT4','T4U','FTI','Class'])


# In[3]:


columns=df.columns


# In[4]:


print(df.shape)
print(df_test.shape)


# In[5]:


df.describe()


# In[6]:


for i in range(len(columns)):
    if(i==0 or i==16 or i==17 or i==18 or i==19 or i==20 ):
        df[columns[i]]=df[columns[i]].astype("float")
        df_test[columns[i]]=df_test[columns[i]].astype("float")
        
    else:
        df[columns[i]]=df[columns[i]].astype("int")
        df_test[columns[i]]=df_test[columns[i]].astype("int")


# In[7]:


df.dtypes


# In[8]:


len(training_dataset[0])


# In[9]:


df.isnull().sum()


# In[10]:


count={}
count_test={}
for i in range(3):
    count[i+1]=0
    count_test[i+1]=0
for i in range(len(df)):
    class_val=df.iloc[i,21]
    count[class_val]=count[class_val]+1
print(count)

for i in range(len(df_test)):
    class_val=int(df_test.iloc[i,21])
    count_test[class_val]=count_test[class_val]+1
print(count_test)    


# In[11]:


count_train=df['Class'].value_counts()
print(count_train)
count_test=df_test['Class'].value_counts()
print(count_test)


# In[13]:


import matplotlib.pyplot as plt

plt.bar('1',count_train[1],color='lightcoral')
plt.bar('2',count_train[2],color='mediumseagreen')
plt.bar('3',count_train[3],color='mediumpurple')
plt.xlabel('Class Label')
plt.ylabel('Class Distribution')
plt.title("Training Data")
plt.show()


# In[14]:


plt.bar('1',count_test[1],color='lightcoral')
plt.bar('2',count_test[2],color='mediumseagreen')
plt.bar('3',count_test[3],color='mediumpurple')
plt.xlabel('Class Label')
plt.ylabel('Class Distribution')
plt.title("Testing Data")
plt.show()


# In[28]:


#Normalise the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(df.iloc[:,0:3])
#X_test=sc_X.fit_transform(df_test.iloc[:,0:21])


# In[29]:


y_train=df.iloc[:,21]


# In[30]:


print(X_train.shape)
print(y_train.shape)


# In[31]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.metrics import roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2,random_state=100)
clf_dt= DecisionTreeClassifier(max_depth=5) 

#Train the model using the training sets
clf_dt.fit(X_train, y_train)

#Predict the response for test dataset
y_dt = clf_dt.predict(X_test)
a=metrics.accuracy_score(y_test,y_dt)
print("Training Validation accuracy: ",a)


# In[40]:


import pickle
pickle.dump(clf_dt, open("model.pkl","wb"))


# In[55]:


#prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,3)
    loaded_model = pickle.load(open("model.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]



@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        
        if int(result)==1:
            prediction='1'
        elif int(result)==2:
            prediction='2'
        else:
            prediction='3'
        
            
        return render_template("result.html",prediction=prediction)




# In[ ]:




