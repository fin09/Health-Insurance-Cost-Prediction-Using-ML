#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd


# In[30]:


data=pd.read_csv("insurance.csv")


# In[31]:


data.info()


# In[32]:


data.head(n=10)


# In[33]:


data.isnull().sum()


# In[34]:


data.describe(include="all")


# In[35]:


data['sex']=data['sex'].map({'female':0,'male':1})


# In[36]:


data['smoker']=data['smoker'].map({'yes':1,'no':0})


# In[37]:


data['region']=data['region'].map({'southwest':0,'southeast':1,'northwest':2,'northeast':3})


# In[38]:


data.info()


# In[39]:


x=data.drop(['charges'],axis=1)
y=data['charges']


# In[40]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[41]:


lr = LinearRegression()
lr.fit(X_train,y_train)
svm = SVR()
svm.fit(X_train,y_train)
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
gr = GradientBoostingRegressor()
gr.fit(X_train,y_train)
y_pred1 = lr.predict(X_test)
y_pred2 = svm.predict(X_test)
y_pred3 = rf.predict(X_test)
y_pred4 = gr.predict(X_test)
df1 = pd.DataFrame({'Actual':y_test,'Lr':y_pred1,
                  'svm':y_pred2,'rf':y_pred3,'gr':y_pred4})


# In[42]:


import matplotlib.pyplot as plt
plt.subplot(221)
plt.plot(df1['Actual'].iloc[0:11],label='Actual')
plt.plot(df1['Lr'].iloc[0:11],label="Lr")
plt.legend()
plt.subplot(222)
plt.plot(df1['Actual'].iloc[0:11],label='Actual')
plt.plot(df1['svm'].iloc[0:11],label="svr")
plt.legend()
plt.subplot(223)
plt.plot(df1['Actual'].iloc[0:11],label='Actual')
plt.plot(df1['rf'].iloc[0:11],label="rf")
plt.legend()
plt.subplot(224)
plt.plot(df1['Actual'].iloc[0:11],label='Actual')
plt.plot(df1['gr'].iloc[0:11],label="gr")
plt.tight_layout()
plt.legend()


# In[43]:


from sklearn import metrics
score1 = metrics.r2_score(y_test,y_pred1)
score2 = metrics.r2_score(y_test,y_pred2)
score3 = metrics.r2_score(y_test,y_pred3)
score4 = metrics.r2_score(y_test,y_pred4)
print(score1,score2,score3,score4)


# In[44]:


s1 = metrics.mean_absolute_error(y_test,y_pred1)
s2 = metrics.mean_absolute_error(y_test,y_pred2)
s3 = metrics.mean_absolute_error(y_test,y_pred3)
s4 = metrics.mean_absolute_error(y_test,y_pred4)
print(s1,s2,s3,s4)


# In[45]:


test = {'age' : 40,
        'sex' : 1,
        'bmi' : 40.30,
        'children' : 4,
        'smoker' : 1,
        'region' : 2}
df = pd.DataFrame(test,index=[0])
df


# In[46]:


new_pred = rf.predict(df)
print("Medical Insurance cost for New Customer is : ",new_pred[0])


# In[47]:


df1


# In[48]:


rf = RandomForestRegressor()
gr.fit(x,y)


# In[49]:


import joblib as jb


# In[50]:


jb.dump(gr,"RandomForestRegressor_model")


# In[51]:


model= jb.load("RandomForestRegressor_model")


# In[52]:


model.predict(df)


# In[53]:


from tkinter import *
import joblib as jb


# In[54]:


def show_entry():
    p1=float(l1.get())
    p2=float(l2.get())
    p3=float(l3.get())
    p4=float(l4.get())
    p5=float(l5.get())
    p6=float(l6.get())
    f_model=jb.load("RandomForestRegressor_model")
    result=f_model.predict([[p1,p2,p3,p4,p5,p6]])
    Label(m,text="Your cost is :").grid(row=7)
    Label(m,text=result).grid(row=8)
 
    
m=Tk()
m.title("insurance cost")
label=Label(m,text="insurance cost",bg="black"
            ,fg="white").grid(row=0,columnspan=2)
Label(m,text="enter your age").grid(row=1)
Label(m,text="enter your genter male or female[1/0]").grid(row=2)
Label(m,text="enter your BMI").grid(row=3)
Label(m,text="enter your number of children").grid(row=4)
Label(m,text="you are smoker yes/no [1/0]").grid(row=5)
Label(m,text="enter your region[1-4]").grid(row=6)

l1=Entry(m)
l2=Entry(m)
l3=Entry(m)
l4=Entry(m)
l5=Entry(m)
l6=Entry(m)
l1.grid(row=1,column=1)
l2.grid(row=2,column=1)
l3.grid(row=3,column=1)
l4.grid(row=4,column=1)
l5.grid(row=5,column=1)
l6.grid(row=6,column=1)

Button(m,text="Your cost is :",command=show_entry).grid()




m.mainloop()


# In[ ]:





# In[ ]:




