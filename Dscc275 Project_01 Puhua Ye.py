
# coding: utf-8

# In[1]:


# Problem1


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm


# In[3]:


# (1)


# In[4]:


df = pd.read_csv("Problem1_DataSet.csv")
df.head(12)


# In[5]:


plt.figure(figsize=[15,10])
plt.plot(df['Miles, in Millions'])
plt.xlabel('Time',fontsize=14)
plt.ylabel('Miles,in Millions',fontsize=14)
x_major_locator=MultipleLocator(2)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.xlim(0,90)
plt.show()


# In[6]:


# (2)


# In[7]:


ACF = plot_acf(df['Miles, in Millions'])
x_major_locator=MultipleLocator(2)
ax=ACF.gca()
ax.xaxis.set_major_locator(x_major_locator)
ACF.show()


# In[8]:


# The seasonal period is 12.


# In[9]:


# (3)


# In[10]:


MA_6 = np.round(df['Miles, in Millions'].rolling(6).mean(),1)
df.insert(df.shape[1],'MA_6',MA_6)
MA_12 = np.round(df['Miles, in Millions'].rolling(12).mean(),1)
df.insert(df.shape[1],'MA_12',MA_12)
MA_24 = np.round(df['Miles, in Millions'].rolling(24).mean(),1)
df.insert(df.shape[1],'MA_24',MA_24)
df


# In[11]:


plt.figure(figsize=[15,10])
plt.plot(df['Miles, in Millions'], label='original data')
plt.plot(df['MA_6'],label='MA_6')
plt.plot(df['MA_12'],label='MA_12')
plt.plot(df['MA_24'],label='MA_24')
plt.legend(loc=2)
x_major_locator=MultipleLocator(2)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.xlim(0,90)
plt.show()


# In[12]:


# I think 12 or 24 are suitable, because they are more smoothing.
# window length = 6 still has seasonal factor.


# In[13]:


# (4)


# In[14]:


# The trend line is increasing.


# In[15]:


# (5)


# In[16]:


first_diff = df['Miles, in Millions'].diff(1)
first_diff.dropna(inplace=True)
ACF = plot_acf(first_diff)
PACF = plot_pacf(first_diff,method="ywmle")


# In[17]:


# significant lags based on ACF are 3,4,5,7,9,12.
# significant lags based on PACF is 2,3,4,5,7,8,11,20.


# In[18]:


#(6)


# In[19]:


seasonal_diff = first_diff.diff(12)
seasonal_diff.dropna(inplace=True)
ACF = plot_acf(seasonal_diff)
PACF = plot_pacf(seasonal_diff)


# In[20]:


# significant lags based on ACF is 2,10,12.
# significant lags based on PACF is 2,4,8,10,11,12,16,19.


# In[21]:


# (7)


# In[22]:


first_six_years = df[:72]
seven_years = df[72:84]


# In[23]:


p = [0,1,2,3]
P = [0,1,2,3]
q = [0,1,2,3]
Q = [0,1,2,3]
d = 1
D = 1
AIC=[]
for i in p:
    for j in q:
        for m in P:
            for n in Q:
                try:
                    model = sm.tsa.statespace.SARIMAX(first_six_years['Miles, in Millions'],
                                                  order=(p[i],d,q[j]), seasonal_order=(P[m],D,Q[n],12)).fit()
                except:
                    print("p:" , p[i] , "d" , d , "q" , q[j] , "P" , P[m] , "D" , D , "Q" , Q[n])
                    continue
                model_parameter = ["p:" , p[i] , "d" , d , "q" , q[j] , "P" , P[m] , "D" , D , "Q" , Q[n]]
                AIC.append(model_parameter)
                AIC.append(model.aic)
AIC_parameter = []
for x in range(1,len(AIC),2):
    AIC_parameter.append(AIC[x]) 
best_AIC = min(AIC_parameter)


# In[24]:


print("The best choice of Parameters are:" , AIC[AIC.index(best_AIC)-1])


# In[25]:


best_model = sm.tsa.statespace.SARIMAX(first_six_years['Miles, in Millions'],
                                                  order=(2,1,3),seasonal_order=(1,1,0,12)).fit()
print(best_model.summary())


# In[26]:


# I use aic as criteria.


# In[27]:


# (8)


# In[28]:


predict = best_model.forecast(12)
predict


# In[29]:


SSE = (seven_years['Miles, in Millions']- predict)**2
SSE = np.sum(SSE)
print("SSE is ", SSE)


# In[30]:


plt.figure(figsize=[15,10])
plt.plot(seven_years['Miles, in Millions'], label='original data')
plt.plot(predict,label='predict')
plt.legend(loc=2)
x_major_locator=MultipleLocator(2)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.xlim(70,88)
plt.show()


# In[31]:


# From the graph, we can see that the predicted data
# are not much different from original data.
# Also, SSE is also great.


# In[32]:


# problem 2


# In[33]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm


# In[34]:


# (a)


# In[35]:


df = pd.read_csv("TotalWine.csv")
df.head(10)


# In[36]:


plt.figure(figsize=[15,10])
plt.plot(df['TotalWine'])
plt.xlabel('Time',fontsize=14)
plt.ylabel('TotalWine',fontsize=14)
x_major_locator=MultipleLocator(2)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.xlim(0,53)
plt.show()


# In[37]:


# (b)


# In[38]:


first_diff = df['TotalWine'].diff(1)
first_diff.dropna(inplace=True)
plt.figure(figsize=[15,10])
plt.plot(first_diff)
plt.xlabel('Time',fontsize=14)
plt.ylabel('TotalWine',fontsize=14)
x_major_locator=MultipleLocator(2)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.xlim(0,53)
plt.show()


# In[39]:


second_diff = df['TotalWine'].diff(2)
second_diff.dropna(inplace=True)
plt.figure(figsize=[15,10])
plt.plot(second_diff)
plt.xlabel('Time',fontsize=14)
plt.ylabel('TotalWine',fontsize=14)
x_major_locator=MultipleLocator(2)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.xlim(0,53)
plt.show()


# In[40]:


fourth_diff = df['TotalWine'].diff(4)
fourth_diff.dropna(inplace=True)
plt.figure(figsize=[15,10])
plt.plot(fourth_diff)
plt.xlabel('Time',fontsize=14)
plt.ylabel('TotalWine',fontsize=14)
x_major_locator=MultipleLocator(2)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.xlim(0,53)
plt.show()


# In[41]:


sixth_diff = df['TotalWine'].diff(6)
sixth_diff.dropna(inplace=True)
plt.figure(figsize=[15,10])
plt.plot(sixth_diff)
plt.xlabel('Time',fontsize=14)
plt.ylabel('TotalWine',fontsize=14)
x_major_locator=MultipleLocator(2)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.xlim(0,53)
plt.show()


# In[42]:


# lag = 4 is most suitable to remove seasonality.


# In[43]:


# (c)


# In[44]:


ACF = plot_acf(df['TotalWine'])
x_major_locator=MultipleLocator(2)
ax=ACF.gca()
ax.xaxis.set_major_locator(x_major_locator)
ACF.show()


# In[45]:


# seasonal period is 4.


# In[46]:


# (d)


# In[47]:


import statsmodels.tsa.api as smt
best_order = smt.AR(df['TotalWine']).select_order(maxlag=10, ic='aic',trend='nc')
print("The best order is :", best_order)


# In[48]:


# (e)


# In[49]:


# i
model = smt.AR(fourth_diff).fit(3)
print(model.summary())


# In[50]:


# ii
predict=model.predict()
predict.head(10)


# In[51]:


# iii
plt.figure(figsize=[15,10])
plt.plot(predict,label='predicted data')
plt.xlabel('Time',fontsize=14)
plt.ylabel('TotalWine',fontsize=14)
x_major_locator=MultipleLocator(2)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.xlim(0,53)
plt.plot(fourth_diff,label='seasonal differencing')
plt.legend(loc=2)
plt.show()


# In[52]:


# iv
MAE = (abs(fourth_diff-predict)).sum()
print(MAE/51)

