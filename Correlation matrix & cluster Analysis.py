#!/usr/bin/env python
# coding: utf-8

# In[75]:


import pandas as pd


df = pd.read_csv('GoHire_data_New.csv')


# In[76]:


df.head()


# In[77]:


df.describe()


# In[78]:


df.info()


# In[79]:


df = df.dropna()


# In[80]:


df = df.drop_duplicates()


# In[81]:


df['Timestamp']= pd.to_datetime(df['Timestamp'])


# In[82]:


df['Start_Time'] = pd.to_datetime(df['Start_Time'])


# In[83]:


df['End_Time']= pd.to_datetime(df['End_Time'])


# In[84]:


df['Date_of_Shift'] = pd.to_datetime(df['Date_of_Shift'], format='%m/%d/%Y')


# In[85]:


print(df.dtypes)


# In[86]:


df.head()


# In[87]:


df['Client']= df['Client'].astype('str')


# In[88]:


print(df['Client'].dtype)


# # correlation matrix

# In[137]:


#selection of relevant columns 
cols = df[['Age','Date_of_Shift', 'Approved_Shift_Hours']]


# In[138]:


cols.head()


# In[139]:


#extract day, month, and year from date column

df['Day']= pd.to_datetime(df['Date_of_Shift']).dt.day
df['Month']= pd.to_datetime(df['Date_of_Shift']).dt.month
df['Year']= pd.to_datetime(df['Date_of_Shift']).dt.year


# In[140]:


#converting start & end times to numeric variabes representing hour of the day.

df["Start_Hour"]= pd.to_datetime(df['Start_Time']).dt.hour
df['End_Hour']=pd.to_datetime(df['End_Time']).dt.hour


# In[141]:


print(df.dtypes)


# In[142]:


df['Start_Hour']= df['Start_Hour'].astype('float')
df['End_Hour']= df['End_Hour'].astype('float')


# In[143]:


print(cols.dtypes)


# In[147]:


#select all relevant columns for orrelation matrix

df_corr= df[['Age','Date_of_Shift', 'Approved_Shift_Hours','Day', 'Month', 'Year', 'Start_Hour', 'End_Hour']]


# In[149]:


df_corr.head()


# In[150]:


corr_matrix = df_corr.corr()


# In[151]:


print(corr_matrix)


# # correlation matrix with heatmap

# In[152]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[161]:


sns.heatmap(corr_matrix, cmap='coolwarm', annot =True) 


# # run a cluster analysis on the "Age" variable from the correlation matrix.

# In[162]:


#extract the "Age" column from the original dataframe
age_data = df[['Age']]


# In[163]:


age_data


# In[165]:


#standardize the data using the StandardScaler

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
age_data_scale = scaler.fit_transform(age_data)


# In[167]:


# cluster analysis using the KMeans function
from sklearn.cluster import KMeans

Kmeans = KMeans(n_clusters=3, random_state=0).fit(age_data_scale)


# In[168]:


df['Age_Cluster'] = Kmeans.labels_


# In[174]:


import seaborn as sns

sns.scatterplot(data=df, x='Age', y='Approved_Shift_Hours', hue='Age_Cluster')


# In[ ]:




