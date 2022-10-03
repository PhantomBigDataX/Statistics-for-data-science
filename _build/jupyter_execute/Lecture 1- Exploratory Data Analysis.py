#!/usr/bin/env python
# coding: utf-8

# <div style="width: 100%; clear: both;">
# <div style="float: right; width: 40%;">
# <img src="https://barcelonatechnologyschool.com//wp-content/uploads/2018/01/BTS-header-color.svg", alt="Logo BTS", align="left", width="500">
# </div>
# <div style="float: left; width: 60%;">
# <p style="margin: 0; padding-top: 10px; font-size:24px; font-weight:bold;">Statistical Foundations for Data Scince</p>
# <p style="margin: 0; padding-top: 10px; font-size:20px;">Master in Big Data & A.I. Solutions​</p>
# <p style="margin: 0; padding-top: 5px; font-size:20px;">2021-2022</p>
# <p style="margin: 0; padding-top: 10px; font-size:20px; font-weight:bold; color:royalblue">Ankit Tewari​</p>
# </div>
# </div>

# In[ ]:





# In[ ]:





# In[1]:


from pandas import read_csv
dating_profiles = read_csv("dating_profiles.csv")
preferred_columns = ['age', 'status', 'sex', 'orientation', 'body_type', 
                     'diet', 'drinks', 'drugs', 'education', 'ethnicity', 
                     'height', 'income', 'job', 'last_online', 'location', 'sign']
dating_profiles = dating_profiles[preferred_columns]
dating_profiles.head(4)


# In[80]:


print("Mean:", dating_profiles["age"].mean())
print("Median:", dating_profiles["age"].median())
print("Max:", dating_profiles["age"].max())
print("Min:", dating_profiles["age"].min())


# In[17]:


dating_profiles.describe()


# In[14]:


dating_profiles[dating_profiles["age"]> 80]


# In[18]:


dating_profiles["age"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])


# In[ ]:





# In[75]:


import seaborn as sns
sns.set(
        style="darkgrid",
        rc={"figure.figsize": (10, 5)})


# In[76]:


sns.boxplot(data=dating_profiles["age"], palette="viridis")


# In[79]:


sns.histplot(data=dating_profiles["age"], 
             bins=range(1, 75),
            )


# In[81]:


dating_profiles.columns


# In[89]:


dating_profiles["status"].unique()


# In[114]:


sns.countplot(data= dating_profiles, 
              y= 'status',
              palette="summer",
             )


# In[117]:


sns.countplot(data= dating_profiles, 
              y= 'status',
              palette="summer",
              hue= 'sex'
             )


# ### Violin Plot
# A violin plot, introduced by [Hintze-Nelson-1998], is an enhancement to the boxplot
# and plots the density estimate with the density on the y-axis. The density is mirrored
# and flipped over, and the resulting shape is filled in, creating an image resembling a
# violin. The advantage of a violin plot is that it can show nuances in the distribution
# that aren’t perceptible in a boxplot. On the other hand, the boxplot more clearly
# shows the outliers in the data.

# ### Multiple Variables

# In[ ]:




