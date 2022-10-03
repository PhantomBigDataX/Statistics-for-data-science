#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Exploratory Data Analysis

# In[1]:


from pandas import read_csv
dating_profiles = read_csv("course-data/dating_profiles.csv")
preferred_columns = ['age', 'status', 'sex', 'orientation', 'body_type', 
                     'diet', 'drinks', 'drugs', 'education', 'ethnicity', 
                     'height', 'income', 'job', 'last_online', 'location', 'sign']
dating_profiles = dating_profiles[preferred_columns]
dating_profiles[['age', 'status', 'sex', 'orientation', 'body_type']].head(4)


# In[2]:


dating_profiles.to_csv("dating_profiles.csv")


# In[3]:


print("Mean:", dating_profiles["age"].mean())
print("Median:", dating_profiles["age"].median())
print("Max:", dating_profiles["age"].max())
print("Min:", dating_profiles["age"].min())


# In[4]:


dating_profiles.describe()


# In[5]:


dating_profiles["age"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])


# In[ ]:





# In[6]:


import seaborn as sns
sns.set(
        style="darkgrid",
        rc={"figure.figsize": (10, 5)})


# In[7]:


sns.boxplot(data=dating_profiles["age"], palette="viridis")


# In[8]:


sns.histplot(data=dating_profiles["age"], 
             bins=range(1, 75),
            )


# In[9]:


dating_profiles.columns


# In[10]:


dating_profiles["status"].unique()


# In[11]:


sns.countplot(data= dating_profiles, 
              y= 'status',
              palette="summer",
             )


# In[12]:


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




