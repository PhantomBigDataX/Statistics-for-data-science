#!/usr/bin/env python
# coding: utf-8

# # Introduction to Statistics

# ## What is Statistics?
# The art and science of answering questions and exploring ideas through the processes of gathering data, describing data, and making generalizations about a population on the basis of a smaller sample. In this process, we may divide our objectives to study the data about some population into the following two-
# 
# - Descriptive: It allows us to describe and organize the inevitable variability among observations;
# - Inferential: It allows us to to generalize beyond limited sets of observations (this limited set of obervations are the ones that we have drawn from the original population) by adjusting for the pervasive effects of variability;

# ## What is Data?
# 
# 
# 
# ## What are the various types of data?
# 
# Generally, in Math and Statistics, the variables may be Quantitative (Numerical) or Qualtitative (Categorical). These two major classfication of data maybe further divided into the following-
# 
# - Numerical
#  - Discrete
#  - Continuous
#  
# - Categorical
#  - Ordinal
#  - Nominal
# 

# **Numerical variables**
# 
# A numerical variable can take a wide range of numerical values, and it is sensible to add, subtract, or take averages with those values.
# 
# - **Discrete**
# 
#  A discrete numerical variable is a variable whose value is obtained by counting. For example, the number of students present in a class or the number of heads when flipping three coins.
# 
# 
# - **Continuous**
# 
#  A continuous variable is a numerical variable whose value is obtained by measuring. For example, the height of the students in class or the time it takes a student to get to school.

# **Categorical variables**
# 
# A categorical variable takes a limited, and usually fixed, number of possible values (categories). The possible values are called the variable's levels.
# - **Binary**:
# A Binary variable is a categorical variable whose outcomes may take only one of the two possible values. For example, Male and Female.
# 
# - **Ordinal**: 
# An ordinal variable is a categorical variable whose levels have a natural notion of order. For example, the age range of the students or the agreement with a statement ("strongly agree", "agree", "disagree", etc).
# 
# - **Nominal**:
# A nominal variable is a categorical variable without any notion of order. For example, we can consider genders or blood type.

# In[ ]:





# ## Collecting and Summarizing Data
# 
# The lesson starts with a discussion about how to collect data and different methods to use. Once the data is collected, we need to summarize the data. How to summarize the data depends on which variable type we have. All of these concepts will be presented here.
# 
# 

# Statistical methods are based on the notion of implied randomness. If observational data is not collected in a random framework from a population, these statistical methods – the estimates and errors associated with the estimates – are not reliable.

# ### Simple random sampling
# 
# In general, a sample is referred to a Simple Random Sampling if each case in the population has an equal chance of being included in the final sample and knowing that a case is included in a sample does not provide useful information about which other cases are included.

# In[1]:


from pandas import read_csv
dating_profiles = read_csv('dating_profiles.csv', index_col= [0])
dating_profiles.sample(5)[["age", "status", "sex", "orientation", "diet", "drinks"]]


# ### Stratified random sampling
# 
# In Stratified Random Sampling the population is divided into groups called strata. The strata are chosen so that similar cases are grouped together, then a second sampling method, usually simple random sampling, is employed within each stratum.

# In[2]:


# Break the code and find more extreme cases
from pandas import read_csv
dating_profiles = read_csv('dating_profiles.csv', index_col= [0])
dating_profiles.groupby("orientation", group_keys=False).apply(lambda x: x.sample(2))[["age", "status", "sex", "orientation", "diet", "drinks"]]


# In[3]:


dating_profiles.groupby('drinks').count()


# In[4]:


from pandas import read_csv
dating_profiles = read_csv('dating_profiles.csv', index_col= [0])
dating_profiles.groupby("diet", group_keys=False).apply(lambda x: x.sample(min(len(x), 2)))[["age", "status", "sex", "orientation", "diet", "drinks"]]


# ### Cluster random sampling
# 
# In a Cluster Random Sample, we break up the population into many groups, called clusters. Then we sample a fixed number of clusters and include all observations from each of those clusters in the sample.

# ### Multistage random sampling
# 
# A Multistage Random Sample is like a cluster sample, but rather than keeping all observations in each cluster, we collect a random sample within each selected cluster.

# ### Sample with and without replacement
# 
# When a sampling unit is drawn from a finite population and is returned to that population, after its characteristics have been recorded, before the next unit is drawn, the sampling is said to be “with replacement”.
# 
# Otherwise, the sampling is said to be "without replacement".

# In[5]:


# sampling with replacement
dating_profiles.head(5)[["age", "status", "sex", "orientation", "diet", "drinks"]]


# In[6]:


# sampling without replacement
df = dating_profiles.sample(n=100)
df.sample(n=20, replace=True).sort_values(by="age")[["age", "status", "sex", "orientation", "diet", "drinks"]]


# In[7]:


df.shape


# ## Measures of Central Tendency
# 
# 

# ### Percentiles
# 
# A percentile is a value below which there is a given percentage of values in the data.

# In[8]:


dating_profiles['age'].describe()


# ### Median 
# 
# The median is the 50th percentile

# In[9]:


import numpy as np
data = [0, 1, 3, 2, 4, 5, 7, 6, 8, 9, 10, 11]
print("Median (odd number of observations):", np.median(data))
print(Series(data).quantile(0.50))

data = [0, 1, 3, 2, 4, 5, 7, 6, 8, 9, 10, 11, 12]
print("Median (odd number of observations):", np.median(data))
print(Series(data).quantile(0.50))


# In[244]:


data = np.arange(12)
print("Data:", data)
print("Length:", len(data))

data = np.sort(data)
middle_element = (len(data) + 1)/2

if len(data)%2==0:
    lower = int(np.floor(middle_element))-1
    upper = int(np.ceil(middle_element))-1
    median = (data[lower] + data[upper])/2
    print("Median:", median)
else:
    index= int(middle_element)-1
    print("Median:", data[index])
    
print("Median computed by numpy:", np.median(data))


# In[245]:


data = np.arange(13)
print("Data:", data)
print("Length:", len(data))

data = np.sort(data)
middle_element = (len(data) + 1)/2

if len(data)%2==0:
    lower = int(np.floor(middle_element))-1
    upper = int(np.ceil(middle_element))-1
    median = (data[lower] + data[upper])/2
    print("Median:", median)
else:
    index= int(middle_element)-1
    print("Median:", data[index])
    
print("Median computed by numpy:", np.median(data))


# ### Mean
# 
# The most common summary statistic is the mean, which is meant to describe the central tendency of the data.
# 
# The population mean is expressed by-
# $$ \mu = \frac{1}{N} \sum_{i=1}^{N} x_i$$
# 
# Similarly, the sample mean is expressed by-
# $$ \overline{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

# In[229]:


data = np.arange(13)
print("Data:", data)

sum_of_observations = np.sum(data)
mean = sum_of_observations/len(data)
print("Mean:", mean)


# ## Measures of Spread (Variability)

# ### Range
# It is the difference between highest and lowest values of a variable

# In[321]:


data = np.arange(5,13)
print("Data:", data)

print("Range:", max(data)-min(data))


# ### Variance
# 
# The variance is a summary statistic intended to describe the variability or spread of a distribution.
# 
# $$\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2$$
# 
# $$S^2 = \frac{1}{n - 1} \sum_{i=1}^{n} (x_i - \overline{x})^2$$
# 
# The term $x_i - \overline{x}$ is called the deviation from the mean, so the variance is the mean squared deviation. 

# In[223]:


data = np.arange(13)
mean = np.mean(data)
mean_deviations = data - mean
mean_deviations_squared = np.square(mean_deviations)
sum_mean_deviations_squared = np.sum(mean_deviations_squared)
variance = (1/(len(data))) * sum_mean_deviations_squared

print("Data:", data)
print("Mean deviations:", mean_deviations)
print("Mean deviations squared:", mean_deviations_squared)
print("Sum of squared deviations:", sum_mean_deviations_squared)
print("Variance:", variance)


# In[219]:


np.var(data)


# ### Standard deviation
# 
# The square root of the variance, S, is the standard deviation.
# $$\sigma = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu})^2$$
# 
# $$S = \sqrt{ \frac{1}{n - 1} \sum_{i=1}^{n} (x_i - \overline{x}})^2$$

# In[224]:


data = np.arange(13)
mean = np.mean(data)
mean_deviations = data - mean
mean_deviations_squared = np.square(mean_deviations)
sum_mean_deviations_squared = np.sum(mean_deviations_squared)
variance = (1/(len(data))) * sum_mean_deviations_squared

print("Data:", data)
print("Mean deviations:", mean_deviations)
print("Mean deviations squared:", mean_deviations_squared)
print("Sum of squared deviations:", sum_mean_deviations_squared)
print("Variance:", variance)
print("Standard deviation:", np.sqrt(variance))


# In[225]:


np.std(data)


# ### Interquartile range
# 
# The IQR is computed as $IQR = Q3 - Q1$, where Q1 and Q3 are the 25th and 75th percentiles.

# In[227]:


from pandas import Series
data = np.arange(13)
Q1 = Series(data).quantile(0.25)
Q3 = Series(data).quantile(0.75)
print("Data:", data)
IQR = Q3 - Q1
print("Inter-Quartile Range:", IQR)


# # Exploratory Data Analysis

# ### Sampling from Population

# In[279]:


from matplotlib import pyplot as plt
flights = read_csv('course-data/flight-delays.csv')
flights.head(5)[['YEAR', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_NUM', 'ORIGIN',
        'DEST', 'DEP_TIME', 'DEP_DELAY', 'ARR_TIME', 'ARR_DELAY']]


# In[277]:


flights.columns


# In[162]:


delay_bins = np.arange(-50, 201, 10)
flights[['ARR_DELAY']].plot.hist(bins = delay_bins, figsize=(10, 5))
plt.title('Population')


# In[161]:


delta = flights.sample(1000)
delay_bins = np.arange(-50, 201, 10)
delta[['ARR_DELAY']].plot.hist(bins = delay_bins, 
                               color = 'skyblue',
                               figsize=(10, 5))
plt.title('Sample: 1000 randomly drawn observations')


# ## Exploring the Dating Profiles

# In[330]:


from pandas import read_csv
dating_profiles = read_csv("course-data/dating_profiles.csv")

preferred_columns = ['age', 'status', 'sex', 'orientation', 'body_type', 
                     'diet', 'drinks', 'drugs', 'education', 'ethnicity', 
                     'height', 'income', 'job', 'last_online', 'location', 'sign']

dating_profiles = dating_profiles[preferred_columns]
dating_profiles[['age', 'status', 'sex', 'orientation', 'body_type']].head(4)


# In[331]:


dating_profiles.shape


# In[371]:


dating_profiles['age']


# In[80]:


print("Mean:", dating_profiles["age"].mean())
print("Median:", dating_profiles["age"].median())
print("Max:", dating_profiles["age"].max())
print("Min:", dating_profiles["age"].min())


# In[17]:


dating_profiles.describe()


# In[18]:


dating_profiles["age"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])


# In[193]:


print('Range of age of individuals present on dating app:', 
      max(dating_profiles["age"])- min(dating_profiles["age"]))


# In[194]:


# perform the analysis of range of income of individuals


# In[ ]:





# In[195]:


Q1 = dating_profiles["age"].quantile(0.25)
Q3 = dating_profiles["age"].quantile(0.75)
IQR = Q3 - Q1
IQR


# In[ ]:





# ## Single Variables

# In[322]:


import seaborn as sns
sns.set(
        style="darkgrid",
        rc={"figure.figsize": (10, 5)})
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


# ## Multiple Variables

# ### Exploring two categorical variables

# In[280]:


cars = read_csv('course-data/audi.csv')
cars.head()


# In[288]:


sns.set(style="darkgrid",
        rc={"figure.figsize": (10, 8)})
sns.catplot(data=cars, x="transmission", y="engineSize", hue="fuelType", kind="bar")


# In[ ]:





# In[312]:


crosstab = cars[["transmission", "fuelType", "engineSize"]].pivot_table(index='transmission', columns='engineSize',
                            aggfunc=lambda x: len(x), margins=True)

df = crosstab.loc['Automatic':'Manual',:].copy()
df


# In[308]:


df['fuelType']['All']


# In[311]:


df.loc[:,'0':'1.5'] = df.loc[:,'0':'1.5'].div(df['fuelType']['All'], axis=0)
df['fuelType']['All'] = df['fuelType']['All'] / sum(df['fuelType']['All'])
perc_crosstab = df
perc_crosstab


# ### Exploring two numerical variables

# In[313]:


cars.plot.scatter(x='price', 
                            y='mileage', 
                            color = 'royalblue',
                            figsize=(10, 4))


# In[384]:


sns.set_style("darkgrid")
sns.regplot(data= cars, 
            x= 'price', y= 'mileage',
            order=5,
            scatter_kws={"color": "skyblue"}, line_kws={"color": "tomato"},
            )


# In[ ]:





# ### Exploring a numeric variable along with a categorical variable

# In[115]:


sns.set(style="darkgrid",
        rc={"figure.figsize": (12, 8)})
sns.boxplot(data= dating_profiles,
            x= 'age', y= 'job')


# In[385]:


dating_profiles[dating_profiles['age']>100]


# In[ ]:




