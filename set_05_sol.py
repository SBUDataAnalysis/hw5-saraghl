#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# The multivariate normal, multinormal or Gaussian distribution is a generalization of the one-dimensional normal distribution to higher dimensions. Such a distribution is specified by its mean and
# covariance matrix. These parameters are analogous to the mean (average or "center") and variance (standard deviation, or "width," squared) of the one-dimensional normal distribution.

# # 
# 
# Read the files labeled as 'sampleX.txt' using numpy or pandas and plot them.

# In[2]:


import pandas as pd


# In[3]:


sample1 = pd.read_csv(r'C:\Users\RTS\Downloads\hw5-saraghl-main\hw5-saraghl-main\sample1.txt',encoding = 'UTF8', delimiter='\t')
sample2 = pd.read_csv(r'C:\Users\RTS\Downloads\hw5-saraghl-main\hw5-saraghl-main\sample2.txt',encoding = 'UTF8', delimiter='\t')
sample3 = pd.read_csv(r'C:\Users\RTS\Downloads\hw5-saraghl-main\hw5-saraghl-main\sample3.txt',encoding = 'UTF8', delimiter='\t')


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(18, 3))

plt.subplot(1,3,1)
plt.scatter(sample1.iloc[:,0],sample1.iloc[:,1],c = 'b')
plt.title('sample1')
plt.xlabel('x')
plt.ylabel('y')

#plot2
plt.subplot(1,3,2)
plt.scatter(sample2.iloc[:,0],sample2.iloc[:,1],c = 'r')
plt.title('sample2')
plt.xlabel('x')
plt.ylabel('y')

#plot3
plt.subplot(1,3,3)
plt.scatter(sample3.iloc[:,0],sample3.iloc[:,1],c = 'g')
plt.title('sample3')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# 
# Determine whether each sample is correlated, anticorrelated or uncorrelated.

# In[5]:


np.corrcoef(sample3.iloc[:,0],sample3.iloc[:,1])[0,1]


# the correlation of the third sample is really close to 0, so it's uncorrelated with a good approximation.

# In[6]:


np.corrcoef(sample2.iloc[:,0],sample2.iloc[:,1])[0,1]


# the correlation of the second sample is negative, so it's anti-correlated.

# In[7]:


np.corrcoef(sample1.iloc[:,0],sample1.iloc[:,1])[0,1]


# the correlation of the first sample is positive, so it's correlated.

# ## 3d Plot
# Plot the joint probability distribution of each sample in 3D. For this you can use 'plot_surface' found in matplotlib library.

# If you want to plot using matplotlib, the codes below will come in handy. First line makes sure that your plots are interactive, second line provides color maps.

# In[8]:


get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import cm


# At the first step try to get the 2D histogram of your data. (Hint: beware of different sizes of arrays!)

# In[9]:


#plot sample1
plt.figure(figsize=(18, 3))

plt.subplot(1,3,1)
plt.hist2d(sample1['x'],sample1['y'], bins =(100,100),cmap=plt.cm.jet)
plt.title('the 2D histogram of sample 1')

#plot sample2
plt.subplot(1,3,2)
plt.hist2d(sample2['x'],sample2['y'], bins =(100,100),cmap=plt.cm.jet)
plt.title('the 2D histogram of sample 2')

#plot sample3
plt.subplot(1,3,3)
plt.hist2d(sample3['x'],sample3['y'], bins =(100,100),cmap=plt.cm.jet)
plt.title('the 2D histogram of sample 3')
plt.show()


# Now you can plot the 3D histogram:

# In[39]:


z1, x1, y1 = np.histogram2d(sample1['x'], sample1['y'])
z2, x2, y2 = np.histogram2d(sample2['x'], sample2['y'])
z3, x3, y3 = np.histogram2d(sample3['x'], sample3['y'])


# In[ ]:


import matplotlib
matplotlib.rc_file_defaults()

fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')

#Make data suitable to 3d plotting
...

#Plot the surface.
...

# Add a color bar which maps values to colors.
...

plt.show()


# In[ ]:


#Plot Sample 2


# In[ ]:


#Plot sample 3


# 
# Using the calculated histograms, now write a code to calculate the marginalized PDFs along both axes and then plot them.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set() #Use this line to plot the marginalized PDFs with seaborn style

plt.figure(figsize=(15, 10))
plt.suptitle('Marginalized PDFs')

plt.subplot(321)
...
...


plt.show()


# ## Extra example:
# 
# You can also combine the two steps above and plot the joint PDF and the marginalized ones altogether using seaborn.

# In[11]:



g1 = sns.jointplot(data=sample1, x='x', y='y', kind='hist')
g1.fig.suptitle('Sample 1')
g1.fig.tight_layout()

g2 = sns.jointplot(data=sample2, x='x', y='y', kind='hist')
g2.fig.suptitle('Sample 2')
g2.fig.tight_layout()

g3 = sns.jointplot(data=sample3, x='x', y='y', kind='hist')
g3.fig.suptitle('Sample 3')
g3.fig.tight_layout()

plt.show()


# ## Contour Plots

# Plot the contours of the datasets showing different values of contours.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
matplotlib.rc_file_defaults() #Use this line to revert back to matplotlib default style

#Plot the contours

plt.show()


# In[ ]:


#Contour od Sample 2


# In[ ]:


#Contour of Sample 3


# 
# ## 3 parts
# In the multivariate case, a gaussian distribution is defined via a mean and a covrience matrix. Here the covarience matrix is the equivalant of varience in higher dimensions. To refresh your mind, take a look at the [Wikipedia page](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Notation_and_parameterization). A correlation matrix is very similiar but has some [subtle differences](https://en.wikipedia.org/wiki/Correlation#Correlation_matrices). 

# Now using those defenitions, find the **covariance** (part 1) and **correlation** (part 2) matrices for each distribution. Are they the same? why? (part 3)
# 
# (Hint 1: You may find scipy.stats useful)
# 
# (Hint 2: Study the lecture note titled: 'parameter estimation 7' once more)
# 
# (Hint 3: [This lecture note](https://pages.ucsd.edu/~rlevy/lign251/fall2007/lecture_4.pdf) may also be useful, althogh the codes aren't written in python)

# In[12]:


#covariance matrix_sample 1

x1 = sample1['x']
y1 = sample1['y']
cov1 = np.vstack((x1,y1))
print(np.cov(cov1))


# In[13]:


#correlation matrix_sample 1

np.array(sample1.corr())


# In[14]:


#covariance matrix_sample 2

x2 = sample1['x']
y2 = sample2['y']
cov2 = np.vstack((x2,y2))
print(np.cov(cov2))


# In[15]:


#correlation matrix_sample 2

np.array(sample2.corr())


# In[16]:


#covariance matrix_sample 3

x3 = sample3['x']
y3 = sample3['y']
cov3 = np.vstack((x3,y3))
print(np.cov(cov3))


# In[17]:


#correlation matrix_sample 3

np.array(sample3.corr())


# no they are not the same. just the sign of their elements are the same.

# 
# ## 2 parts
# 
# Now, only focus on the positievly correlated distribution. If the errors along both of the axes are huge, (as discussed in the lecture 'parameter estimation 7'), Is there a linear combination of the two parameters that can be well constrained? Discuss it (part 1).  Find the mode of the distribution (part 2)

# In[ ]:





# # Real World
# let's apply this to real world data and using house price data. first import house_data.csv

# In[18]:


df = pd.read_csv(r'C:\Users\RTS\Downloads\hw5-saraghl-main\hw5-saraghl-main\House_price.csv',encoding = 'UTF8', delimiter=',')


# you can see detail of your dataframe with the code below

# In[19]:


df.info()


# Now select the columns of the train set with numerical data

# In[20]:


df_num = df.select_dtypes(include='number')


# In[21]:


df_num.head()


# Plot the distribution of all the numerical data

# In[22]:


num_column = df_num.columns
num_column 


# In[ ]:


plt.figure(figsize = (7,10))
for i in num_column :
    sns.countplot(data = num_column[i]), y = num_column[i],palette='Set2',order =pd.Series(num_column).value_counts().index)
    plt.show()


# In[ ]:


count_val = df_num[num_column].nunique().sort_values(ascending=False)
count_val


# plot Heatmap for all the remaining numerical data including the 'SalePrice'

# In[23]:



corrmat = df_num.corr()
cols = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_num[cols].values.T)
sns.set(font_scale=1)
plt.figure(figsize = (5,5))
hm = sns.heatmap(cm, cbar=True,linewidths=.5, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[48]:


print(type(higher1))


# From the distribution of each numerical variables as well as the heatmap you can notice columns that are important and correlated (correlation higher than absolute 0.3) with our target variable 'SalePrice'. select columns where the correlation with 'SalePrice' is higher than |0.3|
# 

# In[24]:


corr_mat = df_num[num_column].corr() 
higher1 = corr_mat['SalePrice'][abs(corr_mat['SalePrice'])>0.3].index.tolist()
higher1


# Now choose Features with high correlation (higher than 0.5) and plot the correlation of each feature with SalePrice

# In[25]:


corr_mat = df_num[num_column].corr()
higher2 = corr_mat['SalePrice'][abs(corr_mat['SalePrice'])>0.5].index.tolist()
higher2


# Check the NaN of dataframe set by ploting percent of missing values per column and plot the result

# In[26]:


list1= []
for col in df.columns :
    nan = df[col].isnull().sum()
    list1.append(nan)


# In[27]:


data = {'number of nan values':list1, 'name':df.columns}
df_new = pd.DataFrame(data)


# In[45]:


plt.figure(figsize =(10,10))
plt.pie(list1, labels= df.columns)
plt.show()


# in the last session I think Amirreza said that droping Nan cells is not suited in many projects cause
# it can remove alots of information of your dataframe. ofcourse he is right and I would like to give a short introduction to the process of handling Nan cells which is called "Imputation". Data imputation is the substitution of estimated values for missing or inconsistent data items (fields). The substituted values are intended to create a data record that does not fail edits. here you can use Simple_

# In[82]:


# Imputation of missing values (NaNs) with SimpleImputer you can check diffrent strategy 
my_imputer = SimpleImputer(strategy="median")
df_num_imputed = pd.DataFrame(my_imputer.fit_transform(df_num))
df_train_imputed.columns = df_train_num.columns


# # Categorical features

# ## Explore and clean Categorical features

# find all Catagorical columns. you can use the code for finding the numerical columns and just using 'object' for dtype.

# In[30]:


df_catagorical=df.select_dtypes(include=['category'])


# Countplot for each of the categorical features in the train set

# In[ ]:


#Code here

