#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# ### Getting Set up

# In[2]:


from sklearn.datasets import load_iris#save data information as variable
iris = load_iris()#view data description and information
print(iris.DESCR)


# ## Putting Data into a DataFrame

# ### Feature Data

# In[3]:


data = pd.DataFrame(iris.data)
data.head()


# In[4]:


data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
#note: it is common practice to use underscores between words, and avoid spaces
data.head()


# ### Target Data

# In[5]:


#put target data into data frame
target = pd.DataFrame(iris.target)#Lets rename the column so that we know that these values refer to the target values
target = target.rename(columns = {0: 'target'})
target.head()


# ### Exploratory Data Analysis (EDA)

# In[7]:


df = pd.concat([data, target], axis = 1)
#note: it is common practice to name your data frame as "df", but you can name it anything as long as you are clear and consistent
#in the code above, axis = 1 tells the data frame to add the target data frame as another column of the data data frame, axis = 0 would add the values as another row on the bottom
df.head()


# ### Data Cleaning

# In[8]:


df.dtypes


# In[9]:


df.isnull().sum()


# In[10]:


df.describe()


# In[11]:


import seaborn as sns
sns.heatmap(df.corr(), annot = True);
#annot = True adds the numbers onto the squares


# In[15]:


import matplotlib.pyplot as plt
# The indices of the features that we are plotting (class 0 & 1)
x_index = 0
y_index = 1
# this formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args:
                              iris.target_names[int(i)])

plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.tight_layout()
plt.show()


# In[16]:


x_index = 2
y_index = 3
# this formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.tight_layout()
plt.show()


# In[17]:


#divide our data into predictors (X) and target values (y)
X = df.copy()
y = X.pop('target')


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify = y)
'''
by stratifying on y we assure that the different classes are represented proportionally to the amount in the total data (this makes sure that all of class 1 is not in the test group only)
'''


# In[20]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# In[21]:


df.target.value_counts(normalize= True)


# In[22]:


from sklearn.linear_model import LogisticRegression
#create the model instance
model = LogisticRegression()
#fit the model on the training data
model.fit(X_train, y_train)
#the score, or accuracy of the model
model.score(X_test, y_test)


# In[23]:


#the test score is already very high, but we can use the cross validated score to ensure the model's strength 
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, cv=10)
print(np.mean(scores))


# ### Understanding the predictions

# In[24]:


df_coef = pd.DataFrame(model.coef_, columns=X_train.columns)
df_coef


# In[25]:


predictions = model.predict(X_test)
#compare predicted values with the actual scores
compare_df = pd.DataFrame({'actual': y_test, 'predicted': predictions})
compare_df = compare_df.reset_index(drop = True)
compare_df


# In[ ]:


from sklearn.metrics import confusion_matrix
pd.DataFrame(confusion_matrix(y_test, predictions, labels=[2, 1, 0]),index=[2, 1, 0], columns=[2, 1, 0])


# In[ ]:





# In[ ]:




