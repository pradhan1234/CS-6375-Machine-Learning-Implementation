
# coding: utf-8

# In[63]:

#tic tac toe game
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle


# In[64]:

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data"
headers = ['top-left', 'top-middle', 'top-right', 'middle-left', 
           'middle-middle', 'middle-right', 'bottom-left', 'bottom-middle', 
           'bottom-right', 'class']


# In[65]:

dataframe = pd.read_csv(url, names=headers)
dataframe = shuffle(dataframe)


# In[66]:

dataframe.describe()


# In[67]:

#sneekpeak into the dataset.
dataframe[1:10]


# In[68]:

labelEncoder = preprocessing.LabelEncoder()

labelEncoder.fit(dataframe['top-left'])
labelEncoder.fit(dataframe['top-middle'])
labelEncoder.fit(dataframe['top-right'])
labelEncoder.fit(dataframe['middle-left'])
labelEncoder.fit(dataframe['middle-middle'])
labelEncoder.fit(dataframe['middle-right'])
labelEncoder.fit(dataframe['bottom-left'])
labelEncoder.fit(dataframe['bottom-middle'])
labelEncoder.fit(dataframe['bottom-right'])
labelEncoder.fit(dataframe['class'])


# In[69]:

dataframe['top-left'] = labelEncoder.fit_transform(dataframe['top-left'])
dataframe['top-middle'] = labelEncoder.fit_transform(dataframe['top-middle'])
dataframe['top-right'] = labelEncoder.fit_transform(dataframe['top-right'])
dataframe['middle-left'] = labelEncoder.fit_transform(dataframe['middle-left'])
dataframe['middle-middle'] = labelEncoder.fit_transform(dataframe['middle-middle'])
dataframe['middle-right'] = labelEncoder.fit_transform(dataframe['middle-right'])
dataframe['bottom-left'] = labelEncoder.fit_transform(dataframe['bottom-left'])
dataframe['bottom-middle'] = labelEncoder.fit_transform(dataframe['bottom-middle'])
dataframe['bottom-right'] = labelEncoder.fit_transform(dataframe['bottom-right'])
dataframe['class'] = labelEncoder.fit_transform(dataframe['class'])


# In[70]:

#dataframe.describe()


# In[71]:

#let's split the dataframe into X and y
X = dataframe[headers[:-1]]
y = dataframe[headers[-1]]

#Split the dataset into training and testing sets resp.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=20)


# In[72]:

scaler = StandardScaler()
scaler.fit(X_train)


# In[73]:

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#X_train


# In[74]:

#Dimensions of training set and testing set, created by splitting dataset.
print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)


# In[75]:

mlp = MLPClassifier(activation = 'relu', solver = 'lbfgs', hidden_layer_sizes=(9,7,5))
mlp.fit(X_train,y_train)


# In[76]:

predictions = mlp.predict(X_train)
print("Confusion Matrix for testing on Training Set:")
print(confusion_matrix(y_train,predictions))


# In[77]:

print("Classification Report on training set\n\n")
print(classification_report(y_train,predictions))


# In[78]:

predictions = mlp.predict(X_test)
print("Confusion Matrix for testing on Testing Set:")
print(confusion_matrix(y_test,predictions))


# In[79]:

print("Classification Report on testing set\n\n")
print(classification_report(y_test,predictions))


# In[ ]:




# In[ ]:



