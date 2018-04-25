
# coding: utf-8

# In[29]:

#Import the Multi Layer Perceptron Model
from sklearn.neural_network import MLPClassifier


# In[30]:

#create a data set for our NN to learn XOR function
'''
y = (x1 AND x2') OR (x1' AND x2)

x1 x2 y
0  0  0
0  1  1
1  0  1
1  1  0
'''

X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]
y = [0, 1, 1, 0]


# In[31]:

#build classifier
classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2,), random_state=20)


# In[32]:

#learn data
classifier.fit(X, y)


# In[33]:

#intercepts represents the biases between layers

#bias from input layer to hidden layer 1
print("Bias from input layer to layer 1:", classifier.intercepts_[0])

#bias from hidden layer 1 to output layer 
print("Bias from input layer 1 to output layer:", classifier.intercepts_[1])


# In[34]:

#weights from input layer to hidden layer 1
print("Weights from i/p layer to layer 1:\n", classifier.coefs_[0])

#weights from hidden layer 1 to output layer
print("\nWeights from layer 1 to output layer:\n", classifier.coefs_[1])


# In[35]:

#loss of the classifier
print("Loss: ", classifier.loss_)


# In[36]:

#predictions
classifier.predict([[0, 0], [0, 1], [1,0], [1,1]])


# In[37]:

print("XOR Predictions:")
print("0 0 ", classifier.predict([[0, 0]]))
print("0 1 ", classifier.predict([[0, 1]]))
print("1 0 ", classifier.predict([[1, 0]]))
print("1 1 ", classifier.predict([[1, 1]]))


# In[ ]:



