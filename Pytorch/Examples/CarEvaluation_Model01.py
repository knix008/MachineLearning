import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
from sklearn.datasets import load_iris
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt #importing graph plotting functionality
# import os
# print(os.listdir("../input"))
# Load dataset
df = pd.read_csv("Datasets/car_evaluation.csv", names = ["buying","maint", "doors", "persons", "lug_boot","safety","class"])

## get_dummies() implementation
category_col =["buying","maint", "doors", "persons", "lug_boot","safety","class"] 
df = pd.get_dummies(df, columns=category_col)
df.to_csv('Datasets/car_evaluation_preprocessed.csv',index=False)
## visualizing processed dataset
print(df.shape)
df.head(10)

X = df.iloc[:, 0:21].values
y = df.iloc[:, 21:].values
## Normalizing data - Normalization refers to rescaling real valued numeric attributes into the range 0 and 1.
X = preprocessing.scale(X)
from sklearn.model_selection import train_test_split
feature_train, feature_test, labels_train, labels_test = train_test_split(X, y, random_state = 42)
print ("Train:%d +  Test:%d = Total:%d"  % (len(feature_train),len(feature_test),len(feature_train)+len(feature_test)))

feature_train_v = Variable(torch.FloatTensor(feature_train), requires_grad = False)
labels_train_v = Variable(torch.FloatTensor(labels_train), requires_grad = False)
feature_test_v = Variable(torch.FloatTensor(feature_test), requires_grad = False)
labels_test_v = Variable(torch.FloatTensor(labels_test), requires_grad = False)

class LinearClassifier(nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.h_layer = nn.Linear(21, 4) #21 input layers and 4 output layers
        self.s_layer = nn.Softmax()
    def forward(self,x):
        y = self.h_layer(x)
        p = self.s_layer(y)
        return p
#declaring the classifier to an object
model = LinearClassifier()   
#calculates the loss
loss_fn = nn.BCELoss()       
optim = torch.optim.SGD(model.parameters(), lr = 0.01)

all_losses = []
for num in range(5000): 
    pred = model(feature_train_v) #predict
    loss = loss_fn(pred, labels_train_v) #calculate loss
    all_losses.append(loss.data)
    optim.zero_grad() #zero gradients to not accumulate
    loss.backward() #update weights based on loss
    optim.step() #update optimiser for next iteration
    

all_losses = np.array(all_losses, dtype = np.float)
all_losses
plt.plot(all_losses)
plt.show()
print(pred[3])
print(labels_train_v[3])
print(all_losses[-1])


from sklearn.metrics import accuracy_score
predicted_values = []
for num in range(len(feature_test_v)):
    predicted_values.append(model(feature_test_v[num]))

    
    score = 0
for num in range(len(predicted_values)):
    if np.argmax(labels_test[num]) == np.argmax(predicted_values[num].data.numpy()):
        score = score + 1
accuracy = float(score / len(predicted_values)) * 100
print ('Testing Accuracy Score is ' + str(accuracy))