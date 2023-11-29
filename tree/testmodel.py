import numpy as np

#%matplotlib inline

import torch

import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd

import os
import io

import torch
from torch.autograd import Variable

# open a file to append logs to
f = open("./logs.txt", "a")


# LINEAR REGRESSION MODEL
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


working = True
model = None




def find_accuracy():
    global model
    global working
    
    inputDim = 3        # takes variable 'x' 
    outputDim = 1       # takes variable 'y'
    learningRate = 0.01 
    epochs = 100
        
    train = pd.read_csv('./train.csv')
    # shuffle data
    train = train.sample(frac=1)
    
    train.dropna(subset=['Age'], inplace=True)
    
    test = pd.read_csv('./test.csv')
    y_train = train["Age"]

    features = ["Pclass", "SibSp", "Parch"]
    x_train = pd.get_dummies(train[features])
    X_test = pd.get_dummies(test[features])
    
    # train to numpy array
    x_train = x_train.to_numpy().astype(np.float32)
    y_train = y_train.to_numpy().astype(np.float32)
    
    # expand y_train so that it iss 714, 1
    y_train = np.expand_dims(y_train, axis=1)

    if model == None:
        # get model if exists
        if os.path.exists('./model.pth'):
            model = torch.load('./model.pth')
        else:
            model = linearRegression(inputDim, outputDim)
        
        if torch.cuda.is_available():
            model.cuda()
            
    
    # find model accuracy
    correct = 0
    total = 0
    
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(x_train).cuda())
        labels = Variable(torch.from_numpy(y_train).cuda())
    else:
        inputs = Variable(torch.from_numpy(x_train))
        labels = Variable(torch.from_numpy(y_train))
        
    
    with torch.no_grad():
        for i, data in enumerate(inputs):
            real = labels[i].item()
            predicted = model(data).item()
            
            print(predicted, real)
            if abs(predicted - real) < 5:
                correct += 1
            total += 1
            
    print("Accuracy: ", round(correct/total * 100, 3))



# TRAIN THE MODEL
def work():
    global model
    global working
    
    inputDim = 3        # takes variable 'x' 
    outputDim = 1       # takes variable 'y'
    learningRate = 0.01 
    epochs = 100
        
    train = pd.read_csv('./train.csv')
    
    train.dropna(subset=['Age'], inplace=True)
    
    test = pd.read_csv('./test.csv')
    y_train = train["Age"]

    features = ["Pclass", "SibSp", "Parch"]
    x_train = pd.get_dummies(train[features])
    X_test = pd.get_dummies(test[features])
    
    # train to numpy array
    x_train = x_train.to_numpy().astype(np.float32)
    y_train = y_train.to_numpy().astype(np.float32)

    # expand y_train so that it iss 714, 1
    y_train = np.expand_dims(y_train, axis=1)
    print(y_train.shape)

    if model == None:
        # get model if exists
        if os.path.exists('./model.pth'):
            model = torch.load('./model.pth')
        else:
            model = linearRegression(inputDim, outputDim)
        
        if torch.cuda.is_available():
            model.cuda()
    
    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    for epoch in range(epochs):
        # Converting inputs and labels to Variable
        if torch.cuda.is_available():
            inputs = Variable(torch.from_numpy(x_train).cuda())
            labels = Variable(torch.from_numpy(y_train).cuda())
        else:
            inputs = Variable(torch.from_numpy(x_train))
            labels = Variable(torch.from_numpy(y_train))

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(inputs)

        # get loss for the predicted output
        loss = criterion(outputs, labels)
        # print(loss)
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()
        
        # log loss
        f.write('epoch {}, loss {}\n'.format(epoch, loss.item()))

        # print('epoch {}, loss {}'.format(epoch, loss.item()))
    
    # save model
    # torch.save(model, './model.pth')    
    
    buffer = io.BytesIO()
    torch.save(model, buffer)

    print("DONE TRAINING")
    working = False
    
work()
find_accuracy()