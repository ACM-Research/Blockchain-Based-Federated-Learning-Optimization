from time import sleep
import requests
import sys
import socket
import random
import threading

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import model

server = "localhost:3000"
address = "XXXXXXXXX"
contract_loc = ""

print(sys.argv)
if len(sys.argv) > 1:
    server = sys.argv[1]

serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("localhost", random.randint(3001, 4000)))

serv.connect(("localhost", 3000))
# send port to server
serv.send(str(s.getsockname()[1]).encode())

parent = {}
children = []
maxChildren = 2

temp = serv.recv(1024).decode()
if temp != "None":
    parent = {"ip": temp.split(" ")[0], "port": temp.split(" ")[1]}
    print("parent", parent)
else:
    print("ROOT")

# CREATE NEW THREAD TO WORK


from sklearn.model_selection import train_test_split
from sklearn import metrics

from model import Helpers, Titanic_Model_1
from torch.utils.data import TensorDataset, random_split, DataLoader

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import os

import torch
from torch.autograd import Variable
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


def work(x = 0):
    
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
        print(loss)
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

        print('epoch {}, loss {}'.format(epoch, loss.item()))
    
    # save model
    torch.save(model, './model.pth')

    print("hi")


# https://www.tensorflow.org/federated/tutorials/building_your_own_federated_learning_algorithm
def fedAvg(m1, m2):
    # average the parameters of the two models
    for param1, param2 in zip(m1.parameters(), m2.parameters()):
        param1.data = (param1.data + param2.data) / 2
    
    return m1


threading.Thread(target=work).start()

while True:
    temp = serv.recv(1024).decode()
    if temp != "None":
        children.append({"ip": temp.split(" ")[0], "port": temp.split(" ")[1]})
        print("children", children)
    if len(children) == maxChildren:
        break

while True:
    temp = serv.recv(1024).decode()






# receive response from server (id, port, parentIp, parentPort)
# print(r.text)
# id, port, parentIp, parentPort = r.text.split(" ")

# create socket








# # connect to parent
# if parentIp != "None":
#     s.connect((parentIp, int(parentPort)))

#     # send id to parent
#     s.send(id.encode())

#     # receive response from parent
#     print(s.recv(1024).decode())

# # accept connections from children
# while True:
#     conn, addr = s.accept()
#     print("connected to", addr)

#     # receive id from child
#     childId = conn.recv(1024).decode()
#     print("received id", childId)

#     # send id to child
#     conn.send(id.encode())

#     # receive response from child
#     print(conn.recv(1024).decode())