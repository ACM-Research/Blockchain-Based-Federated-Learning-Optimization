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
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from model import Helpers, Titanic_Model_1
from torch.utils.data import TensorDataset, random_split, DataLoader

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report



def work(x = 0):
        
    train = pd.read_csv('./train.csv')
    test = pd.read_csv('./test.csv')
    y = train["Survived"]

    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(train[features])
    X_test = pd.get_dummies(test[features])

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)
    predictions = model.predict(X_test)
    train_predictions = model.predict(X)
    
    print(classification_report(y, train_predictions))

    print("hi")


# https://www.tensorflow.org/federated/tutorials/building_your_own_federated_learning_algorithm
def fedAvg(w1, w2):
    return (w1 + w2) / 2


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