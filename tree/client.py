from time import sleep
import requests
import sys
import socket
import random
import threading

# Helper libraries
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

import pickle
import struct

data_identifiers = {"info": 0, "data": 1, "image": 2}
isRoot = False
waiting = True

def send_data(conn, payload, data_id=0):
    """
    @brief: send payload along with data size and data identifier to the connection
    @args[in]:
        conn: socket object for connection to which data is supposed to be sent
        payload: payload to be sent
        data_id: data identifier
    """
    # serialize payload
    serialized_payload = pickle.dumps(payload)
    # send data size, data identifier and payload
    conn.sendall(struct.pack(">I", len(serialized_payload)))
    conn.sendall(struct.pack(">I", data_id))
    conn.sendall(serialized_payload)


def receive_data(conn):
    """
    @brief: receive data from the connection assuming that
        first 4 bytes represents data size,
        next 4 bytes represents data identifier and
        successive bytes of the size 'data size'is payload
    @args[in]:
        conn: socket object for conection from which data is supposed to be received
    """
    # receive first 4 bytes of data as data size of payload
    data_size = struct.unpack(">I", conn.recv(4))[0]
    # receive next 4 bytes of data as data identifier
    data_id = struct.unpack(">I", conn.recv(4))[0]
    # receive payload till received payload size is equal to data_size received
    received_payload = b""
    reamining_payload_size = data_size
    while reamining_payload_size != 0:
        received_payload += conn.recv(reamining_payload_size)
        reamining_payload_size = data_size - len(received_payload)
    payload = pickle.loads(received_payload)
    return (data_id, payload)

server = "localhost:3000"
address = "XXXXXXXXX"
contract_loc = ""

print(sys.argv)
if len(sys.argv) > 1:
    server = sys.argv[1]

serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
parentConn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("localhost", random.randint(3001, 4000)))
s.listen(2)

serv.connect(("localhost", 3000))
# send port to server
serv.send(str(s.getsockname()[1]).encode())

parent = {}
children = []
maxChildren = 2

temp = serv.recv(1024).decode()
if temp != "None":
    parent = {"ip": temp.split(" ")[0], "port": temp.split(" ")[1]}
    parentConn.connect((parent["ip"], int(parent["port"])))
    print("parent", parent)
else:
    print("ROOT")
    isRoot = True


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

working = True
model = None
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

        # print('epoch {}, loss {}'.format(epoch, loss.item()))
    
    # save model
    torch.save(model, './model.pth')    
    
    buffer = io.BytesIO()
    torch.save(model, buffer)
    
    if parent != {} and len(children) != maxChildren:
        send_data(parentConn, buffer.getvalue(), data_identifiers["data"])

    print("DONE TRAINING")
    working = False


# https://www.tensorflow.org/federated/tutorials/building_your_own_federated_learning_algorithm
# def fedAvg2(m1, m2):
#     print("WORKING 2")
#     print(m1)
#     print(m2)
#     # average the parameters of the two models
#     for param1, param2 in zip(m1.parameters(), m2.parameters()):
#         param1.data = (param1.data + param2.data) / 2
    
#     torch.save(m1, './avg.pth')
#     return m1

child1model = None
def fedAvg3(m1, m2, m3):
    global child1model
    
    while working:
        sleep(1)
    
    print("WORKING 3")
    print(m1)
    print(m2)
    print(m3)
    # average the parameters of the two models
    for param1, param2, param3 in zip(m1.parameters(), m2.parameters(), m3.parameters()):
        param1.data = (param1.data + param2.data + param3.data) / 3
    
    torch.save(m1, './avg.pth')
    
    if not isRoot:
        buffer = io.BytesIO()
        torch.save(m1, buffer)
        # SEND UP THE TREE ONCE AVG IS DONE
        send_data(parentConn, buffer.getvalue(), data_identifiers["data"])
    child1model = None
    return m1

def fedAvg2(m1, m2):
    
    while working:
        sleep(1)
    
    print("WORKING 2")
    print(m1)
    print(m2)
    # average the parameters of the two models
    for param1, param2 in zip(m1.parameters(), m2.parameters()):
        param1.data = (param1.data + param2.data) / 2
        
    torch.save(m1, './avg.pth')
    
    if not isRoot:
        buffer = io.BytesIO()
        torch.save(m1, buffer)
        # SEND UP THE TREE ONCE AVG IS DONE
        send_data(parentConn, buffer.getvalue(), data_identifiers["data"])
    return m1

children = []


def start_epoch():
    # CREATE NEW THREAD TO WORK
    threading.Thread(target=work).start()
    
def send_broadcast_down(payload):
    # SEND MSG TO CHILDREN
    for child in children:
        send_data(child["conn"], payload, data_identifiers["info"])
    


def handle_client(conn, conn_name):
    """
    @brief: handle the connection from client at seperate thread
    @args[in]:
        conn: socket object of connection
        con_name: name of the connection
    """
    global child1model
    while True:
        try:
            if isRoot and waiting:
                continue
            
            data_id, payload = receive_data(conn)
            # if data identifier is image then save the image
            if data_id == data_identifiers["image"]:
                print("---Recieved image too ---")
            # otherwise send the data to do something
            elif data_id == data_identifiers["data"]:
                print("---Recieved data too ---")
                # payload bytes to model
                buffer = io.BytesIO(payload)
                buffer.seek(0)
                clientM = torch.load(buffer)
                if len(children) == 1:
                    fedAvg2(model, clientM)
                else:
                    if child1model != None:
                        fedAvg3(model, clientM, child1model)
                    else:
                        child1model = clientM
            else:
                # if data is 'bye' then break the loop and client connection will be closed
                if payload == "bye":
                    print(
                        "[INFO]: {} requested to close the connection".format(conn_name)
                    )
                    print("[INFO]: Closing connection with {}".format(conn_name))
                    break
                elif payload == "start":
                    start_epoch()
                    send_broadcast_down("start")
                else:
                    print(payload)
        except KeyboardInterrupt:
            if (isRoot):
                send_broadcast_down("start")
                start_epoch()
    conn.close()

def connectToChildren():
    while True:
        try:
            # accept client connection
            # if first message from client match the defined message
            # then handle it at seperate thread
            # otherwise close the connection
            conn, (address, port) = s.accept()
            conn_name = "{}|{}".format(address, port)
            children.append({"ip": address, "port": port, "conn": conn})
            print("[INFO]: Accepted the connection from {}".format(conn_name))
            threading.Thread(target=handle_client, args=(conn, conn_name)).start()
            
        # break the while loop when keyboard intterupt is received and server will be closed
        except KeyboardInterrupt:
            print("\n[INFO]: Keyboard Interrupt Received")
            break

    s.close()
    print("[INFO]: Server Closed")




threading.Thread(target=connectToChildren).start()



if isRoot:
    input("Press Enter to start the tree...\n\n")
    start_epoch()
    send_broadcast_down("start")
    waiting = False
else:
    while True:
        # receive from parent
        data_id, payload = receive_data(parentConn)
        if payload == "start":
            start_epoch()
            send_broadcast_down("start")


# LISTEN TO SERVER FOR CHILDREN

# while True:
#     temp = serv.recv(1024).decode()
#     if temp != "None":
#         children.append({"ip": temp.split(" ")[0], "port": temp.split(" ")[1]})
#         print("children", children)
#     if len(children) == maxChildren:
#         break
