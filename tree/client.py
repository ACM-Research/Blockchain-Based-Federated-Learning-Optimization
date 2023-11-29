from time import sleep
import json
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

from brownie import *
from web3 import Web3

import pickle
import struct

import time


# open a csv file to append logs to (for testing)
# model_accuracy = open("./data/model_accuracy.csv", "a")
# gas_costs = open("./data/gas_costs.csv", "a")
# speed = open("./data/speed.csv", "a")

start_time = None

test_server = False
iteration = 0

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
contract_address = "0x3194cBDC3dbcd3E11a07892e7bA5c3394048Cc87"
contract_abi = None
task_id = 0

user_address = "0x71C7656EC7ab88b098defB751B7401B5f6d8976F"

if (os.path.exists("./MainContract.json")):
    with open("./MainContract.json") as f:
        contract_abi = json.load(f)
        print("Loaded contract abi from file")
        contract_abi = contract_abi["abi"]
        # print(contract_abi)
else:
    print("AAA")

print(sys.argv)
if len(sys.argv) > 1:
    server = sys.argv[1]


# IMPORTANT CONNECTIONS
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
parentConn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
port = random.randint(3001, 4000)
s.bind(("localhost", port))
print("IP is ", s.getsockname()[0], "PORT is ", s.getsockname()[1])
s.listen(2)

tree = None
contract = None

if test_server:
    # TEST SERVER CONNECTION
    serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    serv.connect(("localhost", 3000))
    # send port to server
    serv.send(str(s.getsockname()[1]).encode())

    parent = {}
    children = []
    maxChildren = 2

    # GET PARENT FROM SERVER
    temp = serv.recv(1024).decode()
    if temp != "None":
        parent = {"ip": temp.split(" ")[0], "port": temp.split(" ")[1]}
        parentConn.connect((parent["ip"], int(parent["port"])))
        print("parent", parent)
    else:
        print("ROOT")
        isRoot = True
else:
    # connect to web3 brownie contract
    
    
    w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))
    w3.eth.defaultAccount = w3.eth.accounts[0]
    # get user address
    user_address = w3.eth.accounts[0]
    print("User address", user_address)
    
    # Create a contract instance
    contract = w3.eth.contract(address=contract_address, abi=contract_abi)
    
    # get nextTaskId public variable from contract (this is the task id)
    
    event_filter = contract.events.TreeStructureGenerated.createFilter(fromBlock='latest')
    
    tx = contract.functions.addUser(task_id, user_address, str(s.getsockname()[0]) + ":" + str(s.getsockname()[1])).transact()
    # get gas used
    receipt = w3.eth.getTransactionReceipt(tx)
    gas_used = receipt['gasUsed']
    gas_costs = open("./data/gas_costs.csv", "a")
    gas_costs.write(str(port) + ", addUser, " + str(gas_used) + "\n")
    gas_costs.close()
    # Wait for the event
    while True:
        for event in event_filter.get_new_entries():
            # print("Event", event)
            if event["args"]["taskId"] == task_id:
                tree = event["args"]["tree"]
                break
        if tree != None:
            break
        
        sleep(1)
    
    index = 0
    for i, node in enumerate(tree):
        if node[2] == str(s.getsockname()[0]) + ":" + str(s.getsockname()[1]):
            index = i
            break
        
    print("INDEX", index + 1)
    parentIndex = int((index + 1) / 2)
    print("PARENT INDEX", parentIndex)
        
    parent = {}
    children = []
    maxChildren = 2

    if parentIndex != 0:
        sleep(3)
        parent = {"ip": tree[parentIndex-1][2].split(":")[0], "port": tree[parentIndex-1][2].split(":")[1]}
        parentConn.connect((parent["ip"], int(parent["port"])))
        print("parent", parent)
    else:
        print("ROOT")
        isRoot = True
    

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
children = []

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
            
            # print(predicted, real)
            if abs(predicted - real) < 5:
                correct += 1
            total += 1
            
    print("Accuracy: ", round(correct/total * 100, 3))
    model_accuracy = open("./data/model_accuracy.csv", "a")
    model_accuracy.write(str(iteration) + ", " + str(correct/total * 100) + "\n")
    model_accuracy.close()

# TRAIN THE MODEL
def work():
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
    
    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    loss = None
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
    
    # save model
    # torch.save(model, './model.pth')        
    
    buffer = io.BytesIO()
    torch.save(model, buffer)
    
    if parent != {} and len(children) != maxChildren:
        send_data(parentConn, buffer.getvalue(), data_identifiers["data"])

    print("DONE TRAINING")
    working = False

child1model = None
finished = False
# FEDAVG WITH 2 CHILDREN
def fedAvg3(m1, m2, m3):
    global child1model
    global finished
    global model
    
    while working:
        sleep(1)
    
    print("AVERAGING 3 MODELS")
    # print(m1)
    # print(m2)
    # print(m3)
    # average the parameters of the two models
    for param1, param2, param3 in zip(m1.parameters(), m2.parameters(), m3.parameters()):
        param1.data = (param1.data + param2.data + param3.data) / 3
    
    if not isRoot:
        buffer = io.BytesIO()
        torch.save(m1, buffer)
        # SEND UP THE TREE ONCE AVG IS DONE
        send_data(parentConn, buffer.getvalue(), data_identifiers["data"])
    else:
        torch.save(m1, './avg.pth')
        model = m1
        threading.Thread(target=find_accuracy).start()
        
        for child in children:
            send_data(child["conn"], {"type": "model", "model": m1}, data_identifiers["data"])
        finished = True
    child1model = None
    return m1

# FEDAVG WITH 1 CHILD
def fedAvg2(m1, m2):
    global finished
    global model
    
    while working:
        sleep(1)
    
    print("AVERAGING 2 MODELS")
    # print(m1)
    # print(m2)
    # average the parameters of the two models
    for param1, param2 in zip(m1.parameters(), m2.parameters()):
        param1.data = (param1.data + param2.data) / 2
        
    
    if not isRoot:
        buffer = io.BytesIO()
        torch.save(m1, buffer)
        # SEND UP THE TREE ONCE AVG IS DONE
        send_data(parentConn, buffer.getvalue(), data_identifiers["data"])
    else:
        torch.save(m1, './avg.pth')
        model = m1
        threading.Thread(target=find_accuracy).start()
        for child in children:
            send_data(child["conn"], {"type": "model", "model": m1}, data_identifiers["data"])
        finished = True
    return m1





# CREATE NEW THREAD TO TRAIN THE MODEL
def start_epoch():
    threading.Thread(target=work).start()
    
# SEND MESSAGE TO ALL CHILDREN
def send_broadcast_down(payload):
    # SEND MSG TO CHILDREN
    for child in children:
        send_data(child["conn"], payload, data_identifiers["info"])
    
def send_up(payload):
    send_data(parentConn, payload, data_identifiers["info"])

# HANDLE MESSAGES FROM CHILDREN (AWAIT MODEL DATA)
def handle_client(conn, conn_name):
    """
    @brief: handle the connection from client at seperate thread
    @args[in]:
        conn: socket object of connection
        con_name: name of the connection
    """
    global child1model
    while True:
        if restructuring:
            break
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
        except OSError:
            print("[INFO]: {} forcibly closed the connection".format(conn_name))
            return
        sleep(0.1)
    conn.close()

restructuring = False
stopThreads = False

threads = []

# LISTEN FOR CONNECTIONS FROM CHILDREN AND CREATE A THREAD FOR EACH
def connectToChildren(s, stpThread):
    print("[INFO]: Waiting for connections...")
    global children
    global parentConn
    global tree
    global isRoot
    global parent
    global index
    # global s
    global threads
    global restructuring
    
    temp = None
    
    while True:
        if restructuring or s == None or stpThread:
            print("RESTRUCTURING")
            break
        
        try:
            # Listen for connections
            conn, (address, tport) = s.accept()
            conn_name = "{}|{}".format(address, port)
            children.append({"ip": address, "port": tport, "conn": conn})
            print("[INFO]: Accepted the connection from {}".format(conn_name))
            temp = threading.Thread(target=handle_client, args=(conn, conn_name))
            temp.start()
            
        # break the while loop when keyboard intterupt is received and server will be closed
        except OSError:
            print("\n[INFO]: Keyboard Interrupt Received")
            break

        sleep(0.1)

    s.close()
    s = None
    print("[INFO]: Server Closed")

# START A THREAD FOR LISTENING TO CHILDREN (FOR FEDAVG)
listener = threading.Thread(target=connectToChildren, args=(s,stopThreads))
listener.start()

def sendTreeDown():
    global children
    global parentConn
    global tree
    global isRoot
    global parent
    global index
    global s
    global parentIndex
    global listener
    global maxChildren
    global restructuring
    global threads
    global stopThreads
    
    restructuring = True
    stopThreads = True
    
    # stop listening for children
    # listener.join(0)
    
    for child in children:
        send_data(child["conn"], {"type": "tree", "tree": tree}, data_identifiers["data"])
        
    # sleep(1)
    
    listener.join(0)
    for t in threads:
        t.join(0)
        # remove thread from list
    threads = []
        
    # disconnect from everyone
    for child in children:
        child["conn"].close()
    
    children = []
    
    if parentConn != None:
        parentConn.close()
        
    if s != None:
        s.close()
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    parentConn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("localhost", port))
    print("IP is ", s.getsockname()[0], "PORT is ", s.getsockname()[1])
    s.listen(2)
    
    # sleep(5)
    restructuring = False
    stopThreads = False
    listener = threading.Thread(target=connectToChildren, args=(s,stopThreads))
    listener.start()
        
    index = 0
    for i, node in enumerate(tree):
        if node[2] == str(s.getsockname()[0]) + ":" + str(s.getsockname()[1]):
            index = i
            break
        
    print("INDEX", index + 1)
    parentIndex = int((index + 1) / 2)
    print("PARENT INDEX", parentIndex)
        
    parent = {}
    children = []
    maxChildren = 2

    if parentIndex != 0:
        # sleep(10)
        parent = {"ip": tree[parentIndex-1][2].split(":")[0], "port": tree[parentIndex-1][2].split(":")[1]}
        parentConn.connect((parent["ip"], int(parent["port"])))
        print("parent", parent)
        isRoot = False
    else:
        print("ROOT")
        isRoot = True

finishedBlock = False
while True:
    
    # AWAIT FOR START AS ROOT, OTHERWISE ACT AS NODE AND PASS MESSAGES DOWN
    if isRoot:
        # input("Press Enter to start the tree...\n\n")
        # print("Waiting for children")
        # wait for children to connect
        while len(children) != maxChildren:
            if len(tree) == 2 and len(children) == 1:
                break
            sleep(1)
        
        print("SENDING START")
        
        # log the start time
        iteration += 1
        speed = open("./data/speed.csv", "a")
        speed.write(str(iteration) + ", start, " + str(time.time()) + "\n")
        speed.close()
        
        start_epoch()
        send_broadcast_down("start")
        waiting = False
        # send transaction to model
        
        while not finished:
            sleep(0.1)
            
        print("ITERATION COMPLETE")
        speed = open("./data/speed.csv", "a")
        speed.write(str(iteration) + ", end, " + str(time.time()) + "\n")
        speed.close()
        
        tree = None
        event_filter = contract.events.TreeStructureGenerated.createFilter(fromBlock='latest')
        event_filter2 = contract.events.IterationComplete.createFilter(fromBlock='latest')
        tx = contract.functions.completeIteration(task_id).transact()
        # get gas used
        receipt = w3.eth.getTransactionReceipt(tx)
        gas_used = receipt['gasUsed']
        gas_costs = open("./data/gas_costs.csv", "a")
        gas_costs.write(str(iteration) + ", " + str(port) + ", completeIteration, " + str(gas_used) + "\n")
        gas_costs.close()
        restructuring = True
        
        
        while True:
            for event in event_filter.get_new_entries():
                # print("Event", event)
                if event["args"]["taskId"] == task_id:
                    tree = event["args"]["tree"]
                    break
                
            for event in event_filter2.get_new_entries():
                # print("Event", event)
                if event["args"]["taskId"] == task_id and event["args"]["complete"]:
                    finishedBlock = True
                    break
            if tree != None or finishedBlock:
                break
            
            sleep(1)
            print("Waiting for event")
            
        if finishedBlock:
            print("FEDERATED LEARNING COMPLETE")
            send_broadcast_down("stop")
            
            gas_costs = open("./data/gas_costs.csv", "a")
            model_accuracy = open("./data/model_accuracy.csv", "a")
            speed = open("./data/speed.csv", "a")
            gas_costs.write("\n\n\n")
            model_accuracy.write("\n\n\n")
            speed.write("\n\n\n")
            gas_costs.close()
            model_accuracy.close()
            speed.close()
            
            break
        
        sendTreeDown()
        
        
        
        
        
    else:
        while not isRoot:
            
            if not restructuring:
                try:
                    data_id, payload = receive_data(parentConn)
                    # get type of data
                    
                    if data_id == data_identifiers["info"]:
                        if payload == "start":
                            print("RECEIVED START")
                            iteration += 1
                            start_epoch()
                            send_broadcast_down("start")
                        elif payload == "stop":
                            print("RECEIVED STOP")
                            finishedBlock = True
                            break
                    elif data_id == data_identifiers["data"]:
                        if payload["type"] == "model":
                            model = payload["model"]
                            print("RECEIVED GLOBAL MODEL")
                        elif payload["type"] == "tree":
                            # get tree
                            # send_up("received")
                            tree = payload["tree"]
                            print("TREE", tree)
                            sendTreeDown()
                except ConnectionResetError:
                    print("Connection reset")
                    parentConn.close()
                    parentConn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    parentConn.connect((parent["ip"], int(parent["port"])))
                    print("parent", parent)
                    
            
    if finishedBlock:
        break