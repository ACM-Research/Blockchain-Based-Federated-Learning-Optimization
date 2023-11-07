from time import sleep
import requests
import sys
import socket
import random
import threading
# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

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




def work(x = 0):
    # DATA
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() # 28x28 images of hand-written digits 0-9

    # PREPROCESS DATA
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # BUILD MODEL
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)), # input layer
        tf.keras.layers.Dense(128, activation='relu'), # hidden layer
        tf.keras.layers.Dense(10, activation='softmax') # output layer
    ])

    # COMPILE MODEL
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # TRAIN MODEL
    model.fit(x_train, y_train, epochs=5)

    # EVALUATE MODEL
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)

    # MAKE PREDICTIONS
    predictions = model.predict(x_test)
    print(predictions[0])
    print(np.argmax(predictions[0]))
    print(y_test[0])

    # SAVE MODEL
    model.save('model.h5')
    work(x - 1)


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








# connect to parent
if parentIp != "None":
    s.connect((parentIp, int(parentPort)))

    # send id to parent
    s.send(id.encode())

    # receive response from parent
    print(s.recv(1024).decode())

# accept connections from children
while True:
    conn, addr = s.accept()
    print("connected to", addr)

    # receive id from child
    childId = conn.recv(1024).decode()
    print("received id", childId)

    # send id to child
    conn.send(id.encode())

    # receive response from child
    print(conn.recv(1024).decode())