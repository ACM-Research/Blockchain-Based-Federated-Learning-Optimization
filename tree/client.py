import requests
import sys
import socket
import random

server = "localhost:3000"

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

while True:
    temp = serv.recv(1024).decode()
    if temp != "None":
        children.append({"ip": temp.split(" ")[0], "port": temp.split(" ")[1]})
        print("children", children)
    if len(children) == maxChildren:
        break

while True:
    temp = serv.recv(1024).decode()


def work():
    return "hi"



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