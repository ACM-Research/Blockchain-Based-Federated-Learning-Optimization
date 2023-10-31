import requests
import sys
import socket

# host the socket server that handles the tree
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("localhost", 3000))

# create a tree
tree = [None]

# handle connections from clients
# Listen for connections
while True:
    s.listen(100)
    # Accept a connection
    conn, addr = s.accept()
    # Do something with the connection
    print("Received connection from", addr)
    client = {}
    print("Waiting for client to send port")
    client["conn"] = conn
    client["addr"] = addr
    client["ip"] = addr[0]
    client["port"] = conn.recv(1024).decode()
    print("Received port", client["port"])
    # add client to tree
    index = len(tree)
    tree.append(client)
    # get parent
    parent = int(index / 2)
    print("Parent is", parent)
    if parent > 0:
        parent = tree[int(parent)]
        # send parent info to client
        conn.send((str(parent["ip"]) + " " + str(parent["port"])).encode())
        # send client info to parent
        parent["conn"].send((str(client["ip"]) + " " + str(client["port"])).encode())
    else:
        # send client info to parent
        conn.send("None".encode())