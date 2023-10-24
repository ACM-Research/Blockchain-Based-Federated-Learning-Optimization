import requests
import sys
import socket

server = "localhost:3000"

print(sys.argv)
if len(sys.argv) > 1:
    server = sys.argv[1]

# post initial connection to server
r = requests.post("http://" + server + "/connect")

# receive response from server (id, port, parentIp, parentPort)
print(r.text)
id, port, parentIp, parentPort = r.text.split(" ")

# create socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("localhost", int(port)))
s.listen(5)

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