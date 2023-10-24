import requests
import sys

server = "localhost:3000"

print(sys.argv)
if len(sys.argv) > 1:
    server = sys.argv[1]

# post initial connection to server
r = requests.post("http://" + server + "/connect")
print(r.text)
