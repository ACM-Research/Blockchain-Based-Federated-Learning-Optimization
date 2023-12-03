

import threading
from web3 import Web3
import os
import json
import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import asyncio
from websockets.server import serve

#initTask(uint256 requiredUsers, uint256 totalIterations, uint256 fundingAmount)
contract_address = "0x3194cBDC3dbcd3E11a07892e7bA5c3394048Cc87"
contract_abi = None
contract = None

# create copy of MainContract.json in this directory from ./brownie/build/contracts/MainContract.json 
# if it doesn't exist  already

if (os.path.exists("./brownie/build/contracts/MainContract.json")):
    with open("./brownie/build/contracts/MainContract.json") as f:
        contract_abi = json.load(f)
        print("Loaded contract abi from file")
        contract = contract_abi
        contract_abi = contract_abi["abi"]

    with open("./MainContract.json", "w") as f:
        json.dump(contract, f)
        print("Saved contract abi to file")


w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))
w3.eth.defaultAccount = w3.eth.accounts[0]

# Create a contract instance
contract = w3.eth.contract(address=contract_address, abi=contract_abi)

NUM_USERS = 4
TOTAL_ITERATIONS = 20

result = contract.functions.initTask(NUM_USERS, TOTAL_ITERATIONS, 100).transact()

# Get the transaction receipt
receipt = w3.eth.getTransactionReceipt(result)




print("Task Created")

# print(receipt)

# get gas used
gas_used = receipt['gasUsed']

gas_costs = open("./data/gas_costs.csv", "a")
print ("Gas used: ", gas_used)
gas_costs.write("0, initTask, " + str(gas_used) + ", " + str(NUM_USERS) + ", " + str(TOTAL_ITERATIONS) + "\n")

# # Get the return value
# return_value = receipt.return_value

# create a flask app instance and have static_url_path point to docs/src
app = Flask(__name__, static_url_path='/docs')

#allow any origin to make a request
CORS(app)

# GET request to check if the server is running

@app.route('/', defaults=dict(filename=None))
@app.route('/<path:filename>', methods=['GET'])
def serve_static(filename):
    filename = filename or 'index.html'
    if request.method == 'GET':
        return send_from_directory('./docs', filename)

    return jsonify(request.data)



frontend = None
tree = None

async def echo(websocket):
    global frontend
    async for message in websocket:
        data = json.loads(message)
        print(data)
        type = data["type"]
        if type == "front":
            frontend = websocket
        elif type == "tree":
            tree = data["tree"]
            if frontend != None:
                await frontend.send(json.dumps({"type": "tree", "tree": tree}))
        elif type == "accuracy":
            accuracy = data["accuracy"]
            if frontend != None:
                await frontend.send(json.dumps({"type": "accuracy", "accuracy": accuracy}))
        elif type == "gas":
            gas = data["gas"]
            if frontend != None:
                await frontend.send(json.dumps({"type": "gas", "gas": gas}))
        elif type == "iteration":
            iteration = data["iteration"]
            if frontend != None:
                await frontend.send(json.dumps({"type": "iteration", "iteration": iteration}))
        elif type == "finished":
            if frontend != None:
                await frontend.send(json.dumps({"type": "finished"}))
        elif type == "status":
            status = data["status"]
            port = data["port"]
            if frontend != None:
                await frontend.send(json.dumps({"type": "status", "status": status, "port": port}))
        


async def main():
    async with serve(echo, "localhost", 8765):
        await asyncio.Future()  # run forever

def start_server():
    asyncio.run(main())

threading.Thread(target=start_server).start()
# start_server()

# run the app
if __name__ == '__main__':
    app.run(port=3000)