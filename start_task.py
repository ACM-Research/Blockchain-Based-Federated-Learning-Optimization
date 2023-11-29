

from web3 import Web3
import os
import json

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

# print("Create task id:", return_value)
