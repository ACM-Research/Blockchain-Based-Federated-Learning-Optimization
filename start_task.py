

from web3 import Web3
import os
import json

#initTask(uint256 requiredUsers, uint256 totalIterations, uint256 fundingAmount)
contract_address = "0x3194cBDC3dbcd3E11a07892e7bA5c3394048Cc87"
contract_abi = None

if (os.path.exists("./MainContract.json")):
    with open("./MainContract.json") as f:
        contract_abi = json.load(f)
        print("Loaded contract abi from file")
        contract_abi = contract_abi["abi"]
        # print(contract_abi)
else:
    print("AAA")


w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))
w3.eth.defaultAccount = w3.eth.accounts[0]

# Create a contract instance
contract = w3.eth.contract(address=contract_address, abi=contract_abi)

result = contract.functions.initTask(1, 2, 100).transact()

print(result)

