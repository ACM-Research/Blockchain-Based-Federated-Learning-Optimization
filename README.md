# Blockchain-Based-Federated-Learning-Optimization

## How to test run the program
1. download python 3.9.16
```python -p 3.9.16 venv .venv```
2. install dependencies
```python -m pip install -r requirements.txt```
3. go to the brownie folder and init and/or compile the contract(?)
```cd brownie```
```brownie compile```
4. run the local dev blockchain thing with the smart contract
```brownie run scripts/test.py```
5. run the start_task.py script to start the task in the smart contract
```cd ..```
```python start_task.py```
6. run multiple tree/client.py scripts to create new clients for federated learning
```python tree/client.py```
7. badda bing badda boom u have a trained model somewhere

![Research  Poster](poster.jpg)
 
## Things that still need to be done:
1. Restructure tree (not too hard) just need to create a restructure msg handler from old root
2. The fed average is not actually evenly averaged (D:) Fix: divide the node's params by the total num of nodes THEN add them together. idk why i didn't  do it that way to begin with :/
3. Figure out the best way to send the WHOLE model params to tree or just leave it as a local .pth file in the tree folder. Maybe we can upload it to some cloud thing somewhere like an aws or google bucket and store the url bc i don't know if solidity can handle big calls/transactions very well
