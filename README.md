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
 