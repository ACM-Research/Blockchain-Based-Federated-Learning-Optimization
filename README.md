# Blockchain-Based-Federated-Learning-Optimization

## Demo Video
[![Youtube_Video](https://img.youtube.com/vi/JNdhq95s6yI/0.jpg)](https://youtu.be/JNdhq95s6yI)

## Research Poster
![Research  Poster](poster.png)

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
