// SPDX-License-Identifier: MIT
pragma solidity ^0.8.9;
import "hardhat/console.sol";

contract MainContract {
    enum Role { Worker, SubAggregator, RootAggregator }
    
    struct User {
        address userAddress;
        Role role;
    }
    
    struct Task {
        uint256 id;
        uint256 requiredUsers;
        uint256 currentUserCount;
        bytes32 modelUpdateHash;
        User[] users;
        bool isInitialized;
    }

    uint256 public nextTaskId;
    mapping(uint256 => Task) public tasks;

    event TaskInitialized(uint256 taskId);
    event UserAdded(uint256 taskId, address user);
    event ModelUpdated(uint256 taskId, bytes32 modelUpdateHash);

    function initTask(uint256 requiredUsers) public {
        require(requiredUsers > 0, "Number of required users must be greater than 0");

        Task storage newTask = tasks[nextTaskId];
        newTask.id = nextTaskId;
        newTask.requiredUsers = requiredUsers;
        newTask.isInitialized = true;

        emit TaskInitialized(nextTaskId);
        nextTaskId++;
    }

    function addUser(uint256 taskId, address userAddress) public {
        Task storage task = tasks[taskId];
        require(task.isInitialized, "Task is not initialized");
        require(task.currentUserCount < task.requiredUsers, "All users have already been added");

        User memory newUser = User({
            userAddress: userAddress,
            role: Role.Worker
        });

        task.users.push(newUser);
        task.currentUserCount++;

        emit UserAdded(taskId, userAddress);
    }

    function updateModel(uint256 taskId, bytes32 modelUpdateHash) public {
        Task storage task = tasks[taskId];
        require(task.isInitialized, "Task is not initialized");
        require(msg.sender == task.users[0].userAddress, "Only the Root Aggregator can update the model");

        task.modelUpdateHash = modelUpdateHash;
        emit ModelUpdated(taskId, modelUpdateHash);
    }
}