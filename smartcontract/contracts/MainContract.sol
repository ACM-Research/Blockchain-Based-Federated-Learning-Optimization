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

    struct TreeNode {
        address userAddress;
        Role role;
        uint256 left;
        uint256 right;
    }

    mapping(uint256 => TreeNode) public tree;


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

        
    function createTree(uint256 totalMembers, address root, address[] memory subAggregators, address[] memory workers) public {
        require(totalMembers >= 3, "Total members must be at least 3");
        require(totalMembers == 1 + subAggregators.length + workers.length, "Total members must match the sum of root, subAggregators, and workers");
        require(subAggregators.length > 0, "At least one sub-aggregator is required");

        // Root Aggregator
        tree[0] = TreeNode({
            userAddress: root,
            role: Role.RootAggregator,
            left: 1,
            right: 2
        });

        // SubAggregators
        for (uint256 i = 0; i < subAggregators.length; i++) {
            tree[i + 1] = TreeNode({
                userAddress: subAggregators[i],
                role: Role.SubAggregator,
                left: 2 * i + 3,
                right: 2 * i + 4
            });
        }

        // Workers
        for (uint256 i = 0; i < workers.length; i++) {
            tree[subAggregators.length + i + 1] = TreeNode({
                userAddress: workers[i],
                role: Role.Worker,
                left: 0,
                right: 0
            });
        }
    }

}