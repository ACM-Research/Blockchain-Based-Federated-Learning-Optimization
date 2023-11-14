// SPDX-License-Identifier: MIT
pragma solidity ^0.8.9;

contract MainContract {
    enum Role { Worker, SubAggregator, RootAggregator }

    struct User {
        address userAddress;
        Role role;
    }

    struct TreeNode {
        address userAddress;
        Role role;
        uint256 left;
        uint256 right;
    }

    struct Task {
        uint256 id;
        uint256 requiredUsers;
        uint256 currentUserCount;
        User[] users;
        bool isInitialized;
        uint256 modelUpdateCount;
    }

    uint256 public nextTaskId;
    mapping(uint256 => Task) public tasks;
    mapping(uint256 => TreeNode) public tree;

    event TaskInitialized(uint256 taskId);
    event UserAdded(uint256 taskId, address user);
    event ModelUpdateStarted(uint256 taskId);

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
        User memory newUser = User({userAddress: userAddress, role: Role.Worker});
        task.users.push(newUser);
        task.currentUserCount++;
        emit UserAdded(taskId, userAddress);
    }

    function finalizeTaskSetup(uint256 taskId) public {
        Task storage task = tasks[taskId];
        require(task.currentUserCount == task.requiredUsers, "Task does not have required number of users");
        
        // Categorize users
        for (uint256 i = 0; i < task.users.length; i++) {
            if (i == 0) {
                task.users[i].role = Role.RootAggregator;
            } else if (i <= 1 + 2 * (task.users.length - 2) / 3) { // Sub Aggregators
                task.users[i].role = Role.SubAggregator;
            } else {
                task.users[i].role = Role.Worker;
            }
        }
        createTree(taskId);
    }

    function createTree(uint256 taskId) internal {
        Task storage task = tasks[taskId];
        uint256 totalMembers = task.users.length;

        for (uint256 i = 0; i < totalMembers; i++) {
            uint256 leftChild = 2 * i + 1 < totalMembers ? 2 * i + 1 : 0;
            uint256 rightChild = 2 * i + 2 < totalMembers ? 2 * i + 2 : 0;

            tree[i] = TreeNode({
                userAddress: task.users[i].userAddress,
                role: task.users[i].role,
                left: leftChild,
                right: rightChild
            });
        }
    }

    function updateModel(uint256 taskId) public {
        Task storage task = tasks[taskId];
        require(task.isInitialized, "Task is not initialized");
        require(msg.sender == task.users[0].userAddress, "Only the Root Aggregator can initiate model update");

        task.modelUpdateCount++;
        createTree(taskId);
        emit ModelUpdateStarted(taskId);
    }
}
