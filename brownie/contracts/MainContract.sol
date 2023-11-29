// SPDX-License-Identifier: MIT
pragma solidity ^0.8.9;

contract MainContract {
    enum Role { Worker, SubAggregator, RootAggregator }
    enum TaskStatus { Pending, Completed }

    struct User {
        address userAddress;
        Role role;
        string ip;
    }

    struct TreeNode {
        address userAddress;
        Role role;
        string ip;
    }

    struct Task {
        uint256 id;
        uint256 requiredUsers;
        uint256 currentIteration;
        uint256 totalIterations;
        uint256 fundingAmount;
        User[] users;
        TreeNode[] tree;
        bool isInitialized;
        bool isFull; 
        TaskStatus status;
        mapping(address => uint256) contributions;
    }

    uint256 public nextTaskId;
    mapping(uint256 => Task) public tasks;

    event TaskInitialized(uint256 taskId, uint256 requiredUsers, uint256 totalIterations, uint256 fundingAmount);
    event UserAdded(uint256 taskId, address user, string ip);
    event IterationComplete(uint256 taskId, uint256 iteration, address rootAggregator);
    event TreeStructureGenerated(uint256 taskId, address rootAggregator, TreeNode[] tree);

    function initTask(uint256 requiredUsers, uint256 totalIterations, uint256 fundingAmount) public {
        Task storage newTask = tasks[nextTaskId];
        newTask.id = nextTaskId;
        newTask.requiredUsers = requiredUsers;
        newTask.totalIterations = totalIterations;
        newTask.fundingAmount = fundingAmount;
        newTask.isInitialized = true;
        newTask.isFull = false;
        newTask.currentIteration = 0;
        newTask.status = TaskStatus.Pending;

        emit TaskInitialized(nextTaskId, requiredUsers, totalIterations, fundingAmount);
        nextTaskId++;
    }

    function addUser(uint256 taskId, address userAddress, string memory ip) public {
        Task storage task = tasks[taskId];
        require(task.isInitialized, "Task not initialized");
        require(task.users.length < task.requiredUsers, "Task is already full");

        task.users.push(User({ userAddress: userAddress, role: Role.Worker, ip: ip}));
        if (task.users.length == task.requiredUsers) {
            task.isFull = true;
            startIteration(taskId);
        }

        emit UserAdded(taskId, userAddress, ip);
    }

    function startIteration(uint256 taskId) internal {
        Task storage task = tasks[taskId];
        require(task.isFull, "Task is not full");
        generateTreeStructure(taskId);
        task.currentIteration++;
    }

    // function completeIteration(uint256 taskId, address[] memory contributors, uint256[] memory values) public {
    //     Task storage task = tasks[taskId];
    //     require(task.currentIteration <= task.totalIterations, "All iterations completed");
    //     require(msg.sender == getRootAggregatorAddress(taskId), "Only Root Aggregator can complete the iteration");

    //     uint256 totalContributionForIteration = 1000;
    //     for (uint256 i = 0; i < contributors.length; i++) {
    //         uint256 contributionPercentage = (values[i] * 100) / totalContributionForIteration;
    //         task.contributions[contributors[i]] += contributionPercentage;
    //     }

    //     if (task.currentIteration < task.totalIterations) {
    //         startIteration(taskId);
    //     } else {
    //         distributeFunds(taskId);
    //         task.status = TaskStatus.Completed;
    //     }

    //     emit IterationComplete(taskId, task.currentIteration, task.tree[0].userAddress);
    // }

    function completeIteration(uint256 taskId) public {
        Task storage task = tasks[taskId];
        require(task.currentIteration <= task.totalIterations, "All iterations completed");
        require(msg.sender == getRootAggregatorAddress(taskId), "Only Root Aggregator can complete the iteration");

        if (task.currentIteration < task.totalIterations) {
            startIteration(taskId);
        } else {
            // distributeFunds(taskId);
            task.status = TaskStatus.Completed;
        }

        emit IterationComplete(taskId, task.currentIteration, task.tree[0].userAddress);
    }

    function distributeFunds(uint256 taskId) internal {
        Task storage task = tasks[taskId];
        uint256 totalFund = task.fundingAmount;

        for (uint256 i = 0; i < task.users.length; i++) {
            address userAddress = task.users[i].userAddress;
            uint256 userContributionPercentage = task.contributions[userAddress];
            uint256 userShare = (totalFund * userContributionPercentage) / 100;
            payable(userAddress).transfer(userShare);
        }
    }

    uint256 public entropy = block.timestamp;

    function generateTreeStructure(uint256 taskId) internal {
        Task storage task = tasks[taskId];
        uint256 numUsers = task.users.length;

        for (uint256 i = 0; i < numUsers; i++) {
            task.tree.push(TreeNode({
                userAddress: task.users[i].userAddress,
                role: Role.Worker,
                ip: task.users[i].ip
            }));
        }

        for(uint256 i = task.tree.length -1 ; i > 0; i--) {

            uint256 swapIndex = entropy % (task.tree.length - i);

            TreeNode memory temp = TreeNode({
                userAddress: task.tree[i].userAddress,
                role: task.tree[i].role,
                ip: task.tree[i].ip
            });

            task.tree[i] = task.tree[swapIndex];
            task.tree[swapIndex] = temp;
        }

        emit TreeStructureGenerated(taskId, task.tree[0].userAddress, task.tree);
    }

    function getRootAggregatorAddress(uint256 taskId) public view returns (address) {
        Task storage task = tasks[taskId];
        return task.tree[0].userAddress;
    }
}
