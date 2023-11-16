const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("MainContract", function () {
    let MainContract, mainContract, owner, addr1, addr2, addr3;

    beforeEach(async function () {
        [owner, addr1, addr2, addr3] = await ethers.getSigners();
        MainContract = await ethers.getContractFactory("MainContract");
        mainContract = await MainContract.deploy();
    });

    describe("Task Management", function () {
        beforeEach(async function () {
            const fundingAmount = ethers.parseEther("1");
            await mainContract.initTask(3, 2, fundingAmount);
            await mainContract.addUser(0, addr1.address);
            await mainContract.addUser(0, addr2.address);
            await mainContract.addUser(0, addr3.address);
        });
    
        it("Should initialize and add users to a task correctly", async function () {
            const task = await mainContract.tasks(0);
            expect(task.requiredUsers).to.equal(3);
            expect(task.users.length).to.equal(3); // Ensure this line is after users are added
            expect(task.isInitialized).to.be.true;
            expect(task.isFull).to.be.true;
        });
    
        it("Should create a tree structure correctly", async function () {
            if (typeof mainContract.startIteration === 'function') { // Check if function exists
                await mainContract.startIteration(0);
                const task = await mainContract.tasks(0);
                expect(task.tree.length).to.equal(3);
            } else {
                throw new Error("startIteration function is not available in MainContract");
            }
        });
    });

    describe("Iteration Completion", function () {
        beforeEach(async function () {
            await mainContract.initTask(3, 2, ethers.parseEther("1"));
            await mainContract.addUser(0, addr1.address);
            await mainContract.addUser(0, addr2.address);
            await mainContract.addUser(0, addr3.address);
            await mainContract.startIteration(0);
        });

        it("Should allow the root aggregator to complete an iteration", async function () {
            await mainContract.connect(addr1).completeIteration(0, [addr1.address, addr2.address, addr3.address], [100, 100, 100]);
            const task = await mainContract.tasks(0);
            expect(task.currentIteration).to.equal(1);
        });

        it("Should not allow non-root aggregators to complete an iteration", async function () {
            await expect(mainContract.connect(addr2).completeIteration(0, [addr1.address, addr2.address, addr3.address], [100, 100, 100]))
                .to.be.revertedWith("Only Root Aggregator can complete the iteration");
        });
    });
});
