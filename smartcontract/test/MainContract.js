const { expect } = require("chai");

describe("MainContract", function () {
    let MainContract, mainContract, owner, addr1, addr2, addr3;

    beforeEach(async function () {
        [owner, addr1, addr2, addr3] = await ethers.getSigners();
        MainContract = await ethers.getContractFactory("MainContract");
        mainContract = await MainContract.deploy();
        await mainContract.deployed();
    });

    describe("Task Management", function () {
        it("Should initialize and finalize task setup correctly", async function () {
            await mainContract.initTask(3);
            await mainContract.addUser(0, addr1.address);
            await mainContract.addUser(0, addr2.address);
            await mainContract.addUser(0, addr3.address);
            await mainContract.finalizeTaskSetup(0);

            const task = await mainContract.tasks(0);
            expect(task.requiredUsers).to.equal(3);
            expect(task.currentUserCount).to.equal(3);
            expect(task.isInitialized).to.be.true;
        });

        it("Should create a tree structure correctly", async function () {
            await mainContract.initTask(3);
            await mainContract.addUser(0, addr1.address);
            await mainContract.addUser(0, addr2.address);
            await mainContract.addUser(0, addr3.address);
            await mainContract.finalizeTaskSetup(0);

            const root = await mainContract.tree(0);
            const subAggregator = await mainContract.tree(1);
            const worker = await mainContract.tree(2);

            expect(root.userAddress).to.equal(addr1.address);
            expect(subAggregator.userAddress).to.equal(addr2.address);
            expect(worker.userAddress).to.equal(addr3.address);
        });
    });

    describe("Model Update", function () {
        it("Should allow the root aggregator to update the model", async function () {
            await mainContract.initTask(3);
            await mainContract.addUser(0, addr1.address);
            await mainContract.addUser(0, addr2.address);
            await mainContract.addUser(0, addr3.address);
            await mainContract.finalizeTaskSetup(0);

            await mainContract.connect(addr1).updateModel(0);
            const task = await mainContract.tasks(0);
            expect(task.modelUpdateCount).to.equal(1);
        });

        it("Should not allow non-root aggregators to update the model", async function () {
            await mainContract.initTask(3);
            await mainContract.addUser(0, addr1.address);
            await mainContract.addUser(0, addr2.address);
            await mainContract.addUser(0, addr3.address);
            await mainContract.finalizeTaskSetup(0);

            await expect(mainContract.connect(addr2).updateModel(0)).to.be.revertedWith("Only the Root Aggregator can initiate model update");
        });
    });
});
