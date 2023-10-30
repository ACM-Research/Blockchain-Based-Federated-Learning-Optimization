const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("MainContract", function () {
  let MainContract, mainContract, owner, addr1, addr2;

  beforeEach(async function () {
    MainContract = await ethers.getContractFactory("MainContract");
    mainContract = await MainContract.deploy();
    [owner, addr1, addr2] = await ethers.getSigners();
  });

  describe("Deployment", function () {
    it("Should initialize tasks correctly", async function () {
      await mainContract.initTask(3);
      const task = await mainContract.tasks(0);
      expect(task.requiredUsers).to.equal(3);
    });
  });

  describe("User Management", function () {
    it("Should add users to a task", async function () {
      await mainContract.initTask(3);
      await mainContract.addUser(0, addr1.address);
      const task = await mainContract.tasks(0);
      expect(task.currentUserCount).to.equal(1);
    });
  });

  describe("Model Update", function () {
    it("Should update the model correctly", async function () {
      await mainContract.initTask(3);
      await mainContract.addUser(0, owner.address);
      const modelUpdateHash = "0x1234567890123456789012345678901234567890123456789012345678901234";
      await mainContract.updateModel(0, modelUpdateHash);
      const task = await mainContract.tasks(0);
      expect(task.modelUpdateHash).to.equal(modelUpdateHash);
    });
  });
});
