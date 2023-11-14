const hre = require("hardhat");

async function main() {
    const MainContract = await hre.ethers.getContractFactory("MainContract");
    const mainContract = await MainContract.deploy();

    await mainContract.deployed();
    console.log(`MainContract deployed to: ${mainContract.address}`);
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
