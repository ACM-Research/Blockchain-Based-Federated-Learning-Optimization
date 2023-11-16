const hre = require("hardhat");

async function main() {
    const MainContract = await hre.ethers.getContractFactory("MainContract");
    const mainContract = await MainContract.deploy();

    console.log(`MainContract deployed to: ${mainContract.target}`);
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
