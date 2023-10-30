const hre = require("hardhat");

async function main() {
  const MainContract = await hre.ethers.getContractFactory("MainContract");
  const initialCount = 5;
  const mainContract = await MainContract.deploy(initialCount);
  console.log(`MainContract Deployed to: ${mainContract.target}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
