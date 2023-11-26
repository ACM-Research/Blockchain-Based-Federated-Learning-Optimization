import pytest;


from brownie import MainContract, accounts, exceptions, network;
import web3;

network.connect('development')

# Will spit out a mnemonic
me = accounts.add()
alice = accounts.add()
# Give me some Ether
accounts[0].transfer(to=me.address, amount=1000);
txn = {
  "from": me.address,
  "to": alice.address,
  "value": 10,
  "gas": 21000,
  "gasPrice": 0,
  "nonce": me.nonce
}
txn_signed = web3.eth.account.signTransaction(txn, me.private_key)
web3.eth.send_raw_transaction(txn_signed.rawTransaction)
alice.balance()

@pytest.fixture
def counter():
    accounts[0].deploy(MainContract);