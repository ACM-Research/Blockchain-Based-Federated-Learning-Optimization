import pytest;


from brownie import Counter, accounts, exceptions;

#
# These tests are meant to be executed with brownie. To run them:
# * create a brownie project using brownie init
# * in the contract directory, place Counter.sol (maybe using a symlink)
# * in the tests directory, place this script
# * run brownie test
#

@pytest.fixture
def counter():
    return accounts[0].deploy(Counter);

#
# Inquire initial count
#
def test_initialCount(counter):
    count = counter.read();
    assert(0 == count);   

#
# Test increment
#
def test_increment(counter):
    me = accounts[0];
    # Remember old value of counter
    oldCount = counter.read();
    # Increment
    txn_receipt = counter.increment({"from": me});
    newCount = counter.read();
    assert(newCount == oldCount + 1);
    # Verify that this has created an event
    event = txn_receipt.events['Increment'];
    assert(event['sender'] == me.address);
    assert(event['oldValue'] == oldCount);
    assert(event['newValue'] == newCount);
