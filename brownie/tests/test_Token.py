import pytest;
import brownie;

from brownie import Token, accounts, exceptions;

#
# These tests are meant to be executed with brownie. To run them:
# * create a brownie project using brownie init
# * in the contract directory, place Token.sol (maybe using a symlink)
# * in the tests directory, place this script
# * run brownie test
#

@pytest.fixture
def token():
    return accounts[0].deploy(Token);


# Test the name method
def test_name(token):
    assert(token.name() == "MyToken");

# Test the symbol
def test_symbol(token):
    assert(token.symbol() == "MTK");

# Test decimals
def test_decimals(token):
    assert(token.decimals() == 2);

# Test total supply
def test_totalSupply(token):
    assert(token.totalSupply() == 100000);

# Test balanceOf - initially, the entire supply should be 
# in the account of the token owner
def test_balanceOf(token):
    tokenOwner = accounts[0];
    assert(token.balanceOf(tokenOwner.address) == token.totalSupply());

# Test a valid transfer
def test_transfer(token):
    me = accounts[0];
    alice = accounts[1];
    oldBalanceAlice = token.balanceOf(alice.address);
    oldBalanceMe = token.balanceOf(me.address);
    value = 100;
    txn_receipt = token.transfer(alice.address, value, {"from": me.address});
    assert(oldBalanceAlice + value == token.balanceOf(alice.address));
    assert(oldBalanceMe - value == token.balanceOf(me.address));
    # Verify that event has been emitted
    event = txn_receipt.events['Transfer'];
    assert(event['from'] == me.address);
    assert(event['to'] == alice.address);
    assert(event['value'] == value);

# Test a valid transfer of value zero
def test_transferZero(token):
    me = accounts[0];
    alice = accounts[1];
    oldBalanceAlice = token.balanceOf(alice.address);
    oldBalanceMe = token.balanceOf(me.address);
    value = 0;
    txn_receipt = token.transfer(alice.address, value, {"from": me.address});
    assert(oldBalanceAlice + value == token.balanceOf(alice.address));
    assert(oldBalanceMe - value == token.balanceOf(me.address));
    # Verify that event has been emitted
    event = txn_receipt.events['Transfer'];
    assert(event['from'] == me.address);
    assert(event['to'] == alice.address);
    assert(event['value'] == value);

# Test a transfer that exceeds the balance
def test_transfer_insufficientFunds(token):
    me = accounts[0];
    alice = accounts[1];
    oldBalanceAlice = token.balanceOf(alice.address);
    oldBalanceMe = token.balanceOf(me.address);
    value = oldBalanceMe + 1;
    with brownie.reverts("Insufficient balance"):
        token.transfer(alice.address, value, {"from": me.address});

# Test an unauthorized transferFrom
def test_transferFrom_notAuthorized(token):
    me = accounts[0];
    alice = accounts[1];
    bob = accounts[2];
    oldBalanceAlice = token.balanceOf(alice.address);
    oldBalanceMe = token.balanceOf(me.address);
    value = oldBalanceMe + 1;
    with brownie.reverts("Transfer not authorized"):
        token.transferFrom(me.address, alice.address, value, {"from": bob.address});

# Test approval
def test_approve(token):
    me = accounts[0];
    bob = accounts[2];
    value = 100;
    # Allow bob to spend 100 token on my behalf
    txn_receipt = token.approve(bob.address, value, {"from": me.address});
    # Verify that event has been emitted
    event = txn_receipt.events['Approval'];
    assert(event['owner'] == me.address);
    assert(event['spender'] == bob.address);
    assert(event['value'] == value);    
    # Check
    assert(token.allowance(me.address, bob.address) == value);


# Test approval - overwrite old value
def test_approve_overwrite(token):
    me = accounts[0];
    bob = accounts[2];
    value = 100;
    # Allow bob to spend 100 token on my behalf
    token.approve(bob.address, value, {"from": me.address});
    # Check
    assert(token.allowance(me.address, bob.address) == value);
    # Overwrite
    value = 120
    token.approve(bob.address, value, {"from": me.address});
    assert(token.allowance(me.address, bob.address) == value);

# Test a valid withdrawal
def test_transferFrom(token):
    me = accounts[0];
    alice = accounts[1];
    bob = accounts[2];
    oldBalanceAlice = token.balanceOf(alice.address);
    oldBalanceMe = token.balanceOf(me.address);
    value = 100;
    # Authorize bob
    token.approve(bob.address, value + 10);
    txn_receipt = token.transferFrom(me.address, alice.address, value, {"from": bob.address});
    assert(oldBalanceAlice + value == token.balanceOf(alice.address));
    assert(oldBalanceMe - value == token.balanceOf(me.address));
    # Verify that event has been emitted
    event = txn_receipt.events['Transfer'];
    assert(event['from'] == me.address);
    assert(event['to'] == alice.address);
    assert(event['value'] == value);
    # Verify that the approval has been reduced by the transferred value
    assert(token.allowance(me.address, bob.address) == 10);


# Test an invalid withdrawal - allowance not sufficient
def test_transferFrom_insufficientAllowance(token):
    me = accounts[0];
    alice = accounts[1];
    bob = accounts[2];
    oldBalanceAlice = token.balanceOf(alice.address);
    oldBalanceMe = token.balanceOf(me.address);
    value = 100;
    # Authorize bob
    token.approve(bob.address, value - 10);
    with brownie.reverts("Transfer not authorized"):
        token.transferFrom(me.address, alice.address, value, {"from": bob.address});

# Test an invalid withdrawal - balance not sufficient
def test_transferFrom_insufficientBalance(token):
    me = accounts[0];
    alice = accounts[1];
    bob = accounts[2];
    oldBalanceMe = token.balanceOf(me.address);
    value = oldBalanceMe + 1;
    # Authorize bob
    token.approve(bob.address, value);
    with brownie.reverts("Insufficient balance"):
        token.transferFrom(me.address, alice.address, value, {"from": bob.address});
