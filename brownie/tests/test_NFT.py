import pytest;
import hexbytes;
import brownie;

from brownie import NFT, ERC721TokenReceiverImplementation, accounts, exceptions;

#
# These tests are meant to be executed with brownie. To run them:
# * create a brownie project using brownie init
# * in the contract directory, place NFT.sol and ERC721TokenReceiver.sol
# * in the tests directory, place this script
# * run brownie test
#

@pytest.fixture(scope="session")
def token():
    return accounts[0].deploy(NFT, "https://localhost:3000/");

@pytest.fixture(scope="session")
def tokenReceiver():
    return accounts[0].deploy(ERC721TokenReceiverImplementation);

# 
# Some helper functions
#
def _ensureToken(token, tokenID,  owner):
    # Make sure that token does not yet exist
    token._burn(tokenID, {"from": owner});
    # Mint
    token._mint(tokenID, {"from": owner});

#
# Verify that a Transfer event has been logged
#
def _verifyTransferEvent(txn_receipt, _from, to, tokenID):
    event = txn_receipt.events['Transfer'];
    assert(event['tokenID'] == tokenID);
    assert(event['to'] == to);
    assert(event['from'] == _from);

#
# Verify that an Approval event has been logged
#
def _verifyApprovalEvent(txn_receipt, owner, spender, tokenID):
    event = txn_receipt.events['Approval'];
    assert(event['tokenID'] == tokenID);
    assert(event['owner'] == owner);
    assert(event['spender'] == spender);

#
# Verify that an ApprovalForAll event has been logged
#
def _verifyApprovalForAllEvent(txn_receipt, owner, operator, approved):
    event = txn_receipt.events['ApprovalForAll'];
    assert(event['owner'] == owner);
    assert(event['operator'] == operator);
    assert(event['approved'] == approved);


#
# Inquire the balance for the zero address - this should raise an exception
#
def test_balanceOfZeroAddress(token):
    # This should raise an error. Note that we need to provide an address with
    # 40 hex digits, as otherwise the web3 ABI encoder treats the argument as a string
    # and is not able to find the matching ABI entry
    with brownie.reverts("Address 0 is not valid"):
        balance = token.balanceOf("0x"+"0"*40);

#
# Inquire the balance for a non-zero address 
#
def test_balanceOfNonZeroAddress(token):
    balance = token.balanceOf("0x1"+"0"*39);
    assert(0 == balance);   

#
# Mint a token - this also tests balanceOf and
# ownerOf
#
def test_mint(token):
    me = accounts[0];
    tokenID = 1;
    # Make sure that token does not yet exist
    token._burn(tokenID, {"from": me});
    # Remember old balance and mint
    oldBalance = token.balanceOf(me);
    txn_receipt = token._mint(tokenID, {"from": me});
    newBalance = token.balanceOf(me);
    assert(newBalance == oldBalance + 1);
    assert(me == token.ownerOf(tokenID));
    # Verify that minting has created an event
    event = txn_receipt.events['Transfer'];
    assert(event['tokenID'] == tokenID);
    assert(event['from'] == "0x"+"0"*40);
    assert(event['to'] == me);

#
# Only the contract owner can mint
#
def test_mint_notOwner(token):
    me = accounts[0];
    bob = accounts[1]
    tokenID = 1;
    # Make sure that token does not yet exist
    token._burn(tokenID, {"from": me});
    # Try to mint
    with brownie.reverts("Sender not contract owner"):
        token._mint(tokenID, {"from": bob});

#
# Cannot mint an existing token
#
def test_mint_tokenExists(token):
    me = accounts[0];
    bob = accounts[1]
    tokenID = 1;
    _ensureToken(token, tokenID, me);
    # Try to mint
    with brownie.reverts("Token already exists"):
        token._mint(tokenID, {"from": me});


#
# Burn a token
#
def test_burn(token):
    me = accounts[0];
    tokenID = 1;
    # Make sure that token does not yet exist
    token._burn(tokenID, {"from": me});
    # Mint
    token._mint(tokenID, {"from": me});
    # Now burn it
    txn_receipt = token._burn(tokenID, {"from": me});
    # Verify that burning has created an event
    _verifyTransferEvent(txn_receipt, me, "0x"+40*"0", tokenID);

#
# Only the contract owner can burn
#
def test_burn_notOwner(token):
    me = accounts[0];
    bob = accounts[1]
    tokenID = 1;
    _ensureToken(token, tokenID, me);
    # Try to burn
    with brownie.reverts("Sender not contract owner"):
        token._burn(tokenID, {"from": bob});

#
# Get owner of non-existing token
#
def test_ownerOf_invalidTokenID(token):
    me = accounts[0];
    tokenID = 1;
    token._burn(tokenID, {"from": me});
    # Try to call ownerOf
    with brownie.reverts("Invalid token ID"):
        token.ownerOf(tokenID);


#
# Test a valid transfer, initiated by the current owner of the token
#
def test_transferFrom(token):
    me = accounts[0];
    alice = accounts[1];
    tokenID = 1;
    _ensureToken(token, tokenID, me);
    # Remember balances
    oldBalanceMe = token.balanceOf(me);
    oldBalanceAlice = token.balanceOf(alice);
    # Now do the transfer
    txn_receipt = token.transferFrom(me, alice, tokenID, {"from": me});
    # check owner of NFT
    assert(alice == token.ownerOf(tokenID));
    # Check balances
    newBalanceMe = token.balanceOf(me);
    newBalanceAlice = token.balanceOf(alice);
    assert (newBalanceMe + 1 == oldBalanceMe);
    assert (oldBalanceAlice + 1 == newBalanceAlice);
    # Verify that an Transfer event has been logged
    _verifyTransferEvent(txn_receipt, me, alice, tokenID);

#
# Test an invalid transfer - from is not current owner
#
def test_transferFrom_notOwner(token):
    me = accounts[0];
    alice = accounts[1];
    bob = accounts[2];
    tokenID = 1;
    _ensureToken(token, tokenID, me);
    # Now do the transfer
    with brownie.reverts("From not current owner"):
        token.transferFrom(bob, alice, tokenID,{"from": bob});

#
# Test an invalid transfer - to is the zero address
#
def test_transferFrom_toZeroAddress(token):
    me = accounts[0];
    tokenID = 1;
    _ensureToken(token, tokenID, me);
    # Now do the transfer
    with brownie.reverts("Address 0 is not valid"):
        token.transferFrom(me, "0x"+40*"0", tokenID, {"from": me});

#
# Test an invalid transfer - invalid token ID
#
def test_transferFrom_invalidTokenID(token):
    me = accounts[0];
    alice = accounts[1];
    tokenID = 1;
    token._burn(tokenID, {"from": me});
    # Now do the transfer
    with brownie.reverts("Invalid token ID"):
        token.transferFrom(me, alice, tokenID, {"from": me});

#
# Test an invalid transfer - not authorized
#
def test_transferFrom_notAuthorized(token):
    me = accounts[0];
    alice = accounts[1];
    bob = accounts[2];
    tokenID = 1;
    _ensureToken(token, tokenID, me);
    # Now do the transfer
    with brownie.reverts("Sender not authorized"):
        token.transferFrom(me, alice, tokenID, {"from": bob});

#
# Test a valid safe transfer, initiated by the current owner of the token
#
def test_safeTransferFrom(token):
    me = accounts[0];
    alice = accounts[1];
    tokenID = 1;
    _ensureToken(token, tokenID, me);
    # Remember balances
    oldBalanceMe = token.balanceOf(me);
    oldBalanceAlice = token.balanceOf(alice);
    # Now do the transfer
    txn_receipt = token.safeTransferFrom(me, alice, tokenID, hexbytes.HexBytes(""), {"from": me});
    # check owner of NFT
    assert(alice == token.ownerOf(tokenID));
    # Check balances
    newBalanceMe = token.balanceOf(me);
    newBalanceAlice = token.balanceOf(alice);
    assert (newBalanceMe + 1 == oldBalanceMe);
    assert (oldBalanceAlice + 1 == newBalanceAlice);
    # Verify that an Transfer event has been logged
    _verifyTransferEvent(txn_receipt, me, alice, tokenID);

#
# Test an invalid safe transfer - from is not current owner
#
def test_safeTransferFrom_notOwner(token):
    me = accounts[0];
    alice = accounts[1];
    bob = accounts[2];
    tokenID = 1;
    _ensureToken(token, tokenID, me);
    # Now do the transfer
    with brownie.reverts("From not current owner"):
        token.safeTransferFrom(bob, alice, tokenID, hexbytes.HexBytes(""), {"from": bob});

#
# Test an safe invalid transfer - to is the zero address
#
def test_safeTransferFrom_toZeroAddress(token):
    me = accounts[0];
    tokenID = 1;
    _ensureToken(token, tokenID, me);
    # Now do the transfer
    with brownie.reverts("Cannot send to zero address"):
        token.safeTransferFrom(me, "0x"+40*"0", tokenID, hexbytes.HexBytes(""), {"from": me});

#
# Test an invalid safe transfer - invalid token ID
#
def test_safeTransferFrom_toZeroAddress(token):
    me = accounts[0];
    alice = accounts[1];
    tokenID = 1;
    # Make sure that token does not exist
    token._burn(tokenID, {"from": me});
    # Now do the transfer
    with brownie.reverts("Invalid token ID"):
        token.safeTransferFrom(me, alice, tokenID, hexbytes.HexBytes(""), {"from": me});

#
# Test an invalid safe transfer - not authorized
#
def test_safeTransferFrom_notAuthorized(token):
    me = accounts[0];
    alice = accounts[1];
    bob = accounts[2];
    tokenID = 1;
    _ensureToken(token, tokenID, me);
    # Now do the transfer
    with brownie.reverts("Sender not authorized"):
        token.safeTransferFrom(me, alice, tokenID, hexbytes.HexBytes(""), {"from": bob});

#
# Test a valid safe transfer to a contract returning the proper magic value
#
def test_safeTransferFrom(token, tokenReceiver):
    data = "0x1234";
    me = accounts[0];
    tokenID = 1;
    _ensureToken(token, tokenID, me);
    # get current invocation count of test contract
    oldInvocationCount = tokenReceiver.getInvocationCount();
    # Remember balances
    oldBalanceMe = token.balanceOf(me);
    oldBalanceToken = token.balanceOf(tokenReceiver.address);
    # Make sure that the contract returns the correct magic value
    tokenReceiver.setReturnCorrectValue(True);
    # Now do the transfer
    txn_receipt = token.safeTransferFrom(me, tokenReceiver.address, tokenID, hexbytes.HexBytes(data), {"from": me});
    # check owner of NFT
    assert(tokenReceiver.address == token.ownerOf(tokenID));
    # Check balances
    newBalanceMe = token.balanceOf(me);
    newBalanceToken = token.balanceOf(tokenReceiver.address);
    assert (newBalanceMe + 1 == oldBalanceMe);
    assert (oldBalanceToken + 1 == newBalanceToken);
    # get current invocation count of test contract
    newInvocationCount = tokenReceiver.getInvocationCount();
    assert(oldInvocationCount + 1 == newInvocationCount);
    # Check that data has been stored
    assert(tokenReceiver.getData() == data);
    # Verify that an Transfer event has been logged
    _verifyTransferEvent(txn_receipt, me, tokenReceiver.address, tokenID);

#
# Test a valid safe transfer to a contract returning the wrong proper magic value
#
def test_safeTransferFrom_wrongMagicValue(token, tokenReceiver):
    me = accounts[0];
    tokenID = 1;
    _ensureToken(token, tokenID, me);
    # Make sure that the contract returns the wrong magic value
    tokenReceiver.setReturnCorrectValue(False);
    # Now do the transfer
    with brownie.reverts("Did not return magic value"):
        token.safeTransferFrom(me, tokenReceiver.address, tokenID, hexbytes.HexBytes(""), {"from": me});
    # Reset behaviour of test contract
    tokenReceiver.setReturnCorrectValue(True);

#
# Test a valid safe transfer to a contract returning the proper magic value - no data
#
def test_safeTransferFrom_noData(token, tokenReceiver):
    me = accounts[0];
    tokenID = 1;
    _ensureToken(token, tokenID, me);
    # get current invocation count of test contract
    oldInvocationCount = tokenReceiver.getInvocationCount();
    # Remember balances
    oldBalanceMe = token.balanceOf(me);
    oldBalanceToken = token.balanceOf(tokenReceiver.address);
    # Make sure that the contract returns the correct magic value
    tokenReceiver.setReturnCorrectValue(True);
    # Now do the transfer
    txn_receipt = token.safeTransferFrom(me, tokenReceiver.address, tokenID,  {"from": me});
    # check owner of NFT
    assert(tokenReceiver.address == token.ownerOf(tokenID));
    # Check balances
    newBalanceMe = token.balanceOf(me);
    newBalanceToken = token.balanceOf(tokenReceiver.address);
    assert (newBalanceMe + 1 == oldBalanceMe);
    assert (oldBalanceToken + 1 == newBalanceToken);
    # get current invocation count of test contract
    newInvocationCount = tokenReceiver.getInvocationCount();
    assert(oldInvocationCount + 1 == newInvocationCount);
    # Verify that an Transfer event has been logged
    _verifyTransferEvent(txn_receipt, me, tokenReceiver.address, tokenID);

#
# Test an approval which is not authorized
#
def test_approval_notAuthorized(token):
    me = accounts[0];
    alice = accounts[1];
    tokenID = 1;
    _ensureToken(token, tokenID, me);
    with brownie.reverts("Sender not authorized"):
        token.approve(alice, tokenID, {"from": alice});

#
# Test a valid transfer, initiated by an approved sender
#
def test_transferFrom_approved(token):
    me = accounts[0];
    alice = accounts[1];
    bob = accounts[2];
    tokenID = 1;
    _ensureToken(token, tokenID, me);
    # Approve
    token.approve(bob, tokenID, {"from": me});
    # Remember balances
    oldBalanceMe = token.balanceOf(me);
    oldBalanceAlice = token.balanceOf(alice);
    # Now do the transfer
    txn_receipt = token.transferFrom(me, alice, tokenID, {"from": bob});
    # check owner of NFT
    assert(alice == token.ownerOf(tokenID));
    # Check balances
    newBalanceMe = token.balanceOf(me);
    newBalanceAlice = token.balanceOf(alice);
    assert (newBalanceMe + 1 == oldBalanceMe);
    assert (oldBalanceAlice + 1 == newBalanceAlice);
    # Verify that an Transfer event has been logged
    _verifyTransferEvent(txn_receipt, me, alice, tokenID);

#
# Test setting and getting approval
#
def test_approval(token):
    me = accounts[0];
    alice = accounts[1];
    bob = accounts[2];
    tokenID = 1;
    # Make sure that token does not yet exist
    token._burn(tokenID);
    # Get approval - should raise
    with brownie.reverts("Invalid token ID"):
        token.getApproved(tokenID);
    # Approve - should raise
    with brownie.reverts("Invalid token ID"):
        token.approve(bob, tokenID, {"from": me});
    # Mint
    token._mint(tokenID, {"from": me});
    # Approve for bob 
    txn_receipt = token.approve(bob, tokenID, {"from": me});
    # Check
    assert(bob == token.getApproved(tokenID));
    # Verify events
    _verifyApprovalEvent(txn_receipt, me, bob, tokenID);


#
# Test that approval is reset to zero address if token is transferred
#
def test_approval_resetUponTransfer(token):
    me = accounts[0];
    alice = accounts[1];
    bob = accounts[2];
    tokenID = 1;
    _ensureToken(token, tokenID, me);
    # Approve for bob 
    token.approve(bob, tokenID, {"from": me});
    # Check
    assert(bob == token.getApproved(tokenID));
    # Do transfer
    token.transferFrom(me, alice, tokenID, {"from": bob});
    # Check that approval has been reset
    assert(("0x"+40*"0") == token.getApproved(tokenID));

#
# Test setting and clearing the operator flag
#
def test_setGetOperator(token):
    me = accounts[0];
    alice = accounts[1];
    bob = accounts[2];
    assert(False == token.isApprovedForAll(me, bob));
    assert(False == token.isApprovedForAll(me, alice));
    # Declare bob as operator for me 
    txn_receipt = token.setApprovalForAll(bob, True, {"from": me});
    # Check
    assert(True == token.isApprovedForAll(me, bob));
    assert(False == token.isApprovedForAll(me, alice));
    # Check events
    _verifyApprovalForAllEvent(txn_receipt, me, bob, True);
    # Do the same for alice
    txn_receipt = token.setApprovalForAll(alice, True, {"from": me});
    # Check
    assert(True == token.isApprovedForAll(me, bob));
    assert(True == token.isApprovedForAll(me, alice));
    # Check events
    _verifyApprovalForAllEvent(txn_receipt, me, alice, True);
    # Reset both
    txn_receipt = token.setApprovalForAll(bob, False, {"from": me});
    # Check events
    _verifyApprovalForAllEvent(txn_receipt, me, bob, False);
    txn_receipt = token.setApprovalForAll(alice, False, {"from": me});
    # Check events
    _verifyApprovalForAllEvent(txn_receipt, me, alice, False);
    # Check
    assert(False == token.isApprovedForAll(me, bob));
    assert(False == token.isApprovedForAll(me, alice));

#
# Test authorization logic for setting and getting approval
#
def test_approval_authorization(token):
    me = accounts[0];
    alice = accounts[1];
    bob = accounts[2];
    tokenID = 1;
    _ensureToken(token, tokenID, me);
    # Try to approve for bob while not being owner or operator - this should raise an exception
    with brownie.reverts("Sender not authorized"):
        token.approve(bob, tokenID, {"from": alice});
    # Now make alice an operator for me
    token.setApprovalForAll(alice, True, {"from": me});
    # Approve for bob again - this should now work
    txn_receipt = token.approve(bob, tokenID, {"from": alice});
    # Check
    assert(bob == token.getApproved(tokenID));
    # Verify events
    _verifyApprovalEvent(txn_receipt, me, bob, tokenID);
    # Reset
    token.setApprovalForAll(alice, False, {"from": me});

#
# Test a valid transfer, initiated by an operator for the current owner of the token
#
def test_transferFrom_operator(token):
    me = accounts[0];
    alice = accounts[1];
    bob = accounts[2];
    tokenID = 1;
    _ensureToken(token, tokenID, me);
    # Now make bob an operator for me
    token.setApprovalForAll(bob, True, {"from": me});
    # Remember balances
    oldBalanceMe = token.balanceOf(me);
    oldBalanceAlice = token.balanceOf(alice);
    # Now do the transfer
    txn_receipt = token.transferFrom(me, alice, tokenID, {"from": bob});
    # Reset
    token.setApprovalForAll(bob, False, {"from": me});
    # check owner of NFT
    assert(alice == token.ownerOf(tokenID));
    # Check balances
    newBalanceMe = token.balanceOf(me);
    newBalanceAlice = token.balanceOf(alice);
    assert (newBalanceMe + 1 == oldBalanceMe);
    assert (oldBalanceAlice + 1 == newBalanceAlice);
    # Verify that an Transfer event has been logged
    _verifyTransferEvent(txn_receipt, me, alice, tokenID);

#
# Test ERC165 functions
#
def test_ERC615(token):
    # ERC721
    assert(True == token.supportsInterface("0x80ac58cd"));
    # ERC165 itself
    assert(True == token.supportsInterface("0x01ffc9a7"));
    # ERC721 Metadata 
    assert(True == token.supportsInterface("0x5b5e139f"));

#
# Test name and symbol
#
def test_name_symbol(token):
    name = token.name();
    symbol = token.symbol();
    assert(len(name) > 0);
    assert(len(symbol) > 0);

#
# Test tokenURI
#
def test_tokenURI(token):
    me = accounts[0];
    tokenID = 1;
    # Make sure that token does not yet exist
    token._burn(tokenID, {"from": me});
    # Try to get tokenURI of invalid token - should raise exception
    with brownie.reverts("Invalid token ID"):
        token.tokenURI(tokenID);
    # Mint
    token._mint(tokenID, {"from": me});
    # Get base URI
    baseURI = token._getBaseURI();
    # Get token URI
    tokenURI = token.tokenURI(tokenID);
    assert(baseURI + "1" == tokenURI);

#
# Test tokenURI - token ID 0
#
def test_tokenURI_idzero(token):
    me = accounts[0];
    tokenID = 0;
    # Make sure that token does not yet exist
    token._burn(tokenID, {"from": me});
    # Try to get tokenURI of invalid token - should raise exception
    with brownie.reverts("Invalid token ID"):
        token.tokenURI(tokenID);
    # Mint
    token._mint(tokenID, {"from": me});
    # Get base URI
    baseURI = token._getBaseURI();
    # Get token URI
    tokenURI = token.tokenURI(tokenID);
    assert(baseURI + "0" == tokenURI);



