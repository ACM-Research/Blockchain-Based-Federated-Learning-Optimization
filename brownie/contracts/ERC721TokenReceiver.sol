// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0 <0.9.0;


contract ERC721TokenReceiverImplementation {

    bytes4 internal constant magicValue = 0x150b7a02;
    bytes4 internal constant wrongMagicValue = 0x150b7a01;
    uint256 internal invocationCount;
    bool internal returnCorrectValue = true;
    bytes data; 

    function onERC721Received(address, address,  uint256, bytes memory _data ) external payable returns(bytes4) {
        invocationCount++;
        data = _data;
        if (returnCorrectValue) {
            return magicValue;
        }
        return wrongMagicValue;
    }

    function getInvocationCount() public view returns(uint256) {
        return invocationCount;
    }

    function getData() public view returns(bytes memory) {
        return data;
    }

    function setReturnCorrectValue(bool value) external {
        returnCorrectValue = value;
    }

}
