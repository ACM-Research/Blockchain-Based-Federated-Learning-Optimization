// SPDX-License-Identifier: GPL-3.0


/**
 * We want to ensure that we use at least v0.8 of the Solidity compiler
 * where overflows are checked by default, so we no longer need a SafeMath
 * library - see https://docs.soliditylang.org/en/v0.8.6/control-structures.html#checked-or-unchecked-arithmetic
 */
pragma solidity >=0.8.0 <0.9.0;


/** 
 * @title NFT
 * @dev Implements an ERC721 token, see https://eips.ethereum.org/EIPS/eip-721
 */

contract NFT {

    event Transfer(
        address indexed from,
        address indexed to,
        uint256 indexed tokenID
    );

    event Approval(
        address indexed owner,
        address indexed spender,
        uint256 indexed tokenID
    );

    event ApprovalForAll(
        address indexed owner,
        address indexed operator,
        bool approved
    );

    /// magic value returned by a contract
    /// implement ERC721Receiver
    bytes4 private constant magicValue = 0x150b7a02;

    /// Interface IDs
    bytes4 private constant erc165InterfaceID = 0x01ffc9a7;
    bytes4 private constant erc721InterfaceID = 0x80ac58cd;
    bytes4 private constant erc721metadataID = 0x5b5e139f;


    /// The owner of the contract
    address private _contractOwner;

    /// The base URI
    string private _baseURI;

    /// The owner of each token
    mapping (uint256 => address) private _ownerOf;

    /// The balance of NFT for each address
    mapping (address => uint256) private _balances;

    /// Keep track of approvals per tokenID
    mapping (uint256 => address) private _approvals; 

    /// Keep track of operators
    mapping (address => mapping(address => bool)) private _isOperatorFor;

    /// Name and symbol of the contract
    string private constant _name = "Non-fungible token";
    string private constant _symbol = "MNFT";


    /// Messages
    string private constant _requiresOwner = "Sender not contract owner";
    string private constant _invalidTokenID = "Invalid token ID";
    string private constant _invalidAddress = "Address 0 is not valid";
    string private constant _senderNotAuthorized = "Sender not authorized";

    ///
    /// Constructor - remember who the contract owner is and assign initial balance
    /// Also set the baseURI
    ///
    constructor(string memory baseURI)  {
        _baseURI = baseURI;
        _contractOwner = msg.sender;
    }

    /// Modifier to check that the sender of the msg is the 
    /// contract owner

    modifier isContractOwner() {
        require(msg.sender == _contractOwner, _requiresOwner);
        _;
    }

    /// Modifier to check that a token ID is valid
    modifier isValidToken(uint256 _tokenID) {
        require(_ownerOf[_tokenID] != address(0), _invalidTokenID);
        _;
    }

    /// Modifier to check that an address is valid
    modifier isValidAddress(address _address) {
        require(_address != address(0), _invalidAddress);
        _;
    }

    /// Return the count of all NFTs assigned to an owner. Throw
    /// if the queried address is 0
    /// @param owner An address for whom to query the balance
    /// @return The number of NFTs owned by `owner`, possibly zero
    function balanceOf(address owner) external view isValidAddress(owner) returns (uint256) {
        return _balances[owner];
    }

    /// Return the owner of an NFT. If the result is the zero address,
    /// the token is considered invalid and we throw
    /// @param tokenID The identifier for an NFT
    /// @return The address of the owner of the NFT
    function ownerOf(uint256 tokenID) external view isValidToken(tokenID) returns (address)  {
        return _ownerOf[tokenID];
    }

    /// Mint a token. This is a non-standard extension.
    function _mint(uint256 tokenID) external isContractOwner {
        require(_ownerOf[tokenID] == address(0), "Token already exists");
        _balances[_contractOwner] +=1;
        _ownerOf[tokenID] = _contractOwner;
        /// Emit event
        emit Transfer(address(0), _contractOwner, tokenID);
    }

    /// Burn a token. Do nothing if the token does not exist instead
    /// of reverting, so that we can use this to make sure that a token
    /// does not exist. 
    /// IMPORTANT: the contract owner can burn EVERY token, regardless
    /// of who the current owner is - this is nothing you would accept
    /// for a real NFT. DO NOT DO THIS IN PRODUCTION
    function _burn(uint256 tokenID) external isContractOwner {
        address owner = _ownerOf[tokenID];
        if (owner == address(0)) {
            return;
        }
        _balances[owner] -=1;
        _ownerOf[tokenID] = address(0);
        _approvals[tokenID] = address(0);
        /// Emit event
        emit Transfer(owner, address(0), tokenID);
    }


    /// Transfer ownership of an NFT. Throws unless `msg.sender` is the current owner, an authorized
    ///  operator, or the approved address for this NFT. Throws if `from` is
    ///  not the current owner. Throws if `to` is the zero address. Throws if
    ///  `tokenId` is not a valid NFT.
    /// @param from The current owner of the NFT
    /// @param to The new owner
    /// @param tokenID The NFT to transfer
    function transferFrom(address from, address to, uint256 tokenID) external payable {
        _doTransferFrom(from, to, tokenID);
    }

    function _doTransferFrom(address from, address to, uint256 tokenID) private isValidToken(tokenID) isValidAddress(to) {
        address currentOwner = _ownerOf[tokenID];
        require(from == currentOwner, "From not current owner");
        bool authorized = (msg.sender == from) 
                            || (_approvals[tokenID] == msg.sender) 
                            || (_isOperatorFor[currentOwner][msg.sender]);
        require(authorized, _senderNotAuthorized);
        _balances[currentOwner]-=1;
        _balances[to]+=1;
        _ownerOf[tokenID] = to;
        _approvals[tokenID] = address(0);
        /// Emit transfer event. My interpretation of the standard is that this event
        /// is sufficient to also indicate that the approval has been reset. This is in line
        /// with the 0xcert implementation (https://github.com/0xcert/ethereum-erc721/blob/master/src/contracts/tokens/nf-token.sol)
        /// but deviates from the OpenZeppelin implementation, see, however, also this issue
        /// https://github.com/OpenZeppelin/openzeppelin-contracts/issues/1038
        /// which seems to support this point of view
        emit Transfer(from, to, tokenID);
    }

    function _isContract(address addr) private view returns (bool){
        uint32 size;
        assembly {
            size := extcodesize(addr)
        }
        return (size > 0);
    }


    function _invokeOnERC721Received(address to, address operator, address from, uint256 tokenID, bytes memory data) private {
        if (_isContract(to)) {
            ERC721TokenReceiver erc721Receiver = ERC721TokenReceiver(to);
            bytes4 retval = erc721Receiver.onERC721Received(operator, from, tokenID, data);
            require(retval == magicValue, "Did not return magic value");
        }
    }

    /// Transfers the ownership of an NFT from one address to another address
    /// Throws if the sender is not authorized or if from is not the current owner
    ///  Throws if `to` is the zero address. Throws if
    ///  `tokenID` is not a valid NFT. When a transfer is complete, this function
    ///  checks if `to` is a smart contract (code size > 0). If so, it calls
    ///  `onERC721Received` on `to` and throws if the return value is not
    ///  `bytes4(keccak256("onERC721Received(address,address,uint256,bytes)"))`.
    /// @param from The current owner of the NFT
    /// @param to The new owner
    /// @param tokenID The NFT to transfer
    /// @param data Additional data with no specified format, sent in call to `_to`
    function safeTransferFrom(address from, address to, uint256 tokenID, bytes memory data) external payable {
        _doTransferFrom(from, to, tokenID);
        _invokeOnERC721Received(to, msg.sender, from, tokenID, data);
    }

    /// This works identically to the other function with an extra data parameter,
    /// except this function just sets data to "".
    /// @param from The current owner of the NFT
    /// @param to The new owner
    /// @param tokenID The NFT to transfer
    function safeTransferFrom(address from, address to, uint256 tokenID) external payable {
        _doTransferFrom(from, to, tokenID);
        _invokeOnERC721Received(to, msg.sender, from, tokenID, bytes(""));
    }

    /// Change or reaffirm the approved address for an NFT
    /// The zero address indicates there is no approved address.
    /// Throws unless `msg.sender` is the current NFT owner, or an authorized
    /// operator of the current owner.
    /// @param approved The new approved NFT controller
    /// @param tokenID The NFT to approve
    function approve(address approved, uint256 tokenID) external payable isValidToken(tokenID) {
        address currentOwner = _ownerOf[tokenID];
        bool authorized = (msg.sender == currentOwner) 
                           || (_isOperatorFor[currentOwner][msg.sender]);
        require(authorized, _senderNotAuthorized);
        _approvals[tokenID] = approved;
        emit Approval(_ownerOf[tokenID], approved, tokenID);
    }

    /// Get the approved address for a single NFT
    /// Throws if `tokenID` is not a valid NFT.
    /// @param tokenID The NFT to find the approved address for
    /// @return The approved address for this NFT, or the zero address if there is none
    function getApproved(uint256 tokenID) external view isValidToken(tokenID) returns (address) {
        return _approvals[tokenID];
    }

    /// Enable or disable approval for a third party ("operator") to manage
    /// all of `msg.sender`'s assets
    /// Emits the ApprovalForAll event. The contract MUST allow
    /// multiple operators per owner.
    /// @param operator Address to add to the set of authorized operators
    /// @param approved True if the operator is approved, false to revoke approval
    function setApprovalForAll(address operator, bool approved) external {
        _isOperatorFor[msg.sender][operator] = approved;
        emit ApprovalForAll(msg.sender, operator, approved);
    }

    /// Query if an address is an authorized operator for another address
    /// @param owner The address that owns the NFTs
    /// @param operator The address that acts on behalf of the owner
    /// @return True if `operator` is an approved operator for `owner`, false otherwise
    function isApprovedForAll(address owner, address operator) external view returns (bool) {
        return _isOperatorFor[owner][operator];
    }

    /// ERC165 - supportsInterface implementation
    /// see https://github.com/ethereum/EIPs/blob/master/EIPS/eip-165.md
    function supportsInterface(bytes4 interfaceID) external pure returns (bool) {
        return (interfaceID == erc165InterfaceID) 
                || (interfaceID == erc721InterfaceID)
                || (interfaceID == erc721metadataID);
    }

    /// A descriptive name for a collection of NFTs in this contract
    /// part of the ERC721 Metadata extension
    function name() external pure returns (string memory) {
        return _name;
    }

    /// An abbreviated name for NFTs in this contract
    /// part of the ERC721 Metadata extension
    function symbol() external pure returns (string memory) {
        return _symbol;
    }

    function _toString(uint256 value) private pure returns (string memory) {
        /// taken from OpenZeppelin 
        /// https://github.com/OpenZeppelin/openzeppelin-contracts/blob/master/contracts/utils/Strings.sol
        /// MIT licensed
        if (value == 0) {
            return "0";
        }
        uint256 temp = value;
        uint256 digits;
        while (temp != 0) {
            digits++;
            temp /= 10;
        }
        bytes memory buffer = new bytes(digits);
        while (value != 0) {
            digits -= 1;
            buffer[digits] = bytes1(uint8(48 + uint256(value % 10)));
            value /= 10;
        }
        return string(buffer);
    }    

    /// A distinct Uniform Resource Identifier (URI) for a given asset.
    /// Throws if `tokenID` is not a valid NFT. 
    function tokenURI(uint256 tokenID) external view isValidToken(tokenID) returns (string memory) {
        return string(abi.encodePacked(_baseURI, _toString(tokenID)));
    }

    /// A non-standard function to retrieve the baseURI
    function _getBaseURI() external view returns (string memory) {
        return _baseURI;
    }
}
/**
 * ERC-721 interface for accepting safe transfers.
 * See https://github.com/ethereum/EIPs/blob/master/EIPS/eip-721.md.
 */
interface ERC721TokenReceiver
{
  function onERC721Received(address, address, uint256, bytes calldata) external returns(bytes4);
}
