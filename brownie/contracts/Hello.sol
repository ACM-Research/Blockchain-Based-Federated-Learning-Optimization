// SPDX-License-Identifier: MIT


pragma solidity >=0.8.0 <0.9.0;


contract Hello {

    event SayHello();


    /**
     * Say hello by emitting an event
     */
    function sayHello() public {
        emit SayHello();
    }
}
