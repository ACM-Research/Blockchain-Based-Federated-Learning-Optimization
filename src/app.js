import React, { useEffect } from "react";

import { render } from "react-dom";
import "./style.css";

var ws = new WebSocket("ws://localhost:8765");
const App = () => {

	useEffect(() => {
		ws.onmessage = (event) => {
			console.log("message");
			console.log(event.data);
		};
		ws.onopen = () => {
			console.log("connected!");
			ws.send("front");
		};
	}, []);

	return (
		<div>
			<h1>Hello World!</h1>
		</div>
	);
};

render(<App />, document.getElementById("root"));