import React, { useEffect, useRef, useState } from "react";

import { render } from "react-dom";
import "./style.css";
import { CircularProgress } from "@mui/material";

var ws = new WebSocket("ws://localhost:8765");

var clientW = 400;
var clientH = 200;

var true_elements = {};

const App = () => {
	const [elements, setElements] = useState({});
	const [iteration, setIteration] = useState(0);
	const [accuracy, setAccuracy] = useState(0);
	const [gas, setGas] = useState(0);

	useEffect(() => {
		if (true_elements !== elements) {
			console.log(elements);
			setElements(true_elements);
		}
	}, [elements]);

	function createClients(tree) {
		const newElements = {};
		const N = tree.length;

		// get depth of tree
		let height = Math.ceil(Math.log(N + 1) / Math.log(2)) - 1;
		var lastDepth = 0;
		var firstElem = 0;
		// make a vertical list of clients for each depth
		for (let i = 0; i < tree.length; i++) {
			const client = tree[i]; // [address, incentive, name]
			const depth = Math.floor(Math.log(i + 1) / Math.log(2));
			if (depth !== lastDepth) {
				// update position of first element in this depth
				lastDepth = depth;
				firstElem = i;
			}
			const x = depth * (clientW + 100) + 50;
			const y = (i - firstElem) * (clientH + 100) + 50;
			const ip = client[2];
			const port = ip.split(":")[1];
			newElements[port] = new Client(x, y, clientW, clientH, ip);

			// get parent and add to next element
			const parentIndex = Math.floor((i + 1) / 2) - 1;
			if (parentIndex >= 0) {
				const parent = tree[parentIndex];
				const parentPort = parent[2].split(":")[1];
				const parentX = newElements[parentPort].x;
				const parentY = newElements[parentPort].y;

				// draw line to parent
				newElements[port].setNextElement(
					parentX + clientW / 2,
					parentY + clientH / 2
				);
			}
		}

		true_elements = newElements;

		// console.log(newElements);

		setElements(newElements);
	}

	function parseClients(tree) {
		// var tree = Object.values(dict);
		const N = tree.length;

		// get depth of tree
		let height = Math.ceil(Math.log(N + 1) / Math.log(2)) - 1;

		var lastDepth = 0;
		var firstElem = 0;
		// make a vertical list of clients for each depth
		for (let i = 0; i < tree.length; i++) {
			const client = tree[i]; // [address, incentive, name]
			const depth = Math.floor(Math.log(i + 1) / Math.log(2));
			if (depth !== lastDepth) {
				// update position of first element in this depth
				lastDepth = depth;
				firstElem = i;
			}
			const x = depth * (clientW + 100) + 50;
			const y = (i - firstElem) * (clientH + 100) + 50;

			// check if client already exists
			var ip = client[2];
			const port = ip.split(":")[1];

			true_elements[port].setNewPosition(x, y);
			true_elements[port].status = "connecting";

			// get parent and add to next element
			const parentIndex = Math.floor((i + 1) / 2) - 1;
			if (parentIndex >= 0) {
				const parent = tree[parentIndex];
				const parentPort = parent[2].split(":")[1];
				const parentX = true_elements[parentPort].x;
				const parentY = true_elements[parentPort].y;

				// draw line to parent
				true_elements[port].setNextElement(
					parentX + clientW / 2,
					parentY + clientH / 2
				);
			}
		}


		// console.log(newElements);

		setElements(true_elements);
	}

	useEffect(() => {
		ws.onmessage = (event) => {
			console.log("message");
			console.log(event.data);
			const data = JSON.parse(event.data);
			if (data.type === "tree") {
				// turn tree array into dictionary of clients
				var tree = data.tree;
				if (Object.keys(true_elements).length === 0) createClients(tree);
				else parseClients(tree);
			} else if (data.type === "status") {
				// update status of client
				const port = data.port;
				const status = data.status;
				true_elements[port].status = status;
				setElements(true_elements);
			} else if (data.type === "accuracy") {
				// update accuracy of model
				setAccuracy(data.accuracy);
			} else if (data.type === "iteration") {
				// update iterations of model
				setIteration(data.iteration);
			} else if (data.type === "gas") {
				// update gas of model
				setGas(data.gas);
			}
		};
		ws.onopen = () => {
			console.log("connected!");
			ws.send(JSON.stringify({ type: "front" }));
		};
		ws.onclose = () => {
			console.log("disconnected");
		};
	}, []);

	return (
		<div className="flex-page">
			<Canvas elements={elements} />
			<Header iterations={iteration} accuracy={accuracy} gas={gas} />
		</div>
	);
};

function Header(props) {
	return (
		<div className="header">
			{props.iterations === 0 ? <CircularProgress /> : null}
			<h3>BCFL Visualization</h3>
			<h3>Iteration: {props.iterations}</h3>
			<h3>Accuracy: {props.accuracy}%</h3>
			<h3>Gas: {props.gas}</h3>
		</div>
	);
}

var interval = null;
function Canvas(props) {
	// const [elements, setElements] = useState([]);
	const canvasRef = useRef(null);

	function drawGrid(ctx, w, h, step) {
		ctx.beginPath();
		for (let x = 0; x <= w; x += step) {
			ctx.moveTo(x, 0);
			ctx.lineTo(x, h);
		}
		for (let y = 0; y <= h; y += step) {
			ctx.moveTo(0, y);
			ctx.lineTo(w, y);
		}
		ctx.lineWidth = 1;
		ctx.stroke();
	}

	function update() {
		if (!canvasRef.current) return;
		console.log("update");
		const canvas = canvasRef.current;
		const ctx = canvas.getContext("2d");
		// clear canvas
		ctx.clearRect(0, 0, canvas.width, canvas.height);
		// draw grid
		ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
		drawGrid(ctx, canvas.width, canvas.height, 50);
		// draw elements

		for (const [key, value] of Object.entries(props.elements)) {
			value.drawLineTo(ctx);
		}
		for (const [key, value] of Object.entries(props.elements)) {
			value.draw(ctx);
		}
	}

	useEffect(() => {}, []);

	useEffect(() => {
		clearInterval(interval);
		interval = setInterval(() => {
			update();
		}, 1000 / 60);
	}, [props.elements]);

	return (
		<canvas
			id="canvas"
			width={window.innerWidth * 2}
			height={window.innerHeight * 2}
			ref={canvasRef}
		/>
	);
}

class Client {
	constructor(
		x = 0,
		y = 0,
		w = 50,
		h = 50,
		name = "Client",
		status = "connecting"
	) {
		this.x = x;
		this.y = y;
		this.newX = x;
		this.newY = y;
		this.oldX = x;
		this.oldY = y;
		this.w = w;
		this.h = h;
		this.name = name;
		this.status = status;
		this.extra = "";
		this.color = "rgba(35, 35, 35, 1)";
		this.textColor = "rgba(255, 255, 255, 1)";
		this.nextX = x + w / 2;
		this.nextY = y + h / 2;
		this.opacity = 1;
	}

	setNextElement(x, y) {
		this.nextX = x;
		this.nextY = y;
	}

	drawLineTo(ctx) {
		var x = this.nextX;
		var y = this.nextY;
		// draw a bezier curve from the center of the client to the given x and y
		ctx.strokeStyle = "rgba(255, 255, 255, " + 0.5 * this.opacity + ")";
		ctx.beginPath();
		ctx.moveTo(this.x + this.w / 2, this.y + this.h / 2);
		ctx.bezierCurveTo(x, this.y + this.h / 2, this.x + this.w / 2, y, x, y);

		// make line thicker
		ctx.lineWidth = 3;
		ctx.stroke();
	}

	setNewPosition(x, y) {
		this.newX = x;
		this.newY = y;
		this.oldX = this.x;
		this.oldY = this.y;
	}

	draw(ctx) {
		if (this.status === "connecting" || this.status === "training") {
			this.opacity = 0.5;
		} else {
			this.opacity = 1;
		}

		// get direction to move
		var xDist = this.newX - this.x;
		var yDist = this.newY - this.y;
		var dist = Math.sqrt(xDist * xDist + yDist * yDist);
		var xSpeed = xDist / dist;
		var ySpeed = yDist / dist;
		var speed = 15;
		// move towards new position
		if (Math.abs(this.x - this.newX) < 10) this.x = this.newX;
		else this.x += xSpeed * speed;

		if (Math.abs(this.y - this.newY) < 10) this.y = this.newY;
		else this.y += ySpeed * speed;

		// draw client
		ctx.fillStyle = this.color;
		// draw curved corner
		ctx.beginPath();
		ctx.moveTo(this.x + this.w / 2, this.y);
		ctx.arcTo(this.x + this.w, this.y, this.x + this.w, this.y + this.h, 25);
		ctx.arcTo(this.x + this.w, this.y + this.h, this.x, this.y + this.h, 25);
		ctx.arcTo(this.x, this.y + this.h, this.x, this.y, 25);
		ctx.arcTo(this.x, this.y, this.x + this.w, this.y, 25);
		ctx.fill();
		// draw text
		ctx.fillStyle = "rgba(255, 255, 255, " + this.opacity + ")";
		ctx.font = "30px Arial";
		ctx.fillText("Client: " + this.name, this.x + 20, this.y + 50);
		if (this.status === "connecting" || this.status === "training") {
			// red
			ctx.fillStyle = "rgba(255, 0, 0, " + this.opacity + ")";
		} else if (this.status === "averaged" || this.status === "trained") {
			// green
			ctx.fillStyle = "rgba(0, 255, 0, " + this.opacity + ")";
		} else {
			// yellow
			ctx.fillStyle = "rgba(255, 255, 0, " + this.opacity + ")";
		}
		ctx.font = "25px Arial";
		ctx.fillText("Status: " + this.status, this.x + 20, this.y + 90);
	}

	getIfMouseOver(x, y) {
		return (
			x > this.x && x < this.x + this.w && y > this.y && y < this.y + this.h
		);
	}
}

render(<App />, document.getElementById("root"));
