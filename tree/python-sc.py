from flask import Flask

app = Flask(__name__)

@app.route("/")
def echo():
  id = request.args.get("id")
  return {"id": id, "parent": id // 2, "children": [id * 2, id * 2 + 1]}