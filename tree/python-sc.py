from flask import Flask

app = Flask(__name__)

@app.route("/join/<id>")
def join(id):
  id = int(id)
  return {"id": id, "parent": id, "children": [id * 2, id * 2 + 1]}

