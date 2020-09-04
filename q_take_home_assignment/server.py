import flask
from flask import request, jsonify
from TrainAndTest import test
import os
import time

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/api', methods=['POST'])
def home():
    data = request.get_json()["strings"]
    string1 = data["string1"]
    string2 = data["string2"]

    res = test(string1, string2)

    return jsonify(res)


@app.route('/hello', methods=['GET'])
def landing():
    return jsonify("Hello World")


app.run()