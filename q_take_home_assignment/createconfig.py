import json

data = {
    "learning_rate" : 0.0001,
    "threshold" : 0.5,
    "epochs" : 100
}

with open("config.json", "w") as outfile:
    json.dump(data, outfile)