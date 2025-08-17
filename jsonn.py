import json


with open('/Users/anatolii/Desktop/dd.json', 'r') as f:
    dd = json.load(f)

print(len(dd))