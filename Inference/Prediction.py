import json
import requests

data = [
    json.dumps({'Feature1': 42, 'Feature2': 'text value 1'}),
    json.dumps({'Feature1': 60, 'Feature2': 'text value 2'}),
]

api_key = '...'
api_endpoint = '...'

session = requests.Session()
session.headers = {
    'Authorization': 'Bearer {}'.format(api_key),
    'Content-Type': 'text/json',
}

for row in data:
    print(session.post(api_endpoint, data=row).json())
