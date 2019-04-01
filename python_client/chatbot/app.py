import os

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import utils
import redis_db

db_host = os.environ.get('REDIS_IP') or 'localhost'
db_port = os.environ.get('REDIS_PORT') or 6379


app = Flask(__name__)
CORS(app)
redis_db = redis_db.DB(host=db_host, port=db_port)
redis_db.initiate()


@app.route('/chat', methods=['POST'])
def chat():
    req_data = request.get_json()
    message = utils.normalize_string((req_data['message']))
    try:
        indices = utils.get_batched_indices(message)
    except KeyError:
        reply = "I did not understand your language!!, check the spelling perhaps"
    else:
        numpy_array = utils.list2numpy(indices)
        length = utils.get_length(numpy_array)
        reply = redis_db.process(numpy_array, length)

    resp = jsonify(reply=reply)
    return resp


@app.route('/<path:filepath>')
def ui_components(filepath):
    return send_from_directory('static', filepath)


@app.route('/')
def ui():
    return send_from_directory('static', 'index.html')
