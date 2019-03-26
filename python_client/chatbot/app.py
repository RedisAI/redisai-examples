import os

from flask import Flask, request, jsonify
import utils
import redis_db

db_host = os.environ.get('REDIS_IP') or 'localhost'
db_port = os.environ.get('REDIS_PORT') or 6379


app = Flask(__name__)
redis_db = redis_db.DB(host=db_host, port=db_port)
redis_db.initiate()


@app.route('/chat', methods=['POST'])
def chat():
    req_data = request.get_json()
    message = get_bot_reply(req_data['message'])
    return jsonify(response=message)


def get_bot_reply(message):
    message = utils.normalize_string(message)
    indices = utils.get_batched_indices(message)
    numpy_array = utils.list2numpy(indices)
    length = utils.get_length(numpy_array)
    reply = redis_db.process(numpy_array, length)
    return reply
