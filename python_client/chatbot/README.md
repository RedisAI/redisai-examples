## How to run the Demo

Since we wanted to keep the directory structure intact, redisai_examples/models/ has all the models. For our flask app to get the models, we'll have to run the Docker build from the root directory of this example repo. Also, you might have to run the Redis server and bind it to 0.0.0.0 to make it accessible from other network, even if the docker and Redis server are in the same machine. Below given are the steps to follow to make the server up and run the example.

- cd rediai_examples repo's root folder
- Since Dockerfile is in chatbot folder, pass that as `-f` option. Run `docker build -t redisai_chatbot -f python_client/chatbot/Dockerfile .`
- Run `docker run -e REDIS_IP='<Server_IP>' -p 5000:5000 redisai_chatbot` to bring up the flask server inside docker
- Run redis server with `--bind` and bind it to `0.0.0.0` -> `redis-server --loadmodule build/redisai.so --bind 0.0.0.0`

Now the RedisServer and Flask application are up. If you want to try the API, we have only one API endpoint -> `/chat` which accpets `message` as the json key with your message as value.

```
curl http://localhost:5000/chat -H "Content-Type: application/json" -d '{"message": "I think I am crazy"}'
```

But we have made a naive UI also for you to chat from the browser. Go to the URL `http://localhost:5000` to see it in action.
