## Setting up Sentinel

The whole configuration process is scripted in this folder for setting up three node Redis server and three node sentinel instance. We are setting up one sentinel and one Redis-server in one node and hence need three nodes.

- SSH to the server and run `installation.sh`: This will set up the Redis-server, RedisAI, and prerequisites required. This step installs RedisAI for CPU, if you need to try out GPU based sentinel setup, you might have to make a slight tweak in the script.
- Execute setup.py as `python setup.py --master "<master IP>" --slaves "<comma separated slave IPs>" --passphrase <ssh key passphrase> --user <username>` (Make sure you have fabric installed). This copies the configuration (redis.conf and sentinel.conf) to remote machines. The script we made assumes home folder in all the machines are the same (username is the same for all machines)
- Run `run_server.sh` in each machine to bring up the Redis master and slave instances with RedisAI loaded.
- For testing: run `model_set.py` on the master which sets the imagenet - pytorch model on the master instance. Then go ahead and bring down the master instance. This makes the sentinels elect the new master. Now run `model_run.py` (old master is still down) to get the model result.