# redisai-examples

**A set of examples those showcases the features of RedisAI**

## Setting UP
- Clone the repository
```
git clone git@github.com:RedisAI/redisai-examples.git
```

- Install [git lfs](https://help.github.com/en/articles/installing-git-large-file-storage).
- Pull model files and assets with git lfs
```
git-lfs pull
```
- You need python3.6+ for running python examples. Use conda environment file for installing dependencies. Nodejs folder has package.json file for installing dependencies.


### Steps to run python examples

- All the model instance are saved in the `models` folder. But if you want to execute them and create the model file yourself, scripts are also available.
- Once the model is created, you need the full pipeline. Essentially model instance is one spot in your pipeline. A full pipelined model is also created in each corresponding folder inside `models` folder. But again, if you want to run it by yourself, script is available
- For some models, the pipeline is defined as torch module and created a single model file. But for some the pre processing and post processing is kept as `SCRIPT`s.
- Once the model files are ready to be loaded into Redis, python client code is available in `python` folder.

Steps to run the sentinel example is documented inside sentinel directory