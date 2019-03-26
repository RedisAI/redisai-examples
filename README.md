# redisai-examples

**A set of examples those showcases the features of RedisAI**

All the models (both serialized models and the code to generate the same) are kept inside the `models` folder as subfolders with all the dependancies included. Right now we have example client implementations for the models in JS and Python.

### Steps to run python examples

- All the model instance are saved in the `models` folder. But if you want to execute them and create the model file yourself, scripts are also available.
- Once the model is created, you need the full pipeline. Essentially model instance is one spot in your pipeline. A full pipelined model is also created in each corresponding folder inside `models` folder. But again, if you want to run it by yourself, script is available
- For some models, the pipeline is defined as torch module and created a single model file. But for some the pre processing and post processing is kept as `SCRIPT`s.
- Once the model files are ready to be loaded into Redis, python client code is available in `python` folder.

Steps to run the sentinel example is documented inside sentinel directory