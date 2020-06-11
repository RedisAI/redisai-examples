[![license](https://img.shields.io/github/license/RedisAI/redisai-examples.svg)](https://github.com/RedisAI/redisai-examples)
[![Forum](https://img.shields.io/badge/Forum-RedisAI-blue)](https://forum.redislabs.com/c/modules/redisai)
[![Gitter](https://badges.gitter.im/RedisLabs/RedisAI.svg)](https://gitter.im/RedisLabs/RedisAI?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)


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
- You need Python 3.6+ for running Python examples. Use the conda environment file for installing Python dependencies. The Node.js folder contains a package.json file for installing Node.js dependencies.
- Python examples accept device information and connection paramters over command line arguments. Ex: `python torch_imagenet.py --gpu --host aws.com` will run the example on RedisAI GPU hosted at aws.com
- Python examples use the converter package [ml2rt](https://github.com/hhsecond/ml2rt) for loading the models and script 

### ml2rt
ml2rt is a set of machine learning utilities for model conversion, serialization, loading etc. We use ml2rt for
- Saving Tensorflow, PyTorch and ONNX models to disk
- Converting models from other frameworks like spark, sklearn to ONNX
- Loading models and script from disk

Checkout [the repository](https://github.com/hhsecond/ml2rt) for the complete documentation

### Structure

- `models` folder consist of subfolders for each framework/package. Currently we have examples for
    - [PyTorch](./models/pytorch)
    - [Tensorflow](./models/tensorflow)
    - [Scikit-Learn](./models/sklearn)
    - [SparkML](./models/spark)
    - [CoreML (coming soon)](./models/coreml)
    - [XgBoost (coming soon)](./models/xgboost)
- Each framework folder will have specific example folders that has
    - Trained models (should be pulled using `git lfs`)
    - Script we have used to train the models
    - Script you can use to check the output of the models
    - Other assets if required for RedisAI
- Different client examples are placed in the root directory itself. Right now we have examples for three clients although making these examples working for another client library in another language should be no-brainer.
    - [Python client](./python_client)
    - [Go client](./go_client)
    - [NodeJS client](./js_client)
    - [Bash client](./bash_client)
- [Sentinel example](./sentinel) is documented inside the directory itself
