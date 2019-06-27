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
- You need python3.6+ for running python examples. Use conda environment file for installing python dependencies. Nodejs folder has package.json file for installing node dependencies.


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
- [Sentinel example](./sentinel) is documented inside the directory itself