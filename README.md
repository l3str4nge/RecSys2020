<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

## Approach

Our model is based on a deep learning pipeline with three major sources of input: 1) extracted features that describe engaging user, tweet creator and tweet content 2) language model embeddings for tweet content 3) engagement/creation history embeddings for user and tweet creator. These inputs are combined with feed forward neural networks to generate 4-way predictions for each engagement type.

For the language model component, we use pretrained multilingual Transformer-based models [BERT-Base](https://huggingface.co/bert-base-multilingual-cased) and [XLM-Roberta-Large](https://huggingface.co/xlm-roberta-large).


## Execution

We use a hybrid Java-Python pipeline where data parsing and feature extraction is done in Java, and deep learning model training is done in Python. To run the code first execute:
```
./run.sh
```
This script requires path to the main `PROJECT_PATH` directory which must contain a subdirectory `PROJECT_PATH/Data/` with the `training.tsv`, `val.tsv` and `competition_test.tsv` challenge datasets. The script will parse the data, extract features, train baseline XGBoost model using features only and run inference on the leaderboard (`val.tsv`) and test (`competition_test.tsv`) sets. Trained XGBoost models and predictions are outputted to `PROJECT_PATH/Models/XGB/`.


## Github Repo
https://github.com/layer6ai-labs/RecSys2020

