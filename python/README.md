# Python Code Repo for RecSys 

There are 5 parts for the python code repo for RecSys:

1. [unsupservised_language_modeling](https://github.com/layer6ai-labs/RecSys2020/blob/master/python/unsupervised_language_modeling/xlm-r/README.md)
2. [history_model](https://github.com/layer6ai-labs/RecSys2020/blob/master/python/history_model/README.md)
3. [mlp_model](https://github.com/layer6ai-labs/RecSys2020/blob/master/python/mlp_model/README.md)
4. [supervised_bert_model](https://github.com/layer6ai-labs/RecSys2020/tree/master/python/supervised_bert_model)
5. [post_processing](https://github.com/layer6ai-labs/RecSys2020/blob/master/python/post_processing/README.md)

Please refer to the detailed README inside each folder for information on how to run each


## Reproducing the Final Submission (Inference)

1. run each models as described in separate `readme`
2. run `blender.py` and `temperature_scaling.py`(optional) under post_processing


### Model 1

One model is submitted with temperature tuning

1. run "[temperature_scaling.py](https://github.com/layer6ai-labs/RecSys2020/blob/master/python/post_processing/temperature_scaling.py) --submit_folder /path/to/blended/4csvs/test"


### Model 2

The other model is submitted without temperature tuning. The 4 csv files outputted by the blender are directly used.

## The Full Training Pipeline Overview
1. Features are extracted from the Java code
2. Power transforms are applied to the features using [scaler.py](https://github.com/layer6ai-labs/RecSys2020/blob/master/python/mlp_model/scaler.py)
3. The unsupervised MLM models XLM-Roberta and Bert are trained using tweets using [./unsupervised_language_modeling](https://github.com/layer6ai-labs/RecSys2020/tree/master/python/unsupervised_language_modeling)
4. The embeddings are matched to the training dataset and the training data is split into 3 "chunks" so as to fit into RAM, this is done in [./mlp_model/data_splitting](https://github.com/layer6ai-labs/RecSys2020/tree/master/python/mlp_model/data_splitting)
5. [main_feature_embedding.py](https://github.com/layer6ai-labs/RecSys2020/blob/master/python/mlp_model/main_feature_embedding.py) is used to train a neural network to predict the 4 engagements. The inputs are both features extracted from the Java pipeline (with Powertransforms) and the tweet embeddings.
6. A slight variant of our method uses historical tweets as additional context or features to inform the model. [./history_model](https://github.com/layer6ai-labs/RecSys2020/tree/master/python/history_model) has this implementation. The history model is a neural network with attention and is blended into the final predictions.
7. Another variant of our method fine-tunes the Bert model in a supervised way, the implementation is in [supervised_bert_model](https://github.com/layer6ai-labs/RecSys2020/tree/master/python/supervised_bert_model). The model here is also a neural network and it is also blended into the final prediction.
8. Post-processing includes blending model predictions (using a linear regressor) and calibrating the temperature. Implementation can be found in [./post_processing](https://github.com/layer6ai-labs/RecSys2020/tree/master/python/post_processing). See the readme.md in [./post_processing](https://github.com/layer6ai-labs/RecSys2020/tree/master/python/post_processing) for more details.
