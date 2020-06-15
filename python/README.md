# Python Code Repo for RecSys 

There are 5 parts for the python code repo for RecSys:

1. history_model
2. mlp_model
3. [post_processing](https://github.com/layer6ai-labs/RecSys2020/blob/master/python/post_processing/README.md)
4. supervised_bert_model
5. [unsupservised_language_modeling](https://github.com/layer6ai-labs/RecSys2020/blob/master/python/unsupervised_language_modeling/xlm-r/README.md)

Please refer to the detailed README inside each folder for information on how to run each


## Reproducing the Final Submission (Inference)

1. run [./mlp_model/run.sh](https://github.com/layer6ai-labs/RecSys2020/blob/master/python/mlp_model/run.sh)
2. run [blendingfile](blah)


### Model 1

One model is submitted with temperature tuning

1. run "[temperature_scaling.py](https://github.com/layer6ai-labs/RecSys2020/blob/master/python/post_processing/temperature_scaling.py) --submit_folder /path/to/4csvs/test"


### Model 2

The other model is submitted without temperature tuning. The 4 csv files outputted by the blender are directly used.
