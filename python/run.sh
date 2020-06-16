#!/bin/bash

mkdir ./data
mkdir ./out

# Download the following files from S3:
# 1. XLM-Roberta and Bert embeddings, placing them in /data/recsys2020/DL_Data/embeddings/xlm-r and /data/recsys2020/DL_Data//embeddings/bert respectively
# 2. Train.sav, Valid.sav, Submit.sav, Test.sav, placing them in /data/recsys2020/DL_Data/data
# 3. model checkpoints, placing them in /data/recsys2020/DL_Checkpoints

# Train the multilayer perceptron (MLP) models and then perform inference on the Valid, Submit and Test sets
# The 4 csv output/submission files are saved under python/out/train(val)/modelname/engagements.csv
mlp_model/run.sh

# This concludes our "main pipeline". At this point we have 4/9 (+1 for the XGBoost model) of the models used in the final blend.
# The following instructions will describe reproducing 1+(2) other models used in the blend: supervised fine-tuning of Bert
# and using history embeddings.


# Supervised Bert fine-tuning
# Train the MLP model on supervised fine-tuned embeddings and then perform inference on the Valid, Submit and Test sets
supervised_bert_model/mlp_on_embeddings/run.sh

# The 4 csv output/submission files are saved under /data/recsys2020/DL_Ouputs/(train)(val)(submit)(test)/modelname/(engagement).csv


# History embedding model
# Train the MLP model on XLM-R embeddings with history and then perform inference on the Valid, Submit and Test sets
history_model/run_python_codes.sh

# The 4 csv output/submission files are saved under /data/recsys2020/DL_Ouputs/(train)(val)(submit)(test)/modelname/(engagement).csv

# Finally the prediction scores are blended
# Please make sure all the predictions files are in /data/recsys2020/DL_Ouputs in the following format:
# (train)(val)(submit)(test)/modelname/(engagement).csv

# run blender
./post_processing/run_blender.sh
# this should output the predictions from the blended model to /data/recsys2020/DL_Ouputs/test/blended


# run temperature tuning (done only for 1 out of 2 submissions)
./post_processing/run_temperature_scaling.sh
# this should output the scaled predictions to /data/recsys2020/DL_Ouputs/test/blended_and_scaled
