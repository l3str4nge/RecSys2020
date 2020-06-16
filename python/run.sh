#!/bin/bash

mkdir ./data
mkdir ./out

# Download the following files from S3:
# 1. XLM-Roberta and Bert embeddings, placing them in ./data/embeddings/xlm-r and ./data/embeddings/bert respectively
# 2. Train.sav, Valid.sav, Submit.sav, Test.sav, placing them in ./data/unscaled_data

# run the Powertransform scaler, this saves the .sav files to ./data
scaler.sh

# Make sure the output of scaler.sh is stored in python/data, there should be 4 files: Train.sav, Valid.sav, Submit.sav, Test.sav
# Run split_data.sh to chunk Train.sav into 3 chunks (this is because with embeddings, the data is too large to fit into RAM)
# Note: we trained on a 256GB RAM machine with 3 chunks. More chunks may reduce the performance.
data_splitting/splitter.sh


# Train the multilayer perceptron (MLP) models and then perform inference on the Valid, Submit and Test sets
# The 4 csv output/submission files are saved under python/out/train(val)/modelname/engagements.csv
mlp_model/run.sh

# This concludes our "main pipeline". At this point we have 4/9 (+1 for the XGBoost model) of the models used in the final blend.
# The following instructions will describe reproducing 1+(2) other models used in the blend: supervised fine-tuning of Bert
# and using history embeddings.


# Supervised Bert fine-tuning
# We have fine-tuned the Bert model in a supervised way and extracted the embeddings. Download them from S3.
# Place the supervise fine-tuned embedddings in ./data/embeddings/supervised_bert
cp mlp_models/checkpoints/some_checkpoints supervised_bert_model/somewhere

# Train the MLP model on supervised fine-tuned embeddings and then perform inference on the Valid, Submit and Test sets
supervised_bert_model/some_shellscript.sh

# The 4 csv output/submission files are saved under python/out/train(val)/modelname/engagements.csv


# History embedding model
# Train the MLP model on XLM-R embeddings with history and then perform inference on the Valid, Submit and Test sets
history_model/run_python_codes.sh

# The 4 csv output/submission files are saved under python/out/train(val)/modelname/engagements.csv

# Finally the prediction scores are blended
# Please make sure all the predictions files are in .......blahblahblah/folder in the following format:
# /blahblahblah

#run blender
blender/run.sh


