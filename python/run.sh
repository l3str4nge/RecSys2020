#!/bin/bash

mkdir ./data
mkdir ./out

#extract bert/xlm-r embeddings
unsupervised_language_modeling/bert/run_file.sh
unsupervised_language_modeling/xlm-r/run_file.sh

#move embeddings files to ./data
mv unsupervised_language_modeling/bert/bert_embddings_name.p ./data
mv unsupervised_language_modeling/xlm-r/xlm-r_embddings_name.p ./data

#run scaler and save .sav files to ./data
scaler.sh

##Now every .sav files and embeddings are saved under python/data 
#run splitter to chunnk sav files 
data_splitting/splitter.sh
...

#run mlp_models and save output files under python/out/train(val)/modelname/engagements.csv
mlp_model/run.sh

#copy checkpoints from mlp models for supervised bert
cp mlp_models/checkpoints/some_checkpoints supervised_bert_model/somewhere

#run supervised_bert
supervised_bert_model/some_shellscript.sh

# save output files under python/out/train(val)/modelname/engagements.csv
# rename to engagment.csv if needed

#run history_model
history_model/run_python_codes.sh

# save output files under python/out/train(val)/modelname/engagements.csv
# rename to engagment.csv if needed

#run blender
blender/run.sh


