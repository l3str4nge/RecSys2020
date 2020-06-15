# Run preprocess/scaler.py to generate serilized files Train.sav, Valid.sav, Submit.sav and Test.sav

python preprocess/scaler.py

# Run preprocess/split_dataset.py to generate TrainEmbID* files and ValidEmbID, SubmitEmbID and TestEmbID

python preprocess/split_dataset.py

# run preprocess/split_given_id.py to load up the unsupervised pretrained embeddings and split them by ids into TrainEmb*.sav, ValidEmb.sav, SubmitEmb.sav and TestEmb.sav

python preprocess/split_given_id.py

run history_nn_chunk_xlmr.py with train option to train the model and then with valid/submit/test to generate validation scores, submission files and test files

python model/history_nn_chunk_xlmr.py
