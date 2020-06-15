# Steps to run History Model with XLM-R embeddings

1. Use the Java code (RecSys20ModelHistory.java) to genearte TrainXGB.csv, ValidXGB.csv, Submit.csv and Test.csv
2. Run preprocess/scaler.py to generate serilized files Train.sav, Valid.sav, Submit.sav and Test.sav
3. run preprocess/split_dataset.py to generate TrainEmbID* files and ValidEmbID, SubmitEmbID and TestEmbID
4. run preprocess/split_given_id.py to load up the unsupervised pretrained embeddings and split them by ids into TrainEmb*.sav, ValidEmb.sav, SubmitEmb.sav and TestEmb.sav
5. run history_nn_chunk_xlmr.py with train option to train the model and then with valid/submit/test to generate validation scores, submission files and test files
