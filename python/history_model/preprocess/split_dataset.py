import numpy as np
import pandas as pd
import joblib
import glob
import pickle
import gc
from pandas.core.common import flatten

## Split Training
with open('/data/recsys2020/history_nn/Train.sav', 'rb') as f:
    train = joblib.load(f)

indexes = np.arange(len(train['labels']))
np.random.shuffle(indexes)

splitted_indexes = np.array_split(indexes, 4)

for i, index in enumerate(splitted_indexes):
    print("Processing Chunk ", i)
    chunk_dict = {}
    for key, value in train.items():
        chunk_dict[key] = value[index]
    with open('/data/recsys2020/history_nn/TrainChunk' + str(i) + '.sav','wb') as f:
        joblib.dump(chunk_dict, f)

del chunk_dict
del splitted_indexes
del indexes
del train
gc.collect()
## Split Training

## Getting ID Mapping Files
id_map_csv = pd.read_csv('/data/recsys2020/history_nn/TweetIDMap.csv', header=None)
id_map = dict(zip(id_map_csv.iloc[:, 1], id_map_csv.iloc[:, 0]))

del id_map_csv
gc.collect()
## Getting ID Mapping Files

## Get Training Embeddings IDs
train_feature_dict_dir = "/data/recsys2020/history_nn/TrainChunk*"
train_chunks_dirs = list(sorted(glob.glob(train_feature_dict_dir)))

for i, file in enumerate(train_chunks_dirs):
    print(i, file)
    with open(file, 'rb') as f:
        chunk = joblib.load(f)
    setid = set(flatten(chunk['tweet_ids']))
    print('setid', len(setid))
    setengagenum = set(np.unique(chunk['engagement_histories']))
    print('setengagenum', len(setengagenum))
    setengage = {id_map[k] for k in setengagenum if k in id_map}
    print('setengage', len(setengage))
    setid.update(setengage)
    print('setid', len(setid))

    ids_chunk = np.array(list(setid))
    with open("/data/recsys2020/history_nn/TrainEmbID" + str(i), "wb") as f2:
        joblib.dump(ids_chunk, f2)

    del ids_chunk
    del setid
    del setengagenum
    del setengage
    gc.collect()
## Get Training Embeddings IDs

## Get Valid Embedding IDs
with open("/data/recsys2020/history_nn/Valid.sav", 'rb') as f:
    chunk = joblib.load(f)

setid = set(flatten(chunk['tweet_ids']))
print('setid', len(setid))
setengagenum = set(np.unique(chunk['engagement_histories']))
print('setengagenum', len(setengagenum))
setengage = {id_map[k] for k in setengagenum if k in id_map}
print('setengage', len(setengage))
setid.update(setengage)
print('setid', len(setid))

ids_chunk = np.array(list(setid))
with open("/data/recsys2020/history_nn/ValidEmbID", "wb") as f2:
    joblib.dump(ids_chunk, f2)

del ids_chunk
del setid
del setengagenum
del setengage
gc.collect()
## Get Valid Embedding IDs

## Get Submit Embedding IDs
with open("/data/recsys2020/history_nn/Submit.sav", 'rb') as f:
    chunk = joblib.load(f)

setid = set(flatten(chunk['tweet_ids']))
print('setid', len(setid))
setengagenum = set(np.unique(chunk['engagement_histories']))
print('setengagenum', len(setengagenum))
setengage = {id_map[k] for k in setengagenum if k in id_map}
print('setengage', len(setengage))
setid.update(setengage)
print('setid', len(setid))

ids_chunk = np.array(list(setid))
with open("/data/recsys2020/history_nn/SubmitEmbID", "wb") as f2:
    joblib.dump(ids_chunk, f2)

del ids_chunk
del setid
del setengagenum
del setengage
gc.collect()
## Get Submit Embedding IDs

## Get Test Embedding IDs
with open("/data/recsys2020/history_nn/Test.sav", 'rb') as f:
    chunk = joblib.load(f)

setid = set(flatten(chunk['tweet_ids']))
print('setid', len(setid))
setengagenum = set(np.unique(chunk['engagement_histories']))
print('setengagenum', len(setengagenum))
setengage = {id_map[k] for k in setengagenum if k in id_map}
print('setengage', len(setengage))
setid.update(setengage)
print('setid', len(setid))

ids_chunk = np.array(list(setid))
with open("/data/recsys2020/history_nn/TestEmbID", "wb") as f2:
    joblib.dump(ids_chunk, f2)

del ids_chunk
del setid
del setengagenum
del setengage
gc.collect()
## Get Test Embedding IDs