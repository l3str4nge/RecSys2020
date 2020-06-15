import numpy as np
import pandas as pd
import joblib
import glob
import pickle
import gc
from tqdm import tqdm
from pandas.core.common import flatten

## ----------------- Load Embedding Files -----------------
emb_dir = "/home/layer6/joey/recsys/chunks/xlmr/trainval/*.p"
emb_chunk_files = glob.glob(emb_dir)

for i, file in enumerate(tqdm(emb_chunk_files)):

    if i == 0:
        embedding = pickle.load(open(file, "rb"))
    else:
        embedding.update(pickle.load(open(file, "rb")))

## ----------------- Load Submit Embedding Files -----------------
emb_dir = "/home/layer6/joey/recsys/chunks/xlmr/submit/*.p"
emb_chunk_files = glob.glob(emb_dir)
for i, file in enumerate(tqdm(emb_chunk_files)):
    embedding.update(pickle.load(open(file, "rb")))
## ----------------- Load Submit Embedding Files -----------------

## ----------------- Load Test Embedding Files -----------------
emb_dir = "/home/layer6/joey/recsys/chunks/xlmr/test/*.p"
emb_chunk_files = glob.glob(emb_dir)
for i, file in enumerate(tqdm(emb_chunk_files)):
    embedding.update(pickle.load(open(file, "rb")))
## ----------------- Load Test Embedding Files -----------------

# embs = np.array(list(embedding.values()))
emb_ids = np.array(list(embedding.keys()))
emb_dict = {k: i for i, k in enumerate(emb_ids)}
gc.collect()
## ----------------- Load Embedding Files -----------------

## ----------------- Get Training Embeddings -----------------
train_emb_dir_list = "/home/layer6/joey/recsys/history_nn/TrainEmbID*"
train_emb_dirs = list(sorted(glob.glob(train_emb_dir_list)))

for i, file in enumerate(tqdm(train_emb_dirs)):

    print(i, file)
    with open(file, 'rb') as f:
        chunk = joblib.load(f)
    print(len(chunk))

    embs_chunk = np.array([embedding[k] for k in chunk])
    print(embs_chunk.shape)

    with open("/home/layer6/joey/recsys/history_nn/TrainEmb" + str(i) + ".sav", "wb") as f1:
        joblib.dump(embs_chunk, f1)
    del embs_chunk
    gc.collect()

    del chunk
    gc.collect()
## -----------------  Get Training Embeddings -----------------

## ----------------- Get Valid Embeddings -----------------
with open("/home/layer6/joey/recsys/history_nn/ValidEmbID", 'rb') as f:
    chunk = joblib.load(f)
print(len(chunk))

embs_chunk = np.array([embedding[k] for k in chunk])
print(embs_chunk.shape)

with open("/home/layer6/joey/recsys/history_nn/ValidEmb.sav", "wb") as f1:
    joblib.dump(embs_chunk, f1)

del embs_chunk
del chunk
gc.collect()
## ----------------- Get Valid Embeddings -----------------

## ----------------- Get Submit Embeddings -----------------
with open("/home/layer6/joey/recsys/history_nn/SubmitEmbID", 'rb') as f:
    chunk = joblib.load(f)
print(len(chunk))

embs_chunk = np.array([embedding[k] for k in chunk])
print(embs_chunk.shape)

with open("/home/layer6/joey/recsys/history_nn/SubmitEmb.sav", "wb") as f1:
    joblib.dump(embs_chunk, f1)

del embs_chunk
del chunk
gc.collect()
## ----------------- Get Submit Embeddings -----------------

# ----------------- Get Test Embeddings -----------------
with open("/home/layer6/joey/recsys/history_nn/TestEmbID", 'rb') as f:
    chunk = joblib.load(f)
print(len(chunk))

embs_chunk = np.array([embedding[k] for k in chunk])
print(embs_chunk.shape)

with open("/home/layer6/joey/recsys/history_nn/TestEmb.sav", "wb") as f1:
    joblib.dump(embs_chunk, f1)

del embs_chunk
del chunk
gc.collect()
# ----------------- Get Test Embeddings -----------------