import pickle
from tqdm import tqdm
from os.path import join
import numpy as np

emb_size = 768
mode = "test" # "submit" or "test"
source-embeddings_file = "/home/layer6/recsys/clean/embeddings/supervised_mean_emb_{}.p".format(mode)
tid_to_row_file = "new_split/tweet_id_to_row_{}.p".format(mode)


all_embeddings = pickle.load(open(source_embeddings_file, "rb"))
tid_to_row = pickle.load(open(tid_to_row_file, "rb"))


emb_matrix = np.zeros([len(tid_to_row), emb_size], dtype=np.float32)
    
assigned_rows = 0
processed_tweet_ids = set()
processed_rows = set()

for tid, row_num in tqdm(tid_to_row.items()):

    embedding = all_embeddings[tid]
    emb_matrix[row_num, :] = embedding

    assigned_rows += 1
    assert row_num not in processed_rows
    processed_tweet_ids.add(tid)
    processed_rows.add(row_num)


assert assigned_rows == len(processed_rows)
print("Added {} unique tweet ids to {} rows".format(len(processed_tweet_ids), len(processed_rows)))

print("Saving ... ")
np_mmp = np.memmap(join("./{}_emb.p".format(mode)), dtype='float32', mode='w+', shape=emb_matrix.shape)

print("Shape of embedding matrix : {}".format(emb_matrix.shape))

np_mmp[:] = emb_matrix[:]

del np_mmp
