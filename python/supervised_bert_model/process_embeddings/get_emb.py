import pickle
from tqdm import tqdm
from os.path import join
import numpy as np

NUM_CHUNKS = 3
emb_size = 768
source_embeddings_file = "/home/layer6/recsys/clean/embeddings/supervised_mean_emb_trainval.p"

all_embeddings = pickle.load(open(source_embeddings_file, "rb"))
print("Total # embeddings: {}".format(len(all_embeddings.keys())))
all_tids = set()

# train

for c in range(NUM_CHUNKS):

    tid_to_row = pickle.load(open("new_split/tweet_id_to_row{}.p".format(c), "rb"))
    print(c, "# ids: {}".format(len(tid_to_row)))

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

    all_tids.update(processed_tweet_ids)

    assert assigned_rows == len(processed_rows)
    print("Added {} unique tweet ids to {} rows".format(len(processed_tweet_ids), len(processed_rows)))

    print("Saving ... ")
    np_mmp = np.memmap(join("./train_emb{}.memmap".format(c)), dtype='float32', mode='w+', shape=emb_matrix.shape)

    print("Shape of embedding matrix : {}".format(emb_matrix.shape))

    np_mmp[:] = emb_matrix[:]

    del np_mmp


print("{} unique tweet ids in Train".format(len(all_tids)))

# val

tid_to_row = pickle.load(open("new_split/tweet_id_to_row_val.p", "rb"))

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
np_mmp = np.memmap(join("./val_emb.memmap"), dtype='float32', mode='w+', shape=emb_matrix.shape)

print("Shape of embedding matrix : {}".format(emb_matrix.shape))

np_mmp[:] = emb_matrix[:]

del np_mmp
