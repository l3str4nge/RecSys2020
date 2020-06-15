import pickle
from tqdm import tqdm
from os.path import join
import numpy as np
import glob


emb_size = 1024
# all_embeddings = pickle.load(open("/data2/xlmr_checkpoint192000/submit_mean.p", "rb"))
input_folder = "/data0/xlmr_new_checkpoint146000/test"
segments = sorted(glob.glob(join(input_folder, "*mean*.p")))
print("Extracting embeddings from {} segments ...".format(len(segments)))

tid_to_row = pickle.load(open("/data7/final_data/100percentdata/tweet_id_to_row_test.p", "rb"))

emb_matrix = np.zeros([len(tid_to_row), emb_size], dtype=np.float32)
    
assigned_rows = 0
processed_tweet_ids = set()
processed_rows = set()

for s in tqdm(segments):
    
    seg_embeddings = pickle.load(open(s, "rb"))

    for tid, emb in tqdm(seg_embeddings.items()):

        if tid in tid_to_row:
            row_num = tid_to_row[tid]
            emb_matrix[row_num, :] = emb
            assigned_rows += 1
            processed_tweet_ids.add(tid)
            processed_rows.add(row_num)

# for tid, row_num in tqdm(tid_to_row.items()):

#     embedding = all_embeddings[tid]
#     emb_matrix[row_num, :] = embedding

#     assigned_rows += 1
#     assert row_num not in processed_rows
#     processed_tweet_ids.add(tid)
#     processed_rows.add(row_num)


assert len(tid_to_row) == assigned_rows
assert assigned_rows == len(processed_rows)
print("Added {} unique tweet ids to {} rows".format(len(processed_tweet_ids), len(processed_rows)))

print("Saving ... ")
np_mmp = np.memmap(join("/data7/final_data/split_embeddings/final_data_xlmr_new/test_emb.p"), dtype='float32', mode='w+', shape=emb_matrix.shape)

print("Shape of embedding matrix : {}".format(emb_matrix.shape))

np_mmp[:] = emb_matrix[:]

del np_mmp
