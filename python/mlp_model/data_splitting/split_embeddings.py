import pickle
from tqdm import tqdm
from os.path import join
import numpy as np
import glob


emb_size = 1024
input_folder = "/data0/xlmr_new_checkpoint146000/trainval"
segments = sorted(glob.glob(join(input_folder, "*mean*.p")))
print("Extracting embeddings from {} segments from folder {}...".format(len(segments), input_folder))


tid_to_row0 = pickle.load(open("/data6/old_split_3chunks/kevin_data/tweet_id_to_row0.p", "rb"))
tid_to_row1 = pickle.load(open("/data6/old_split_3chunks/kevin_data/tweet_id_to_row1.p", "rb"))
tid_to_row2 = pickle.load(open("/data6/old_split_3chunks/kevin_data/tweet_id_to_row2.p", "rb"))
tid_to_rowval = pickle.load(open("/data6/old_split_3chunks/kevin_data/tweet_id_to_row_val.p", "rb"))

emb_matrix0 = np.zeros([len(tid_to_row0), emb_size], dtype=np.float32)
emb_matrix1 = np.zeros([len(tid_to_row1), emb_size], dtype=np.float32)
emb_matrix2 = np.zeros([len(tid_to_row2), emb_size], dtype=np.float32)
emb_matrixval = np.zeros([len(tid_to_rowval), emb_size], dtype=np.float32)


assigned_rows0 = 0
processed_tweet_ids0 = set()
processed_rows0 = set()
assigned_rows1 = 0
processed_tweet_ids1 = set()
processed_rows1 = set()
assigned_rows2 = 0
processed_tweet_ids2 = set()
processed_rows2 = set()
assigned_rowsval = 0
processed_tweet_idsval = set()
processed_rowsval = set()


for s in tqdm(segments):
    
    seg_embeddings = pickle.load(open(s, "rb"))

    for tid, emb in tqdm(seg_embeddings.items()):

        if tid in tid_to_row0:
            row_num = tid_to_row0[tid]
            emb_matrix0[row_num, :] = emb
            assigned_rows0 += 1
            processed_tweet_ids0.add(tid)
            processed_rows0.add(row_num)

        if tid in tid_to_row1:
            row_num = tid_to_row1[tid]
            emb_matrix1[row_num, :] = emb
            assigned_rows1 += 1
            processed_tweet_ids1.add(tid)
            processed_rows1.add(row_num)

        if tid in tid_to_row2:
            row_num = tid_to_row2[tid]
            emb_matrix2[row_num, :] = emb
            assigned_rows2 += 1
            processed_tweet_ids2.add(tid)
            processed_rows2.add(row_num)

        if tid in tid_to_rowval:
            row_num = tid_to_rowval[tid]
            emb_matrixval[row_num, :] = emb
            assigned_rowsval += 1
            processed_tweet_idsval.add(tid)
            processed_rowsval.add(row_num)


assert assigned_rows0 == len(tid_to_row0)
assert assigned_rows0 == len(processed_rows0)
print("Added {} unique tweet ids to {} rows".format(len(processed_tweet_ids0), len(processed_rows0)))
print("Saving ... ")
np_mmp = np.memmap("/data6/split_embeddings_3_chunks_old_data/xlmr_146000/train_emb0.memmap", dtype='float32', mode='w+', shape=emb_matrix0.shape)
print("Shape of embedding matrix : {}".format(emb_matrix0.shape))
np_mmp[:] = emb_matrix0[:]
del np_mmp

assert assigned_rows1 == len(tid_to_row1)
assert assigned_rows1 == len(processed_rows1)
print("Added {} unique tweet ids to {} rows".format(len(processed_tweet_ids1), len(processed_rows1)))
print("Saving ... ")
np_mmp = np.memmap("/data6/split_embeddings_3_chunks_old_data/xlmr_146000/train_emb1.memmap", dtype='float32', mode='w+', shape=emb_matrix1.shape)
print("Shape of embedding matrix : {}".format(emb_matrix1.shape))
np_mmp[:] = emb_matrix1[:]
del np_mmp

assert assigned_rows2 == len(tid_to_row2)
assert assigned_rows2 == len(processed_rows2)
print("Added {} unique tweet ids to {} rows".format(len(processed_tweet_ids2), len(processed_rows2)))
print("Saving ... ")
np_mmp = np.memmap("/data6/split_embeddings_3_chunks_old_data/xlmr_146000/train_emb2.memmap", dtype='float32', mode='w+', shape=emb_matrix2.shape)
print("Shape of embedding matrix : {}".format(emb_matrix2.shape))
np_mmp[:] = emb_matrix2[:]
del np_mmp

assert assigned_rowsval == len(tid_to_rowval)
assert assigned_rowsval == len(processed_rowsval)
print("Added {} unique tweet ids to {} rows".format(len(processed_tweet_idsval), len(processed_rowsval)))
print("Saving ... ")
np_mmp = np.memmap("/data6/split_embeddings_3_chunks_old_data/xlmr_146000/val_emb.memmap", dtype='float32', mode='w+', shape=emb_matrixval.shape)
print("Shape of embedding matrix : {}".format(emb_matrixval.shape))
np_mmp[:] = emb_matrixval[:]
del np_mmp


all_tids = processed_tweet_ids0.union(processed_tweet_ids1).union(processed_tweet_ids2)
print("{} unique tweet ids in Train".format(len(all_tids)))
print("{} unique tweet ids in Val".format(len(processed_tweet_idsval)))

print("done")