from pathlib import Path
import os
import pickle
import time
import gc
from tqdm import tqdm
import joblib


mode = "test"
source_path = "/home/layer6/recsys/unsupervised_kevin/checkpoints/supervised_difflr/checkpoint-21000/{}_embeddings/".format(mode)
tid_to_row_file = "new_split/tweet_id_to_row_{}.p".format(mode)
out_path = "/home/layer6/recsys/clean/embeddings/supervised_mean_emb_{}.p".format(mode)

#files = sorted([str(x) for x in Path(source_path).rglob("*.p")])
files = sorted([str(source_path + x) for x in os.listdir(source_path)])

print(files)
print(len(files))

submit_ids = pickle.load(open(tid_to_row_file, "rb"))
print("there are {} tweets IDs".format(len(submit_ids.keys())))

embedding_map = {}

for f in tqdm(files):
    print(f)

    s = time.time()
    data = pickle.load(open(f, 'rb'))
    print("loaded")
    print(time.time()-s)

    s = time.time()
    skipped = 0
    for key in data.keys():
        if (key in submit_ids.keys()):
            embedding_map[key] = data[key]
        else:
            skipped += 1
    print("updated {}, skipped: {}".format(len(embedding_map), skipped))
    print(time.time()-s)
    
    del data 
    gc.collect()
    time.sleep(2)

    print("There are {} embeddings".format(len(embedding_map.keys())))

gc.collect()
pickle.dump(embedding_map, open(out_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
print("compiled the embeddings!")