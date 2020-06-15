
import joblib
import pickle
import numpy as np
import random
import time

NUM_CHUNKS = 3
def inhour(elapsed):
    return time.strftime('%H:%M:%S', time.gmtime(elapsed))

start_time = time.time()
all_dataset = joblib.load(open("/data4/final_data/100percentdata/Train.sav", "rb"))
# NOTE: only necessary if Train.sav didn't get the tweet_ids
all_dataset_tid = joblib.load(open("/data4/final_data/100percentdata/Train_tid.sav", "rb"))
all_dataset['tweet_ids'] = np.squeeze(all_dataset_tid['tweet_ids'])
del all_dataset_tid

print("LOAD Elapsed: {0}".format(inhour(time.time() - start_time)))

all_indices = list(range(len(all_dataset["tweet_ids"])))

random.shuffle(all_indices)

chunk_size = len(all_indices)//NUM_CHUNKS


h = 0
for c in range(NUM_CHUNKS):
    t = h + chunk_size
    if c == NUM_CHUNKS - 1:
        t = len(all_indices)
    dataset = {}
    
    dataset['features'] = all_dataset["features"][all_indices[h:t]]
    dataset['tokens'] = all_dataset["tokens"][all_indices[h:t]]
    dataset['labels'] = all_dataset["labels"][all_indices[h:t]]
    #for submit
    #dataset['labels'] = []
    dataset['ids'] = all_dataset["ids"][all_indices[h:t]]
    dataset['tweet_ids'] = all_dataset["tweet_ids"][all_indices[h:t]]
    print("ASSIGN Elapsed: {0}".format(inhour(time.time() - start_time)))
    unique_tweet_ids = list(set(dataset['tweet_ids']))
    tweet_id_to_row = {tid:i for i, tid in enumerate(unique_tweet_ids)}
    dataset['tweet_row'] = np.array([tweet_id_to_row[tid] for tid in dataset['tweet_ids']])
    print("SPLIT Elapsed: {0}".format(inhour(time.time() - start_time)))

    del dataset['tweet_ids']

    with open('/data4/final_data/100percentdata/Train{}.sav'.format(c), 'wb') as f:
        joblib.dump(dataset, f)
    with open("/data4/final_data/100percentdata/tweet_id_to_row{}.p".format(c), "wb") as f:
        pickle.dump(tweet_id_to_row, f, protocol=4)
    print("SAVE Elapsed: {0}".format(inhour(time.time() - start_time)))
    
    h = t
    print(h)
