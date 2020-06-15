
import joblib
import pickle
import numpy as np
import random
import time

def inhour(elapsed):
    return time.strftime('%H:%M:%S', time.gmtime(elapsed))

start_time = time.time()
all_dataset = joblib.load(open("/data5/final_data/Submit.sav", "rb"))
# TODO: only necessary if tweet_ids wasn't extracted
# all_dataset_tid = joblib.load(open("/data4/final_data/100percentdata/Valid_tid.sav", "rb"))
# all_dataset['tweet_ids'] = np.squeeze(all_dataset_tid['tweet_ids'])
# del all_dataset_tid

#for submit
#all_dataset['tweet_ids'] = np.squeeze(all_dataset['tweet_ids'])


print("LOAD Elapsed: {0}".format(inhour(time.time() - start_time)))

unique_tweet_ids = list(set(np.squeeze(all_dataset['tweet_ids'])))
tweet_id_to_row = {tid:i for i, tid in enumerate(unique_tweet_ids)}

all_dataset['tweet_row'] = np.array([tweet_id_to_row[tid] for tid in np.squeeze(all_dataset['tweet_ids'])])

print("SPLIT Elapsed: {0}".format(inhour(time.time() - start_time)))

with open('/data7/final_data/100percentdata/Submit_withrow.sav', 'wb') as f:
    joblib.dump(all_dataset, f)
with open("/data7/final_data/100percentdata/tweet_id_to_row_submit.p", "wb") as f:
    pickle.dump(tweet_id_to_row, f, protocol=4)


print("SAVE Elapsed: {0}".format(inhour(time.time() - start_time)))
