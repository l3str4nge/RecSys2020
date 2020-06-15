from tqdm import tqdm
import pandas as pd
import numpy as np
import joblib
import gc
import os


def generate_dict_np(path):
    # num_rows = 200
    num_rows = 88592400
    print(num_rows)

    ret = {
        "labels":np.zeros((num_rows,4),dtype=np.bool_),
        "tokens":np.zeros((num_rows,297),dtype=np.float32),
        "features":np.zeros((num_rows,138),dtype=np.float32),
        "tweet_ids":np.zeros((num_rows,1),dtype='U32'),
        "ids":np.zeros((num_rows,2),dtype=np.int32)
    }

    type_map = {
        **{k: np.bool_ for k in range(0, 4)},
        **{k: np.float32 for k in range(4, 4+297)},
        **{k: np.float32 for k in range(301, 301+138)},
        **{k: np.int32 for k in range(440, 442)},
    }


    df_iterator = pd.read_csv(path, dtype=type_map, header=None, chunksize=1000000)
    h = 0
    for df in tqdm(df_iterator):
        t = h + len(df)
        ret['labels'][h:t] = df[range(0, 4)].values
        ret['tokens'][h:t] = df[range(4, 4+297)].values
        ret['features'][h:t] = df[range(301, 301+138)].values
        ret['tweet_ids'][h:t] = df[range(439, 440)].values
        ret['ids'][h:t] = df[range(440, 442)].values
        h = t
        del df
        gc.collect()
        
    print(t)
    
    del df_iterator
    gc.collect()

    return ret


num_datasets = 5
train_path = "/media/kevin/datahdd/data/recsys/Hojin/chunked/chunk12/Train.sav"
output_path = "/media/kevin/datahdd/data/recsys/Hojin/chunked/chunk12/splits/Train"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

print("Loading the train file, this will take a while ... ")
# tr_dict = generate_dict_np(train_path)
tr_dict = joblib.load(open(os.path.join(train_path), 'rb'))

dataset_size = 1 + len(tr_dict['labels']) // num_datasets

total_tweet_ids = len(set(np.squeeze(tr_dict['tweet_ids'], 1)))
total_engager_ids = len(set(tr_dict['ids'][:, 0]))
total_creator_ids = len(set(tr_dict['ids'][:, 1]))

gc.collect()

redundant_tweet_id = 0
redundant_engager_id = 0
redundant_creator_id = 0

# for figuring out what chunk the sample falls into
row_ids = np.arange(len(tr_dict['labels']), dtype=np.int32)
np.random.shuffle(row_ids)


for i in tqdm(range(num_datasets)):

    dataset = {}
    start_idx = i*dataset_size
    end_idx = min((i+1)*dataset_size, len(row_ids))
    split_row_idxs = row_ids[start_idx:end_idx]

    print("There are {} samples in dataset {} ... ".format(end_idx-start_idx, i))

    dataset['labels'] = tr_dict['labels'][split_row_idxs]
    dataset['tokens'] = tr_dict['tokens'][split_row_idxs]
    dataset['features'] = tr_dict['features'][split_row_idxs]
    dataset['tweet_ids'] = tr_dict['tweet_ids'][split_row_idxs]
    dataset['ids'] = tr_dict['ids'][split_row_idxs]
    dataset['row_id'] = split_row_idxs

    dataset_tweet_ids = len(set(np.squeeze(dataset['tweet_ids'], 1)))
    dataset_engager_ids = len(set(dataset['ids'][:, 0]))
    dataset_creator_ids = len(set(dataset['ids'][:, 1]))

    redundant_tweet_id += dataset_tweet_ids
    redundant_engager_id += dataset_engager_ids
    redundant_creator_id += dataset_creator_ids

    # report statistics
    print("There are {}/{} tweet ids in dataset {}/{} ... ".format(dataset_tweet_ids, total_tweet_ids, i, num_datasets))
    print("There are {}/{} engager ids in dataset {}/{} ... ".format(dataset_engager_ids, total_engager_ids, i, num_datasets))
    print("There are {}/{} creator ids in dataset {}/{} ... ".format(dataset_creator_ids, total_creator_ids, i, num_datasets))

    joblib.dump(dataset, open(output_path + "_dataset_{}.sav".format(i), 'wb'))
    joblib.dump({k: dataset[k] for k in ['tweet_ids', 'ids']}, open(output_path + "_ids_{}.sav".format(i), 'wb'))

    del dataset, dataset_tweet_ids, dataset_engager_ids, dataset_creator_ids
    gc.collect()


print("There are {}/{} tweet ids across datasets, this is {}% redundancy ... ".format(
    redundant_tweet_id, total_tweet_ids, float(redundant_tweet_id - total_tweet_ids)/total_tweet_ids))
print("There are {}/{} engager ids across datasets, this is {}% redundancy ... ".format(
    redundant_engager_id, total_engager_ids, float(redundant_engager_id - total_engager_ids)/total_engager_ids))
print("There are {}/{} creator ids across datasets, this is {}% redundancy ... ".format(
    redundant_creator_id, total_creator_ids, float(redundant_creator_id - total_creator_ids)/total_creator_ids))

print("done")
