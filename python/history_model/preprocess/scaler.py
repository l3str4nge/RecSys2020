import pandas as pd
import gc
import sklearn
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
import pickle
import joblib
import numpy as np
import time

def inhour(elapsed):
    return time.strftime('%H:%M:%S', time.gmtime(elapsed))

N_LABEL = 4
N_TOKEN = 297
N_FEAT = 170

#####################Train / Valid##################################
def generate_dict(path):
    type_map = {
        **{k: np.bool_ for k in range(0, N_LABEL)},
        **{k: np.float32 for k in range(N_LABEL, N_LABEL + N_TOKEN)},
        **{k: np.float32 for k in range(N_LABEL + N_TOKEN, N_LABEL + N_TOKEN + N_FEAT)},
        **{k: object for k in range(N_LABEL + N_TOKEN + N_FEAT, N_LABEL + N_TOKEN + N_FEAT + 1)},
        **{k: object for k in range(N_LABEL + N_TOKEN + N_FEAT + 1, N_LABEL + N_TOKEN + N_FEAT + 5)},
        **{k: np.int32 for k in range(N_LABEL + N_TOKEN + N_FEAT + 5, N_LABEL + N_TOKEN + N_FEAT + 7)},
        **{k: object for k in range(N_LABEL + N_TOKEN + N_FEAT + 7, N_LABEL + N_TOKEN + N_FEAT + 8)},
    }

    df = pd.read_csv(path, dtype=type_map, header=None)

    dic = {
        "labels": np.array(df[range(0, N_LABEL)]),
        # "tokens": np.array(df[range(N_LABEL, N_LABEL + N_TOKEN)]),
        "features": np.array(df[range(N_LABEL + N_TOKEN, N_LABEL + N_TOKEN + N_FEAT)]),
        "tweet_ids": np.array(df[range(N_LABEL + N_TOKEN + N_FEAT, N_LABEL + N_TOKEN + N_FEAT + 1)]),
        "engagement_histories": np.array(df[range(N_LABEL + N_TOKEN + N_FEAT + 1, N_LABEL + N_TOKEN + N_FEAT + 5)]),
        "ids": np.array(df[range(N_LABEL + N_TOKEN + N_FEAT + 5, N_LABEL + N_TOKEN + N_FEAT + 7)]),
        "lb_user_ids": np.array(df[range(N_LABEL + N_TOKEN + N_FEAT + 7, N_LABEL + N_TOKEN + N_FEAT + 8)]),
    }

    del df
    gc.collect()
    return dic


def generate_dict_np(path):
    # num_rows = 0
    # for _ in open(path):
    #    num_rows += 1
    # num_rows = 112904170
    # num_rows = 88592400
    # num_rows = int(113452296*0.9)
    # num_rows = int(109418789 * 0.95)
    num_rows = 109418789
    print("# rows: ", num_rows)

    ret = {
        "labels": np.zeros((num_rows, N_LABEL), dtype=np.bool_),
        # "tokens": np.zeros((num_rows, N_TOKEN), dtype=np.float32),
        "features": np.zeros((num_rows, N_FEAT), dtype=np.float32),
        "tweet_ids": np.zeros((num_rows, 1), dtype=object),
        "engagement_histories": np.zeros((num_rows, 4), dtype=object),
        "ids": np.zeros((num_rows, 2), dtype=np.int32),
    }

    type_map = {
        **{k: np.bool_ for k in range(0, N_LABEL)},
        **{k: np.float32 for k in range(N_LABEL, N_LABEL + N_TOKEN)},
        **{k: np.float32 for k in range(N_LABEL + N_TOKEN, N_LABEL + N_TOKEN + N_FEAT)},
        **{k: object for k in range(N_LABEL + N_TOKEN + N_FEAT, N_LABEL + N_TOKEN + N_FEAT + 1)},
        **{k: object for k in range(N_LABEL + N_TOKEN + N_FEAT + 1, N_LABEL + N_TOKEN + N_FEAT + 5)},
        **{k: np.int32 for k in range(N_LABEL + N_TOKEN + N_FEAT + 5, N_LABEL + N_TOKEN + N_FEAT + 7)},
    }

    df_iterator = pd.read_csv(path, dtype=type_map, header=None, chunksize=1000000)
    h = 0
    for df in tqdm(df_iterator):
        # nlen = int(len(df) * 0.95)
        # nlen = int(len(df))
        # t = h + nlen
        t = h + len(df)
        ret['labels'][h:t] = df[range(0, N_LABEL)].values
        # ret['tokens'][h:t] = df[range(N_LABEL, N_LABEL + N_TOKEN)].values[:nlen]
        ret['features'][h:t] = df[range(N_LABEL + N_TOKEN, N_LABEL + N_TOKEN + N_FEAT)].values
        ret['tweet_ids'][h:t] = df[range(N_LABEL + N_TOKEN + N_FEAT, N_LABEL + N_TOKEN + N_FEAT + 1)].values
        ret['engagement_histories'][h:t] = df[range(N_LABEL + N_TOKEN + N_FEAT + 1, N_LABEL + N_TOKEN + N_FEAT + 5)].values
        ret['ids'][h:t] = df[range(N_LABEL + N_TOKEN + N_FEAT + 5, N_LABEL + N_TOKEN + N_FEAT + 7)].values
        # ret['chunks'][h:t] = df[442].values
        h = t
        del df
        gc.collect()
    print(t)

    del df_iterator
    gc.collect()

    return ret

##############################Train########################################
train_path = "/data/recsys2020/history_nn/TrainXGB.csv"
train_dict = generate_dict_np(train_path)

##Fit scalers
scaler_f = PowerTransformer(copy=False)
start_time = time.time()
s = len(train_dict['features'])
scaler_f.fit(train_dict['features'][np.random.choice(s, int(0.1 * s))].astype(np.float64, copy=False))
print("Elapsed: {0}".format(inhour(time.time() - start_time)))
print("fit feature scaler")
##Save scalers
with open('/data/recsys2020/history_nn/f_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_f, f, protocol=4)
##Fit scalers

## Load scalers
# with open('/data/recsys2020/history_nn/f_scaler.pkl', 'rb') as f:
#    scaler_f = pickle.load(f)
## Load scalers

##Apply scalers to train set
start_time = time.time()
train_dict['features'] = scaler_f.transform(train_dict['features'])
print("Elapsed: {0}".format(inhour(time.time() - start_time)))
print("Transformed feature")

engagement_histories = np.full((len(train_dict['features']), 10 * 4), -1)
for i, row in enumerate(train_dict['engagement_histories']):
    for j, r in enumerate(row):
        if isinstance(r, float):
            temp = []
        else:
            temp = r.split('|')[:-1]
        starting_index = j * 10
        ending_index = (j+1) * 10
        if temp != []:
            l = min(len(temp), 10)
            engagement_histories[i, starting_index: starting_index + l] = temp[:l]
train_dict['engagement_histories'] = engagement_histories

print("finish feature transformation")
print("Elapsed: {0}".format(inhour(time.time() - start_time)))

with open('/data/recsys2020/history_nn/Train.sav', 'wb') as f:
    joblib.dump(train_dict, f)
del (train_dict)
print("saved!")
##############################Train########################################

##############################Valid########################################
valid_path = "/data/recsys2020/history_nn/Valid.csv"

## Load scalers
with open('/data/recsys2020/history_nn/f_scaler.pkl', 'rb') as f:
   scaler_f = pickle.load(f)
## Load scalers

##Apply scaler to valid set
valid_dict = generate_dict(valid_path)

start_time = time.time()
valid_dict['features'] = scaler_f.transform(valid_dict['features'])
print("finish feature transformation")
print("Elapsed: {0}".format(inhour(time.time() - start_time)))

engagement_histories = np.full((len(valid_dict['features']), 10 * 4), -1)
for i, row in enumerate(valid_dict['engagement_histories']):
    for j, r in enumerate(row):
        if isinstance(r, float):
            temp = []
        else:
            temp = r.split('|')[:-1]
        starting_index = j * 10
        ending_index = (j+1) * 10
        if temp != []:
            l = min(len(temp), 10)
            engagement_histories[i, starting_index: starting_index + l] = temp[:l]
valid_dict['engagement_histories'] = engagement_histories

with open('/data/recsys2020/history_nn/Valid.sav', 'wb') as f:
    joblib.dump(valid_dict, f)
del (valid_dict)
print("saved!")
##############################Valid########################################

##############################Submit########################################
def generate_lb_dict(path):
    type_map = {
        **{k: np.float32 for k in range(0, N_TOKEN)},
        **{k: np.float32 for k in range(N_TOKEN, N_TOKEN + N_FEAT)},
        **{k: object for k in range(N_TOKEN + N_FEAT, N_TOKEN + N_FEAT + 1)},
        **{k: object for k in range(N_TOKEN + N_FEAT + 1, N_TOKEN + N_FEAT + 5)},
        **{k: np.int32 for k in range(N_TOKEN + N_FEAT + 5, N_TOKEN + N_FEAT + 7)},
        **{k: object for k in range(N_TOKEN + N_FEAT + 7, N_TOKEN + N_FEAT + 8)},
    }

    df = pd.read_csv(path, dtype=type_map, header=None)
    print(df.shape)

    dic = {
        # "tokens": np.array(df[range(0, N_TOKEN)]),
        "features": np.array(df[range(N_TOKEN, N_TOKEN + N_FEAT)]),
        "tweet_ids": np.array(df[range(N_TOKEN + N_FEAT, N_TOKEN + N_FEAT + 1)]),
        "engagement_histories": np.array(df[range(N_TOKEN + N_FEAT + 1, N_TOKEN + N_FEAT + 5)]),
        "ids": np.array(df[range(N_TOKEN + N_FEAT + 5, N_TOKEN + N_FEAT + 7)]),
        "lb_user_ids": np.array(df[range(N_TOKEN + N_FEAT + 7, N_TOKEN + N_FEAT + 8)]),
    }

    del df
    gc.collect()
    return dic


lb_path = "/data/recsys2020/history_nn/Submit.csv"
lb_dict = generate_lb_dict(lb_path)

with open('/data/recsys2020/history_nn/f_scaler.pkl', 'rb') as f:
   scaler_f = pickle.load(f)

start_time = time.time()
lb_dict['features'] = scaler_f.transform(lb_dict['features'])
print("finish feature transformation")
print("Elapsed: {0}".format(inhour(time.time() - start_time)))

engagement_histories = np.full((len(lb_dict['features']), 10 * 4), -1)
for i, row in enumerate(lb_dict['engagement_histories']):
    for j, r in enumerate(row):
        if isinstance(r, float):
            temp = []
        else:
            temp = r.split('|')[:-1]
        starting_index = j * 10
        ending_index = (j+1) * 10
        if temp != []:
            l = min(len(temp), 10)
            engagement_histories[i, starting_index: starting_index + l] = temp[:l]
lb_dict['engagement_histories'] = engagement_histories

with open('/data/recsys2020/history_nn/Submit.sav', 'wb') as f:
    joblib.dump(lb_dict, f)
print("saved!")
##############################Submit########################################

##############################Test########################################
def generate_test_dict(path):
    type_map = {
        **{k: np.float32 for k in range(0, N_TOKEN)},
        **{k: np.float32 for k in range(N_TOKEN, N_TOKEN + N_FEAT)},
        **{k: object for k in range(N_TOKEN + N_FEAT, N_TOKEN + N_FEAT + 1)},
        **{k: object for k in range(N_TOKEN + N_FEAT + 1, N_TOKEN + N_FEAT + 5)},
        **{k: np.int32 for k in range(N_TOKEN + N_FEAT + 5, N_TOKEN + N_FEAT + 7)},
        **{k: object for k in range(N_TOKEN + N_FEAT + 7, N_TOKEN + N_FEAT + 8)},
    }

    df = pd.read_csv(path, dtype=type_map, header=None)
    print(df.shape)

    dic = {
        # "tokens": np.array(df[range(0, N_TOKEN)]),
        "features": np.array(df[range(N_TOKEN, N_TOKEN + N_FEAT)]),
        "tweet_ids": np.array(df[range(N_TOKEN + N_FEAT, N_TOKEN + N_FEAT + 1)]),
        "engagement_histories": np.array(df[range(N_TOKEN + N_FEAT + 1, N_TOKEN + N_FEAT + 5)]),
        "ids": np.array(df[range(N_TOKEN + N_FEAT + 5, N_TOKEN + N_FEAT + 7)]),
        "lb_user_ids": np.array(df[range(N_TOKEN + N_FEAT + 7, N_TOKEN + N_FEAT + 8)]),
    }

    del df
    gc.collect()
    return dic


test_path = "/data/recsys2020/history_nn/Test.csv"
test_dict = generate_test_dict(test_path)

with open('/data/recsys2020/history_nn/f_scaler.pkl', 'rb') as f:
   scaler_f = pickle.load(f)

start_time = time.time()
test_dict['features'] = scaler_f.transform(test_dict['features'])
print("finish feature transformation")
print("Elapsed: {0}".format(inhour(time.time() - start_time)))

engagement_histories = np.full((len(test_dict['features']), 10 * 4), -1)
for i, row in enumerate(test_dict['engagement_histories']):
    for j, r in enumerate(row):
        if isinstance(r, float):
            temp = []
        else:
            temp = r.split('|')[:-1]
        starting_index = j * 10
        ending_index = (j+1) * 10
        if temp != []:
            l = min(len(temp), 10)
            engagement_histories[i, starting_index: starting_index + l] = temp[:l]
test_dict['engagement_histories'] = engagement_histories

with open('/data/recsys2020/history_nn/Test.sav', 'wb') as f:
    joblib.dump(test_dict, f)
print("saved!")
##############################Test########################################