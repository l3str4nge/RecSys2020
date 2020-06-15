import pandas as pd
import gc
import sklearn
from tqdm import tqdm
from utils.progress import WorkSplitter, inhour
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
import pickle
import joblib
import numpy as np
import time
import os

N_LABEL = 4
N_TOKEN = 297
N_FEAT = 170 
NUM_ROWS = 109418789

train_path = "/data/recsys2020/Models/DL/TrainXGB.csv"
valid_path = "/data/recsys2020/DL/ValidXGB.csv"
lb_path = "/data/recsys2020/DL/Submit.csv"
test_path = "/data/recsys2020/DL/Test.csv"

def generate_dict(path):

    type_map = {
        **{k: np.bool_ for k in range(0, N_LABEL)},
        **{k: np.float32 for k in range(N_LABEL, N_LABEL + N_TOKEN)},
        **{k: np.float32 for k in range(N_LABEL + N_TOKEN, N_LABEL + N_TOKEN + N_FEAT)},
        **{k: np.int32 for k in range(N_LABEL + N_TOKEN + N_FEAT + 1, N_LABEL + N_TOKEN + N_FEAT + 3)},
    }
    df = pd.read_csv(path, dtype=type_map, header=None)
    dic = {
        "labels": np.array(df[range(0, N_LABEL)]),
        "tokens": np.array(df[range(N_LABEL, N_LABEL + N_TOKEN)]),
        "features": np.array(df[range(N_LABEL + N_TOKEN, N_LABEL + N_TOKEN + N_FEAT)]),
        "tweet_ids": np.array(df[range(N_LABEL + N_TOKEN + N_FEAT, N_LABEL + N_TOKEN + N_FEAT + 1)]).astype("U32"),
        "ids": np.array(df[range(N_LABEL + N_TOKEN + N_FEAT + 1, N_LABEL + N_TOKEN + N_FEAT + 3)]),
        "lb_user_ids": np.array(df[range(N_LABEL + N_TOKEN + N_FEAT+3, N_LABEL + N_TOKEN + N_FEAT+4)]),
    }

    del df
    gc.collect()
    return dic

def generate_dict_np(path):

    #num_rows = 0
    #for _ in open(path):
    #    num_rows += 1
    num_rows = NUM_ROWS
    print("# rows: ", num_rows)

    ret = {
        "labels":np.zeros((num_rows,N_LABEL),dtype=np.bool_),
        "tokens":np.zeros((num_rows,N_TOKEN),dtype=np.float32),
        "features":np.zeros((num_rows,N_FEAT),dtype=np.float32),
        "tweet_ids":np.zeros((num_rows,1),dtype='U32'),
        "ids":np.zeros((num_rows,2),dtype=np.int32),
     }

    type_map = {
        **{k: np.bool_ for k in range(0, N_LABEL)},
        **{k: np.float32 for k in range(N_LABEL, N_LABEL + N_TOKEN)},
        **{k: np.float32 for k in range(N_LABEL + N_TOKEN, N_LABEL + N_TOKEN + N_FEAT)},
        **{k: np.int32 for k in range(N_LABEL + N_TOKEN + N_FEAT + 1, N_LABEL + N_TOKEN + N_FEAT + 3)},
    }
    
    df_iterator = pd.read_csv(path, dtype=type_map, header=None, chunksize=1000000)
    h = 0

    for df in tqdm(df_iterator):
        t = h + len(df)
        ret['labels'][h:t] = df[range(0, N_LABEL)].values
        ret['tokens'][h:t] = df[range(N_LABEL, N_LABEL + N_TOKEN)].values
        ret['features'][h:t] = df[range(N_LABEL + N_TOKEN, N_LABEL + N_TOKEN + N_FEAT)].values
        ret['tweet_ids'][h:t] = df[range(N_LABEL + N_TOKEN + N_FEAT, N_LABEL + N_TOKEN + N_FEAT + 1)].values
        ret['ids'][h:t] = df[range(N_LABEL + N_TOKEN + N_FEAT + 1, N_LABEL + N_TOKEN + N_FEAT + 3)].values
        h = t
        del df
        gc.collect()
    
    del df_iterator
    gc.collect()

    return ret

def generate_lb_dict(path):

    type_map = {
        **{k: np.float32 for k in range(0,N_TOKEN)},
        **{k: np.float32 for k in range(N_TOKEN, N_TOKEN + N_FEAT)},
        **{k: np.int32 for k in range(N_TOKEN + N_FEAT+1, N_TOKEN + N_FEAT+3)}
    }


    df = pd.read_csv(path, dtype=type_map, header=None)
    print(df.shape)

    dic = {
        "tokens": np.array(df[range(0,N_TOKEN)]),
        "features": np.array(df[range(N_TOKEN, N_TOKEN + N_FEAT)]),
        "tweet_ids": np.array(df[range(N_TOKEN + N_FEAT,  N_TOKEN + N_FEAT+1)]),
        "ids": np.array(df[range(N_TOKEN + N_FEAT+1, N_TOKEN + N_FEAT+3)]),
        "lb_user_ids": np.array(df[range(N_TOKEN + N_FEAT+3, N_TOKEN + N_FEAT+4)]),
        "labels" : []
    }

    del df
    gc.collect()
    return dic

if not os.path.exists("./data"):
    os.mkdir("./data")

train_dict = generate_dict_np(train_path)
t = {'tweet_ids':train_dict['tweet_ids']}

with open('./data/Train_tid.sav', 'wb') as f:
    joblib.dump(t, f)

##Create scalers
scaler_f = PowerTransformer(copy=False)
start_time = time.time()
s = len(train_dict['features'])
scaler_f.fit(train_dict['features'][np.random.choice(s, int(0.1*s))].astype(np.float64,copy=False))
print("Elapsed: {0}".format(inhour(time.time() - start_time)))
print("fit feature scaler")

scaler_t = MinMaxScaler(copy=False)
start_time = time.time()
scaler_t.fit(train_dict['tokens'][np.random.choice(s, int(0.1*s))])
print("Elapsed: {0}".format(inhour(time.time() - start_time)))
print("fit token scaler")

##Save scalers
with open('./data/f_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_f, f, protocol=4)

with open('./data/t_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_t, f, protocol=4)

##Apply scalers to train set 
start_time = time.time()
train_dict['features'] = scaler_f.transform(train_dict['features'])

print("finish feature transformation")
print("Elapsed: {0}".format(inhour(time.time() - start_time)))

start_time = time.time()
train_dict['tokens'] = scaler_t.transform(train_dict['tokens'])
print("finish token transformation")
print("Elapsed: {0}".format(inhour(time.time() - start_time)))

with open('./data/Train.sav', 'wb') as f:
    joblib.dump(train_dict, f)
del(train_dict)
print("saved!")

'''
with open('./data/f_scaler.pkl', 'rb') as f:
    scaler_f = pickle.load(f)
​
with open('./data/t_scaler.pkl', 'rb') as f:
    scaler_t = pickle.load(f)
'''

#Apply scaler to valid set 
valid_dict = generate_dict(valid_path)
t = {'tweet_ids':valid_dict['tweet_ids']}
with open('./data/Valid_tid.sav', 'wb') as f:
    joblib.dump(t, f)

start_time = time.time()
valid_dict['features'] = scaler_f.transform(valid_dict['features'])
print("finish feature transformation")
print("Elapsed: {0}".format(inhour(time.time() - start_time)))

start_time = time.time()
valid_dict['tokens'] = scaler_t.transform(valid_dict['tokens'])
print("finish token transformation")
print("Elapsed: {0}".format(inhour(time.time() - start_time)))


with open('./data/Valid.sav', 'wb') as f:
    joblib.dump(valid_dict, f)
del(valid_dict)
print("saved!")

###submit
lb_dict = generate_lb_dict(lb_path)

start_time = time.time()
lb_dict['features'] = scaler_f.transform(lb_dict['features'])
print("finish feature transformation")
print("Elapsed: {0}".format(inhour(time.time() - start_time)))

start_time = time.time()
lb_dict['tokens'] = scaler_t.transform(lb_dict['tokens'])
print("finish token transformation")
print("Elapsed: {0}".format(inhour(time.time() - start_time)))

with open('./data/Submit.sav', 'wb') as f:
    joblib.dump(lb_dict, f)
del(lb_dict)
print("saved!")


###test
test_dict = generate_lb_dict(test_path)

start_time = time.time()
test_dict['features'] = scaler_f.transform(test_dict['features'])
print("finish feature transformation")
print("Elapsed: {0}".format(inhour(time.time() - start_time)))

start_time = time.time()
test_dict['tokens'] = scaler_t.transform(test_dict['tokens'])
print("finish token transformation")
print("Elapsed: {0}".format(inhour(time.time() - start_time)))

with open('./data/Test.sav', 'wb') as f:
    joblib.dump(test_dict, f)
del(test_dict)
print("saved!")