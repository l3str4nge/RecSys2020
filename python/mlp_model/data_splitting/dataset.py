import torch
from torch.utils.data import DataLoader, Dataset
import random
import os
import pandas as pd
import numpy as np
import pickle
# from utils.io import get_csr_matrix
import joblib
from pathlib import Path


class TwitterDataset(Dataset):
    """Wrapper, convert <user, creator, rating> Tensor into Pytorch Dataset"""
    def __init__(self, token_tensor, feature_tensor, target_tensor, chunk, row_id=None):
        self.token_tensor = token_tensor
        self.feature_tensor = feature_tensor
        self.target_tensor = target_tensor
        self.chunk = chunk
        self.row_id = row_id

    def __getitem__(self, index):

        if self.row_id is None:
            return self.token_tensor[index], self.feature_tensor[index], self.target_tensor[index], self.chunk[index]
        else:
            return self.token_tensor[index], self.feature_tensor[index], self.target_tensor[index], self.chunk[self.row_id[index]]

    def __len__(self):
        return self.token_tensor.shape[0]

class LBDataset(Dataset):
    """Wrapper, convert <user, creator, tweet_lb, user_lb> Tensor into Pytorch Dataset"""
    def __init__(self, token_tensor, feature_tensor, tweet_lb, user_lb, chunk):
        self.token_tensor = token_tensor
        self.feature_tensor = feature_tensor
        self.user_lb = user_lb
        self.tweet_lb = tweet_lb
        self.chunk = chunk

    def __getitem__(self, index):
        return self.token_tensor[index], self.feature_tensor[index], self.tweet_lb[index], self.user_lb[index], self.chunk[index]

    def __len__(self):
        return self.token_tensor.shape[0]


class Data:
    def __init__(self, dpath, val_file_name, is_lb=False):

        self.train_files = sorted([str(x) for x in Path(os.path.join(dpath, "splits")).glob("*dataset*.sav")])
        self.val_file = os.path.join(dpath, val_file_name)

        self.current_split = 0
        self.num_splits = len(self.train_files)

        # load val data
        with open(self.val_file, 'rb') as f:
            val_dict = joblib.load(f)
        
        self.val_labels = np.array(val_dict['labels']).astype(np.float)
        #b = []
        #for i in [43,44,45,46]:
        #    b.append(val_dict['features'][:,i] > np.min(val_dict['features'][:,i]))
        #self.val_features = np.concatenate((val_dict['features'], np.vstack(b).T), axis=1)
        self.val_features = val_dict['features']
        self.val_tokens = val_dict['tokens']
        self.val_chunk = np.zeros(len(self.val_features),dtype=np.int64)
        if 'lb_user_ids' in val_dict.keys():
            self.val_lb_users = val_dict['lb_user_ids']
            self.val_lb_tweets = val_dict['tweet_ids']
            
        self.n_feature = self.val_features.shape[1]
        self.n_token = self.val_tokens.shape[1]


        if not is_lb:
                        #chunk_point = [3123598*2, 3700539*2, 4065998*2, 3171717*2, 3745257*2, \
            #                4185078*2, 3332611*2, 3762057*2, 4159071*2, 3131109*2,\
            #                3753093*2, 4166072*2, 3433993*2, 4750430*2, 3971462*2]
            chunk_point = [3123598*2, 3700539*2, 4065998*2, 3171717*2, \
                            3745257*2, 4185078*2, 3332611*2, 3762057*2, \
                            4159071*2, 3131109*2, 3753093*2, 4166072*2]
            
            chunk_index = np.zeros(np.sum(chunk_point),dtype=np.int64)

            s = 0
            idx = 0

            for i in chunk_point:
                t = s + i
                chunk_index[s:t] = idx
                idx += 1
                s = t

            self.tr_chunk = chunk_index


    def initialize_split(self):

        if self.current_split % self.num_splits == 0:
            print("This is the start of {} splits, shuffling the split list ... ".format(self.num_splits))
            random.shuffle(self.train_files)

        split_idx = self.current_split % self.num_splits
        self.current_train_file = self.train_files[split_idx]

        print("Initializing train split {} ... ".format(self.current_train_file))
        

        # TODO load the embeddings
        # embeddings = pickle.load(f_split_x)
        

        with open(self.current_train_file, 'rb') as f:
            tr_dict = joblib.load(f)


        self.tr_labels = np.array(tr_dict['labels']).astype(np.float)

        #b = []
        #for i in [43,44,45,46]:
        #    b.append(tr_dict['features'][:,i] > np.min(tr_dict['features'][:,i]))
        #self.tr_features = np.concatenate((tr_dict['features'], np.vstack(b).T), axis=1)
        self.tr_features = tr_dict['features']
        self.tr_tokens = tr_dict['tokens']
        self.row_id = tr_dict["row_id"]

        self.current_split += 1


    def instance_a_train_loader(self, batch_size):
        '''
        chunk_point = [3123598*2, 3700539*2, 4065998*2, 3171717*2, \
                        3745257*2, 4185078*2, 3332611*2, 3762057*2, \
                        4159071*2, 3131109*2, 3753093*2, 4166072*2]        
        '''

        self.initialize_split()

        #s = 0
        #idx = np.random.permutation(12)
        #for i in range(12):
        #    s = s + int(chunk_point[i]/2)
        #    t = s + int(chunk_point[i]//2)
        #    self.tr_chunk[s:t] = idx[i]
        #    s = t
        
        #self.tr_chunk = self.tr_chunk.astype(np.int64,copy=False)
        dataset = TwitterDataset(token_tensor=self.tr_tokens,
                                   feature_tensor=self.tr_features,
                                   target_tensor=self.tr_labels,
                                   chunk = self.tr_chunk,
                                   row_id=self.row_id)
        #del self.tr_tokens
        #del self.tr_features
        #del self.tr_labels
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers =20, pin_memory=True)


    def instance_a_valid_loader(self, batch_size):
        dataset = TwitterDataset(token_tensor=self.val_tokens,
                                   feature_tensor=self.val_features,
                                   target_tensor=self.val_labels,
                                   chunk=self.val_chunk)
        del self.val_tokens
        del self.val_features
        del self.val_labels
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers =20, pin_memory=True)
 

    def instance_a_lb_loader(self, batch_size):
        dataset =LBDataset(feature_tensor=self.val_features,
                                   token_tensor=self.val_tokens,
                                   tweet_lb=self.val_lb_tweets.tolist(),
                                   user_lb = self.val_lb_users.tolist(),
                                   chunk=self.val_chunk)
        del self.val_tokens
        del self.val_features
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers =20, pin_memory=True)