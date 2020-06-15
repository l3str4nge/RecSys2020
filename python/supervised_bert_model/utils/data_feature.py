import torch
from torch.utils.data import DataLoader, Dataset
import random
import os
import pandas as pd
import numpy as np
import pickle
from utils.io import get_csr_matrix
import joblib

class TwitterDataset(Dataset):
    """Wrapper, convert <user, creator, rating> Tensor into Pytorch Dataset"""
    def __init__(self, token_tensor, feature_tensor, target_tensor): #, chunk):
        self.token_tensor = token_tensor
        self.feature_tensor = feature_tensor
        self.target_tensor = target_tensor
        #self.chunk = chunk

    def __getitem__(self, index):
        return self.token_tensor[index], self.feature_tensor[index], self.target_tensor[index]#, self.chunk[index]

    def __len__(self):
        return self.token_tensor.shape[0]

class LBDataset(Dataset):
    """Wrapper, convert <user, creator, tweet_lb, user_lb> Tensor into Pytorch Dataset"""
    def __init__(self, token_tensor, feature_tensor, tweet_lb, user_lb): #, chunk):
        self.token_tensor = token_tensor
        self.feature_tensor = feature_tensor
        self.user_lb = user_lb
        self.tweet_lb = tweet_lb
        #self.chunk = chunk

    def __getitem__(self, index):
        return self.token_tensor[index], self.feature_tensor[index], self.tweet_lb[index], self.user_lb[index]#, self.chunk[index]

    def __len__(self):
        return self.token_tensor.shape[0]


class Data:
    def __init__(self, dpath, tr_name, val_name,is_lb=False):


        if not is_lb:
            with open(os.path.join(dpath, tr_name), 'rb') as f:
                tr_dict = joblib.load(f)
            
            self.tr_labels = np.array(tr_dict['labels']).astype(np.float)
            self.tr_features = tr_dict['features']
            self.tr_tokens = tr_dict['tokens']
            #self.tr_chunks = tr_dict['chunks'].astype(np.int64)

        with open(os.path.join(dpath, val_name), 'rb') as f:
            val_dict = joblib.load(f)
        
        self.val_labels = np.array(val_dict['labels']).astype(np.float)
        self.val_features = val_dict['features']
        self.val_tokens = val_dict['tokens']
        #self.val_chunks = np.zeros(len(self.val_features),dtype=np.int64)

        if 'lb_user_ids' in val_dict.keys():
            self.val_lb_users = val_dict['lb_user_ids']
            self.val_lb_tweets = val_dict['tweet_ids']
            
        self.n_feature = self.val_features.shape[1] 
        self.n_token = self.val_tokens.shape[1]


    def instance_a_train_loader(self, batch_size):
        dataset = TwitterDataset(token_tensor=self.tr_tokens,
                                   feature_tensor=self.tr_features,
                                   target_tensor=self.tr_labels)
                                   #chunk = self.tr_chunks)

        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers =20, pin_memory=True)


    def instance_a_valid_loader(self, batch_size):
        dataset = TwitterDataset(token_tensor=self.val_tokens,
                                   feature_tensor=self.val_features,
                                   target_tensor=self.val_labels)
                                   #chunk=self.val_chunks)
        del self.val_tokens
        del self.val_features
        del self.val_labels
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers =20, pin_memory=True)
 

    def instance_a_lb_loader(self, batch_size):
        dataset =LBDataset(feature_tensor=self.val_features,
                                   token_tensor=self.val_tokens,
                                   tweet_lb=self.val_lb_tweets.tolist(),
                                   user_lb = self.val_lb_users.tolist())
                                   #chunk=self.val_chunks)
        del self.val_tokens
        del self.val_features
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers =20, pin_memory=True)