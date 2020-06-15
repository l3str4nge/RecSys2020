import torch
import random
import os
import pandas as pd
import numpy as np
import pickle
import joblib

from torch.utils.data import DataLoader, Dataset



class TwitterDataset(Dataset):
    """Wrapper, convert <user, creator, rating> Tensor into Pytorch Dataset"""
    def __init__(self, token_tensor, feature_tensor, target_tensor, tweetid, userid, text_dict): #, chunk):
        self.token_tensor = token_tensor
        self.feature_tensor = feature_tensor
        self.target_tensor = target_tensor
        self.tweetid = tweetid
        self.userid = userid
        self.text_dict = text_dict
        #self.chunk = chunk

    def __getitem__(self, index):

        inputs = torch.tensor(self.text_dict[self.tweetid[index]])
        #inputs = torch.tensor([150]*np.random.randint(10,90), dtype = torch.int64)

        return self.token_tensor[index], self.feature_tensor[index], self.target_tensor[index], inputs, self.tweetid[index], self.userid[index]

    def __len__(self):
        return self.token_tensor.shape[0]

class LBDataset(Dataset):
    """Wrapper, convert <user, creator, tweet_lb, user_lb> Tensor into Pytorch Dataset"""
    def __init__(self, token_tensor, feature_tensor, tweetid, userid, text_dict): #, chunk):
        self.token_tensor = token_tensor
        self.feature_tensor = feature_tensor
        self.tweetid = tweetid
        self.userid = userid
        self.text_dict = text_dict
        #self.chunk = chunk

    def __getitem__(self, index):

        inputs = torch.tensor(self.text_dict[self.tweetid[index]], dtype = torch.int64)
        #inputs = torch.tensor([150]*np.random.randint(10,90), dtype=torch.int64)

        return self.token_tensor[index], self.feature_tensor[index], inputs, self.tweetid[index], self.userid[index]

    def __len__(self):
        return self.token_tensor.shape[0]


class Data:
    def __init__(self, args, dpath, tr_name, val_name,is_lb=False):

        print("Loading tweet tokens mapping ... ")
        self.tid_to_text_dict = {}
        loaded = pickle.load(open(args.tweet_id_to_text_file, "rb"))
        for i in range(len(loaded["tweet_id"])):
            self.tid_to_text_dict[loaded["tweet_id"][i]] = loaded["tokens"][i]
        #print(self.tid_to_text_dict)
        #self.tid_to_text_dict = {}
        print("There are {} tweet IDS in the dictionary".format(len(self.tid_to_text_dict.keys())))
        #raise Exception

        print("Loading train and val sets ... ")
        if not is_lb:
            with open(os.path.join(dpath, tr_name), 'rb') as f:
                tr_dict = joblib.load(f)

            self.tr_labels = np.array(tr_dict['labels']).astype(np.float)
            self.tr_features = tr_dict['features']
            self.tr_tokens = tr_dict['tokens']
            #self.tr_tweetid = tr_dict['tweet_ids']
            self.tr_tweetid = []
            tweet_rows = tr_dict['tweet_row']
            tweet_ids = pickle.load(open(os.path.join(dpath, 'old_split/tweet_row_to_id_val.p'), 'rb'))
            for i in range(len(tweet_rows)):
                self.tr_tweetid.append(tweet_ids[tweet_rows[i]])
            print(len(self.tr_tweetid))
            #self.tr_chunks = tr_dict['chunks'].astype(np.int64)
            self.tr_userid = tr_dict['ids'][:,0]

            print(self.tr_labels.shape, self.tr_features.shape, len(self.tr_tokens), len(self.tr_tweetid))

        with open(os.path.join(dpath, val_name), 'rb') as f:
            val_dict = joblib.load(f)
        
        self.val_labels = np.array(val_dict['labels']).astype(np.float)
        self.val_features = val_dict['features']
        self.val_tokens = val_dict['tokens']
        #self.val_tweetid = val_dict['tweet_ids']
        self.val_tweetid = []
        tweet_rows = val_dict['tweet_row']
        tweet_ids = pickle.load(open(os.path.join(dpath, 'old_split/tweet_row_to_id_val.p'), 'rb'))
        for i in range(len(tweet_rows)):
            self.val_tweetid.append(tweet_ids[tweet_rows[i]])
        print(len(self.val_tweetid))
        #self.val_chunks = np.zeros(len(self.val_features),dtype=np.int64)
        self.val_userid = val_dict['ids'][:,0]

        print(self.val_labels.shape, self.val_features.shape, len(self.val_tokens), len(self.val_tweetid))

        # if 'lb_user_ids' in val_dict.keys():
        #     self.val_lb_users = val_dict['lb_user_ids']
        #     self.val_lb_tweets = val_dict['tweet_ids']
            
        self.n_feature = self.val_features.shape[1] 
        self.n_token = self.val_tokens.shape[1]


    def instance_a_train_dataset(self):
        dataset = TwitterDataset(token_tensor=self.tr_tokens,
                                feature_tensor=self.tr_features,
                                target_tensor=self.tr_labels,
                                tweetid=self.tr_tweetid,
                                userid=self.tr_userid,
                                text_dict=self.tid_to_text_dict)
                                #chunk = self.tr_chunks)

        return dataset


    def instance_a_valid_dataset(self):
        dataset = TwitterDataset(token_tensor=self.val_tokens,
                                feature_tensor=self.val_features,
                                target_tensor=self.val_labels,
                                tweetid=self.val_tweetid,
                                userid=self.val_userid,
                                text_dict=self.tid_to_text_dict)
                                #chunk=self.val_chunks)
        return dataset

    def instance_a_lb_dataset(self):
        dataset = LBDataset(token_tensor=self.val_tokens,
                            feature_tensor=self.val_features,
                            tweetid=self.val_tweetid,
                            userid=self.val_userid,
                            text_dict=self.tid_to_text_dict)
                            #chunk=self.val_chunks)
        return dataset

    def instance_a_train_loader(self, batch_size):
        dataset = TwitterDataset(token_tensor=self.tr_tokens,
                                feature_tensor=self.tr_features,
                                target_tensor=self.tr_labels,
                                tweetid=self.tr_tweetid,
                                userid=self.tr_userid,
                                text_dict=self.tid_to_text_dict)
                                #chunk = self.tr_chunks)

        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers =20, pin_memory=True)


    def instance_a_valid_loader(self, batch_size):
        dataset = TwitterDataset(token_tensor=self.val_tokens,
                                feature_tensor=self.val_features,
                                target_tensor=self.val_labels,
                                tweetid=self.val_tweetid,
                                userid=self.val_userid,
                                text_dict=self.tid_to_text_dict)
                                #chunk=self.val_chunks)
        del self.val_tokens
        del self.val_features
        del self.val_labels
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers =20, pin_memory=True)
 

    def instance_a_lb_loader(self, batch_size):
        dataset = LBDataset(token_tensor=self.val_tokens,
                            feature_tensor=self.val_features,
                            tweetid=self.val_tweetid,
                            userid=self.val_userid,
                            text_dict=self.tid_to_text_dict)
                            #chunk=self.val_chunks)
        del self.val_tokens
        del self.val_features
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers =20, pin_memory=True)
