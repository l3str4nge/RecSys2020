import torch
from torch.utils.data import DataLoader, Dataset
import random
import os
import pandas as pd
import numpy as np
import pickle
from utils.io import get_csr_matrix
from fbpca import pca
import joblib

class TwitterDataset(Dataset):
    """Wrapper, convert <user, creator, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, creator_tensor, token_tensor, feature_tensor, target_tensor):
        self.user_tensor = user_tensor
        self.creator_tensor = creator_tensor
        self.token_tensor = token_tensor
        self.feature_tensor = feature_tensor
        self.target_tensor = target_tensor
        #self.chunk_tensor = chunk_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.creator_tensor[index], \
            self.token_tensor[index], self.feature_tensor[index], self.target_tensor[index]#, self.chunk_tensor[index]

    def __len__(self):
        return self.user_tensor.size

class LBDataset(Dataset):
    """Wrapper, convert <user, creator, tweet_lb, user_lb> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, creator_tensor, token_tensor, feature_tensor, tweet_lb, user_lb):
        self.user_tensor = user_tensor
        self.creator_tensor = creator_tensor
        self.token_tensor = token_tensor
        self.feature_tensor = feature_tensor
        self.user_lb = user_lb
        self.tweet_lb = tweet_lb

    def __getitem__(self, index):
        return self.user_tensor[index], self.creator_tensor[index], \
            self.token_tensor[index], self.feature_tensor[index], self.tweet_lb[index], self.user_lb[index]

    def __len__(self):
        return self.user_tensor.size


class Data:
    def __init__(self, dpath, tr_name, val_name, u_threshold=7, c_threshold=7,is_lb=False):
        if not is_lb:
            with open(os.path.join(dpath, tr_name), 'rb') as f:
                tr_dict = joblib.load(f)
            with open(os.path.join(dpath, val_name), 'rb') as f:
                val_dict = joblib.load(f)
            val_engages = pd.DataFrame(data=val_dict['ids'], columns=["userId", "creatorId"])
            tr_engages = pd.DataFrame(data=tr_dict['ids'], columns=["userId", "creatorId"])
            tr_values = (np.sum(tr_dict['labels'],axis=1)>0).astype(np.float32)
            M = get_csr_matrix(tr_engages,'userId','creatorId',tr_values)
            print("startpca")
            self.P, _, self.Q = pca(M,
                   k=25,
                   n_iter=10,
                   raw=True)
            self.Q = self.Q.T
            print("endpca")
            #overlab check
            def overlab_check(unique_uId, unique_cId):
                intersect = len(np.intersect1d(unique_cId, unique_uId))
                union = len(np.union1d(unique_cId, unique_uId))
                print("ratio an id being in  both user and creator, P(u&c):",intersect/union)
                print("ratio an id being in user given it is in creator, P(u|c):", intersect/len(unique_uId))
                print("ratio an id being in creator given it is in user, P(c|u):" ,intersect/len(unique_cId))
            
            print("Before taking out ids with few examples")
            overlab_check(tr_engages['userId'].unique(), tr_engages['creatorId'].unique())

            pos_index = np.sum(tr_dict['labels'],axis=1)>0

            #_cut_by_counts
            if u_threshold > 0:
                unique_uId = self._cut_by_counts(tr_engages[pos_index], 'userId', u_threshold)         
            if c_threshold > 0:
                unique_cId = self._cut_by_counts(tr_engages[pos_index], 'creatorId', c_threshold)
            self.P = self.P[unique_uId]
            self.Q = self.Q[unique_cId]
            print("After taking out ids with few examples")
            overlab_check(unique_uId, unique_cId)
        
            #reindex ids to increase one by one from zero to reduce size of embedding matrix 
            uId2trans_uId = {idx:num for num,idx in enumerate(unique_uId)}
            cId2trans_cId = {idx:num for num,idx in enumerate(unique_cId)}
            s = [self.P, self.Q, uId2trans_uId, cId2trans_cId]
            ##Save scalers
            with open('./data/cf.sav', 'wb') as f:
                joblib.dump(s, f)

            format = lambda x: uId2trans_uId.get(x, len(uId2trans_uId))
            tr_engages.userId = tr_engages['userId'].map(format)
            val_engages.userId = val_engages['userId'].map(format)

            format = lambda x: cId2trans_cId.get(x, len(cId2trans_cId))
            tr_engages.creatorId = tr_engages['creatorId'].map(format)
            val_engages.creatorId = val_engages['creatorId'].map(format)

            self.tr_engages = tr_engages.to_numpy()
            self.tr_labels = np.array(tr_dict['labels']).astype(np.float)

            self.tr_features = tr_dict['features']
            self.tr_tokens = tr_dict['tokens']
            #self.tr_chunks = tr_dict['chunks'].astype(np.int64)
        
            self.val_engages = val_engages.to_numpy()
            self.val_labels = np.array(val_dict['labels']).astype(np.float)
            
            self.val_features = val_dict['features']
            self.val_tokens = val_dict['tokens']
            #self.val_chunks = np.zeros(len(self.val_features),dtype=np.int64)

            self.n_user = len(uId2trans_uId)
            self.n_creator = len(cId2trans_cId)
            
            self.n_feature = self.val_features.shape[1]
            self.n_token = self.val_tokens.shape[1]

            print("num user:",self.n_user)
            print("num creator:",self.n_creator)
            print("num of row for train engage:", len(self.tr_engages))
            print("num of row for valid(or LB) engage:", len(self.val_engages))

            u_coldstart = np.sum(self.tr_engages[:,0] == self.n_user)/len(self.tr_engages)
            c_coldstart = np.sum(self.tr_engages[:,1] == self.n_creator)/len(self.tr_engages)
            a_coldstart = np.sum(np.sum(self.tr_engages[:,:2],axis=-1) == \
                (self.n_user+self.n_creator))/len(self.tr_engages)
            
            print("Training engagement: ")
            print("user coldstart", u_coldstart)
            print("creator coldstart", c_coldstart)
            print("user-creator coldstart", a_coldstart)

            u_coldstart = np.sum(self.val_engages[:,0] == self.n_user)/len(self.val_engages)
            c_coldstart = np.sum(self.val_engages[:,1] == self.n_creator)/len(self.val_engages)
            a_coldstart = np.sum(np.sum(self.val_engages[:,:2],axis=-1) == \
                (self.n_user+self.n_creator))/len(self.val_engages)

            print("Validation(or LB) enagement: ")
            print("user coldstart", u_coldstart)
            print("creator coldstart", c_coldstart)
            print("user-creator coldstart", a_coldstart)
        else:
            print("LB")
            with open(os.path.join(dpath, val_name), 'rb') as f:
                val_dict = joblib.load(f)
            val_engages = pd.DataFrame(data=val_dict['ids'], columns=["userId", "creatorId"])
            with open('./data/cf.pkl', 'rb') as f:
                ids = pickle.load(f)
            #reindex ids to increase one by one from zero to reduce size of embedding matrix 
            uId2trans_uId = ids[0]
            cId2trans_cId = ids[1]

            format = lambda x: uId2trans_uId.get(x, len(uId2trans_uId))
            val_engages.userId = val_engages['userId'].map(format)

            format = lambda x: cId2trans_cId.get(x, len(cId2trans_cId))
            val_engages.creatorId = val_engages['creatorId'].map(format)

            self.val_lb_users = val_dict['lb_user_ids']
            self.val_lb_tweets = val_dict['tweet_ids']

            self.val_engages = val_engages.to_numpy()
            self.val_labels = np.array(val_dict['labels']).astype(np.float)
            
            self.val_features = val_dict['features']
            self.val_tokens = val_dict['tokens']
            #self.val_chunks = np.zeros(len(self.val_features),dtype=np.int64)

            self.n_user = len(uId2trans_uId)
            self.n_creator = len(cId2trans_cId)
            self.n_feature = self.val_features.shape[1]
            self.n_token = self.val_tokens.shape[1]
    
            u_coldstart = np.sum(self.val_engages[:,0] == self.n_user)/len(self.val_engages)
            c_coldstart = np.sum(self.val_engages[:,1] == self.n_creator)/len(self.val_engages)
            a_coldstart = np.sum(np.sum(self.val_engages[:,:2],axis=-1) == \
                (self.n_user+self.n_creator))/len(self.val_engages)

            print("Validation(or LB) enagement: ")
            print("user coldstart", u_coldstart)
            print("creator coldstart", c_coldstart)
            print("user-creator coldstart", a_coldstart)

    def _cut_by_counts(self, engages, criterion, threshold):
        counter = engages[criterion].value_counts()
        valid_index = counter[counter>=threshold].index.values
        return valid_index

    def instance_a_train_loader(self, batch_size):
        dataset = TwitterDataset(user_tensor=self.tr_engages[:,0],
                                   creator_tensor=self.tr_engages[:,1],
                                   token_tensor=self.tr_tokens,
                                   feature_tensor=self.tr_features,
                                   target_tensor=self.tr_labels)
                                   #chunk_tensor = self.tr_chunks)
        del self.tr_tokens
        del self.tr_engages
        del self.tr_features
        del self.tr_labels
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers =20, pin_memory=True)


    def instance_a_valid_loader(self, batch_size):
        dataset = TwitterDataset(user_tensor=self.val_engages[:,0],
                                   creator_tensor=self.val_engages[:,1],
                                   token_tensor=self.val_tokens,
                                   feature_tensor=self.val_features,
                                   target_tensor=self.val_labels)
                                   #chunk_tensor = self.val_chunks)
        del self.val_tokens
        del self.val_engages
        del self.val_features
        del self.val_labels
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers =20, pin_memory=True)
 

    def instance_a_lb_loader(self, batch_size):
        dataset =LBDataset(user_tensor=self.val_engages[:,0],
                                   creator_tensor=self.val_engages[:,1],
                                   feature_tensor=self.val_features,
                                   token_tensor=self.val_tokens,
                                   tweet_lb=self.val_lb_tweets.tolist(),
                                   user_lb = self.val_lb_users.tolist())
        del self.val_tokens
        del self.val_engages
        del self.val_features
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers =20, pin_memory=True)