import sys
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import BertTokenizer, BertConfig, BertModel
import pickle
from tokenizers import BertWordPieceTokenizer
from transformers import BertForMaskedLM

device = "cuda" if torch.cuda.is_available() else "cpu"


import numpy as np
from random import shuffle
from torch.utils.data import Sampler
import torch
import math
from tqdm import tqdm
import pandas as pd
import gc


class SortedSampler(Sampler):

    def __init__(self, data_source):
        self.data_source = data_source
        print("Initializing the sorted sampler ... ")

        self.ind_n_len = [(i, len(p[0])) for i, p in enumerate(data_source)]

    def __iter__(self):
        print("Sorting samples by length... ")

        return iter([pair[0] for pair in sorted(self.ind_n_len, key=lambda tup: tup[1], reverse=True)])

    def __len__(self):
        return len(self.data_source)


class BySequenceLengthSampler(Sampler):

    def __init__(self, data_source,  
                bucket_boundaries, batch_size=64, drop_last=True):
        self.data_source = data_source
        print("Initializing the bucket sampler ... please wait up to 2 minutes ... ")

        self.ind_n_len = [(i, len(p[0])) for i, p in enumerate(data_source)]
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size
        self.drop_last = drop_last

        if self.drop_last:
            print("WARNING: drop_last=True, dropping last non batch-size batch in every bucket ... ")

        self.boundaries = list(self.bucket_boundaries)
        self.buckets_min = torch.tensor([np.iinfo(np.int32).min] + self.boundaries)
        self.buckets_max = torch.tensor(self.boundaries + [np.iinfo(np.int32).max])
        self.boundaries = torch.tensor(self.boundaries)

    def shuffle_tensor(self, t):
        return t[torch.randperm(len(t))]
        
    def __iter__(self):
        print("Constructing the bucket sampler iterator to build batches of similar length ... ")

        df = pd.DataFrame(self.ind_n_len, columns=["p", "seq_len"])
        # boundaries are now ( ] and they don't handle out-of-boundary values so please specify them properly!
        print("WARNING: boundaries are now ( ] and they don't handle out-of-boundary values so please specify them properly!")
        df['bin'] = pd.cut(df["seq_len"], self.boundaries, labels=False, right=False)

        data_buckets = df.groupby('bin')['p'].apply(list).to_dict()

        for k in tqdm(data_buckets.keys()):

            data_buckets[k] = torch.tensor(data_buckets[k])

        iter_list = []
        for k in tqdm(data_buckets.keys()):

            t = self.shuffle_tensor(data_buckets[k])
            batch = torch.split(t, self.batch_size, dim=0)

            if self.drop_last and len(batch[-1]) != self.batch_size:
                batch = batch[:-1]

            iter_list += batch

        shuffle(iter_list) # shuffle all the batches so they arent ordered by bucket
        # size
        for i in iter_list: 
            yield i.numpy().tolist() # as it was stored in an array
    
    def __len__(self):
        return len(self.data_source)
    
    def element_to_bucket_id(self, x, seq_length):

        valid_buckets = (seq_length >= self.buckets_min)*(seq_length < self.buckets_max)
        bucket_id = valid_buckets.nonzero()[0].item()

        return bucket_id
    
""" 
As it is numpy functions you’ll need to keep it on the CPU for now. And as your BatchSampler already creates the batches, your DataLoader should have a batch size of 1.

Also, buckets for values smaller and larger than your buckets are also created so you won’t lose any data.

NB. Currently the batch size must be smaller than smallest number of sequences in any bucket so you may have to adjust your bucket boundaries depending on your batch sizes.
"""

def save_dicts(output_file, max_dict, mean_dict, cls_dict, segment_num):

    # with open(output_file+"_max_seg{}.p".format(segment_num), "wb") as f:
    #     pickle.dump(max_dict, f)
    #     print("saved the max embeddings!")

    with open(output_file+"_mean_seg{}.p".format(segment_num), "wb") as f:
        pickle.dump(mean_dict, f)
        print("saved the mean embeddings!")

    # with open(output_file+"_cls_seg{}.p".format(segment_num), "wb") as f:
    #     pickle.dump(cls_dict, f)
    #     print("saved the cls embeddings!")


class TwitterDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, "rb") as f:
            self.data = pickle.load(f)
            self.token = self.data["tokens"]
            self.tweet_id = self.data["tweet_id"]

    def __getitem__(self, i):

        return torch.tensor(self.token[i]), self.tweet_id[i]

    def __len__(self):
        return len(self.token)


if __name__ == "__main__":

    chunk_name = "clean_val"
    dataset_dir = "/home/kevin/Projects/RecSys2020_NLP/data/{}_tweet_tokens.p".format(chunk_name)
    output_file = "/home/kevin/Projects/RecSys2020_NLP/data/{}_tweet_embs".format(chunk_name)
    model_checkpoint = "/media/kevin/datahdd/data/recsys/checkpoints/pretrain/bert_multi/checkpoint-1500"
    vocab_file = "/home/kevin/Projects/RecSys2020_NLP/pretrain/token/wordpiece-multi-100000-8000-vocab.txt"
    batch_size = 10

    tokenizer = BertTokenizer(vocab_file)
    model = BertForMaskedLM.from_pretrained(model_checkpoint)
    model = model.bert

    def collate(batch):
        tokens = [b[0] for b in batch]
        lens = [len(x) for x in tokens]

        tokens = pad_sequence(tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = (tokens != tokenizer.pad_token_id).int()

        return tokens, attention_mask, [b[1] for b in batch], torch.tensor(lens).unsqueeze(1)

    def mean_emb_no_pad(H, L):
        mask = torch.arange(H.shape[1]).repeat(H.shape[0], 1)
        mask = (mask < L).float()
        mask[:, 0] = 0
        masked_h = H * mask.unsqueeze(2)
        mean_emb = (masked_h.sum(dim=1)/L)
        return mean_emb

    dataset = TwitterDataset(file_path=dataset_dir)
    # sampler = BySequenceLengthSampler(dataset, batch_size=batch_size, drop_last=False)
    sampler = SortedSampler(dataset)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        batch_sampler=sampler,
                        num_workers=0,
                        collate_fn=collate,
                        drop_last=False,
                        pin_memory=False)

    model.to(device)
    model = torch.nn.DataParallel(model)
    model.eval()

    cls_dict = {}
    max_dict = {}
    mean_dict = {}
    chunk_segment_samples_seen = 0
    segment_counter = 0

    for batch in tqdm(loader):
        
        inputs, masks, tweet_ids, lens = batch

        inputs, masks, lens  = [x.to(device) for x in [inputs, masks, lens]]
 
        with torch.no_grad():
            # torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
            (h_sequence, h_cls) = model(input_ids=inputs, attention_mask=masks)
            h_sequence = h_sequence.detach().cpu()

            max_emb = torch.max(h_sequence[:, 1:, :], dim=1)[0]
            # mean_emb = torch.mean(h_sequence[:, 1:, :], dim=1)
            mean_emb = mean_emb_no_pad(h_sequence, lens)
            cls_emb = h_sequence[:, 0, :]

            max_emb, mean_emb, cls_emb = [x.numpy() for x in [max_emb, mean_emb, cls_emb]]

        # max_dict.update(zip(tweet_ids, max_emb))
        mean_dict.update(zip(tweet_ids, mean_emb))
        # cls_dict.update(zip(tweet_ids, cls_emb))

        chunk_segment_samples_seen += len(tweet_ids)

        if chunk_segment_samples_seen > 2700000:  # you probably can't fit more than 2.7M samples in RAM
            print("Saving segment with {} samples ... ".format(chunk_segment_samples_seen))
            save_dicts(output_file, max_dict, mean_dict, cls_dict, segment_counter)
            cls_dict = {}
            max_dict = {}
            mean_dict = {}
            chunk_segment_samples_seen = 0
            segment_counter += 1
            gc.collect()


    save_dicts(output_file, max_dict, mean_dict, cls_dict, segment_counter)