import sys
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import pickle
from transformers import XLMRobertaForMaskedLM

device = "cuda" if torch.cuda.is_available() else "cpu"


import numpy as np
from random import shuffle
from torch.utils.data import Sampler
import torch
import math
from tqdm import tqdm
import pandas as pd
import gc
from bunch import FakeTokenizer


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


def save_dicts(output_file, max_dict, mean_dict, cls_dict, segment_num):
    with open(output_file+"/emb_max_seg{}.p".format(segment_num), "wb") as f:
        pickle.dump(max_dict, f)

    with open(output_file+"/emb_mean_seg{}.p".format(segment_num), "wb") as f:
        pickle.dump(mean_dict, f)

    with open(output_file+"/emb_cls_seg{}.p".format(segment_num), "wb") as f:
        pickle.dump(cls_dict, f)


class TwitterDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, "rb") as f:
            self.data = pickle.load(f)
            self.data = [[k, v[1]] for k, v in self.data.items()]

    def __getitem__(self, i):

        return torch.tensor(self.data[i][1]), self.data[i][0]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":

    batch_size = 10
    # TODO min_num_words = 3, should we use this?
    # tokens_file = "/home/layer6/recsys/unsupervised_data/xlmr_trainvalsubmit_only_tokens.p"  # don't compute them if not in train/val/submit set
    # output_file = "/home/layer6/recsys/embeddings/xlmr/"
    # model_checkpoint = "/home/layer6/recsys/xlm-r/checkpoints/run/checkpoint-1"
    tokens_file = "/home/kevin/Projects/xlm-r/data/xlmr_all_tweet_tokens_leaderboard.p"  # don't compute them if not in train/val/submit set
    output_dir = "/media/kevin/datahdd/data/embeddings"
    model_checkpoint = "/home/kevin/Projects/xlm-r/out/checkpoint-500"


    model = XLMRobertaForMaskedLM.from_pretrained(model_checkpoint)
    model = model.roberta

    tokenizer = FakeTokenizer()

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


    dataset = TwitterDataset(file_path=tokens_file)
    sampler = SortedSampler(dataset)

    loader = DataLoader(dataset, batch_size=batch_size, 
                        sampler=sampler, 
                        num_workers=0, 
                        collate_fn=collate,
                        drop_last=False, pin_memory=False)

    model.to(device)
    model = torch.nn.DataParallel(model)
    model.eval()

    cls_dict = {}
    max_dict = {}
    mean_dict = {}
    chunk_segment_samples_seen = 0
    segment_counter = 0
    processed_ids = set()

    for batch in tqdm(loader):
        
        inputs, masks, tweet_ids, lens = batch

        inputs, masks  = [x.to(device) for x in [inputs, masks]]
 
        with torch.no_grad():
            # torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
            (h_sequence, h_cls) = model(input_ids=inputs, attention_mask=masks)
            h_sequence = h_sequence.detach().cpu()

            max_emb = torch.max(h_sequence[:, 1:, :], dim=1)[0]
            mean_emb = mean_emb_no_pad(h_sequence, lens)
            cls_emb = h_sequence[:, 0, :]

            max_emb, mean_emb, cls_emb = [x.numpy() for x in [max_emb, mean_emb, cls_emb]]

        processed_ids.update(set(tweet_ids))
        max_dict.update(zip(tweet_ids, max_emb))
        mean_dict.update(zip(tweet_ids, mean_emb))
        cls_dict.update(zip(tweet_ids, cls_emb))

        chunk_segment_samples_seen += len(tweet_ids)

        if chunk_segment_samples_seen > 2500000:  # you probably can't fit more than 2.5M samples in RAM
            print("Saving segment with {} samples ... ".format(chunk_segment_samples_seen))
            save_dicts(output_dir, max_dict, mean_dict, cls_dict, segment_counter)
            cls_dict = {}
            max_dict = {}
            mean_dict = {}
            chunk_segment_samples_seen = 0
            segment_counter += 1
            gc.collect()


    save_dicts(output_dir, max_dict, mean_dict, cls_dict, segment_counter)
    # Sanity check
    assert len(processed_ids) == len(dataset)