import sys
import numpy as np
import torch
import pickle
import argparse
import math
import pandas as pd
import gc

from torch.utils.data import Sampler
from transformers import XLMRobertaForMaskedLM
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from random import shuffle
from time import time

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from dataset_bert import Data
from model import MasterNet




device = "cuda" if torch.cuda.is_available() else "cpu"


MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
}


class SortedSampler(Sampler):

    def __init__(self, data_source):
        self.data_source = data_source
        print("Initializing the sorted sampler ... ")
    
        #t1 = time()
        #for i, p in tqdm(enumerate(data_source)):
        #    #print("*******************", i)
        #    #print(p[0])
        #    if i == 10000:
        #        break
        #t2 = time()
        #print("time for 10k: {:.4f}".format(t2-t1))
        #raise Exception

        self.ind_n_len = [(i, len(p[0])) for i, p in enumerate(data_source)]

        #self.ind_n_len = []
        #for i, p in tqdm(enumerate(data_source)):
        #    self.ind_n_len.append((i, len(p[0])))
        #    if (i > 0) and i % 100000 == 0:
        #        print(i)
        #        break

    def __iter__(self):
        print("Sorting samples by length... ")

        return iter([pair[0] for pair in sorted(self.ind_n_len, key=lambda tup: tup[1], reverse=True)])

    def __len__(self):
        return len(self.data_source)


def save_dicts(output_file, max_dict, mean_dict, cls_dict, segment_num):
    # with open(output_file + "/emb_max_seg{}.p".format(segment_num), "wb") as f:
    #     pickle.dump(max_dict, f)

    with open(output_file + "/emb_mean_seg{}.p".format(segment_num), "wb") as f:
        pickle.dump(mean_dict, f)
        print("saved to {} file # {}".format(output_file, segment_num))

    # with open(output_file + "/emb_cls_seg{}.p".format(segment_num), "wb") as f:
    #     pickle.dump(cls_dict, f)


class TwitterDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, "rb") as f:
            loaded = pickle.load(f)
            self.data = []
            for i in range(len(loaded["tweet_id"])):
                self.data.append([loaded["tweet_id"][i], loaded["tokens"][i]])
            #self.data = [[k, v] for k, v in self.data.items()]

    def __getitem__(self, i):
        #print(self.data[i])
        #print(type(self.data[i][1]))
        return torch.tensor(self.data[i][1]), self.data[i][0]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file", default=None, type=str, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",
    )

    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )

    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--use_bucket_iterator", type=int, default=0)

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    parser.add_argument('--emb_size', type=int, default=768)
    parser.add_argument('--mlp_checkpoint', type=str, default='./checkpoint-blah')
    parser.add_argument('--mlp_data_path', default="/home/layer6/recsys/kevin_data")
    parser.add_argument('--mlp_train', default='Train.sav')
    parser.add_argument('--mlp_val', default='Valid.sav')
    parser.add_argument('--tweet_id_to_text_file', default='file.p')
    parser.add_argument('--run_name', default='test')

    parser.add_argument("--continue_training_path", default = "../../checkpoints/supervised_difflr/checkpoint-21000/")
    parser.add_argument("--mlp_learning_rate", default=0.00005)
    parser.add_argument("--dataset_to_export", default="train")
    parser.add_argument("--batch_size", default=10, type=int)

    args = parser.parse_args()


    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]


    # chunk_name = "clean_val"
    # dataset_dir = "/home/kevin/Projects/RecSys2020_NLP/data/{}_tweet_tokens.p".format(chunk_name)
    # model_checkpoint = "/media/kevin/datahdd/data/recsys/checkpoints/pretrain/bert_multi/checkpoint-1500"
    # vocab_file = "/home/kevin/Projects/RecSys2020_NLP/pretrain/token/wordpiece-multi-100000-8000-vocab.txt"
    # batch_size = 10
    output_file = args.output_dir

    # device

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # tokenizer

    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    # data

    data_generator = Data(args, args.mlp_data_path, args.mlp_train, args.mlp_val)

    # config

    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = config_class()

    # model

    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        model = model_class(config=config)

    # mlp_model = BertMLPNet(data_generator.n_token, data_generator.n_feature, args.emb_size, [1024, 2000, 1000, 500, 100], corruption=0.25)
    mlp_model = MasterNet(data_generator.n_token, data_generator.n_feature, args.emb_size, [1024, 2000, 1000, 500, 100], corruption=0.25)
    mlp_model.mlp_net.load_state_dict(torch.load(args.mlp_checkpoint))
    print("Loaded MLP model from {}".format(args.mlp_checkpoint))

    mlp_model.bert = model.bert
    mlp_model.to(args.device)

    mlp_model = torch.nn.DataParallel(mlp_model)
    mlp_model.load_state_dict(torch.load(args.continue_training_path + "model.bin"))
    print("Loaded BertMLP model from {}".format(args.continue_training_path))

    model = mlp_model.module.bert
    model = torch.nn.DataParallel(model)


    # TODO min_num_words = 3, should we use this?
    # tokens_file = "/home/layer6/recsys/unsupervised_data/xlmr_trainvalsubmit_only_tokens.p"  # don't compute them if not in train/val/submit set
    # output_file = "/home/layer6/recsys/embeddings/xlmr/"
    # model_checkpoint = "/home/layer6/recsys/xlm-r/checkpoints/run/checkpoint-1"
    # tokens_file = "/home/kevin/Projects/xlm-r/data/xlmr_all_tweet_tokens_leaderboard.p"  # don't compute them if not in train/val/submit set
    # output_dir = "/media/kevin/datahdd/data/embeddings"
    # model_checkpoint = "/home/kevin/Projects/xlm-r/out/checkpoint-500"
    batch_size = 512
    output_dir = args.continue_training_path
    tokens_file = args.tweet_id_to_text_file


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
        mean_emb = (masked_h.sum(dim=1) / L)
        return mean_emb

    t1 = time()
    dataset = TwitterDataset(file_path=tokens_file)
    t2 = time()
    print("time to create the dataset: {:.4f}".format(t2-t1))

    t1 = time()
    sampler = SortedSampler(dataset)
    t2 = time()
    print("time to create the sampler: {:.4f}".format(t2-t1))

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        sampler=sampler,
                        num_workers=0,
                        collate_fn=collate,
                        drop_last=False,
                        pin_memory=False)

    #model.to(device)
    #model = torch.nn.DataParallel(model)
    #model.eval()

    cls_dict = {}
    max_dict = {}
    mean_dict = {}
    chunk_segment_samples_seen = 0
    segment_counter = 0
    processed_ids = set()

    for bi, batch in tqdm(enumerate(loader)):

        inputs, masks, tweet_ids, lens = batch

        inputs, masks = [x.long().to(device) for x in [inputs, masks]]

        with torch.no_grad():
            # torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
            (h_sequence, h_cls) = model(input_ids=inputs, attention_mask=masks)
            h_sequence = h_sequence.detach().cpu()

            # max_emb = torch.max(h_sequence[:, 1:, :], dim=1)[0]
            mean_emb = mean_emb_no_pad(h_sequence, lens)
            # cls_emb = h_sequence[:, 0, :]

            # max_emb, mean_emb, cls_emb = [x.numpy() for x in [max_emb, mean_emb, cls_emb]]
            mean_emb = mean_emb.numpy()

        processed_ids.update(set(tweet_ids))
        # max_dict.update(zip(tweet_ids, max_emb))
        mean_dict.update(zip(tweet_ids, mean_emb))
        # cls_dict.update(zip(tweet_ids, cls_emb))

        chunk_segment_samples_seen += len(tweet_ids)

        if (bi % 100 == 0):
            print(bi)

        if chunk_segment_samples_seen > 100000:  # you probably can't fit more than 2.5M samples in RAM
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
    print("# processed IDs: {}".format(len(processed_ids)))
    print("# dataset: {}".format(len(dataset)))
    assert len(processed_ids) == len(dataset)
