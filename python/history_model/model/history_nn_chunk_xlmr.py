# import transformers
# from transformers import BertTokenizer, BertConfig
import glob
# import sys
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm, trange
# from transformers import BertTokenizer, BertConfig, BertModel
import pickle
import joblib
import argparse
from sklearn.metrics import precision_recall_curve, auc, log_loss
import os
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from pandas.core.common import flatten
import pandas as pd
from datetime import datetime
import gc

def compute_prauc(pred, gt):
  prec, recall, _ = precision_recall_curve(gt, pred)
  prauc = auc(recall, prec)
  return prauc

def calculate_ctr(gt):
  positive = len([x for x in gt if x == 1])
  ctr = positive/float(len(gt))
  return ctr

def compute_rce(pred, gt):
    cross_entropy = log_loss(gt, pred)
    data_ctr = calculate_ctr(gt)
    strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
    return (1.0 - cross_entropy/strawman_cross_entropy) * 100.0

id_map_csv = pd.read_csv('/data/recsys2020/history_nn/TweetIDMap.csv', header=None)
id_map = dict(zip(id_map_csv.iloc[:, 1], id_map_csv.iloc[:, 0]))

del id_map_csv

class TwitterDataset(Dataset):
    def __init__(self, emb_dir, emb_id_dir, feature_dir):
        with open(emb_dir, "rb") as f:
            print("Emb Dir: ", emb_dir)
            print("Loading Emb Dir: ", datetime.now().time())
            self.embedding_matrix = joblib.load(f)
            print("Done Loading Emb Dir: ", datetime.now().time())

        with open(emb_id_dir, "rb") as f:
            print("Emb ID Dir: ", emb_id_dir)
            print("Loading Emb ID Dir: ", datetime.now().time())
            emb_ids = joblib.load(f)
            emb_dict = {k: i for i, k in enumerate(emb_ids)}
            self.embedding_dict = emb_dict
            print("Done Loading Emb ID Dir: ", datetime.now().time())

        with open(feature_dir, "rb") as f:
            print("Feature Dir: ", feature_dir)
            print("Loading Feature Dict: ", datetime.now().time())
            feature_dict = joblib.load(f)
            print("Done Loading Feature Dict: ", datetime.now().time())

        self.features = feature_dict['features']
        self.labels = feature_dict['labels']
        self.tweet_ids = feature_dict['tweet_ids']
        self.tweet_history = feature_dict['engagement_histories']

        self.id_map = id_map

        del feature_dict
        del emb_ids

    def __getitem__(self, index):

        feature = self.features[index]
        labels = self.labels[index]

        tweets = self.tweet_ids[index][0]
        tweet_histories = self.tweet_history[index]

        history_feats = np.zeros((40, 1024))
        history_masks = np.ones(40)

        for i, e in enumerate(tweet_histories):

            # put one to be none padding to prevent attention blow up
            if i % 10 == 0:
                history_masks[i] = 0

            if e != -1:
                id = self.id_map[e]

                history_feats[i] = self.embedding_matrix[self.embedding_dict[id]]
                history_masks[i] = 0

        tweet_feat = self.embedding_matrix[self.embedding_dict[tweets]]

        return torch.FloatTensor(feature),\
               torch.FloatTensor(tweet_feat),\
               torch.FloatTensor(history_feats),\
               torch.BoolTensor(history_masks),\
               torch.FloatTensor(labels)

    def __len__(self):
        return len(self.features)

class TwitterDatasetValid(Dataset):
    def __init__(self, emb_dir, emb_id_dir, feature_dir):
        with open(emb_dir, "rb") as f:
            print("Emb Dir: ", emb_dir)
            print("Loading Emb Dir: ", datetime.now().time())
            self.embedding_matrix = joblib.load(f)
            print("Done Loading Emb Dir: ", datetime.now().time())

        with open(emb_id_dir, "rb") as f:
            print("Emb ID Dir: ", emb_id_dir)
            print("Loading Emb ID Dir: ", datetime.now().time())
            emb_ids = joblib.load(f)
            emb_dict = {k: i for i, k in enumerate(emb_ids)}
            self.embedding_dict = emb_dict
            print("Done Loading Emb ID Dir: ", datetime.now().time())

        with open(feature_dir, "rb") as f:
            print("Feature Dir: ", feature_dir)
            print("Loading Feature Dict: ", datetime.now().time())
            feature_dict = joblib.load(f)
            print("Done Loading Feature Dict: ", datetime.now().time())

        self.features = feature_dict['features']
        self.labels = feature_dict['labels']
        self.tweet_ids = feature_dict['tweet_ids']
        self.tweet_history = feature_dict['engagement_histories']

        self.user_ids = feature_dict['lb_user_ids']

        self.id_map = id_map

        del feature_dict
        del emb_ids

    def __getitem__(self, index):

        feature = self.features[index]
        labels = self.labels[index]

        tweets = self.tweet_ids[index][0]
        tweet_histories = self.tweet_history[index]

        user_id = self.user_ids[index][0]

        history_feats = np.zeros((40, 1024))
        history_masks = np.ones(40)

        for i, e in enumerate(tweet_histories):

            # put one to be none padding to prevent attention blow up
            if i % 10 == 0:
                history_masks[i] = 0

            if e != -1:
                id = self.id_map[e]

                history_feats[i] = self.embedding_matrix[self.embedding_dict[id]]
                history_masks[i] = 0

        tweet_feat = self.embedding_matrix[self.embedding_dict[tweets]]

        return torch.FloatTensor(feature),\
               torch.FloatTensor(tweet_feat),\
               torch.FloatTensor(history_feats),\
               torch.BoolTensor(history_masks),\
               torch.FloatTensor(labels), \
               tweets,\
               user_id

    def __len__(self):
        return len(self.features)

class TwitterDatasetSubmit(Dataset):
    def __init__(self, emb_dir, emb_id_dir, feature_dir):
        with open(emb_dir, "rb") as f:
            print("Emb Dir: ", emb_dir)
            print("Loading Emb Dir: ", datetime.now().time())
            self.embedding_matrix = joblib.load(f)
            print("Done Loading Emb Dir: ", datetime.now().time())

        with open(emb_id_dir, "rb") as f:
            print("Emb ID Dir: ", emb_id_dir)
            print("Loading Emb ID Dir: ", datetime.now().time())
            emb_ids = joblib.load(f)
            emb_dict = {k: i for i, k in enumerate(emb_ids)}
            self.embedding_dict = emb_dict
            print("Done Loading Emb ID Dir: ", datetime.now().time())

        with open(feature_dir, "rb") as f:
            print("Feature Dir: ", feature_dir)
            print("Loading Feature Dict: ", datetime.now().time())
            feature_dict = joblib.load(f)
            print("Done Loading Feature Dict: ", datetime.now().time())

        self.features = feature_dict['features']
        self.tweet_ids = feature_dict['tweet_ids']
        self.tweet_history = feature_dict['engagement_histories']
        self.user_ids = feature_dict['lb_user_ids']
        self.id_map = id_map

        del feature_dict
        del emb_ids

    def __getitem__(self, index):

        feature = self.features[index]

        tweets = self.tweet_ids[index][0]
        tweet_histories = self.tweet_history[index]

        history_feats = np.zeros((40, 1024))
        history_masks = np.ones(40)

        user_id = self.user_ids[index][0]

        for i, e in enumerate(tweet_histories):

            # put one to be none padding to prevent attention blow up
            if i % 10 == 0:
                history_masks[i] = 0

            if e != -1:
                id = self.id_map[e]

                history_feats[i] = self.embedding_matrix[self.embedding_dict[id]]
                history_masks[i] = 0

        tweet_feat = self.embedding_matrix[self.embedding_dict[tweets]]

        return torch.FloatTensor(feature),\
               torch.FloatTensor(tweet_feat),\
               torch.FloatTensor(history_feats),\
               torch.BoolTensor(history_masks),\
               tweets,\
               user_id
               # torch.FloatTensor(tokens),\

    def __len__(self):
        return len(self.features)


class SeqAttnMatch(nn.Module):
    def __init__(self, args):
        super(SeqAttnMatch, self).__init__()

        self.linear_q = nn.Linear(args.emb_dim, args.hidden_dim)
        self.linear_k = nn.Linear(args.emb_dim, args.hidden_dim)
        self.linear_v = nn.Linear(args.emb_dim, args.hidden_dim)

        self.emb_dropout_layer = nn.Dropout(p=args.emb_dropout)
        self.dropout_layer = nn.Dropout(p=args.dropout)

        self.ff1 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.ff2 = nn.Linear(args.hidden_dim, args.hidden_dim)

    def forward(self, tweet, history, history_mask):
        """
        Args:
            tweet: batch * hdim
            history: batch * len2 * hdim
            history_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * hdim
        """
        # Project vectors

        tweet = self.emb_dropout_layer(tweet)
        tweet_project = self.linear_k(tweet)

        history = self.emb_dropout_layer(history)
        history_project = self.linear_q(history)
        history_values = self.linear_v(history)

        # Compute scores
        scores = tweet_project.unsqueeze(1).bmm(history_project.transpose(2, 1))

        # Mask padding
        history_mask = history_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(history_mask.data, -float('inf'))

        # Normalize with softmax
        alpha = torch.softmax(scores, dim=2)

        # Take weighted average
        matched_seq = alpha.bmm(history_values).squeeze()

        out = self.ff2(self.dropout_layer(torch.relu(self.ff1(matched_seq))))

        return out

class SeqAttnMatchV1(nn.Module):
    def __init__(self, args):
        super(SeqAttnMatchV1, self).__init__()

        self.linear = nn.Linear(args.emb_dim, args.emb_dim)

        self.emb_dropout_layer = nn.Dropout(p=args.emb_dropout)

    def forward(self, tweet, history, history_mask):
        """
        Args:
            tweet: batch * hdim
            history: batch * len2 * hdim
            history_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * hdim
        """
        # Project vectors

        tweet = self.emb_dropout_layer(tweet)
        tweet_project = torch.relu(self.linear(tweet))

        history = self.emb_dropout_layer(history)
        history_project = torch.relu(self.linear(history))

        # Compute scores
        scores = tweet_project.unsqueeze(1).bmm(history_project.transpose(2, 1))

        # Mask padding
        history_mask = history_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(history_mask.data, -float('inf'))

        # Normalize with softmax
        alpha = torch.softmax(scores, dim=2)

        # Take weighted average
        matched_seq = alpha.bmm(history_project).squeeze()

        return matched_seq

class FFNetHistory(nn.Module):

    def __init__(self, args):
        super(FFNetHistory, self).__init__()
        self.args = args

        input_dim = args.hidden_dim * 6

        self.feature_layer = nn.Linear(args.feature_dim, args.hidden_dim)
        self.embedding_layer = nn.Linear(args.emb_dim, args.hidden_dim)

        self.attns = nn.ModuleList()
        self.attn_fcs = nn.ModuleList()

        for k in range(4):
            self.attns.append(SeqAttnMatch(args))
            self.attn_fcs.append(nn.Linear(args.hidden_dim, args.hidden_dim))

        layers = [input_dim, 5000, 3000, 2000, 1000, 500, 4]

        self.fc = nn.ModuleList()

        for k in range(len(layers) - 1):
            self.fc.append(nn.Linear(layers[k], layers[k + 1]))

        self.dropout_layer = nn.Dropout(p=args.dropout)
        self.emb_dropout_layer = nn.Dropout(p=args.emb_dropout)

    def forward(self, feature, tweet_feat, history_feats, history_masks):

        feature = torch.relu(self.feature_layer(feature))
        tweet_feat_out = torch.relu(self.embedding_layer(self.emb_dropout_layer(tweet_feat)))

        history_feats_list = torch.split(history_feats, 10, dim=1)
        history_masks_list = torch.split(history_masks, 10, dim=1)

        history_outputs = []

        for i, attn in enumerate(self.attns):
            out = attn(tweet_feat, history_feats_list[i], history_masks_list[i])
            out = torch.relu(self.attn_fcs[i](self.dropout_layer(out)))
            history_outputs.append(out)

        history = torch.cat(history_outputs, 1)

        output = torch.cat((feature, tweet_feat_out, history), dim=1)

        for layer in self.fc[:-1]:
            output = self.dropout_layer(output)
            output = layer(output)
            output = torch.relu(output)

        output = self.fc[-1](output)

        return output

def train(args, train_chunk_dirs, model, embs_dirs, emb_id_dirs,
          valid_chunk_dir=None, valid_emb_dir=None, valid_emb_dict_dir=None):

    tb_writer = SummaryWriter()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

    if args.fp16:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    epochs_trained = 0
    tr_loss, logging_loss, printing_loss = np.zeros(4), np.zeros(4), np.zeros(4)

    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch")

    loss = nn.BCEWithLogitsLoss(reduction='none')

    # results = {}

    for _ in train_iterator:

        for train_chunk_dir, emb_dir, emb_id_dir in zip(train_chunk_dirs, embs_dirs, emb_id_dirs):

            train_dataset = TwitterDataset(emb_dir, emb_id_dir, train_chunk_dir)

            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(
                train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                num_workers=10, pin_memory=True
            )

            epoch_iterator = tqdm(train_dataloader, desc="Iteration")

            for step, batch in enumerate(epoch_iterator):

                model.train()

                batch = tuple(t.to(args.device) for t in batch)
                output = model(batch[0], batch[1], batch[2], batch[3])

                labels = batch[4]

                train_loss = loss(output, labels).mean(0)

                tr_loss += train_loss.detach().cpu().numpy()

                train_loss = train_loss.mean()

                # if args.n_gpu > 1:
                #     train_loss = train_loss.mean()
                # if args.gradient_accumulation_steps > 1:
                #     loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(train_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    train_loss.backward()

                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                model.zero_grad()
                global_step += 1

                if global_step % args.loss_steps == 0:
                    tmp_loss = (tr_loss - printing_loss) / args.loss_steps
                    printing_loss = np.copy(tr_loss)
                    for i, action in enumerate(["reply", "retweet", "comment", "like"]):
                        tb_writer.add_scalar("train_losses_{}".format(action), tmp_loss[i], global_step)

                # if global_step % args.logging_steps == 0:
                #     results = evaluate(args, valid_dataset, model)
                #
                #     for key, value in results.items():
                #         tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                if global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    torch.save({
                        'step': global_step,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        # 'results': results,
                    }, output_dir)

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break

            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break

    return global_step, tr_loss / global_step

def evaluate(args, valid_dataset, model):

    eval_sampler = SequentialSampler(valid_dataset)
    eval_dataloader = DataLoader(
        valid_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
        num_workers=10, pin_memory=True
    )
    nb_eval_steps = 0

    gt = []
    p = []

    tids = []
    uids = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        b = tuple(t.to(args.device) for t in batch[:4])

        with torch.no_grad():
            output = model(b[0], b[1], b[2], b[3])
            labels = batch[4]

            tid = batch[5]
            uid = batch[6]

            gt.append(labels)
            predict = torch.sigmoid(output)
            p.append(predict.detach().cpu().numpy())

            tids.append(tid)
            uids.append(uid)

        nb_eval_steps += 1

    # gt = np.concatenate(gt, axis=0)
    # p = np.float64(np.concatenate(p, axis=0))
    gt = np.vstack(gt)
    p = np.float64(np.vstack(p))

    results = {}

    tids = list(flatten(tids))
    uids = list(flatten(uids))

    for i, action in enumerate(["reply", "retweet", "comment", "like"]):
        gti = gt[:, i]
        pi = p[:, i]

        results["prauc_{}".format(action)] = compute_prauc(pi, gti)
        results["rce_{}".format(action)] = compute_rce(pi, gti)

        tmp_df = pd.DataFrame({'tid':tids, 'uid':uids, 'pred':list(pi)})
        # file_name = os.path.join(args.blend_dir, action + str(args.step) + "valid" + ".csv")
        file_name = os.path.join(args.blend_dir + "/valid/", action + ".csv")
        tmp_df.to_csv(file_name, index=False, header=False)

    return results

def submit(args, submit_dataset, model, submit=True):

    submit_sampler = SequentialSampler(submit_dataset)
    submit_dataloader = DataLoader(
        submit_dataset, sampler=submit_sampler, batch_size=args.eval_batch_size,
        num_workers=10, pin_memory=True
    )
    nb_eval_steps = 0

    preds = []
    tids = []
    uids = []

    for batch in tqdm(submit_dataloader, desc="Submitting"):
        model.eval()
        b = tuple(t.to(args.device) for t in batch[:4])

        tid = batch[4]
        uid = batch[5]

        with torch.no_grad():
            output = model(b[0], b[1], b[2], b[3])
            predict = torch.sigmoid(output).detach().cpu().numpy()

            preds.append(predict)
            tids.append(tid)
            uids.append(uid)

        nb_eval_steps += 1

    all_preds = np.concatenate(preds, 0)
    tids = list(flatten(tids))
    uids = list(flatten(uids))

    for i, action in enumerate(["reply", "retweet", "comment", "like"]):
        tmp_pred = list(all_preds[:, i])
        tmp_df = pd.DataFrame({'tid':tids, 'uid':uids, 'pred': tmp_pred})
        if submit:
            # file_name = os.path.join(args.submit_dir, action + str(args.step) +"submit" + ".csv")
            file_name = os.path.join(args.submit_dir + "/submit/", action + ".csv")
        else:
            # file_name = os.path.join(args.submit_dir, action + str(args.step) +"test" + ".csv")
            file_name = os.path.join(args.submit_dir + "/test/", action + ".csv")
        tmp_df.to_csv(file_name, index=False, header=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_feature_dict_dir",
                        default="/data/recsys2020/history_nn/TrainChunk*",
                        type=str)
    parser.add_argument("--valid_feature_dict_dir",
                        default="/data/recsys2020/history_nn/Valid.sav",
                        type=str)
    parser.add_argument("--submit_feature_dict_dir",
                        default="/data/recsys2020/history_nn/Submit.sav",
                        type=str)
    parser.add_argument("--test_feature_dict_dir",
                        default="/data/recsys2020/history_nn/Test.sav",
                        type=str)
    parser.add_argument("--output_dir",
                        default="/data/recsys2020/output/",
                        type=str)
    parser.add_argument("--emb_dir",
                        default="/data/recsys2020/history_nn/TrainEmb*.sav",
                        type=str)
    parser.add_argument("--emb_id_dir",
                        default="/data/recsys2020/history_nn/TrainEmbID*",
                        type=str)
    parser.add_argument("--valid_emb_dir",
                        default="/data/recsys2020/history_nn/ValidEmb.sav",
                        type=str)
    parser.add_argument("--valid_emb_id_dir",
                        default="/data/recsys2020/history_nn/ValidEmbID",
                        type=str)
    parser.add_argument("--submit_emb_dir",
                        default="/data/recsys2020/history_nn/SubmitEmb.sav",
                        type=str)
    parser.add_argument("--submit_emb_id_dir",
                        default="/data/recsys2020/history_nn/SubmitEmbID",
                        type=str)
    parser.add_argument("--test_emb_dir",
                        default="/data/recsys2020/history_nn/TestEmb.sav",
                        type=str)
    parser.add_argument("--test_emb_id_dir",
                        default="/data/recsys2020/history_nn/TestEmbID",
                        type=str)
    parser.add_argument("--model_dir",
                        default="/data/recsys2020/output/",
                        type=str)
    parser.add_argument("--submit_dir",
                        default="/data/recsys2020/DL_Ouputs/",
                        type=str)
    parser.add_argument("--blend_dir",
                        default="/data/recsys2020/DL_Ouputs/",
                        type=str)

    parser.add_argument("--train_batch_size", default=4000, type=int)
    parser.add_argument("--eval_batch_size", default=4000, type=int)
    parser.add_argument("--logging_steps", default=10000, type=int)
    parser.add_argument("--save_steps", default=5000, type=int)
    parser.add_argument("--loss_steps", default=1000, type=int)
    parser.add_argument("--max_steps", default=0, type=int)

    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=200.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--emb_dim", default=1024, type=int)
    parser.add_argument("--hidden_dim", default=512, type=int)
    parser.add_argument("--seq_length", default=10, type=int)
    parser.add_argument("--feature_dim", default=170, type=int)
    parser.add_argument("--token_dim", default=297, type=int)

    parser.add_argument("--num_outputs", default=4, type=int)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--dropout", default=0.35, type=float)
    parser.add_argument("--emb_dropout", default=0.5, type=float)

    parser.add_argument("--train", default=True, type=bool)
    parser.add_argument("--evaluate", default=True, type=bool)
    parser.add_argument("--submit", default=True, type=bool)
    parser.add_argument("--test", default=True, type=bool)

    parser.add_argument(
        "--fp16",
        default=True,
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )

    args = parser.parse_args()
    print(args)

    device = torch.device("cuda")
    args.n_gpu = torch.cuda.device_count()

    train_chunks_dirs = list(sorted(glob.glob(args.train_feature_dict_dir)))
    emb_id_dirs = list(sorted(glob.glob(args.emb_id_dir)))
    embs_dirs = list(sorted(glob.glob(args.emb_dir)))

    model = FFNetHistory(args)
    model.to(args.device)

    already_data_paralleled = False

    if args.train:
        print("Training")

        train(args, train_chunks_dirs, model, embs_dirs, emb_id_dirs)

    if args.evaluate:
        print("Evaluating")

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1 and not already_data_paralleled:
            model = torch.nn.DataParallel(model)
            already_data_paralleled = True

        model_file_dirs = sorted(glob.glob(os.path.join(args.model_dir, "checkpoint*")))
        valid_dataset = TwitterDatasetValid(args.valid_emb_dir, args.valid_emb_id_dir, args.valid_feature_dict_dir)
        for file in model_file_dirs:

            print(file)

            model_dict = torch.load(file)
            model.load_state_dict(model_dict['state_dict'])
            step = model_dict['step']

            args.step = step

            results = evaluate(args, valid_dataset, model)

            print(step)
            print(results)

            with open(os.path.join(args.model_dir, "results.txt"), "a") as myfile:
                myfile.write("Step: " + str(step) + "\n")
                myfile.write("Results: " + str(results) + "\n")

    if args.submit:
        print("Submiting")

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1 and not already_data_paralleled:
            model = torch.nn.DataParallel(model)
            already_data_paralleled = True

        model_file_dirs = sorted(glob.glob(os.path.join(args.model_dir, "checkpoint*")))
        print(model_file_dirs)
        submit_dataset = TwitterDatasetSubmit(args.submit_emb_dir, args.submit_emb_id_dir, args.submit_feature_dict_dir)
        for file in model_file_dirs:

            print(file)

            model_dict = torch.load(file)
            model.load_state_dict(model_dict['state_dict'])
            step = model_dict['step']

            args.step = step
            print(step)

            submit(args, submit_dataset, model, submit=True)

    if args.test:
        print("Testing")

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1 and not already_data_paralleled:
            model = torch.nn.DataParallel(model)
            already_data_paralleled = True

        model_file_dirs = sorted(glob.glob(os.path.join(args.model_dir, "checkpoint*")))
        print(model_file_dirs)
        submit_dataset = TwitterDatasetSubmit(args.test_emb_dir, args.test_emb_id_dir, args.test_feature_dict_dir)
        for file in model_file_dirs:

            print(file)

            model_dict = torch.load(file)
            model.load_state_dict(model_dict['state_dict'])
            step = model_dict['step']

            args.step = step
            print(step)

            submit(args, submit_dataset, model, submit=False)
