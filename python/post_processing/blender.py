import os
import argparse
from os.path import join, isfile
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader, Dataset
from utils.eval import compute_prauc, compute_rce
from torch.utils.tensorboard import SummaryWriter
from model.BlenderNet import *

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


TASKS = ["reply", "retweet", "comment", "like"]


def inverse_sigmoid(x):
    return np.log(x/(1-x))


class SimpleDataset(Dataset):
    def __init__(self, X, y): #, chunk):
        self.X = X
        self.y = y

    def __getitem__(self, index):

        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


class SubmitDataset(Dataset):
    def __init__(self, X, tid, uid): #, chunk):
        self.X = X
        self.tid = tid
        self.uid = uid

    def __getitem__(self, index):

        return self.X[index], self.tid[index], self.uid[index]

    def __len__(self):
        return len(self.X)


# load val csvs
def load_submission_files(args, run_val_folders, submit=False):
    
    cache_file = join(args.submit_folder, "dataset_cache.p") if submit else join(args.validation_folder, "dataset_cache.p")
    print(cache_file)
    if isfile(cache_file):
        print("WARNING: using cached validation predictions, please delete the cache file if the predictions have been updated ... ")
        d = joblib.load(open(cache_file, "rb"))
        inputs, tweet_ids, user_ids = d['inputs'], d['tweet_ids'], d['user_ids']
    else:
        dataset = None
        for val_folder in tqdm(run_val_folders):
            for T in tqdm(TASKS):

                field_name = "{}_prob_{}".format(val_folder.split("/")[-1], T)
                print(field_name)
                dtypes = {field_name:np.float32}

                df = pd.read_csv(join(val_folder, "{}.csv".format(T)),
                names=["tweet_id", "user_id", field_name], dtype=dtypes)

                dataset = df if dataset is None else dataset.merge(df, on=['tweet_id', 'user_id'])

        cols = list(dataset.columns)
        prob_cols = cols[2:]

        inputs = dataset[prob_cols].to_numpy()
        tweet_ids = dataset['tweet_id'].to_numpy()
        user_ids = dataset['user_id'].to_numpy()

        joblib.dump({
            "inputs": inputs,
            "tweet_ids": tweet_ids,
            "user_ids": user_ids
        }, open(cache_file, "wb"))

        print("{} tweet ids...".format(len(tweet_ids)))

    return inputs, tweet_ids, user_ids


def load_labels(args, row_lookup):

    cache_file = join(args.validation_folder, "labels_cache.p")

    if isfile(cache_file):
        print("WARNING: using cached labels, please delete the cache file if the data has been updated ... ")
        d = joblib.load(open(cache_file, "rb"))
        labels = d['labels']
    else:

        val_dict = joblib.load(open(args.label_file, "rb"))

        labels = np.zeros([len(row_lookup), 4], dtype=np.float32)
        assigned_rows = set()

        for tid, uid, lab in tqdm(zip(val_dict['tweet_ids'], val_dict['lb_user_ids'][:, 0], val_dict['labels'].astype(np.float32))):
            row = row_lookup[(tid, uid)]

            labels[row, :] = lab
            assigned_rows.add(row)


        assert len(assigned_rows) == len(labels)

        joblib.dump({
            "labels": labels,
        }, open(cache_file, "wb"))


    print("{} labels ...".format(len(labels)))
    return labels


def evaluate(args, model, writer, eval_loader, epoch, train=False):
    model.eval()
    with torch.no_grad():
        preds, labels = [], []
        for _, batch in enumerate(tqdm(eval_loader)):        
            X, y = batch
            # X = X.cuda()
            
            pred = torch.sigmoid(model(X)).detach().cpu().numpy()
            labels.append(y.numpy())
            preds.append(pred)

    labels = np.vstack(labels)
    preds = np.float64(np.vstack(preds))

    for i, engage in enumerate(TASKS):
        _prauc = compute_prauc(preds[:,i],labels[:,i])
        _rce = compute_rce(preds[:,i],labels[:,i])

        print(engage+":")
        print(_prauc)
        print(_rce)

        suffix = 'train' if train else 'val'
        writer.add_scalar('PRAUC_{}/{}'.format(suffix, engage), _prauc, epoch)
        writer.add_scalar('RCE_{}/{}'.format(suffix, engage), _rce, epoch)
    
    if not os.path.exists("./checkpoint/{}".format(args.run_name)):
        os.makedirs("./checkpoint/{}".format(args.run_name))

    torch.save(model.state_dict(), "./checkpoint/{}/epoch{}.ckpt".format(args.run_name, epoch))

# training
def train(args):

    NUM_BLENDS = len(args.csv_subfolders.split(","))
    writer = SummaryWriter(log_dir=join('./logs_blender', args.run_name))

    # load val predictions
    run_val_folders = [join(args.validation_folder, x) for x in args.csv_subfolders.split(",")]
    inputs, tweet_ids, user_ids = load_submission_files(args, run_val_folders)
    row_lookup = {(tid, uid): i for i, (tid, uid) in enumerate(zip(tweet_ids, user_ids))}

    # lets use logit inputs and logit outputs
    inputs = inverse_sigmoid(inputs)

    # load Valid.sav for labels
    labels = load_labels(args, row_lookup)

    # split train and eval
    NUM_SAMPLES = len(labels)
    NUM_EVAL = int(args.eval_percent*NUM_SAMPLES)

    all_indices = list(range(NUM_SAMPLES))
    random.shuffle(all_indices)

    train_indices = all_indices[NUM_EVAL:]
    eval_indices = all_indices[:NUM_EVAL]


    X_tr, y_tr = inputs[train_indices], labels[train_indices]
    X_ev, y_ev = inputs[eval_indices], labels[eval_indices]

    # construct dataset and dataloader
    train_dataset = SimpleDataset(X_tr, y_tr)
    eval_dataset = SimpleDataset(X_ev, y_ev)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers =20, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers =20, pin_memory=True)

    # construct model and optimizer

    #model = TaskIndependentNet(NUM_BLENDS)
    model = BlenderNetSmall(NUM_BLENDS)
    if (model.__class__.__name__ == "TaskIndependentNet"):
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
                if "weight" in name:
                    for j in range(param.data.shape[1]):

                        writer.add_scalar("{}_weight/{}".format(j, name), param.data[0, j], 0)
    
    # model.cuda()
    print(model)
    
    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr = args.lr)

    # training loop
    print('eval:')
    evaluate(args, model, writer, eval_loader, 0)
    print('train:')
    evaluate(args, model, writer, train_loader, 0, train=True)
    global_step = 0

    model.train()
    for epoch in range(1, args.epochs):

        model.train()
        for _, batch in enumerate(tqdm(train_loader)):
            X, y = batch
            # X, y = X.cuda(), y.cuda()
            optim.zero_grad()
            logit = model(X)
            loss = criterion(logit, y)
            loss.backward()
            optim.step()

            if global_step % 100 == 0:
                writer.add_scalar('train_loss', loss.item(), global_step)

            global_step += 1
        if epoch % 10 == 0:
            if (model.__class__.__name__ == "TaskIndependentNet"):
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        print(name, param.data)
                        if "weight" in name:
                            for j in range(param.data.shape[1]):
                                writer.add_scalar("{}_weight/{}".format(j, name), param.data[0, j], epoch+1) 
            print("epoch:",epoch)
            print("eval:")  
            evaluate(args, model, writer, eval_loader, epoch+1)
            print("train:")
            evaluate(args, model, writer, train_loader, epoch+1, train=True)


# submission
def submit(args):
    NUM_BLENDS = len(args.csv_subfolders.split(","))

    # load submit predictions
    run_submit_folders = [join(args.submit_folder, x) for x in args.csv_subfolders.split(",")]
    inputs, tweet_ids, user_ids = load_submission_files(args, run_submit_folders, submit=True)

    # lets use logit inputs and logit outputs
    inputs = inverse_sigmoid(inputs)

    # construct dataset and dataloader
    submit_dataset = SubmitDataset(inputs, tweet_ids, user_ids)

    submit_loader = DataLoader(submit_dataset, batch_size=args.batch_size, shuffle=False, num_workers =20, pin_memory=True)

    # construct model and optimizer

    #model = TaskIndependentNet(NUM_BLENDS)
    model = BlenderNetSmall(NUM_BLENDS)
    model.load_state_dict(torch.load(args.checkpoint))
    print("Loaded model from {}".format(args.checkpoint))
    # model.cuda()
    print(model)

    model.eval()

    lbs = {'user_lb': list(), 'tweet_lb': list()}
    preds = []

    with torch.no_grad():
        lb_iterator = tqdm(submit_loader, desc="lb")

        for _, batch in enumerate(lb_iterator):

            X, tweet_lb, user_lb = batch
            # X = X.cuda()
            logit = model(X)
            pred = torch.sigmoid(logit).detach().cpu().numpy()
            lbs['tweet_lb'] += tweet_lb
            lbs['user_lb'] += user_lb
            preds.append(pred)

        final_csv = pd.DataFrame(lbs)
        preds = np.float64(np.vstack(preds))

        out_dir = join(args.submit_folder, "blend")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        print("Generating CSVs...")
        for i, engage in enumerate(["reply", "retweet", "comment", "like"]):
            final_csv[engage] = preds[:,i]
            final_csv[['tweet_lb','user_lb',engage]].to_csv(join(args.submit_folder, "blend", engage+'.csv'),index=False, header=False)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="P")

    parser.add_argument('--submit_folder', type=str, default="../out/submit")
    parser.add_argument('--validation_folder', type=str, default="../out/val")
    parser.add_argument('--csv_subfolders', type=str, default="bert_e30,history340k,supervised21k,xlmr_final_new_epoch50,xlmr_scheduler_epoch27,featurenet,history470k,XGB3000,xlmr_highway_epoch52")
    parser.add_argument('--label_file', type=str, default="Valid.sav")
    parser.add_argument('--batch_size', type=int, default=16384)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval_percent', type=float, default=0.4)
    parser.add_argument('--run_name', type=str, default="blend")
    parser.add_argument('--checkpoint', type=str, default="./checkpoint/blend/epoch51.ckpt")
    parser.add_argument('--submit', action='store_ture')

    args = parser.parse_args()

    if args.submit:
        submit(args)
    else:
        train(args)
    
