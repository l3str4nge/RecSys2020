import numpy as np
import argparse
import time
import gc
import torch
import torch.nn as nn
import os

from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter

from utils.argcheck import *
from utils.eval import compute_prauc, compute_rce
from utils.progress import WorkSplitter, inhour
from utils.data_feature_embedding_multiple import Data

from model.EmbeddingNet import EmbeddingNet



def main(args):
    writer = SummaryWriter(log_dir=os.path.join('./logs', args.run_name))

    if args.emb_type == 'bert':
        emb_size = 768
    elif args.emb_type == 'xlmr':
        emb_size = 1024

    # Progress bar
    progress = WorkSplitter()
    if not os.path.exists("./checkpoint"):
        os.mkdir("./checkpoint")

    # Load Data
    progress.section("Load Data")
    start_time = time.time()
    print("Embedding size is set to", emb_size)
    data = Data(args, args.path, args.emb_path, args.train, args.valid, emb_size)
    print("Elapsed: {0}".format(inhour(time.time() - start_time)))

    # build model
    progress.section("Build Model")
    model = EmbeddingNet(data.n_token, data.n_feature, emb_size, [1024, 2000, 1000, 500, 100],
                         corruption=args.corruption)
    model.cuda()
    # model.load_state_dict(torch.load("./checkpoint/featurenet_v12_8.ckpt"))
    print(model)

    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.lamb)
    valid_loader = data.instance_a_valid_loader(args.batch)
    # train_loader = data.instance_a_train_loader(args.batch)

    global_step = 0
    progress.section("Train model")
    scores = []

    scores = validate(0, valid_loader, model, writer, scores, args)

    for epoch in range(1, args.epoch):

        total_loss = 0
        epoch_step = 0
        model.train()

        for split_i in range(args.num_splits):

            train_loader = data.instance_a_train_loader(args.batch)

            global_step, total_loss, epoch_step = train(epoch_step, global_step, train_loader, model, optim, criterion, total_loss, writer)

            del train_loader
            gc.collect()

        print("epoch{0} loss:{1:.4f}".format(epoch, total_loss))

        if epoch % 1 == 0:

            scores = validate(epoch, valid_loader, model, writer, scores, args)

    print("Elapsed: {0}".format(inhour(time.time() - start_time)))


def train(epoch_step, global_step, train_loader, model, optim, criterion, total_loss, writer):

    epoch_iterator = tqdm(train_loader, desc="Iteration")
    for _, batch in enumerate(epoch_iterator):
        token, feature, label, embedding = batch[0].float().cuda(), batch[1].float().cuda(), batch[
            2].float().cuda(), batch[3].float().cuda()  # , batch[3].cuda()
        optim.zero_grad()
        logit = model(token, feature, embedding)
        loss = criterion(logit, label)
        loss.backward()
        optim.step()
        total_loss += loss.item()

        if (global_step % 5000 == 0):
            avg_loss = total_loss / (epoch_step + 1)
            print("Batch: {}, training loss: {:.4f}".format(global_step, avg_loss))
            writer.add_scalar('Loss/train_running_avg', total_loss / (epoch_step + 1), global_step)
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)

        global_step += 1
        epoch_step += 1

    del epoch_iterator
    gc.collect()

    return global_step, total_loss, epoch_step


def validate(epoch, valid_loader, model, writer, scores, args):

    model.eval()

    with torch.no_grad():
        preds, labels = [], []
        valid_iterator = tqdm(valid_loader, desc="Validation")
        for _, batch in enumerate(valid_iterator):
            token, feature, label, embedding = batch[0].float().cuda(), batch[1].float().cuda(), batch[2], \
                                               batch[3].float().cuda()  # ,batch[3].cuda()
            pred = torch.sigmoid(model(token, feature, embedding)).detach().cpu().numpy()
            labels.append(label)
            preds.append(pred)

    labels = np.vstack(labels)
    preds = np.float64(np.vstack(preds))

    auprs = []
    for i, engage in enumerate(["reply", "retweet", "comment", "like"]):
        _prauc = compute_prauc(preds[:, i], labels[:, i])
        _rce = compute_rce(preds[:, i], labels[:, i])

        print("Engagement: {}, AUPR: {:.4f}, RCE: {:.4f}".format(engage, _prauc, _rce))
        auprs.append(_prauc)

        writer.add_scalar('PRAUC/{}_val'.format(engage), _prauc, epoch)
        writer.add_scalar('RCE/{}_val'.format(engage), _rce, epoch)

    m_aupr = np.mean(np.array(auprs))
    scores.append(m_aupr)
    best = np.max(np.array(scores))
    print("Mean AUPR: {:.4f} (Best: {:.4f})".format(m_aupr, best))
    if m_aupr == best:
        print("New best score!")

    torch.save(model.state_dict(), "{}featurenet_{}_".format(args.run_name, args.exp_name) + str(epoch) + ".ckpt")

    model.train()

    return scores


if __name__ == "__main__":
    
    # Commandline arguments
    
    parser = argparse.ArgumentParser(description="P")
    
    parser.add_argument('-e', dest='epoch', type=check_int_positive, default=1000)
    parser.add_argument('-b', dest='batch', type=check_int_positive, default=4096)
    parser.add_argument('-lr', dest='lr', type=check_float_positive, default=1e-4)
    parser.add_argument('-l', dest='lamb', type=check_float_positive, default=0)
    parser.add_argument('-c', dest='corruption', type=check_float_positive, default=0.25)
    
    # feeatures
    parser.add_argument('-d', dest='path', default="/data/recsys2020/DL_Data/")
    parser.add_argument('--num_splits', type=int, default=3)
    parser.add_argument('-tr', dest='train', default='Train')
    parser.add_argument('-v', dest='valid', default='Valid.sav')
    
    # embeddings
    parser.add_argument('-ed', dest='emb_path', default="/data/recsys2020/DL_Data/embeddings/bert/supervised/")
    parser.add_argument('--emb_file', type=str, default='train_emb')
    parser.add_argument('--emb_type', type=check_emb_type, required=True)
  
    # run
    parser.add_argument('--run_name', type=str, default="/data/recsys2020/DL_Checkpoints/supervised_bert_model/")
    parser.add_argument('--exp_name', type=str, default='supervised')
   
    args = parser.parse_args()

    main(args)
