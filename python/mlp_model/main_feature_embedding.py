import numpy as np
from utils.progress import WorkSplitter, inhour
from utils.data_feature_embedding import Data
import argparse
import time
from utils.argcheck import *
from utils.eval import compute_prauc, compute_rce

from model.EmbeddingNet import EmbeddingNet, EmbeddingHighWayNet
import gc
import torch
import torch.nn as nn
import os 

from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter


def main(args):
    
    writer = SummaryWriter(log_dir=os.path.join('./logs', args.run_name))

    # Progress bar
    progress = WorkSplitter()
    if not os.path.exists("./checkpoint"):
        os.mkdir("./checkpoint")
    
    if args.emb_type == 'bert':
        emb_size = 768
    elif args.emb_type == 'xlmr':
        emb_size = 1024

    # Load Data
    progress.section("Load Data")
    print("Embedding size is set to", emb_size)
    start_time = time.time()
    data = Data(args, args.path, args.train, args.valid,emb_size)
    print("Elapsed: {0}".format(inhour(time.time() - start_time)))
    
    #build model
    progress.section("Build Model")
    if args.network_architecture == 'embedding_net':
        model = EmbeddingNet(data.n_token, data.n_feature, emb_size, [1024, 2000, 1000, 500, 100],corruption=args.corruption)
    elif args.network_architecture == 'embedding_highway_net':
        model = EmbeddingHighWayNet(data.n_token, data.n_feature, emb_size, [1024, 2000, 1000, 500, 100])
    else:
        raise NotImplementedError('either use embedding_net or embedding_highway_net')
    model.cuda()
    print(model)
    
    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay=args.lamb)
    valid_loader = data.instance_a_valid_loader(args.batch)
    # train_loader = data.instance_a_train_loader(args.batch)

    global_step = 0
    progress.section("Train model")
    for epoch in range(1,args.epoch):

        total_loss = 0
        epoch_step = 0
        model.train()

        start_split = time.time()

        for split_i in range(args.num_splits):

            train_loader = data.instance_a_train_loader(args.batch)

            epoch_iterator = tqdm(train_loader, desc="Iteration")
            for _, batch in enumerate(epoch_iterator) :
                token, feature, label, embedding = batch[0].float().cuda(), batch[1].float().cuda(), batch[2].float().cuda(), batch[3].float().cuda()#, batch[3].cuda()
                optim.zero_grad()
                logit = model(token, feature, embedding)
                loss = criterion(logit, label)
                loss.backward()
                optim.step()
                total_loss += loss.item()

                if global_step % 5000 == 0:
                    writer.add_scalar('Loss/train_running_avg', total_loss/(epoch_step+1), global_step)
                    writer.add_scalar('Loss/train_batch', loss.item(), global_step)

                global_step += 1
                epoch_step += 1
                
            del train_loader
            gc.collect()                

        print("This split took {} seconds ...".format(time.time()-start_split))
        print("epoch{0} loss:{1:.4f}".format(epoch, total_loss))


        if epoch %1 == 0:

            model.eval()

            with torch.no_grad():
                preds, labels = [], []
                valid_iterator = tqdm(valid_loader, desc="Validation")
                for _, batch in enumerate(valid_iterator):        
                    token, feature, label, embedding = batch[0].float().cuda(), batch[1].float().cuda(), batch[2], batch[3].float().cuda()#,batch[3].cuda()
                    pred = torch.sigmoid(model(token, feature, embedding)).detach().cpu().numpy()
                    labels.append(label)
                    preds.append(pred)

            labels = np.vstack(labels)
            preds = np.float64(np.vstack(preds))

            prauc_all = []
            rce_all = []
        
            for i, engage in enumerate(["reply", "retweet", "comment", "like"]):
                _prauc = compute_prauc(preds[:,i],labels[:,i])
                _rce = compute_rce(preds[:,i],labels[:,i])

                print(engage+":")
                print(_prauc)
                print(_rce)
                prauc_all.append(_prauc)
                rce_all.append(_rce)

                writer.add_scalar('PRAUC/{}_val'.format(engage), _prauc, epoch)
                writer.add_scalar('RCE/{}_val'.format(engage), _rce, epoch)

            writer.add_scalar('PRAUC/mean_val', np.mean(prauc_all), epoch)
            writer.add_scalar('RCE/mean_val', np.mean(rce_all), epoch)

            torch.save(model.state_dict(), "./checkpoint/{}_{}.ckpt".format(args.run_name, epoch))
            
    print("Elapsed: {0}".format(inhour(time.time() - start_time)))


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="P")
    parser.add_argument('-e', dest='epoch', type=check_int_positive, default=1000)
    parser.add_argument('-b', dest='batch', type=check_int_positive, default=4096)
    parser.add_argument('-lr', dest='lr', type=check_float_positive, default=1e-4)
    parser.add_argument('-l', dest='lamb', type=check_float_positive, default=0)
    parser.add_argument('-c', dest='corruption', type=check_float_positive, default=0.25)
    parser.add_argument('-d', dest='path', default="/home/kevin/Projects/MLPHojin/data/new_split")
    parser.add_argument('-tr', dest='train', default='Train')
    parser.add_argument('-v', dest='valid', default='Valid.sav')

    parser.add_argument('--emb_folder', type=str, default='/home/kevin/Projects/MLPHojin/data/xlmr_new')
    parser.add_argument('--num_splits', type=int, default=3)
    parser.add_argument('--emb_file', type=str, default='train_emb')
    parser.add_argument('--run_name', type=str, default='xlmr_final_new')
    parser.add_argument('--emb_type', type=check_emb_type, required=True)
    parser.add_argument('--net_arch', dest='network_architecture', default='embedding_net')
    args = parser.parse_args()

    main(args)
