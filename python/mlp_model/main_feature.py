import numpy as np
from utils.progress import WorkSplitter, inhour
from utils.data_feature import Data
import argparse
import time
from utils.argcheck import check_float_positive, check_int_positive
from utils.eval import compute_prauc, compute_rce

from model.FeatureNet import FeatureNet
import gc
import torch
import torch.nn as nn
import os 

from tqdm import tqdm, trange


def main(args):
    # Progress bar
    progress = WorkSplitter()
    if not os.path.exists("./checkpoint"):
        os.mkdir("./checkpoint")

    # Load Data
    progress.section("Load Data")
    start_time = time.time()
    data = Data(args.path, args.train, args.valid)
    print("Elapsed: {0}".format(inhour(time.time() - start_time)))
    
    #build model
    progress.section("Build Model")
    model = FeatureNet(data.n_token, data.n_feature, [1024, 2000, 1000, 500, 100],corruption=args.corruption)
    model.cuda()
    print(model)
    
    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay=args.lamb)
    valid_loader = data.instance_a_valid_loader(args.batch)
    train_loader = data.instance_a_train_loader(args.batch)

    progress.section("Train model")
    for epoch in range(1,args.epoch):
        
        total_loss = 0
        model.train()   
        epoch_iterator = tqdm(train_loader, desc="Iteration")
        for _, batch in enumerate(epoch_iterator) :
            token, feature, label = batch[0].float().cuda(), batch[1].float().cuda(), batch[2].float().cuda()#, batch[3].cuda()
            optim.zero_grad()
            logit = model(token, feature)
            loss = criterion(logit, label)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        
        print("epoch{0} loss:{1:.4f}".format(epoch, total_loss))
        if epoch %1 == 0:
            model.eval()
            with torch.no_grad():
                preds, labels = [], []
                valid_iterator = tqdm(valid_loader, desc="Validation")
                for _, batch in enumerate(valid_iterator):        
                    token, feature, label = batch[0].float().cuda(), batch[1].float().cuda(), batch[2]#,batch[3].cuda()
                    pred = torch.sigmoid(model(token, feature)).detach().cpu().numpy()
                    labels.append(label)
                    preds.append(pred)

            labels = np.vstack(labels)
            preds = np.float64(np.vstack(preds))
        
            for i, engage in enumerate(["reply", "retweet", "comment", "like"]):
                print(engage+":")
                print(compute_prauc(preds[:,i],labels[:,i]))
                print(compute_rce(preds[:,i],labels[:,i]))                
            torch.save(model.state_dict(), "./checkpoint/{}_{}.ckpt".format(args.run_name, epoch))

    print("Elapsed: {0}".format(inhour(time.time() - start_time)))


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="P")
    parser.add_argument('-e', dest='epoch', type=check_int_positive, default=60)
    parser.add_argument('-b', dest='batch', type=check_int_positive, default=4096)
    parser.add_argument('-lr', dest='lr', type=check_float_positive, default=1e-4)
    parser.add_argument('-l', dest='lamb', type=check_float_positive, default=0)
    parser.add_argument('-c', dest='corruption', type=check_float_positive, default=0.25)
    parser.add_argument('-d', dest='path', default="data/")
    parser.add_argument('-tr', dest='train', default='Train.sav')
    parser.add_argument('-v', dest='valid', default='Valid.sav')

    parser.add_argument('--run_name', type=str, default='featurenet')
    args = parser.parse_args()

    main(args)
