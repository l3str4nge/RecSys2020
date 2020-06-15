import numpy as np
import argparse
import time
import torch
import torch.nn as nn
import os
import pandas as pd

from tqdm import tqdm, trange

from model.EmbeddingNet import EmbeddingNet  # , MLP_combine

from utils.argcheck import *
from utils.eval import compute_prauc, compute_rce
from utils.progress import WorkSplitter, inhour
from utils.data_feature_embedding_multiple import Data




def main(args):
    # Progress bar
    progress = WorkSplitter()
    if args.emb_type == 'bert':
        emb_size = 768
    elif args.emb_type == 'xlmr':
        emb_size = 1024

    # Load Data
    progress.section("Load Data")
    print("Embedding size is set to",emb_size)
    start_time = time.time()
    data = Data(args, args.path, args.emb_path, args.train, args.valid, emb_size, is_lb=True)
    print("Elapsed: {0}".format(inhour(time.time() - start_time)))

    # build model
    progress.section("Build Model")

    model = EmbeddingNet(data.n_token, data.n_feature, emb_size, [1024, 2000, 1000, 500, 100])
    model.cuda()
    print(model)

    model.cuda()
    model.load_state_dict(torch.load(args.checkpoint))

    print(model)
    lb_loader = data.instance_a_lb_loader(args.batch)

    lbs = {'user_lb': list(), 'tweet_lb': list()}
    preds = []
    model.eval()
    with torch.no_grad():
        lb_iterator = tqdm(lb_loader, desc="lb")
        for _, batch in enumerate(lb_iterator):
            token, feature, tweet_lb, user_lb, embedding = batch[0].float().cuda(), batch[1].float().cuda(), batch[2], \
                                                           batch[3], batch[4].float().cuda()  # ,batch[4].cuda()
            pred = torch.sigmoid(model(token, feature, embedding)).detach().cpu().numpy()
            lbs['tweet_lb'] += tweet_lb[0]
            lbs['user_lb'] += user_lb[0]
            preds.append(pred)

        final_csv = pd.DataFrame(lbs)
        preds = np.float64(np.vstack(preds))
        if not os.path.exists(args.spath):
            os.makedirs(args.spath)

        print("Generating CSVs...")
        for i, engage in enumerate(["reply", "retweet", "comment", "like"]):
            final_csv[engage] = preds[:, i]
            final_csv[['tweet_lb', 'user_lb', engage]].to_csv(os.path.join(args.spath, engage + '.csv'), index=False,
                                                              header=False)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="P")
    parser.add_argument('-ut', dest='u_threshold', type=check_float_positive, default=0)
    parser.add_argument('-ct', dest='c_threshold', type=check_float_positive, default=0)
    parser.add_argument('-u', dest='is_unified', action='store_true')
    parser.add_argument('-e', dest='epoch', type=check_int_positive, default=1000)
    parser.add_argument('-b', dest='batch', type=check_int_positive, default=4096)
    parser.add_argument('-lr', dest='lr', type=check_float_positive, default=0.0001)
    parser.add_argument('-l', dest='lamb', type=check_float_positive, default=100)
    parser.add_argument('-r', dest='rank', type=check_int_positive, default=20)
    parser.add_argument('-s', dest='seed', type=check_int_positive, default=1)
    parser.add_argument('-d', dest='path', default="/home/layer6/recsys/kevin_data/")
    parser.add_argument('--emb_path', default="/home/layer6/recsys/kevin_data/embeddings/supervised_bert_difflr_checkpoint_21000")
    parser.add_argument('-tr', dest='train', default='Train')
    parser.add_argument('-v', dest='valid', default='Submit.sav')
    parser.add_argument('-sp', dest='spath', default="predictions/supervised_bert_difflr_checkpoint_21000_ep_32")
    parser.add_argument('--num_splits', type=int, default=3)
    parser.add_argument('--emb_file', type=str, default='train_emb')
    parser.add_argument('--emb_type', type=check_emb_type, required=True)
    parser.add_argument('--checkpoint', type=str, default="./checkpoint/featurenet_supervised_32.ckpt")

    args = parser.parse_args()

    main(args)
