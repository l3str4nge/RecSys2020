import os
import argparse
from os.path import join
from blender import inverse_sigmoid
from tqdm import tqdm
import pandas as pd
import joblib
import numpy as np
from utils.eval import compute_prauc, compute_rce
from torch.utils.tensorboard import SummaryWriter


TASKS = ["reply", "retweet", "comment", "like"]


def sigmoid(x):
    return 1.0/(1+np.exp(-x))


# load val csvs
def load_submission_files(validation_folder, submit=False):

    dataset = None

    for T in tqdm(TASKS):

        field_name = "prob_{}".format(T)
        dtypes = {field_name:np.float32} if submit else {"user_id": np.int32, field_name:np.float32}

        df = pd.read_csv(join(validation_folder, "{}.csv".format(T)),
        names=["tweet_id", "user_id", field_name], dtype=dtypes)

        dataset = df if dataset is None else dataset.merge(df, on=['tweet_id', 'user_id'])

    cols = list(dataset.columns)
    prob_cols = cols[2:]

    inputs = dataset[prob_cols].to_numpy()
    tweet_ids = dataset['tweet_id'].to_numpy()
    user_ids = dataset['user_id'].to_numpy()

    return inputs, tweet_ids, user_ids


def load_labels(args, row_lookup):

    val_dict = joblib.load(open(args.label_file, "rb"))

    labels = np.zeros([len(row_lookup), 4], dtype=np.float32)
    assigned_rows = set()

    for tid, uid, lab in tqdm(zip(val_dict['tweet_ids'], val_dict['lb_user_ids'][:, 0], val_dict['labels'].astype(np.float32))):
        row = row_lookup[(tid, uid)]

        labels[row, :] = lab
        assigned_rows.add(row)


    assert len(assigned_rows) == len(labels)

    return labels


def tune(args):

    writer = SummaryWriter(log_dir=join('./logs_scaler', args.run_name))

    # load val predictions
    inputs, tweet_ids, user_ids = load_submission_files(args.validation_folder, submit=True)
    row_lookup = {(tid, uid): i for i, (tid, uid) in enumerate(zip(tweet_ids, user_ids))}

    # lets use logit inputs and logit outputs
    logits = inverse_sigmoid(inputs)

    # load Valid.sav for labels
    labels = load_labels(args, row_lookup)

    for T in tqdm(np.arange(0.8, 1.3, 0.05)):

        print(logits[:, 0].mean())
        print(logits[:, 1].mean())
        print(logits[:, 2].mean())
        print(logits[:, 3].mean())

        scaled_logits = logits.copy()

        m0 = scaled_logits[:, 0].mean()
        m1 = scaled_logits[:, 1].mean()
        m2 = scaled_logits[:, 2].mean()
        m3 = scaled_logits[:, 3].mean()

        scaled_logits[:, 0] = scaled_logits[:, 0] - m0
        scaled_logits[:, 1] = scaled_logits[:, 1] - m1
        scaled_logits[:, 2] = scaled_logits[:, 2] - m2
        scaled_logits[:, 3] = scaled_logits[:, 3] - m3

        print(scaled_logits[:, 0].mean())
        print(scaled_logits[:, 1].mean())
        print(scaled_logits[:, 2].mean())
        print(scaled_logits[:, 3].mean())

        
        scaled_logits = scaled_logits/T

        scaled_logits[:, 0] = scaled_logits[:, 0] + m0
        scaled_logits[:, 1] = scaled_logits[:, 1] + m1
        scaled_logits[:, 2] = scaled_logits[:, 2] + m2
        scaled_logits[:, 3] = scaled_logits[:, 3] + m3

        print(scaled_logits[:, 0].mean())
        print(scaled_logits[:, 1].mean())
        print(scaled_logits[:, 2].mean())
        print(scaled_logits[:, 3].mean())


        preds = sigmoid(scaled_logits)



        print("========== TEMPERATURE = {} ==========".format(T))
        for i, engage in enumerate(TASKS):
            # _prauc = compute_prauc(preds[:,i],labels[:,i])
            _rce = compute_rce(preds[:,i],labels[:,i])

            print(engage+":")
            # print(_prauc)
            print(_rce)

            # writer.add_scalar('PRAUC/{}'.format(engage), _prauc, T)
            writer.add_scalar('RCE/{}'.format(engage), _rce, T*100)


def apply(args, TEMPS=[1, 1, 1, 1]):
    inputs, tweet_ids, user_ids = load_submission_files(args.submit_folder, submit=True)
    
    # lets use logit inputs and logit outputs
    logits = inverse_sigmoid(inputs)

    print(logits[:, 0].mean())
    print(logits[:, 1].mean())
    print(logits[:, 2].mean())
    print(logits[:, 3].mean())

    scaled_logits = logits.copy()

    m0 = scaled_logits[:, 0].mean()
    m1 = scaled_logits[:, 1].mean()
    m2 = scaled_logits[:, 2].mean()
    m3 = scaled_logits[:, 3].mean()

    scaled_logits[:, 0] = scaled_logits[:, 0] - m0
    scaled_logits[:, 1] = scaled_logits[:, 1] - m1
    scaled_logits[:, 2] = scaled_logits[:, 2] - m2
    scaled_logits[:, 3] = scaled_logits[:, 3] - m3

    print(scaled_logits[:, 0].mean())
    print(scaled_logits[:, 1].mean())
    print(scaled_logits[:, 2].mean())
    print(scaled_logits[:, 3].mean())

    
    scaled_logits = scaled_logits/TEMPS

    scaled_logits[:, 0] = scaled_logits[:, 0] + m0
    scaled_logits[:, 1] = scaled_logits[:, 1] + m1
    scaled_logits[:, 2] = scaled_logits[:, 2] + m2
    scaled_logits[:, 3] = scaled_logits[:, 3] + m3 - 0.1

    print(scaled_logits[:, 0].mean())
    print(scaled_logits[:, 1].mean())
    print(scaled_logits[:, 2].mean())
    print(scaled_logits[:, 3].mean())


    lbs = {'user_lb': user_ids, 'tweet_lb': tweet_ids}
    preds = sigmoid(scaled_logits).astype(np.float64)
    #preds = np.clip(preds, 0.00, 0.99)

    final_csv = pd.DataFrame(lbs)

    out_dir = join(args.submit_folder, "scaled")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print("Generating CSVs...")
    for i, engage in enumerate(["reply", "retweet", "comment", "like"]):
        final_csv[engage] = preds[:,i]
        final_csv[['tweet_lb','user_lb',engage]].to_csv(join(args.output_folder, engage+'.csv'),index=False, header=False)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="P")

    parser.add_argument('--output_folder', type=str, default="../out/test/scaled")
    parser.add_argument('--submit_folder', type=str, default="../out/test")
    parser.add_argument('--validation_folder', type=str, default='../out/val')
    parser.add_argument('--label_file', type=str, default='Valid.sav')
    parser.add_argument('--run_name', type=str, default="test")

    args = parser.parse_args()

    # tune(args)
    apply(args, TEMPS=[1.05, 1.15, 1.05, 1.25])
