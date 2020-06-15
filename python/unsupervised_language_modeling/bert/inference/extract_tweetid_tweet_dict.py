"""
    Construct indexed-matched arrays of (tweet_id, tweet string)
"""
from train_tokenizer import load_bert_tokenizer
import os
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np


def main():

    chunk_name = "test"
    in_file = "/home/layer6/recsys/clean/{}.p".format(chunk_name)
    out_file = "/home/layer6/recsys/clean/tweets_only/{}_tweet_tokens.p".format(chunk_name)
    block_size = 100
    min_words_in_sentence = 0

    tokenizer = load_bert_tokenizer("/home/layer6/recsys/pretrained_models/model/vocab.txt", block_size, "multi")

    df = pd.read_pickle(in_file)
    original_len = len(df)
    # TODO filter by language if necessary
    # df = df[df["lang_str"].isin(supported_languages)]
    # print("{}/{} samples left after filtering for {}".format(len(df), original_len, supported_languages))

    df = df[df["word_count"] >= min_words_in_sentence]
    print("{}/{} samples left after filtering for word count".format(len(df), original_len))

    id_n_tweet = df[["tweet_id", "tweet_clean"]].drop_duplicates().to_numpy()

    print("Running tokenization on {} sentences ... ".format(len(id_n_tweet)))

    tokens = []
    chunk_size = 1000000

    for i in tqdm(range(0, len(id_n_tweet), chunk_size)):
        chunk = id_n_tweet[i: min(i+chunk_size, len(id_n_tweet)), 1]
        tokens.extend([x.ids for x in tokenizer.encode_batch(list(chunk))])

    with open(out_file, "wb") as handle:
        pickle.dump({
            "tweet_id": id_n_tweet[:, 0],
            "tokens": np.array(tokens)
        }, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
