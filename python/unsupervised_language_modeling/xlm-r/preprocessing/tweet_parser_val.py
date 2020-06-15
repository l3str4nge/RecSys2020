#!/usr/bin/env python
# coding: utf-8


import time
import pickle
from tqdm import tqdm
import pandas as pd
from itertools import islice
from transformers import BertTokenizer, BertConfig, BertModel
tqdm.pandas()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
start = time.time()


all_features = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",                "tweet_type","language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",               "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",               "enaging_user_id", "enaging_user_follower_count", "enaging_user_following_count", "enaging_user_is_verified",               "enaging_user_account_creation", "engagee_follows_engager"]
all_features_to_idx = dict(zip(all_features, range(len(all_features))))
labels_to_idx = ["reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]


from tweet_parser import get_language, get_tweet_string, get_clean_tweet, url_check, hashtag_check


file_path = "/home/layer6/recsys/raw/val.tsv"

with open(file_path, encoding="utf-8") as f:

    print("Reading the file... this will take a while")
    lines = list(islice(f, 50000000))
    lines = [x.strip().split("\x01") for x in lines]

    assert not (not lines or len(lines) < 1)

print("{} lines in the val file ... ".format(len(lines)))

df = pd.DataFrame(lines, columns=all_features)

# Remove unnecessary fields
df = df[["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "language", "enaging_user_id"]]

# Get language
df = get_language(df)
print(df["lang_str"].value_counts())

# Get tweet text using Bert tokenizer
df = get_tweet_string(df)

# Parse mentions, hashtags, urls from tweet strings
df = get_clean_tweet(df, "tweet_string")

# Check fidelity of extracted hashtag strings and url strings
url_check(df)
hashtag_check(df)

# Save only tweet string, its features, and sample identifier
df = df[["tweet_id", "enaging_user_id", "lang_str", "tweet_string", "tweet_clean", "hashtag_strings", "url_string", "main_mention", "other_mentions"]]
print(df.head(5))
df.to_pickle("/home/layer6/recsys/clean/xlmr/val.p", protocol=pickle.HIGHEST_PROTOCOL)
# df.to_csv("/media/kevin/datahdd/data/recsys/tweetstring/val/val.tsv", sep='\t')

print("done")