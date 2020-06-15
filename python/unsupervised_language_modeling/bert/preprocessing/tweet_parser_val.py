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


def get_language(df):
    # top 25 languages, there are 66 in total
    #     TODO: handle other 66-25 languages
    languages = {
        "english": "D3164C7FBCF2565DDF915B1B3AEFB1DC",
        "japanese": "22C448FF81263D4BAF2A176145EE9EAD",
        "spanish": "06D61DCBBE938971E1EA0C38BD9B5446",
        "portuguese": "ECED8A16BE2A5E8871FD55F4842F16B1",
        "croatian": "B9175601E87101A984A50F8A62A1C374",
        "turkish": "4DC22C3F31C5C43721E6B5815A595ED6",
        "arabic": "167115458A0DBDFF7E9C0C53A83BAC9B",
        "thai": "022EC308651FACB02794A8147AEE1B78",
        "french": "FA3F382BC409C271E3D6EAF8BE4648DD",
        "korean": "125C57F4FA6D4E110983FB11B52EFD4E",
        "indonesian": "9BF3403E0EB7EA8A256DA9019C0B0716",
        "hindi": "975B38F44D65EE42A547283787FF5A21",
        "filipino": "2996EB2FE8162C076D070A4C8D6532CD",
        "german": "FF60A88F53E63000266F8B9149E35AD9",
        "italian": "717293301FE296B0B61950D041485825",
        "russian": "3820C29CBCA409A33BADF68852057C4A",
        "persian": "3E16B11B7ADE3A22DDFC4423FBCEAD5D",
        "polish": "9ECD42BC079C20F156F53CB3B99E600E",
        "chinese": "76B8A9C3013AE6414A3E6012413CDC3B",
        "urdu": "AEF22666801F0A5846D853B9CEB2E327",
        "catalan": "190BA7DA361BC06BC1D7E824C378064D",
        "dutch": "1FFD2FE4297F5E70EBC6C3230D95CB9C",
        "tamil": "D413F5FE5236E5650A46FD983AB39212",
        "telugu": "A0C7021AD8299ADF0C9EBE326C115F6F",
        "swahili": "48236EC80FDDDFADE99420ABC9210DDF"
    }

    lang_map = {v:k for k,v in languages.items()}
    df["lang_str"] = df.language.map(lang_map).fillna("other")

    return df


def get_tweet_string(df):

    config_name = 'bert-base-multilingual-cased'
    tokenizer = BertTokenizer.from_pretrained(config_name)
    df["tweet_string"] = df["text_tokens"].progress_apply(lambda x: tokenizer.decode([int(i) for i in x.split("\t")]))

    return df


def get_mentions(df, col_name):
    
    main_mention_reg = r'RT \@ ([^,:]+) :'
    other_mentions_reg = r'\@ (?P<item>\S*)'
    
    df["main_mention"] = df[col_name].str.extract(main_mention_reg)
    df["tweet_clean"] = df[col_name].str.replace(main_mention_reg, "")
    df["other_mentions"] = df["tweet_clean"].str.extractall(other_mentions_reg).groupby(level=0)["item"].apply(list)
    df["tweet_clean"] = df["tweet_clean"].str.replace(other_mentions_reg, "")

    return df

def get_hashtags(df, col_name):
    
    hashtag_reg = r'\# (?P<item>\S*)'
    
    df["hashtag_strings"] = df[col_name].str.extractall(hashtag_reg).groupby(level=0)["item"].apply(list)
    df["tweet_clean"] = df[col_name].str.replace(hashtag_reg, "")
#     TODO: hashtags in other languages can be multiple words separated by spaces (chinese)
    
    return df

def get_urls(df, col_name):
    
    url_reg = r'(https \: \/ \/ t. co \/ \S*)'
    
    df["url_string"] = df[col_name].str.extract(url_reg)
    df["tweet_clean"] = df[col_name].str.replace(url_reg, "")
    
    return df

def get_clean_tweet(df, col_name):
    return get_urls(get_hashtags(get_mentions(df, col_name), "tweet_clean"), "tweet_clean")


def url_check(df):
    # TODO : https : / / t. co [UNK] is a special case that isn't being caught
    # TODO : this is also a special case [UNK] : / / t. co / vbexugmVpX
    # found extra url
    extra = (df["present_media"] == "") & (df["present_links"] == "") & (~df["url_string"].isna())
    # missed url
    missed = ( (df["present_media"] != "") | (df["present_links"] != "") ) & (df["url_string"].isna())
    total = float(len(df))
    
    print(extra.sum())
    print(missed.sum())
    print(total)
    return (extra.sum()+missed.sum())/total


def hashtag_check(df):
    # found extra hashtags
    extra = (df["hashtags"] == "") & (~df["hashtag_strings"].isna())
    # missed hashtag
    missed = (df["hashtags"] != "") & (df["hashtag_strings"].isna())
    total = float(len(df))
    
    print(extra.sum())
    print(missed.sum())
    print(total)
    return (extra.sum()+missed.sum())/total


file_path = "/home/layer6/recsys/out/test.tsv"

with open(file_path, encoding="utf-8") as f:

    print("Reading the file... this will take a while")
    lines = list(islice(f, 50000000))
    lines = [x.strip().split("\x01") for x in lines]

    assert not (not lines or len(lines) < 1)

print("{} lines in the test file ... ".format(len(lines)))

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
df.to_pickle("/home/layer6/recsys/out/test.p", protocol=pickle.HIGHEST_PROTOCOL)
#df.to_csv("/home/layer6/recsys/out/val.tsv", sep='\t')

print("done")
