import os
import pickle
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer


data_out_file = "/home/kevin/Projects/xlm-r/data/xlmr_all_tweet_tokens.p"
train_out_file = "/home/kevin/Projects/xlm-r/data/xlmr_all_languages_unique_tweets_tokens.p"
dataframe_path = "/home/kevin/Projects/xlm-r/data/"
paths = sorted([str(x) for x in Path(dataframe_path).glob("chunk*.p")])

# data_out_file = "/media/kevin/datahdd/data/recsys/tweetstring/tweets/xlmr_all_tweet_tokens_test.p"
# train_out_file = "/media/kevin/datahdd/data/recsys/tweetstring/tweets/xlmr_all_languages_unique_tweets_test.p"
# dataframe_path = "/media/kevin/datahdd/data/recsys/tweetstring/train/"
# paths = sorted([str(x) for x in Path(dataframe_path).glob("clean*.p")])

min_words_in_sentence = 7
block_size = 100

if not os.path.isdir(os.path.split(data_out_file)[0]):
    print("Make sure the output directories exist!")
    assert False

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large", use_fast=False, cache_dir=None)  # I think it's the same tokenizer for base and large
tokenizer.add_tokens(["<hashtag>", "<url>", "<mention>"])


processed_tweet_ids = set()
df_master = None

for p in tqdm(paths):

    print("Extracting text from {} ... ".format(p))

    df = pd.read_pickle(p)
    print(df["lang_str"].value_counts(normalize=True))

    df = df[["tweet_id", "tweet_clean", "word_count"]].drop_duplicates()

    start_len = len(processed_tweet_ids)

    if df_master is None:
        df_master = df
    else:
        df = df[~df['tweet_id'].isin(processed_tweet_ids)]
        df_master = pd.concat([df_master, df], axis=0)

    processed_tweet_ids.update(set(df['tweet_id']))

    # sanity check
    assert len(df_master) == len(processed_tweet_ids)

    print("Added {} unique tweet ids ... ".format(len(processed_tweet_ids) - start_len))
    print("There are now {} tweet ids in total ... ".format(len(processed_tweet_ids)))
    print(df_master.shape)

# tokenize
np_arr = df_master.to_numpy()
tweet_id, tweet_clean, word_count = list(np_arr[:, 0]), list(np_arr[:, 1]), list(np_arr[:, 2])

print("Running tokenization on {} sentences ... ".format(len(tweet_id)))
tweet_token = []
chunk_size = 100000

for i in tqdm(range(0, len(tweet_clean), chunk_size)):
    chunk = tweet_clean[i: min(i+chunk_size, len(tweet_clean))]
    tweet_token.extend(tokenizer.batch_encode_plus(chunk, add_special_tokens=True, max_length=block_size)['input_ids'])


print("Converting to dictionary ... ")
tweet_id_to_data = {a: (b, c, d) for a, b, c, d in zip(tweet_id, tweet_clean, tweet_token, word_count)}

print("Saving ... ")
pickle.dump(tweet_id_to_data, open(data_out_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)


print("Making unsupervised dataset ... ")
training_dataset = [v[1] for k,v in tweet_id_to_data.items() if v[2] >= min_words_in_sentence]
print("{}/{} samples left after filtering for word count".format(len(training_dataset), len(tweet_id_to_data)))


print("Saving ... ")
pickle.dump(training_dataset, open(train_out_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

print("done")