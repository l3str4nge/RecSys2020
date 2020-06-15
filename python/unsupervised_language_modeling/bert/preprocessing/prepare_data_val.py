import os
import pandas as pd
from pathlib import Path
import random

out_file = "/home/layer6/recsys/unsupervised_data/multi_top10_langs_unique_tweets_leaderboard.txt"
min_words_in_sentence = 7


dataframe_path = "/home/layer6/recsys/clean/val.p"

df = pd.read_pickle(dataframe_path)

original_len = len(df)


df = df[df["word_count"] >= min_words_in_sentence]
print("{}/{} samples left after filtering for word count".format(len(df), original_len))

print(df["lang_str"].value_counts(normalize=True))


all_lines = set(df["tweet_clean"].to_numpy())
all_lines = list(all_lines)
random.shuffle(all_lines)

print("Total number of training sentences: {}".format(len(all_lines)))

os.makedirs(os.path.dirname(out_file), exist_ok=True)

with open(out_file, "w") as f:
    f.write('\n'.join(all_lines))

print("done")
