import os
import pandas as pd
from pathlib import Path


out_file = "/home/layer6/recsys/unsupervised_data/multi_top10_langs_unique_tweets.txt"
min_words_in_sentence = 7


dataframe_path = "/home/layer6/recsys/clean/"
paths = [str(x) for x in Path(dataframe_path).glob("chunk*.p")]


all_lines = set()

for p in paths:

    print("Extracting text from {} ... ".format(p))
    df = pd.read_pickle(p)

    original_len = len(df)

    # at least 7 words or else masking things out 20% of the time won't make much sense
    df = df[df["word_count"] >= min_words_in_sentence]
    print("{}/{} samples left after filtering for word count".format(len(df), original_len))

    print(df["lang_str"].value_counts(normalize=True))

    # input format is just 1 sample per line
    all_lines.update(set(df["tweet_clean"].to_numpy()))
    print("New size of data: {} lines".format(len(all_lines)))

all_lines = list(all_lines)

print("Total number of unique training sentences: {}".format(len(all_lines)))

os.makedirs(os.path.dirname(out_file), exist_ok=True)

with open(out_file, "w") as f:
    f.write('\n'.join(all_lines))


print("done")
