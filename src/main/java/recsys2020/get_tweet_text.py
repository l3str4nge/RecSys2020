from transformers import BertTokenizer
from tqdm import tqdm
import sys
import pandas as pd

tqdm.pandas()

in_path = sys.argv[1]
data_files = ['training.tsv', 'val.tsv', 'competition_test.tsv']

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

tweet_ids = []
tweet_strs = []
tweet_id_set = set()

for data_file in data_files:
    with open(in_path + data_file, encoding="utf-8") as f:
        for line in tqdm(f):
            line = line.strip()
            features = line.split("\x01")
            tweet_id = features[2]

            if tweet_id not in tweet_id_set:
                tweets_raw = features[0]
                tweets = [int(x) for x in tweets_raw.split('\t')]

                tweet_str = tokenizer.decode(tweets)

                tweet_ids.append(tweet_id)
                tweet_strs.append(tweet_str)
                tweet_id_set.add(tweet_id)

print(len(tweet_ids))

token_df = pd.DataFrame({'tweet_id': tweet_ids, 'tweet_strings': tweet_strs})
token_df.to_csv(in_path + 'decoded_strings.csv', header=False, index=False)
