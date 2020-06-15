# Steps to reproduce

See the unsupervised_language_modeling/xlm-r folder for more details. The Bert pipeline and the xlm-r pipeline is almost the same.

# Steps to train bert on tweet text in MLM loss

1. run ./preprocessing/tweet_parser.py to get raw tweet strings
2. run ./preprocessing/clean_data.py to clean up the raw tweet strings
3. run ./preprocessing/prepare_data.py to dump all the tweets into a single newline-separated file
4. run run_file.sh which calls ./run_language_modeling.py


# Steps to extract embeddings for training downstream
1. run ./inference/extract_tweetid_tweet_dict.py to build a dictionary of tweet id to tokenized tweet text
2. run ./inference/extract_embedding.py to extract embeddings for downstream MLP training on the 4 tasks