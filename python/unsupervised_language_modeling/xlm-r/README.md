# Steps to reproduce

Note: we used 2 different language models in our pipeline, Bert base and XLM-Roberta. Below are instructions for reproducing
the pipeline for unsupervised training for XLM-Roberta. The Bert pipeline is almost the same, but with the paths changed.


The input of this pipeline is the raw train.tsv, val.tsv, test.tsv downloaded from the Recsys competition webpage. The
starting point of the pipeline is in tweet_parser.py (explained below). The ending point of the pipeline is in
extract_embeddings.py (explained below). The ouptut of the pipeline is extracted tweet embeddings from a model that is 
finetuned using the Mask Language Modeling (MLM) loss on the tweet data.


## Preprocessing (files located in ./preprocessing)
1. tweet_parser.py - this extracts all the tweet strings from train.tsv and decodes them back to plain text
2. clean_data.py - cleans up special tokens and other extracted fields (note: we parse hashtags, mentions, urls but they are not actually used except the obfuscated <mention>, <hashtag>, <url> tags)
3.  tokenize_data.py - tokenize plain text into tokens for language model
4.  construct_sets.py - make train/eval split for MLM training by getting unique tweet tokens


## Training
1. train.py - main file, loads models, data, tokenizer
2. trainer.py - training and evaluation functions
3. training_args.py - arguments for training
4. dataset.py - skeleton dataset just used to pre-tokenized data and return each element
5. run_file.sh - script to start training


## Inference (files located in ./inference)
1. extract_embeddings.py - extract embeddings from language model