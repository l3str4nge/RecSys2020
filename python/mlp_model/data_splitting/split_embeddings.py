from pathlib import Path
from tqdm import tqdm
from os.path import join
import joblib
import numpy as np
import pickle

"""
    Unfortunately we need to check every embedding file for each data split.
"""


embedding_dir = ""
split_dir = ""
output_path = ""


emb_files = sorted([str(x) for x in Path(join(embedding_dir, "splits")).glob("*.p")])
split_files = sorted([str(x) for x in Path(join(split_dir, "splits")).glob("*ids*.sav")])


for split in tqdm(split_files):

    print("Processing split {} ... ".format(split))

    dataset = joblib.load(open(split, 'rb'))
    tweet_ids = set(np.squeeze(dataset['tweet_ids'], 1))

    emb_for_split = {}
    
    for emb_file in tqdm(emb_files):
    
        embeddings = pickle.load(open(emb_file, "rb"))
        emb_for_split.update({k: v for k, v in embeddings.items() if k in tweet_ids})
    
    print("There are {} embeddings in split {}".format(len(emb_for_split), split))

    joblib.dump(emb_for_split, open(output_path + "_embeddings_{}.sav".format(i), 'wb'))

    del dataset, tweet_ids, emb_for_split, embeddings
    gc.collect()

    