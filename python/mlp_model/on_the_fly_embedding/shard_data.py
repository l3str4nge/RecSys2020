from os.path import join
from tqdm import tqdm
import pickle
import numpy as np
import h5py
import joblib


def construct_data_shards(embeddings_map, data_file, output_dir, num_shards):

    config = pickle.load(open(embeddings_map, 'rb'))
    shard_map = config['map']


    # TODO might load data differently if saved in another format
    data = joblib.load(open(data_file, 'rb'))
            
    labels = np.array(data['labels'])
    features = data['features']
    tokens = data['tokens']
    tweet_ids = data['tweet_ids']

    shard_size = len(labels)//num_shards + 1


    all_samples = []

    for shard_num in tqdm(range(num_shards)):

        start_idx = shard_num*shard_size
        end_idx = min((shard_num+1)*shard_size, len(tweet_ids))

        # tuples (sample shard, sample row, embed shard, embed row)
        shard_tweet_ids = tweet_ids[start_idx:end_idx]

        all_samples.extend(
            [ [shard_num, row_num] + list(shard_map[twt_id]) for row_num, twt_id in enumerate(shard_tweet_ids)]
        )
        
        # Save
        print("Saving ...")
        writer = h5py.File(join(output_dir, 'shard{}.h5'.format(shard_num)), 'w')
        writer.create_dataset('labels', data=labels[start_idx:end_idx])
        writer.create_dataset('features', data=features[start_idx:end_idx])
        writer.create_dataset('tokens', data=tokens[start_idx:end_idx])
        writer.close()       


if __name__ == "__main__":

    num_shards = 3
    embeddings_map = "/media/kevin/datahdd/data/recsys/tweetstring/embeddings/universal_sentence_encoder/multi/shards/map.p"
    data_file = "./data/Valid.sav"
    output_dir = "./data/shards"

    construct_data_shards(embeddings_map, data_file, output_dir, num_shards)