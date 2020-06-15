"""
    Unpack a list of large embedding chunks into single files.
"""
from os.path import join, exists
from tqdm import tqdm
from pathlib import Path
import pickle
import gc
import numpy as np
import h5py


"""
    Construct 2 things:
    1. a map of (tweet_id) --> (shard_id, row_id)
    2. a map of (shard_num) --> set of tweet_ids belonging to shard
"""
def construct_embedding_shard_map(embeddings_dir, output_dir, num_shards):

    cache = join(output_dir, "map.p")

    if exists(cache):
        return pickle.load(open(cache, 'rb'))


    embedding_files = sorted([str(x) for x in Path(embeddings_dir).glob("*.p")])


    tweet_ids = set()


    for emb_file in tqdm(embedding_files):

        emb_map = pickle.load(open(emb_file, "rb"))

        # TODO uncomment depending on what your embedding file looks like
        # tweet_ids.update(set(emb_map.keys()))
        tweet_ids.update(set(emb_map['lookup'].keys()))

        del emb_map
        gc.collect()


    tweet_ids = sorted(list(tweet_ids))
    print("There are {} unique tweet ids, splitting them into {} shards ... ".format(len(tweet_ids), num_shards))


    shard_size = len(tweet_ids)//num_shards + 1


    shard_id_sets = {}
    shard_map = {}

    for shard_num in tqdm(range(num_shards)):

        start_idx = shard_num*shard_size
        end_idx = min((shard_num+1)*shard_size, len(tweet_ids))

        shard_ids = tweet_ids[start_idx:end_idx]

        shard_id_sets[shard_num] = set(shard_ids)

        for row_num, twt_id in enumerate(tqdm(shard_ids)):
    
            shard_map[twt_id] = (shard_num, row_num)


    assert sum([len(x) for x in shard_id_sets.values()]) == len(tweet_ids)
    assert len(shard_map) == len(tweet_ids)


    print("Saving ...")

    pickle.dump({
        'map': shard_map,
        'sets': shard_id_sets
    }, open(cache, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


"""
    For each shard we need to:
    1. Loop over all the embedding files
    2. Extract the embeddings that belong to the shard
    3. Save the shard as a h5py file

"""
def construct_embedding_shards(embeddings_dir, output_dir, num_shards, embedding_size=512):

    config_file = join(output_dir, "map.p")
    config = pickle.load(open(config_file, 'rb'))

    shard_id_sets = config['sets']
    shard_map = config['map']
    embedding_files = sorted([str(x) for x in Path(embeddings_dir).glob("*.p")])


    for shard_num in tqdm(range(num_shards)):

        # let's find the embedding vectors and put them into the right place

        shard_ids = shard_id_sets[shard_num]
        shard_data = np.zeros([len(shard_ids), embedding_size], dtype=np.float32)
        assigned_rows = 0  # sanity check
        processed_tweet_ids = set()


        for emb_file in tqdm(embedding_files):

            emb_map = pickle.load(open(emb_file, "rb"))

            # TODO uncomment depending on what your embedding file looks like
            # for key, embedding in tqdm(emb_map.items()):
            #     if key in shard_ids and not key in processed_tweet_ids:

            #         row_num = shard_map[key][1]  # row number
            #         shard_data[row_num, :] = embedding
            #         assigned_rows += 1
            #         processed_tweet_ids.add(key)

            for key, row in tqdm(emb_map['lookup'].items()):
                if key in shard_ids and not key in processed_tweet_ids:

                    embedding = emb_map['embedding'][row]

                    row_num = shard_map[key][1]  # row number
                    shard_data[row_num, :] = embedding
                    assigned_rows += 1
                    processed_tweet_ids.add(key)


            del emb_map
            gc.collect()


        # Sanity checks
        print("Assigned {}/{} rows in shard data matrix ... ".format(assigned_rows, len(shard_data)))
        assert assigned_rows == len(shard_data)
        print("The following embedding vector should be non-zero ....")
        print(shard_data[0, :])
        

        # Save
        print("Saving ...")
        writer = h5py.File(join(output_dir, 'shard{}.h5'.format(shard_num)), 'w')
        writer.create_dataset('embedding', data=shard_data)
        writer.close()


        # Clean up
        del shard_data, processed_tweet_ids
        gc.collect()


if __name__ == "__main__":

    num_shards = 3
    embeddings_dir = "/media/kevin/datahdd/data/recsys/tweetstring/embeddings/universal_sentence_encoder/multi/train"
    output_dir = "/media/kevin/datahdd/data/recsys/tweetstring/embeddings/universal_sentence_encoder/multi/shards"

    # construct_embedding_shard_map(embeddings_dir, output_dir, num_shards)
    construct_embedding_shards(embeddings_dir, output_dir, num_shards)