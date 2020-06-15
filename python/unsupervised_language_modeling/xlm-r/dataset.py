import logging
import os
import pickle
import time
from tqdm import tqdm

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset


logger = logging.getLogger(__name__)



class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, args, tokenizer, file_path: str, block_size: int):

        assert os.path.isfile(file_path)

        logger.info("Loading features from cached file %s", file_path)
        with open(file_path, "rb") as handle:
            self.examples = pickle.load(handle)

        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)

        # directory, filename = os.path.split(file_path)
        # cached_features_file = os.path.join(
        #     directory, "cached_lm_" + str(block_size) + "_" + filename
        # )

        # if os.path.exists(cached_features_file) and not args.overwrite_cache:
        #     logger.info("Loading features from cached file %s", cached_features_file)
        #     with open(cached_features_file, "rb") as handle:
        #         self.examples = pickle.load(handle)

        # else:

        #     logger.info("Creating features from dataset file at %s", file_path)

        #     with open(file_path, encoding="utf-8") as f:
        #         lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        #     logger.info("Running tokenization on {} sentences ... ".format(len(lines)))

        #     self.examples = []
        #     chunk_size = 100000

        #     for i in tqdm(range(0, len(lines), chunk_size)):
        #         chunk = lines[i: min(i+chunk_size, len(lines))]
        #         self.examples.extend(tokenizer.batch_encode_plus(chunk, add_special_tokens=True, max_length=block_size)['input_ids'])

        #     logger.info("Saving features into cached file %s", cached_features_file)
        #     with open(cached_features_file, "wb") as handle:
        #         pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)
