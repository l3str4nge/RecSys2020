import h5py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from os.path import join
import pickle


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.net = torch.nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),            
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        self.fc = nn.Linear(1024, 4)

    def forward(self, x):
        h = self.net(x)
        logit = self.fc(h)
        return logit


class RandomBatchSampler(torch.utils.data.Sampler):
    """Samples batches order randomly, without replacement.
    Args:
        data_source (Dataset): dataset to sample from
        batch_size (int): Batch size (number of examples)
    Note:
        Last batch is always composed of dataset tail.
    """

    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        return iter([
            list(range(x * self.batch_size, (x + 1) * self.batch_size))
            for x in torch.randperm(self.__len__() - 1).tolist()
        ] + [list(range((self.__len__() - 1) * self.batch_size, len(self.data_source)))])

    def __len__(self):
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, path, num_shards):
        self.file_path = path
        self.datasets = [None]*num_shards

        config = pickle.load(open(join(self.file_path, 'map.p'), 'rb'))
        self.data = list(config['map'].values())

    def __getitem__(self, i):

        shard, index = self.data[i]

        if self.datasets[shard] is None:
            shard_file_name = "{}/shard{}.h5".format(self.file_path, shard)
            print("Opening file {}".format(shard_file_name))
            self.datasets[shard] = h5py.File(shard_file_name, 'r')["embedding"]
            
        return self.datasets[shard][index]

    def __len__(self):
        return len(self.data)



# embeddings_dir = '/media/kevin/datahdd/data/recsys/tweetstring/embeddings/universal_sentence_encoder/multi/shards'
embeddings_dir = '/home/kevin/Projects/HojinChunks/data/shards'

num_shards = len([str(x) for x in Path(embeddings_dir).glob("*.h5")])


dataset = H5Dataset(embeddings_dir, num_shards)


# data_loader = DataLoader(
#     dataset,
#     batch_sampler=RandomBatchSampler(dataset, 4096),
#     num_workers=20,
#     pin_memory=True
# )


data_loader = DataLoader(
    dataset,
    batch_size=4096,
    shuffle=True,
    num_workers=20,
    pin_memory=True
)


model = Net().cuda()
criterion = nn.BCEWithLogitsLoss()
optim = torch.optim.Adam(model.parameters())


with_model = True

for batch in tqdm(data_loader):

    if with_model:

        batch = batch.cuda()
        label = torch.ones([4096, 4]).float().cuda()

        logit = model(batch)
        loss = criterion(logit, label)
        loss.backward()
        optim.step()
    else:
        
        pass


print("done")

