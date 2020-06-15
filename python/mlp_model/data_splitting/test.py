from dataset import Data
from tqdm import tqdm
import argparse
import random

random.seed(99)


def main(args):

    # shuffle splits
    data = Data(args.path, args.valid)

    valid_loader = data.instance_a_valid_loader(args.batch)
    
    print("WARNING: 1 epoch is 1 split, if you want N epochs over the entire dataset, set the epoch argument to N*num_splits ... ")

    for epoch in range(1, args.epoch+1):
        total_loss = 0
        # model.train()

        print("Initializing dataset split {} ... ".format(data.current_split))
        train_loader = data.instance_a_train_loader(args.batch)
        
        epoch_iterator = tqdm(train_loader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator) :
            # do stuff
            pass


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="P")
    parser.add_argument('-e', dest='epoch', type=int, default=5*2)
    parser.add_argument('-b', dest='batch', type=int, default=4096)
    parser.add_argument('-d', dest='path', default="/media/kevin/datahdd/data/recsys/Hojin/chunked/chunk12")
    parser.add_argument('-v', dest='valid', default='Valid.sav')
    args = parser.parse_args()

    main(args)