import gc
import pickle
import joblib
from os.path import join


data_folder = "/data/MLP_data"
unsupervised_folder = "/data/unsupervised_data"


train_set = pickle.load(open(join(data_folder, "train_tweetids.p"), "rb"))

val_set = pickle.load(open(join(data_folder, "val_tweetids.p"), "rb"))

submit_set = pickle.load(open(join(data_folder, "submit_tweetids.p"), "rb"))

TrainValToks = pickle.load(open(join(unsupervised_folder, "xlmr_all_tweet_tokens.p"), "rb"))
LBToks = pickle.load(open(join(unsupervised_folder, "xlmr_all_tweet_tokens_leaderboard.p"), "rb"))

TrainLBToks = {k: v for k,v in TrainValToks.items() if k not in val_set}
TrainLBToks.update({k: v for k,v in LBToks.items() if k not in val_set})

ValToks = {k: v for k,v in TrainValToks.items() if k in val_set}  # val set should take samples that appear
ValToks.update({k: v for k,v in LBToks.items() if k in val_set})  #  in multiple sets but train won't have them


print("Size of XGB train set : {}".format(len(train_set)))
print("Size of XGB val set : {}".format(len(val_set)))
print("Size of XGB submit set : {}".format(len(submit_set)))

print("Size of Unsupervised Data train+submit set : {}".format(len(TrainLBToks)))
print("Size of Unsupervised Data val set : {}".format(len(ValToks)))


min_words_in_sentence = 7

print("Making unsupervised train dataset ... ")
training_dataset = [v[1] for k,v in TrainLBToks.items() if v[2] >= min_words_in_sentence]
print("{}/{} samples left after filtering for word count".format(len(training_dataset), len(TrainLBToks)))

training_dataset = list(set([tuple(x) for x in training_dataset]))
training_dataset = [list(x) for x in training_dataset]
print("{}/{} samples left after filtering for redundancy".format(len(training_dataset), len(TrainLBToks)))

print(training_dataset[:2])


print("Making unsupervised val dataset ... ")
val_dataset = [v[1] for k,v in ValToks.items() if v[2] >= min_words_in_sentence]
print("{}/{} samples left after filtering for word count".format(len(val_dataset), len(ValToks)))

val_dataset = list(set([tuple(x) for x in val_dataset]))
val_dataset = [list(x) for x in val_dataset]
print("{}/{} samples left after filtering for redundancy".format(len(val_dataset), len(ValToks)))

print(val_dataset[:2])

print("Saving ... ")

pickle.dump(training_dataset, open(join(unsupervised_folder, "xlmr_TrainLB.p"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(val_dataset, open(join(unsupervised_folder, "xlmr_Val.p"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)


print("done")