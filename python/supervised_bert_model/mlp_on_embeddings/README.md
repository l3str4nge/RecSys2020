1 - To **train** the MLP model on the supervised Bert embeddings, run the following:

`main_feature_embedding.py`

Assumes that you have the data files in ```-d```, the embedding files in ```-ed```

2 - To **infer** the MLP model on the supervised Bert embeddings on validation, leaderboard or test set, run the following:

`run.sh` which calls `main_feature_embedding_lb.py`

Assumes that you have the data files in ```-d```, the embeddings files in ```--emb_path``` and the freshly trained MLP checkpoint in ```--checkpoint```
