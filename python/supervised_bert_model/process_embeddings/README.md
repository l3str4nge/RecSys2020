1 - Compile all raw embeddings pickle file into a unique pickle file:

`compile_embeddings.py`

The ```source_path``` variable is the output of the `extract_embeddings.py` from the [bert_mlp](https://github.com/layer6ai-labs/RecSys2020/tree/master/python/supervised_bert_model/bert_mlp) module. 

2 - Transform the unique embeddings pickle file into a memmap file:

For training/validation sets:

`get_emb.py`

For public leaderboard/test sets:

`get_emb_submit.py`
