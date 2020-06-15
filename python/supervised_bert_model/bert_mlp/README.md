1 - Train the joint Bert+MLP model (where Bert is initialized from unsupervised pre-training, and MLP from training on unsupervised embeddings):

`run_supervised.py`

Assumes that you have a checkpoint of unsupervised Bert at ```--model_name_or_path```, a checkpoint of the pretraining MLP at ```--mlp_checkpoint```, the data at ```--mlp_data_path``` and the text at ```--tweet_id_to_text_file```

2 - Extract the embeddings of the Bert+MLP model:

`extract_embeddings.py`
