# steps to reproduce

Note: we used 2 different language models in our pipeline, Bert base and XLM-Roberta. Below are instructions for reproducing
the pipeline for data splitting for XLM-Roberta. The Bert pipeline is almost the same, and the files have been omitted
for simplicity.


The input of this pipeline is the transformed training data Train.sav. The starting point of the pipeline is in split.py (explained below). The ending point of the pipeline is in split_embeddings.py.py (explained below). There are 2 outputs of this
pipeline. First is the training data which has been split into 3 chunks (due to RAM constraint of training machines). The second
is the language model embeddings which have also been split into their appropriate 3 chunks.


The outputs of this pipeline are used to train the multilayer perceptron (MLP) model with embeddings.


1. split.py - split training data into 3 chunks
2. split_embeddings.py - split embddings into 3 chunks based on tweet ids from data chunks
3. split_val.py, split_submit.py, split_embeddings_submit.py, submit_embeddings_test.py are basically doing the same thing but for different sets.