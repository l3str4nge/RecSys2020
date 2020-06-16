#!/bin/bash


# to train:
python main_feature_embedding.py --emb_type bert --path /data/recsys2020/DL_Data/data --emb_path /data/recsys2020/DL_Data/embeddings/bert/supervised

# to infer results on validation, leaderboard (submit) and test:
python main_feature_embedding_lb.py -v Valid.sav -sp /data/recsys2020/DL_Output/val/supervised_bert_model --emb_type bert --path /data/recsys2020/DL_Data/data --checkpoint /data/recsys2020/DL_Checkpoints/supervised_bert_model/featurenet_supervised_difflr_new_split_32.ckpt --emb_path /data/recsys2020/DL_Data/embeddings/bert/supervised
python main_feature_embedding_lb.py -v Submit.sav -sp /data/recsys2020/DL_Output/submit/supervised_bert_model --emb_type bert --path /data/recsys2020/DL_Data/data --checkpoint /data/recsys2020/DL_Checkpoints/supervised_bert_model/featurenet_supervised_difflr_new_split_32.ckpt --emb_path /data/recsys2020/DL_Data/embeddings/bert/supervised
python main_feature_embedding_lb.py -v Test.sav -sp /data/recsys2020/DL_Output/test/supervised_bert_model --emb_type bert --path /data/recsys2020/DL_Data/data --checkpoint /data/recsys2020/DL_Checkpoints/supervised_bert_model/featurenet_supervised_difflr_new_split_32.ckpt --emb_path /data/recsys2020/DL_Data/embeddings/bert/supervised



