#!/bin/bash


# to infer results on validation, leaderboard (Submit) and test:
python main_feature_embedding_lb.py -v Valid.sav -sp out/val --emb_type bert --path /data/recsys2020/DL_Data/data -- checkpoint /data/recsys2020/DL_Checkpoints/supervised_bert_model/featurenet_supervised_difflr_new_split_32.ckpt --emb_path /data/recsys2020/DL_Data/embeddings/bert/supervised/
python main_feature_embedding_lb.py -v Submit.sav -sp out/submit --emb_type bert --path /data/recsys2020/DL_Data/data -- checkpoint /data/recsys2020/DL_Checkpoints/supervised_bert_model/featurenet_supervised_difflr_new_split_32.ckpt --emb_path /data/recsys2020/DL_Data/embeddings/bert/supervised/
python main_feature_embedding_lb.py -v Test.sav -sp out/test --emb_type bert --path /data/recsys2020/DL_Data/data -- checkpoint /data/recsys2020/DL_Checkpoints/supervised_bert_model/featurenet_supervised_difflr_new_split_32.ckpt --emb_path /data/recsys2020/DL_Data/embeddings/bert/supervised/



