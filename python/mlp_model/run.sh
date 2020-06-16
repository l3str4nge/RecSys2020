#!/bin/bash


#run scaler
python scaler.py

#run featurenet
python main_feature.py --run_name featurenet -e 40 --path /data/recsys2020/DL_Data/data
python main_feature_lb.py -v Valid.sav -sp out/val/featurenet --path /data/recsys2020/DL_Data/data
python main_feature_lb.py -v Submit.sav -sp out/submit/featurenet --path /data/recsys2020/DL_Data/data
python main_feature_lb.py -v Test.sav -sp out/test/featurenet --path /data/recsys2020/DL_Data/data

#run embeddingnet-bert
python main_feature_embedding.py --run_name bert --emb_type bert --net_arch embedding_net -e 30 --path /data/recsys2020/DL_Data/data
python main_feature_embedding_lb.py -v Valid.sav -sp out/val/bert --emb_type bert --path /data/recsys2020/DL_Data/data --checkpoint /data/recsys2020/DL_Checkpoints/bert_30.ckpt --net_arch embedding_net --emb_folder /data/recsys2020/DL_Data/embeddings/bert
python main_feature_embedding_lb.py -v Submit.sav -sp out/submit/bert --emb_type bert --path /data/recsys2020/DL_Data/data --checkpoint /data/recsys2020/DL_Checkpoints/bert_30.ckpt --net_arch embedding_net --emb_folder /data/recsys2020/DL_Data/embeddings/bert
python main_feature_embedding_lb.py -v Test.sav -sp out/test/bert --emb_type bert --path /data/recsys2020/DL_Data/data --checkpoint /data/recsys2020/DL_Checkpoints/bert_30.ckpt --net_arch embedding_net --emb_folder /data/recsys2020/DL_Data/embeddings/bert

#run embeddingnet-xlmr
python main_feature_embedding.py --run_name xlmr --emb_type xlmr --net_arch embedding_net -e 50 --path /data/recsys2020/DL_Data/data --emb_folder /data/recsys2020/DL_Data/embeddings/xlm-r
python main_feature_embedding_lb.py -v Valid.sav -sp out/val/xlmr --emb_type xlmr --path /data/recsys2020/DL_Data/data --checkpoint /data/recsys2020/DL_Checkpoints/xlmr_50.ckpt --net_arch embedding_net --emb_folder /data/recsys2020/DL_Data/embeddings/xlm-r
python main_feature_embedding_lb.py -v Submit.sav -sp out/submit/xlmr --emb_type xlmr --path /data/recsys2020/DL_Data/data --checkpoint /data/recsys2020/DL_Checkpoints/bert_50.ckpt --net_arch embedding_net --emb_folder /data/recsys2020/DL_Data/embeddings/xlm-r
python main_feature_embedding_lb.py -v Test.sav -sp out/test/xlmr --emb_type xlmr --path /data/recsys2020/DL_Data/data --checkpoint /data/recsys2020/DL_Checkpoints/bert_50.ckpt --net_arch embedding_net --emb_folder /data/recsys2020/DL_Data/embeddings/xlm-r

#run embeddingnet-bert with residual network 
python main_feature_embedding.py --run_name xlmr --emb_type xlmr --net_arch embedding_highway_net -e 52 --path /data/recsys2020/DL_Data/data --emb_folder /data/recsys2020/DL_Data/embeddings/xlm-r
python main_feature_embedding_lb.py -v Valid.sav -sp out/val/xlmr --emb_type xlmr --path /data/recsys2020/DL_Data/data --checkpoint /data/recsys2020/DL_Checkpoints/xlmr_50.ckpt --net_arch embedding_highway_net --emb_folder /data/recsys2020/DL_Data/embeddings/xlm-r
python main_feature_embedding_lb.py -v Submit.sav -sp out/submit/xlmr --emb_type xlmr --path /data/recsys2020/DL_Data/data --checkpoint /data/recsys2020/DL_Checkpoints/bert_50.ckpt --net_arch embedding_highway_net --emb_folder /data/recsys2020/DL_Data/embeddings/xlm-r
python main_feature_embedding_lb.py -v Test.sav -sp out/test/xlmr --emb_type xlmr --path /data/recsys2020/DL_Data/data --checkpoint /data/recsys2020/DL_Checkpoints/bert_50.ckpt --net_arch embedding_highway_net --emb_folder /data/recsys2020/DL_Data/embeddings/xlm-r
