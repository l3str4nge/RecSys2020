#!/bin/bash


#run scaler
python scaler.py

#run featurenet
python main_feature.py --run_name featurenet -e 40
python main_feature_lb.py -v Valid.sav -sp out/val/featurenet
python main_feature_lb.py -v Submit.sav -sp out/submit/featurenet
python main_feature_lb.py -v Test.sav -sp out/test/featurenet

#run embeddingnet-bert
python main_feature_embedding.py --run_name bert --emb_type bert --net_arch embedding_net -e 30
python main_feature_embedding_lb.py -v Valid.sav -sp out/val/bert --emb_type bert --checkpoint ./checkpoint/bert_30.ckpt --net_arch embedding_net
python main_feature_embedding_lb.py -v Submit.sav -sp out/submit/bert --emb_type bert --checkpoint ./checkpoint/bert_30.ckpt --net_arch embedding_net
python main_feature_embedding_lb.py -v Test.sav -sp out/test/bert --emb_type bert --checkpoint ./checkpoint/bert_30.ckpt --net_arch embedding_net

#run embeddingnet-xlmr
python main_feature_embedding.py --run_name xlmr --emb_type xlmr --net_arch embedding_net -e 50
python main_feature_embedding_lb.py -v Valid.sav -sp out/val/xlmr --emb_type xlmr --checkpoint ./checkpoint/xlmr_50.ckpt --net_arch embedding_net
python main_feature_embedding_lb.py -v Submit.sav -sp out/submit/xlmr --emb_type xlmr --checkpoint ./checkpoint/bert_50.ckpt --net_arch embedding_net
python main_feature_embedding_lb.py -v Test.sav -sp out/test/xlmr --emb_type xlmr --checkpoint ./checkpoint/bert_50.ckpt --net_arch embedding_net

#run embeddingnet-bert with residual network 
python main_feature_embedding.py --run_name xlmr --emb_type xlmr --net_arch embedding_highway_net -e 52
python main_feature_embedding_lb.py -v Valid.sav -sp out/val/xlmr --emb_type xlmr --checkpoint ./checkpoint/xlmr_50.ckpt --net_arch embedding_highway_net
python main_feature_embedding_lb.py -v Submit.sav -sp out/submit/xlmr --emb_type xlmr --checkpoint ./checkpoint/bert_50.ckpt --net_arch embedding_highway_net
python main_feature_embedding_lb.py -v Test.sav -sp out/test/xlmr --emb_type xlmr --checkpoint ./checkpoint/bert_50.ckpt --net_arch embedding_highway_net
