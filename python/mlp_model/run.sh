#!/bin/bash


#run scaler
python scaler.py

#run featurenet
python main_feature.py --run_name featurenet
python main_feature_lb.py -v Valid.sav -sp out/val/featurenet
python main_feature_lb.py -v Submit.sav -sp out/submit/featurenet
python main_feature_lb.py -v Test.sav -sp out/test/featurenet

#run embeddingnet-bert
#TODO need to specify directories
python main_feature_embedding.py --run_name bert --emb_type bert 
python main_feature_embedding_lb.py -v Valid.sav -ps out/val/bert --emb_type bert --checkpoint ./checkpoint/bert_30.ckpt
python main_feature_embedding_lb.py -v Submit.sav -ps out/submit/bert --emb_type bert --checkpoint ./checkpoint/bert_30.ckpt
python main_feature_embedding_lb.py -v Test.sav -ps out/test/bert --emb_type bert --checkpoint ./checkpoint/bert_30.ckpt

#run embeddingnet-xlmr
python main_feature_embedding.py --run_name xlmr --emb_type xlmr 
python main_feature_embedding_lb.py -v Valid.sav -ps out/val/xlmr --emb_type xlmr --checkpoint ./checkpoint/xlmr_50.ckpt
python main_feature_embedding_lb.py -v Submit.sav -ps out/submit/xlmr --emb_type xlmr --checkpoint ./checkpoint/bert_50.ckpt
python main_feature_embedding_lb.py -v Test.sav -ps out/test/xlmr --emb_type xlmr --checkpoint ./checkpoint/bert_50.ckpt

##TODO need to add xlmr residual version
