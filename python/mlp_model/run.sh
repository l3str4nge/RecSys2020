#!/bin/bash


#run scaler
#python scaler.py


dpath="/data/recsys2020/DL_Data"
opath="/data/recsys2020/DL_Output"
cpath="/data/recsys2020/DL_Checkpoints"

#run featurenet
#python main_feature.py --run_name featurenet -e 40 --path /data/recsys2020/DL_Data/data

python main_feature_lb.py -v Valid.sav -sp $opath"/val/featurenet_final_40" --path $dpath"/data" --checkpoint $cpath"/mlp_model/featurenet_final_40.ckpt"
python main_feature_lb.py -v Submit.sav -sp $opath"/submit/featurenet_final_40" --path $dpath"/data" --checkpoint $cpath"/mlp_model/featurenet_final_40.ckpt"
python main_feature_lb.py -v Test.sav -sp $opath"/test/featurenet_final_40" --path $dpath"/data" --checkpoint $cpath"/mlp_model/featurenet_final_40.ckpt"

#run embeddingnet-bert
#python main_feature_embedding.py --run_name bert --emb_type bert --net_arch embedding_net -e 30 --path /data/recsys2020/DL_Data/data
python main_feature_embedding_lb.py -v Valid.sav -sp $opath"/val/bert_final_30" --emb_type bert --path $dpath"/data" --checkpoint $cpath"/mlp_model/bert_final_30.ckpt" --net_arch embedding_net --emb_folder $dpath"/embeddings/bert"
python main_feature_embedding_lb.py -v Submit.sav -sp $opath"/submit/bert_final_30" --emb_type bert --path $dpath"/data" --checkpoint $cpath"/mlp_model/bert_final_30.ckpt" --net_arch embedding_net --emb_folder $dpath"/embeddings/bert"
python main_feature_embedding_lb.py -v Test.sav -sp $opath"/test/bert_final_30" --emb_type bert --path $dpath"/data" --checkpoint $cpath"/mlp_model/bert_final_30.ckpt" --net_arch embedding_net --emb_folder $dpath"/embeddings/bert"

#run embeddingnet-xlmr
#python main_feature_embedding.py --run_name xlmr --emb_type xlmr --net_arch embedding_net -e 50 --path /data/recsys2020/DL_Data/data --emb_folder /data/recsys2020/DL_Data/embeddings/xlm-r
python main_feature_embedding_lb.py -v Valid.sav -sp $opath"/val/xlmr_final_new_50" --emb_type xlmr --path $dpath"/data" --checkpoint $cpath"/mlp_model/xlmr_final_new50.ckpt" --net_arch embedding_net --emb_folder $dapth"/embeddings/xlm-r"
python main_feature_embedding_lb.py -v Submit.sav -sp $opath"/submit/xlmr_final_new_50" --emb_type xlmr --path $dpath"/data" --checkpoint $cpath"/mlp_model/xlmr_final_new50.ckpt" --net_arch embedding_net --emb_folder $dapth"/embeddings/xlm-r"
python main_feature_embedding_lb.py -v Test.sav -sp $opath"/test/xlmr_final_new_50" --emb_type xlmr --path $dpath"/data" --checkpoint $cpath"/mlp_model/xlmr_final_new50.ckpt" --net_arch embedding_net --emb_folder $dapth"/embeddings/xlm-r"

#run embeddingnet-xlmr with scheduler
#python main_feature_embedding.py --run_name xlmr --emb_type xlmr --net_arch embedding_net -e 50 --path /data/recsys2020/DL_Data/data --emb_folder /data/recsys2020/DL_Data/embeddings/xlm-r
python main_feature_embedding_lb.py -v Valid.sav -sp $opath"/val/xlmr_scheduler_27" --emb_type xlmr --path $dpath"/data" --checkpoint $cpath"/mlp_model/xlmr_scheduler_27.ckpt" --net_arch embedding_net --emb_folder $dapth"/embeddings/xlm-r"
python main_feature_embedding_lb.py -v Submit.sav -sp $opath"/submit/xlmr_scheduler_27" --emb_type xlmr --path $dpath"/data" --checkpoint $cpath"/mlp_model/xlmr_scheduler_27.ckpt" --net_arch embedding_net --emb_folder $dapth"/embeddings/xlm-r"
python main_feature_embedding_lb.py -v Test.sav -sp $opath"/test/xlmr_scheduler_27" --emb_type xlmr --path $dpath"/data" --checkpoint $cpath"/mlp_model/xlmr_scheduler_27.ckpt" --net_arch embedding_net --emb_folder $dapth"/embeddings/xlm-r"

#run embeddingnet-xlmr with residual network 
#python main_feature_embedding.py --run_name xlmr --emb_type xlmr --net_arch embedding_highway_net -e 52 --path /data/recsys2020/DL_Data/data --emb_folder /data/recsys2020/DL_Data/embeddings/xlm-r
python main_feature_embedding_lb.py -v Valid.sav -sp $opath"/val/xlmr_highway_52" --emb_type xlmr --path $dpath"/data" --checkpoint $cpath"/mlp_model/xlmr_highway_52.ckpt" --net_arch embedding_highway_net --emb_folder $dapth"/embeddings/xlm-r"
python main_feature_embedding_lb.py -v Submit.sav -sp $opath"/submit/xlmr_highway_52" --emb_type xlmr --path $dpath"/data" --checkpoint $cpath"/mlp_model/xlmr_highway_52.ckpt" --net_arch embedding_highway_net --emb_folder $dapth"/embeddings/xlm-r"
python main_feature_embedding_lb.py -v Test.sav -sp $opath"/test/xlmr_highway_52" --emb_type xlmr --path $dpath"/data" --checkpoint $cpath"/mlp_model/xlmr_highway_52.ckpt" --net_arch embedding_highway_net --emb_folder $dapth"/embeddings/xlm-r"

