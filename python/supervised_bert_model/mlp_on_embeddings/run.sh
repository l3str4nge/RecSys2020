#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python3 main_feature_embedding_lb.py \
cd ..-d=../../../recsys_submission_data/training_inference_data/ \
--emb_path=../../../recsys_submission_data/supervised_embeddings/bert/ \
--checkpoint=../../../recsys_submission_data/model_weights/featurenet_supervised_difflr_new_split_32.ckpt \
--emb_type=bert

