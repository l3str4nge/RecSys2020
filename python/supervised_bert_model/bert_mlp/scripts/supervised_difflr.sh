#!/bin/bash
# we will log every logging-steps*per-gpu-train-batch-size*gradient-accumulation-steps samples

cd ..
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_supervised.py \
--model_type=bert \
--output_dir=/home/layer6/recsys/unsupervised_kevin/checkpoints/supervised_difflr \
--model_name_or_path=/home/layer6/recsys/unsupervised_kevin/checkpoints/pretrained_fp16_epoch2/checkpoint-288000 \
--do_train \
--tokenizer_name=/home/layer6/recsys/pretrained_models/model \
--mlm \
--block_size=100 \
--save_total_limit=20 \
--evaluate_during_training \
--line_by_line \
--overwrite_output_dir \
--use_bucket_iterator=0 \
--save_steps=7000 \
--logging_steps=7000 \
--gradient_accumulation_steps=1 \
--per_gpu_train_batch_size=320 \
--per_gpu_eval_batch_size=2048 \
--num_train_epochs=1000 \
--learning_rate=0.00001 \
--mlp_learning_rate=0.00005 \
--warmup_steps=100 \
--emb_size=768 \
--mlp_checkpoint=/home/layer6/recsys/MLPHojin/checkpoint/featurenet_15_newfeat_15.ckpt \
--mlp_data_path=/home/layer6/recsys/kevin_data \
--mlp_train=Train.sav \
--mlp_val=Valid.sav \
--tweet_id_to_text_file=/home/layer6/recsys/clean/tweets_only/all_tokens.p \
--fp16
