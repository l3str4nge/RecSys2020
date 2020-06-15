#!/bin/bash
# we will log every logging-steps*per-gpu-train-batch-size*gradient-accumulation-steps samples
# the tiny validation file has 100,000 lines

cd ..
CUDA_VISIBLE_DEVICES=0,1 python run_language_model.py \
--model_type=bert \
--output_dir=/home/layer6/recsys/unsupervised_kevin/checkpoints/pretrained_fp16 \
--model_name_or_path=/home/layer6/recsys/pretrained_models/model \
--do_train \
--train_data_file=/home/layer6/recsys/unsupervised_data/multi_all_langs_unique_tweets.txt \
--eval_data_file=/home/layer6/recsys/unsupervised_data/multi_all_langs_unique_tweets_leaderboard.txt \
--tokenizer_name=/home/layer6/recsys/pretrained_models/model \
--mlm \
--mlm_probability=0.15 \
--block_size=100 \
--save_total_limit=20 \
--save_steps=2000 \
--logging_steps=2000 \
--num_train_epochs=2 \
--learning_rate=0.0001 \
--warmup_steps=10000 \
--gradient_accumulation_steps=2 \
--per_gpu_train_batch_size=70 \
--per_gpu_eval_batch_size=70 \
--evaluate_during_training \
--line_by_line \
--overwrite_output_dir \
--use_bucket_iterator=0 \
--fp16
