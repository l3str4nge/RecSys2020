#!/bin/bash
# we will log every logging-steps*per-gpu-train-batch-size*gradient-accumulation-steps samples
# the tiny validation file has 100,000 lines

cd ..
python -m torch.distributed.launch \
--nproc_per_node 8 run_language_modeling.py \
--model_name_or_path=xlm-roberta-large \
--output_dir=/data/xlm-r/checkpoints/large_aws \
--do_train \
--do_eval \
--train_data_file=/data/unsupervised_data/xlmr_TrainLB.p \
--eval_data_file=/data/unsupervised_data/xlmr_Val.p \
--mlm \
--block_size=100 \
--save_total_limit=40 \
--save_steps=2000 \
--logging_steps=2000 \
--num_train_epochs=1 \
--learning_rate=0.00005 \
--warmup_steps=4000 \
--gradient_accumulation_steps=2 \
--per_gpu_train_batch_size=24 \
--per_gpu_eval_batch_size=24 \
--evaluate_during_training \
--line_by_line \
--overwrite_output_dir