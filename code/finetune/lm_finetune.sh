#!/bin/sh
export CUDA_VISIBLE_DEVICES=6
export TRAIN_FILE=/shared/2/projects/framing/data/lm_data_08-25/cased/train.raw
export TEST_FILE=/shared/2/projects/framing/data/lm_data_08-25/cased/dev.raw
export OUTPUT_DIR=/shared/2/projects/framing/models/finetune/roberta_cased_09-01-20/


python run_language_modeling.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=roberta \
    --model_name_or_path=roberta-base \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --num_train_epochs=60 \
    --block_size=64 \
    --per_gpu_train_batch_size=32   \


