#!/usr/bin/env bash

# args: task_name, max_train_samples, epochs, warmup_ratio, bsz, num_gpus, learning_rate, model_name_or_path, port
max_train_samples=64
epochs=5
warmup_ratio=1
bsz=32
num_gpus=1
learning_rate=0.0001
model_name_or_path="facebook/opt-125m"
port=11111
echo model_name_or_path: $model_name_or_path
# we log at the end of every epoch
logging_steps=$((max_train_samples / (bsz * num_gpus)))

for seed in "0"
do
    for data_seed in "0" "1" "2" "3" "4" "5" "6" "7" "8" "9"
    do
        python ft.py \
        --wandb_project_name llmft-experiments \
        --wandb_group_name vanilla-ft \
        --model_name_or_path $model_name_or_path \
        --cache_dir /cache_model \
        --task_name mnli \
        --pattern "{text1} ?" \
        --dataset_cache_dir /cache_data \
        --max_seq_length 256 \
        --output_dir /output \
        --overwrite_output_dir \
        --do_train \
        --max_train_samples $max_train_samples \
        --per_device_train_batch_size $bsz \
        --gradient_accumulation_steps 1 \
        --num_train_epochs $epochs \
        --warmup_ratio $warmup_ratio \
        --logging_first_step true \
        --logging_steps $logging_steps \
        --learning_rate $learning_rate \
        --weight_decay 0.0 \
        --do_eval \
        --evaluation_strategy epoch \
        --per_device_eval_batch_size 10 \
        --eval_on_cola_ood \
        --cola_ood_file data/cola_ood/dev.tsv \
        --save_strategy no \
        --seed $seed \
        --data_seed $data_seed \
        --report_to "none"
    done
done