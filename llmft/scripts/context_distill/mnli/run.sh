#!/usr/bin/env bash
# args: task_name, max_train_samples, epochs, warmup_ratio, bsz, num_gpus, learning_rate, model_name_or_path, port
model_name_or_path="facebook/opt-125m"
port=11111
# we log at the end of every epoch
logging_steps=$((max_train_samples / (bsz * num_gpus)))

for data_seed in "0" "1" "2" "3" "4" "5" "6" "7" "8" "9"
do
    $PYTHON_BIN/deepspeed \
        --include localhost:0,1,2,3,4,5,6,7 \
        --master_port $port \
        $PROJECT_DIR/ft.py \
        --wandb_project_name llmft-experiments \
        --wandb_group_name mnli-ft-context-distillation \
        --model_name_or_path $model_name_or_path \
        --cache_dir "./cache_model" \
        --task_name mnli \
        --target_tokens "ĠYes,ĠNo" \
        --pattern "{text1} question: {text2} Yes or No?" \
        --dataset_cache_dir "./cache_dataset" \
        --max_seq_length 256 \
        --max_context_len 1024 \
        --output_dir "./output" \
        --overwrite_output_dir \
        --do_train \
        --max_train_samples 16 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --num_train_epochs 40 \
        --warmup_ratio 0.5 \
        --logging_first_step false \
        --logging_steps -1 \
        --learning_rate 1e-5 \
        --weight_decay 0.0 \
        --do_eval \
        --evaluation_strategy epoch \
        --per_device_eval_batch_size 10 \
        --eval_on_paws_qqp \
        --paws_qqp_file llmft/data/paws_qqp/dev_and_test.tsv \
        --save_strategy no \
        --seed 0 \
        --data_seed 0 \
        --deepspeed $PROJECT_DIR/deepspeed_configs/ds_config_zero3.json \
        --context_distill \
        --num_shots 16 \
        --context_targets "ĠYes,ĠNo" \
        --input_pattern "{text1} question: {text2} Yes or No?" \
        --separate_shots_by "\n\n" \
        --target_prefix " answer: " \
        --balanced  \
        --fp16 \
        --shuffle  \
        --report_to "none" 
done