#!/usr/bin/env bash

export PROJECT_DIR=llmft
source $PROJECT_DIR/scripts/misc/setup.sh
conda init
conda activate cs7643-finalproj

# args: task_name, max_train_samples, bsz, num_gpus, model_name_or_path, port
ls
bash scripts/context_distill/cola/run.sh mnli 64 32 1 facebook/opt-125m 60000