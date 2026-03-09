#!/bin/bash

gpu=5
dataset=cifar10
attack=wanet
poisoned_rate=0.1
target_label=0
resnet=resnet18

log_folder=experiments
name=$log_folder/${dataset}_${attack}_${poisoned_rate}_${target_label}_${resnet}

poisoned_set=./data/${dataset}_poisoned_${poisoned_rate}_${target_label}
python extract_features_from_simclr.py \
	--dataset $dataset \
	--resnet $resnet \
	--attack $attack \
	--poisoned_rate $poisoned_rate \
	--target_label $target_label \
	--experiment_dir $name \
	--train_poisoned_set ${poisoned_set}_train \
	--test_poisoned_set ${poisoned_set}_test \
	--projection_dim 128 \
	--device cuda:$gpu 
