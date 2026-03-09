#!/bin/bash

cd "$(dirname "$0")"/..;

dataset=cifar10
num_classes=10
attack=badnets
poisoned_rate=0.001
target_label=0
resnet=resnet18

log_dir=experiments
experiment_name=${dataset}_${attack}_${poisoned_rate}_${target_label}_${resnet}

model=nflow
steps=8
inflate_coef=2

alpha=0.15
beta_nd=0.6
beta_d=0.05
lambda=0.75

cthr=$alpha
pthr=$alpha

gpu=0
poisoned_set=data/${dataset}_poisoned_${poisoned_rate}_${target_label}_train
python extract_indices_from_nflows.py \
	--dataset $dataset \
	--attack $attack \
	--experiment_dir $log_dir/$experiment_name \
	--original_dataset_dir ./data \
	--num_classes $num_classes \
	--target_label $target_label \
	--poisoned_set $poisoned_set \
	--selfsup_backbone $resnet \
	--model $model \
	--steps $steps \
	--inflate_coef $inflate_coef \
	--clean_threshold $cthr \
	--poisoned_threshold $pthr \
	--beta_nd $beta_nd \
	--beta_d $beta_d \
	--lambda_ $lambda \
	--device cuda:$gpu;
