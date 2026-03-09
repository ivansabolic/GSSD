#!/bin/bash
cd "$(dirname "$0")"/..;

dataset=cifar100
attack=badnets
poisoned_rate=0.1
target_label=0
resnet=resnet18

log_dir=experiments
experiment_name=${dataset}_${attack}_${poisoned_rate}_${target_label}_${resnet}

model=nflow
steps=2
inflate_coef=1

gpu=2
alpha=0.05
beta_nd=0.6
beta_d=0.05
lambda=0.75

cthr=$alpha
pthr=$alpha

indices=$log_dir/$experiment_name/nflows_${model}_${steps}_${inflate_coef}/indices_cthr-${cthr}_pthr-${pthr}_beta_nd-${beta_nd}_beta_d-${beta_d}_lambda-${lambda}.npz
# indices=""


epochs=200
poisoned_set=data/${dataset}_poisoned_${poisoned_rate}_${target_label}
python -m retraining \
	--experiment_dir $log_dir/$experiment_name \
	--dataset $dataset \
	--target_label $target_label \
	--indices $indices \
	--attack $attack \
	--epochs $epochs \
	--poisoned_set ${poisoned_set}_train \
	--test_poisoned_set ${poisoned_set}_test \
	--poisoned_rate $poisoned_rate \
	--device cuda:$gpu



