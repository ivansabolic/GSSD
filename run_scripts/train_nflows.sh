#!/bin/bash
cd "$(dirname "$0")"/..;

dataset=cifar10
num_classes=9
attack=badnets
poisoned_rate=0.1
target_label=0
resnet=resnet18_cifar

log_dir=experiments
experiment_name=${dataset}_${attack}_${poisoned_rate}_${target_label}_${resnet}
experiment_dir=${log_dir}/$experiment_name

model=nflow
steps=8
inflate_coef=2

gpu=5

target_dir=nflows_${model}_${steps}_${inflate_coef}
mkdir -p $experiment_dir/$target_dir

set -e
# per class
for oneclass in $(seq 0 $num_classes)
do
	name=MF_oneclass_${model}_${dataset}_${attack}_class-${oneclass}
	python train_features.py \
		--epochs 50 \
		--eval_every 50 \
		--check_every 25 \
		--batch_size 16 \
		--optimizer adam \
		--lr 1e-3  \
		--gamma 0.9975 \
		--warmup 5000  \
		--model $model \
		--steps $steps \
		--inflate_coef $inflate_coef \
		--dataset $dataset \
		--experiment_dir $experiment_dir \
		--oneclass $oneclass \
		--selfsup_backbone $resnet \
		--name $name \
		--project train_nflows \
		--num_workers 1 \
		--device cuda:$gpu;

	cp -r $experiment_dir/$name $experiment_dir/$target_dir/; 
	rm -r $experiment_dir/$name
done

# all classes
name=MF_allclasses_${model}
python train_features.py \
	--epochs 1 \
	--eval_every 50 \
	--check_every 5 \
	--batch_size 16 \
	--optimizer adam \
	--lr 1e-4  \
	--gamma 0.9975 \
	--warmup 5000  \
	--model $model \
	--steps $steps \
	--inflate_coef $inflate_coef \
	--dataset $dataset \
	--experiment_dir $experiment_dir \
	--selfsup_backbone $resnet \
	--name $name \
	--num_workers 1 \
	--device cuda:$gpu;


cp -r $experiment_dir/$name $experiment_dir/$target_dir/; 
rm -r $experiment_dir/$name
