#!/bin/bash

gpu=0

cd "$(dirname "$0")"/..;
python train_simclr.py \
	--epochs 1 \
	--device cuda:$gpu
