#!/bin/bash
#
# LEBD Pipeline — CIFAR-10 Experiments (Table 1)
# ================================================
# Reproduces the backdoor defense results from:
#   "Backdoor Defense through Self-Supervised and Generative Learning" (BMVC 2024)
#
# The pipeline has 6 steps per attack:
#   1. Train SimCLR (self-supervised feature extractor)
#   2. Extract features from SimCLR
#   3. Train normalizing flows (per-class density models)
#   4. Detect poisoned samples using likelihood scores
#   5. Retrain classifier on cleansed data
#   6. Fine-tune with relabeled suspicious samples
#
# Prerequisites:
#   uv (https://docs.astral.sh/uv/) — installs dependencies from pyproject.toml
#
# Usage:
#   bash run_scripts/run_cifar10_experiments.sh [GPU_ID]
#
# Examples:
#   bash run_scripts/run_cifar10_experiments.sh        # Run on GPU 0
#   bash run_scripts/run_cifar10_experiments.sh 2      # Run on GPU 2
#

set -e
cd "$(dirname "$0")"/..

# ==============================================================================
# USER CONFIGURATION — Edit this section to customize your experiment
# ==============================================================================

# GPU to use (can also be passed as first argument)
gpu=${1:-0}

# Dataset settings
dataset=cifar10
target_label=0
num_classes=10

# Attacks to run. Comment/uncomment lines to select which attacks to evaluate.
# Format: "attack_name  poison_rate  backbone  image_size  clean_thr  poison_thr"
#
#   attack_name : badnets | blended | wanet | labelconsistent | issba | sig | ...
#   poison_rate : fraction of training set that is poisoned (e.g. 0.1 = 10%)
#   backbone    : resnet18_cifar (CIFAR-sized ResNet) or resnet18 (ImageNet-sized)
#   image_size  : input resolution (32 for CIFAR, 224 for ImageNet)
#   clean_thr   : likelihood threshold for clean samples (cthr)
#   poison_thr  : likelihood threshold for poisoned samples (pthr)
attacks=(
    "badnets  0.1  resnet18_cifar  32  0.3  0.15"
    "blended  0.1  resnet18_cifar  32  0.3  0.15"
    "wanet    0.1  resnet18_cifar  32  0.3  0.15"
)

# Normalizing flow architecture
flow_model=nflow       # Flow model type
flow_steps=2           # Number of flow steps
flow_inflate_coef=1    # Inflation coefficient

# Detection parameters (shared across attacks)
beta_nd=0.6            # Weight for non-detected class likelihoods
beta_d=0.05            # Weight for detected class likelihoods
lambda_=0.75           # Mixing parameter for clean/poisoned scores

# ==============================================================================
# END OF USER CONFIGURATION — No need to edit below this line
# ==============================================================================

device="cuda:$gpu"

echo "========================================================================"
echo "LEBD Pipeline — CIFAR-10 Experiments"
echo "Device: $device"
echo "Attacks: ${#attacks[@]} configured"
echo "========================================================================"
echo ""

for attack_config in "${attacks[@]}"; do
    read -r attack poisoned_rate resnet image_size cthr pthr <<< "$attack_config"

    echo "========================================================================"
    echo " Attack: $attack | Poison rate: $poisoned_rate | Backbone: $resnet"
    echo "========================================================================"

    name=${dataset}_${attack}_${poisoned_rate}_${target_label}_${resnet}
    experiment_dir=experiments/$name
    poisoned_set=data/${dataset}_poisoned_${poisoned_rate}_${target_label}
    target_dir=nflows_${flow_model}_${flow_steps}_${flow_inflate_coef}

    mkdir -p $experiment_dir

    # ---- Step 1/6: Train SimCLR ----
    echo "[Step 1/6] Training SimCLR..."
    uv run python train_simclr.py \
        --dataset $dataset \
        --resnet $resnet \
        --attack $attack \
        --poisoned_rate $poisoned_rate \
        --target_label $target_label \
        --train_poisoned_set ${poisoned_set}_train \
        --test_poisoned_set ${poisoned_set}_test \
        --epochs 100 \
        --batch_size 256 \
        --image_size $image_size \
        --projection_dim 128 \
        --device $device

    # ---- Step 2/6: Extract features ----
    echo "[Step 2/6] Extracting SimCLR features..."
    uv run python extract_features_from_simclr.py \
        --dataset $dataset \
        --resnet $resnet \
        --attack $attack \
        --poisoned_rate $poisoned_rate \
        --target_label $target_label \
        --experiment_dir $experiment_dir \
        --train_poisoned_set ${poisoned_set}_train \
        --test_poisoned_set ${poisoned_set}_test \
        --projection_dim 128 \
        --image_size $image_size \
        --device $device

    # ---- Step 3/6: Train normalizing flows ----
    echo "[Step 3/6] Training normalizing flows (${num_classes} per-class + 1 all-class)..."
    mkdir -p $experiment_dir/$target_dir

    for oneclass in $(seq 0 $((num_classes - 1))); do
        flow_name=MF_oneclass_${flow_model}_${dataset}_${attack}_class-${oneclass}
        uv run python train_features.py \
            --epochs 50 \
            --eval_every 50 \
            --check_every 50 \
            --batch_size 16 \
            --optimizer adam \
            --lr 1e-3 \
            --gamma 1.0 \
            --model $flow_model \
            --steps $flow_steps \
            --inflate_coef $flow_inflate_coef \
            --dataset $dataset \
            --experiment_dir $experiment_dir \
            --oneclass $oneclass \
            --selfsup_backbone $resnet \
            --name $flow_name \
            --project train_nflows \
            --num_workers 1 \
            --projection_dim 128 \
            --device $device

        cp -r $experiment_dir/$flow_name $experiment_dir/$target_dir/
        rm -r $experiment_dir/$flow_name
    done

    # All-classes flow (0 epochs = just initialize, used as baseline)
    flow_name=MF_allclasses_${flow_model}
    uv run python train_features.py \
        --epochs 0 \
        --eval_every 50 \
        --check_every 5 \
        --batch_size 16 \
        --optimizer adam \
        --lr 1e-3 \
        --gamma 1.0 \
        --model $flow_model \
        --steps $flow_steps \
        --inflate_coef $flow_inflate_coef \
        --dataset $dataset \
        --experiment_dir $experiment_dir \
        --selfsup_backbone $resnet \
        --name $flow_name \
        --num_workers 1 \
        --projection_dim 128 \
        --device $device

    cp -r $experiment_dir/$flow_name $experiment_dir/$target_dir/
    rm -r $experiment_dir/$flow_name

    # ---- Step 4/6: Detect poisoned samples ----
    echo "[Step 4/6] Detecting poisoned samples (cthr=$cthr, pthr=$pthr)..."
    uv run python extract_indices_from_nflows.py \
        --dataset $dataset \
        --attack $attack \
        --experiment_dir $experiment_dir \
        --original_dataset_dir ./data \
        --target_label $target_label \
        --poisoned_set ${poisoned_set}_train \
        --selfsup_backbone $resnet \
        --model $flow_model \
        --steps $flow_steps \
        --inflate_coef $flow_inflate_coef \
        --clean_threshold $cthr \
        --poisoned_threshold $pthr \
        --beta_nd $beta_nd \
        --beta_d $beta_d \
        --lambda_ $lambda_ \
        --projection_dim 128 \
        --device $device

    # ---- Step 5/6: Retrain on cleansed data ----
    echo "[Step 5/6] Retraining classifier on cleansed data (200 epochs)..."
    indices_path=$experiment_dir/$target_dir/indices_cthr-${cthr}_pthr-${pthr}_beta_nd-${beta_nd}_beta_d-${beta_d}_lambda-${lambda_}.npz

    uv run python retraining.py \
        --experiment_dir $experiment_dir \
        --dataset $dataset \
        --dataset_dir ./data \
        --target_label $target_label \
        --indices $indices_path \
        --attack $attack \
        --epochs 200 \
        --poisoned_set ${poisoned_set}_train \
        --test_poisoned_set ${poisoned_set}_test \
        --poisoned_rate $poisoned_rate \
        --defend \
        --clean_only \
        --work_dir $experiment_dir/$target_dir \
        --device $device

    # ---- Step 6/6: Fine-tune with relabeled samples ----
    echo "[Step 6/6] Fine-tuning with relabeled suspicious samples (2 epochs)..."
    uv run python retraining.py \
        --experiment_dir $experiment_dir \
        --dataset $dataset \
        --dataset_dir ./data \
        --target_label $target_label \
        --indices $indices_path \
        --attack $attack \
        --epochs 2 \
        --poisoned_set ${poisoned_set}_train \
        --test_poisoned_set ${poisoned_set}_test \
        --poisoned_rate $poisoned_rate \
        --defend \
        --finetune_with_relabeled \
        --work_dir $experiment_dir/$target_dir \
        --device $device

    echo ""
    echo "Finished: $attack"
    echo ""
done

echo "========================================================================"
echo "All experiments complete!"
echo "Results are in: experiments/cifar10_<attack>_*/defense_clean_*/log.txt"
echo "========================================================================"
