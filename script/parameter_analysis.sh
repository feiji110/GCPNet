#!/bin/bash

dropout_rates=("0" "0.05" "0.15" "0.20" "0.25")
for dropout_rate in "${dropout_rates[@]}"; do
    python main.py --config_file ./config.yml --task_type CV --dataset_name 2d --project_name "GCPNet_2d_dp_$dropout_rate" --epochs 300 --dropout_rate "$dropout_rate"
done

batch_sizes=("64" "100" "128" "256" "512")
for batch_size in "${batch_sizes[@]}"; do
    python main.py --config_file ./config.yml --task_type CV --dataset_name 2d --project_name "GCPNet_2d_bs_$batch_size" --epochs 300 --batch_size "$batch_size"
done

learning_rates=("0.001" "0.002" "0.003" "0.004" "0.005")
for learning_rate in "${learning_rates[@]}"; do
    python main.py --config_file ./config.yml --task_type CV --dataset_name 2d --project_name "GCPNet_2d_lr_$learning_rate" --epochs 300 --lr "$learning_rate"
done

Ns=("0" "1" "2" "3" "4")
for N in "${Ns[@]}"; do
    python main.py --config_file ./config.yml --task_type CV --dataset_name 2d --project_name "GCPNet_2d_N_$N" --epochs 300 --firstUpdateLayers "$N" --secondUpdateLayers "$N"
done