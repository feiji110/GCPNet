#!/bin/bash

hidden_features=("32" "48" "64" "96" "128")
for feature in "${hidden_features[@]}"; do
    python main.py --config_file ./config.yml --task_type train --dataset_name "surface" --hidden_features "$feature"
done

points=("100" "2000" "5000" "10000" "20000" "30000")
for point in "${points[@]}"; do
    python main.py --config_file ./config.yml --task_type train --dataset_name "surface" --points "$point"
done