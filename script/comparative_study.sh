#!/bin/bash
dataset_names=('surface' '2d' 'pt' 'mof' 'mp18')

for dataset_name in "${dataset_names[@]}"; do
    if [ "$dataset_name" == 'mp18' ]; then
        python main.py --config_file ./config.yml --task_type train --dataset_name "$dataset_name" --target_name 'band_gap' --hidden_features 64
    elif [ "$dataset_name" == 'surface' ]; then
        python main.py --config_file ./config.yml --task_type train --dataset_name "$dataset_name" --hidden_features 128
    else
        python main.py --config_file ./config.yml --task_type train --dataset_name "$dataset_name"
    fi
done