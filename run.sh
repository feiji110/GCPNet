#### 1. Comparative Study
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

#### 2. Parameter Analysis
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


#### 3. Model evaluation
#!/bin/bash

hidden_features=("32" "48" "64" "96" "128")
for feature in "${hidden_features[@]}"; do
    python main.py --config_file ./config.yml --task_type train --dataset_name "surface" --hidden_features "$feature"
done

points=("100" "2000" "5000" "10000" "20000" "30000")
for point in "${points[@]}"; do
    python main.py --config_file ./config.yml --task_type train --dataset_name "surface" --points "$point"
done
