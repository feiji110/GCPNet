# GCPNet

Github repository for our paper - **"GCPNet: An Interpretable Generic Crystal Pattern Graph Neural Network for Predicting Material Properties."**

# Table of Contents
* [Software Architecture](#architecture)
* [Necessary Installations](#installation)
* [Datasets](#dataset)
* [Usage](#usage)
* [Rreproducibility](#reproducibility)
* [Contributors](#contributors)
* [Acknowledgement](#acknowledgement)

## Software Architecture
![Alt text](figs/1.jpg)

We have developed a novel Generic Crystal Pattern graph neural Network (GCPNet) model, which is based on crystal pattern graphs and utilizes the Graph Convolutional Attention Operator (GCAO). This model effectively learns the atoms and their intricate interactions within materials, resulting in a significant improvement in the prediction accuracy of various material properties. The file descriptions in the software code repository are as follows
```
.
├── README.md
├── baseModule.py             # baseModule for construct the model
├── config.yml                # default settings
├── figs 
│   └── image.jpg
├── main.py                   # main file for running the code
├── model.py                  # GCPNet model for training
└── utils
    ├── dataset_utils.py      # dataset utils for loading data
    ├── fastprogress.py       # fastprogress for progress bar
    ├── flags.py              # flags for command line arguments
    ├── helpers.py            # helpers for construct crystal pattern graph
    ├── keras_callbacks.py    # keras_callbacks for training
    ├── node_representations  
    │   └── atom_init.json    # the initial atom representations
    ├── train_utils.py        # train_utils for training
    └── transforms.py         # pyg transforms for data processing
```````
<a name="installation"></a>
## Necessary Installations
We use the PyTorch Framework for our code. Please install the following packages if not already installed.Also you can a virtual environment using conda or pip for this purpose (recommended). We will give you a brief example on how to install the packages.

1. Create the virtual environment:
    ```bash
    conda create -n py310 python=3.10
    conda activate py310
    ```
2. Install the packages use Conda:
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    conda install --channel conda-forge pymatgen
    conda install --channel conda-forge ase
    conda install pyg -c pyg
    conda install pandas
    conda install -c conda-forge  wandb torchmetrics accelerate
    conda install -y -c conda-forge tensorboard 
    ```
3. Install the other packages use pip:
    ```bash
    pip install torch_sparse -f https://pytorch-geometric.com/whl/torch_sparse-0.6.17+pt20cu118-cp310-cp310-linux_x86_64.whl
    pip install torch_scatter -f https://pytorch-geometric.com/whl/torch_scatter-2.1.1+pt20cu118-cp310-cp310-linux_x86_64.whl
    pip install torch_cluster -f https://pytorch-geometric.com/whl/torch_cluster-1.6.1+pt20cu118-cp310-cp310-linux_x86_64.whl
    pip install torch_spline_conv -f https://pytorch-geometric.com/whl/torch_spline_conv-1.2.2+pt20cu118-cp310-cp310-linux_x86_64.whl
    ```

<a name="dataset"></a>
## Datasets
The experimental dataset is stored at [GCPNet](https://figshare.com/articles/dataset/GCPNet_data/23957907). Please download the dataset and place it in the code repository in the following format:

```
.
├── 2d
│   └── raw
│       └── 2d.2023.5.19.json.zip
├── mof
│   └── raw
│       └── mof.2023.5.19.json.zip
├── mp18
│   └── raw
│       └── mp.2018.6.1.json.zip
├── pt
│   └── raw
│       └── pt.2023.5.19.json.zip
└── surface
    └── raw
        └── surface.2023.5.19.json.zip
```

<a name="usage"></a>
## Usage

### I. Training
To train a model on a dataset, the command should be something like the following:
```bash
python main.py --config_file ./config.yml --task_type train 
```
Default settings are saved in a file named `config.yml`. **config_file** is a required parameter used to specify the file path for `config.yml`. Another essential parameter is task_type, which supports five task types: `train`, `test`, `predict`, `visualize`, `hyperparameter`, `CV`.

Support for the **log_enable** parameter (default is True) is available to record the training process, but it can also be manually disabled to save training time. If we want to specify dataset name, the total number of data points, training epochs, project_name, here is an example of modifying these default parameters (order can vary):

```bash
python main.py --config_file ./config.yml --task_type train --dataset_name 2d --points 2000  --epochs 300  --project_name GCPNet_2d --log_enable False
```
Here, all the arguments are taken from the deafult settings of the config.yml file if not specified. For example, if we do not specify the batch_size, it will take the value 64 by default from the config file. By default the data format is `json.zip`. But we can use other types of data as well, such as `cif` files. We just need to set the **dataset_name** toc `cif`. The model file that performs best on the validation set during the training process is by default saved in the folder specified by the *output_dir** parameter. 

### II. Cross validation
When the dataset has a small sample size, we can use cross-validation by simply changing the value of the **task_type** parameter to `CV`. Additionally, the **num_folds** parameter can be specified to determine the number of folds for cross-validation. An example command is as follows:
```bash
python main.py --config_file ./config.yml --task_type CV --dataset_name 2d --epochs 300 --project_name GCPNet_cv 
```
5 fold cross validation (default from the config file) result on the 2d-materials dataset for our model will be record in the log file.

### III.Prediction
#### Predition on a dataset
If we have a model path file (.pth file), we can make prediction on a dataset using that saved model path file. The command should be something like the following command:
```bash
python main.py --config_file ./config.yml --task_type predict --model_path 'path/to/trained/model.pth'
```
For example, we want to predict the results on the 2d-materials dataset using a pretrained model which is saved in a file named 'dir/to/saving/trained/model', then the command would be:
 
```bash
python main.py --config_file ./config.yml --task_type predict --dataset_name 2d --dataset_path ./data --model_path 'path/to/trained/model.pth' --output_path output.csv
```
then you will get an output file named 'outputs.csv'. the output file will contain the predicted results of the 2d-materials dataset.(format：material_id, target_value, predicted_value)

#### Predition on an unseen dataset
We can also use out trained model files to make prediction an unseen dataset. We have to just include the dataset path in the argument. The command should be something like this:
```bash
python main.py --config_file ./config.yml --dataset_path 'path/to/unseen/dataset' --dataset_name 'name/of/dataset' --project_name 'Predition Unseen' --task_type 'predict' --model_path 'path/to/trained/model'
```
#### Property prediction of a new material
Let's say, we have a new material and we want to predict some propeties of it. Then we need to place the structure file of that material inside the 'data' folder (such as, a .cif file). Currently there are 3 structure files already inside the data folder which can be used for testing purpose. We need the following command:
<!-- 3 structure files already -->
```bash
python main.py --config_file ./config.yml --dataset_path 'dir/to/cif/files' --dataset_name cif --project_name 'cif predictions' --task_type predict --model_path 'path/to/trained/model/file'
```
then you will get an output file named 'outputs.csv'. the output file will contain the predicted results of the new materials.(format：material_id, target_value(Set all to 0), predicted_value)

### IV. Visualization
After we finish model training, we can gain insights through t-SNE visualization. We can perform visualization by specifying the  **model_path** parameter to indicate the trained model file, the  **dataset_path** parameter to specify the dataset's path, the  **dataset_name** parameter to specify the dataset's name, and setting the  **task_type** parameter to `visualize`. For more default parameters, please refer to the **visualize_args** section in the 'config.yml' file. An example command is as follows:

```bash
python main.py --config_file ./config.yml --task_type visualize --model_path '/path/to/trained/model' --dataset_path ./data --dataset_name cubic
```

### V. Hyperparameter tuning
We can also support hyperparameter search by modifying or specifying the **entity** parameter to get permission. The default hyperparameters for tuning include **lr**, **batch_size**, **dropout_rate**,**hidden_features** etc. (see config.yml). Other parameters like **n_neighbors**,**optimizer**  can also be manually added to the config.yml file in **sweep_args**.An example command is as follows:

```bash
python main.py --config_file ./config.yml --dataset_path ./data --dataset_name 2d --task_type hyperparameter --project_name tuning --entity 'your_entity' 
```
<a name="reproducibility"></a>
## Rreproducibility
To facilitate the reproduction of experimental results, we provide the shell script used in the experiment. The command for running the script is as follows:
```bash
bash script/comparative_study.sh
```

## Contribution

1.  Fork the repository
2.  Create Feat_xxx branch
3.  Commit your code
4.  Create Pull Request

<a name="contributors"></a>
## Contributors

1. hengda Gao
2. Genglin Li
3. Xiao-W. Guo