#!/usr/bin/python
# -*- encoding: utf-8 -*-

import datetime
import os
import time

import torch
import wandb
import torch.nn as nn
import torchmetrics
from torch_geometric.transforms import Compose

from model import GCPNet
from utils.keras_callbacks import WandbCallback
from utils.dataset_utils import MP18, dataset_split, get_dataloader, get_dataloader_4matbench
from utils.flags import Flags
from utils.train_utils import KerasModel, LRScheduler
from utils.transforms import GetAngle, ToFloat

os.environ["NUMEXPR_MAX_THREADS"] = "24"
debug = True 

import logging
from logging.handlers import RotatingFileHandler

def log_config(log_file='test.log'):
  LOG_FORMAT = '[%(asctime)s][%(levelname)s]: %(message)s'
  level = logging.INFO
  logging.basicConfig(level=level, format=LOG_FORMAT)
  log_file_handler = RotatingFileHandler(filename=log_file, maxBytes=2*1024*1024, backupCount=3)
  formatter = logging.Formatter(LOG_FORMAT)
  log_file_handler.setFormatter(formatter)
  logging.getLogger('').addHandler(log_file_handler)

def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_dataset(config):
    dataset = MP18(root=config.dataset_path, name=config.dataset_name, transform=Compose([GetAngle(), ToFloat(
        )]), r=config.max_edge_distance, n_neighbors=config.n_neighbors, edge_steps=config.edge_input_features, image_selfloop=True, points=config.points, target_name=config.target_name)
    return dataset

def setup_model(dataset, config):
    net = GCPNet(
            data=dataset,
            firstUpdateLayers=config.firstUpdateLayers,
            secondUpdateLayers=config.secondUpdateLayers,
            atom_input_features=config.atom_input_features,
            edge_input_features=config.edge_input_features,
            triplet_input_features=config.triplet_input_features,
            embedding_features=config.embedding_features,
            hidden_features=config.hidden_features,
            output_features=config.output_features,
            min_edge_distance=config.min_edge_distance,
            max_edge_distance=config.max_edge_distance,
            link=config.link,
            dropout_rate=config.dropout_rate,
        )
    return net

def setup_optimizer(net, config):
    optimizer = getattr(torch.optim, config.optimizer)(
        net.parameters(),
        lr=config.lr,
        **config.optimizer_args
    )
    if config.debug:
        print(f"optimizer: {optimizer}")
    return optimizer

def setup_schduler(optimizer, config):
    scheduler = LRScheduler(optimizer, config.scheduler, config.scheduler_args)
    return scheduler

def build_keras(net, optimizer, scheduler):
    model = KerasModel(net=net, loss_fn=nn.L1Loss(), metrics_dict={"mae": torchmetrics.MeanAbsoluteError(
    ), "mape": torchmetrics.MeanAbsolutePercentageError()}, optimizer=optimizer, lr_scheduler=scheduler)
    return model

def train(config, printnet=False):
    name = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if config.log_enable:
        wandb.init(project=config.project_name, name = name, save_code=False)
        
    if config.task_type.lower() == 'hyperparameter':
        for k, v in wandb.config.items():
            setattr(config, k, v)
            print(f"searched keys: {k}: {v}")

    # # 1. load data
    dataset = setup_dataset(config)
    train_dataset, val_dataset, test_dataset = dataset_split(
        dataset, train_size=0.8, valid_size=0.15, test_size=0.05, seed=config.seed, debug=debug) 
    train_loader, val_loader, test_loader = get_dataloader(
        train_dataset, val_dataset, test_dataset, config.batch_size, config.num_workers)

    # 2. load net
    rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = setup_model(dataset, config).to(rank)
    if config.debug:
        print(net)

    # 3. Set-up optimizer & scheduler
    optimizer = setup_optimizer(net, config)
    scheduler = setup_schduler(optimizer, config)

    # 4. Start training
    if config.log_enable:
        callbacks = [WandbCallback(project=config.project_name,config=config)]
    else:
        callbacks = None
    model = build_keras(net, optimizer, scheduler)
    model.fit(train_loader, val_loader, ckpt_path=os.path.join(config.output_dir, config.net+'.pth'), epochs=config.epochs,
            monitor='val_loss', mode='min', patience=config.patience, plot=True, callbacks=callbacks)
    print(model.evaluate(test_loader))
    
    if config.log_enable:
        wandb.log({"test_mae":model.evaluate(test_loader)['val_mae'], "test_mape":model.evaluate(test_loader)['val_mape'], "total_params":model.total_params()})
        wandb.finish()

    return model

import logging
from logging.handlers import RotatingFileHandler


def train_CV(config):

    #  1. load data
    from utils.dataset_utils import loader_setup_CV, split_data_CV
    dataset = setup_dataset(config)
    ## Split datasets
    cv_dataset = split_data_CV(dataset, num_folds=config.num_folds, seed=config.seed)
    cv_error = []

    for index in range(0, len(cv_dataset)):

        ## Set up work dir
        if not(os.path.exists(output_dir := f"{config.output_dir}/{index}")):
            os.makedirs(output_dir)

        ## Set up loader
        train_loader, test_loader, train_dataset, _ = loader_setup_CV(
            index, config.batch_size, cv_dataset, num_workers=config.num_workers
        )
        
        # 2. load net
        rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = setup_model(dataset, config).to(rank)
        
        # 3. Set-up optimizer & scheduler
        optimizer = setup_optimizer(net, config)
        scheduler = setup_schduler(optimizer, config)


        # 4. Start training
        model = build_keras(net, optimizer, scheduler)
        model.fit(train_loader, None, ckpt_path=os.path.join(output_dir, config.net+'.pth'), epochs=config.epochs,
                  monitor='train_loss', mode='min', patience=config.patience, plot=True)

        test_error = model.evaluate(test_loader)['val_mae']
        logging.info("fold: {:d}, Test Error: {:.5f}".format(index+1, test_error)) 
        cv_error.append(test_error)
    import numpy as np
    mean_error = np.array(cv_error).mean()
    std_error = np.array(cv_error).std()
    logging.info("CV Error: {:.5f}, std Error: {:.5f}".format(mean_error, std_error))
    return cv_error
    
    
def predict(config):
    
    # # 1. load data
    dataset = setup_dataset(config)
    from torch_geometric.loader import DataLoader
    test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=False,)

    # 2. load net
    rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = setup_model(dataset, config).to(rank)

    # 3. Set-up optimizer & scheduler
    optimizer = setup_optimizer(net, config)
    scheduler = setup_schduler(optimizer, config)

    # 4. Start Predict
    model = build_keras(net, optimizer, scheduler)
    model.predict(test_loader, ckpt_path=config.model_path, test_out_path=config.output_path)

def visualize(config):
    
    # # 1. load data
    from utils.dataset_utils import MP18, dataset_split, get_dataloader
    from utils.transforms import GetAngle, ToFloat

    dataset = MP18(root=config.dataset_path,name=config.dataset_name,transform=Compose([GetAngle(),ToFloat()]), r=config.max_edge_distance, n_neighbors=config.n_neighbors, edge_steps=config.edge_input_features, image_selfloop=True,points=config.points,target_name=config.target_name)

    train_dataset, val_dataset, test_dataset = dataset_split(dataset,train_size=0.8,valid_size=0.15,test_size=0.05,seed=config.seed, debug=debug)### 调试,不按照论文中的比列，按照 0.8：0.15; 0.05
    train_loader, val_loader, test_loader = get_dataloader(train_dataset, val_dataset, test_dataset, config.batch_size, config.num_workers)

    ## 2. load net
    rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = setup_model(dataset, config).to(rank)

    ##3. Set-up optimizer & scheduler
    optimizer = setup_optimizer(net, config)
    scheduler = setup_schduler(optimizer, config)
    print("optimizer:",optimizer)

    ##4. Start analysis
    model = KerasModel(net=net, loss_fn=nn.L1Loss(), metrics_dict={"mae":torchmetrics.MeanAbsoluteError(),"mape":torchmetrics.MeanAbsolutePercentageError()},optimizer=optimizer,lr_scheduler = scheduler)
    data_loader, _, _ = get_dataloader(dataset, val_dataset, test_dataset, config.batch_size, config.num_workers)
   
    model.analysis(net_name=config.net, test_data=data_loader,ckpt_path=config.model_path,tsne_args=config.visualize_args)
    #############

    return model

def matbench(config):
    name = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if config.log_enable:
        wandb.init(project=config.project_name, name = name, save_code=False)
        
    if config.task_type.lower() == 'hyperparameter':
        for k, v in wandb.config.items():
            setattr(config, k, v)
            print(f"searched keys: {k}: {v}")
            
    # # 1. load data
    from matbench import MatbenchBenchmark
    import numpy as np
    matbenchTask = np.array(['matbench_dielectric',
                    'matbench_jdft2d',
                    'matbench_log_gvrh',
                    'matbench_log_kvrh',
                    'matbench_mp_e_form',
                    'matbench_mp_gap',
                    'matbench_perovskites',
                    'matbench_phonons',
                    ])
    currentTask = []
    if type(config.matbenchTasklist) == int:   
        currentTask.append(matbenchTask[config.matbenchTasklist].tolist())
    else:
        currentTask.extend(matbenchTask[config.matbenchTasklist].tolist())
    mb = MatbenchBenchmark(subset= currentTask,autoload=False)
    import json
    import pandas as pd
    for task in mb.tasks:
        task.load()
        for fold in task.folds:
            trainX, trainY = task.get_train_and_val_data(fold)
            testX, testY = task.get_test_data(fold, include_target=True)
            rawData = {'trainX': trainX, 'trainY': trainY, 'testX': testX, 'testY': testY}
            data_cache_dir = os.path.join(config.dataset_path, 'matbench', task.dataset_name, f"_{fold}")
            dataset = MP18(root=data_cache_dir, name=config.dataset_name, matbenchRaw=rawData, transform=Compose([GetAngle(), ToFloat(
                )]), r=config.max_edge_distance, n_neighbors=config.n_neighbors, edge_steps=config.edge_input_features, image_selfloop=True, points=config.points, target_name=trainY.name)
            train_loader, val_loader, test_loader = get_dataloader_4matbench(dataset, val_ratio=0.05, batch_size=config.batch_size, num_workers=config.num_workers)

            # 2. load net
            rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            net = setup_model(dataset, config).to(rank)
            if config.debug:
                print(net)

            # 3. Set-up optimizer & scheduler
            optimizer = setup_optimizer(net, config)
            scheduler = setup_schduler(optimizer, config)

            # 4. Start training
            if config.log_enable:
                callbacks = [WandbCallback(project=config.project_name,config=config)]
            else:
                callbacks = None
            best_model_path = os.path.join(config.output_dir, task.dataset_name+"_"+config.net+'.pth')
            model = build_keras(net, optimizer, scheduler)
            model.fit(train_loader, val_loader, ckpt_path=best_model_path, epochs=config.epochs,
                    monitor='val_loss', mode='min', patience=config.patience, plot=True, callbacks=callbacks)
            
            preds = model.predict(test_loader, best_model_path)
            preds = torch.tensor(preds)
            task.record(fold, preds)
           

            if config.log_enable:
                wandb.log({"test_mae":model.evaluate(test_loader)['val_mae'], "test_mape":model.evaluate(test_loader)['val_mape'], "total_params":model.total_params()})
                wandb.finish()
        # save to file
        outputfile = config.output_dir+"/"+task.dataset_name +".json.gz"
        mb.to_file(outputfile)
        # show the results
        resultList = []
        with open(outputfile) as f:
            tmp = json.load(f)
            for i in range(5):
                # print(tmp['tasks'][task.dataset_name]['results']['fold_'+str(i)]['scores'])
                resultList.append(tmp['tasks'][task.dataset_name]['results']['fold_'+str(i)]['scores'])
        resultDf = pd.DataFrame(resultList)
        print(resultDf)
        resultDf.to_csv(config.output_dir+"/"+task.dataset_name +"_"+"results.csv")
        mean_std = pd.DataFrame([resultDf.mean(),resultDf.std()])
        print(mean_std)
        mean_std.to_csv(config.output_dir+"/"+task.dataset_name +"_"+"mean_std.csv")

    mb.to_file("./results.json.gz")

if __name__ == "__main__":

    import warnings
    warnings.filterwarnings('ignore', '.*TypedStorage is deprecated.*')
    
    flags = Flags()
    config = flags.updated_config
    
    name = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    config.output_dir = os.path.join(config.output_dir, name)
    if not(os.path.exists(config.output_dir)):
        os.makedirs(config.output_dir)
    set_seed(config.seed)

    if config.task_type.lower() == 'train':
        train(config)
        
    elif config.task_type.lower() == 'hyperparameter':
        sweep_id = wandb.sweep(config.sweep_args, config.entity, config.project_name)
        def wandb_train():
            return train(config)
        wandb.agent(sweep_id, wandb_train, count=config.sweep_count)
        
    elif config.task_type.lower() == 'visualize':
        visualize(config)
        
    elif config.task_type.lower() == 'cv':
        log_file = config.project_name + '.log'
        log_config(log_file)
        train_CV(config)
        
    elif config.task_type.lower() == 'predict':
        predict(config)
        
    elif config.task_type.lower() == 'matbench':
        matbench(config)

    else:
        raise NotImplementedError(f"Task type {config.task_type} not implemented. Supported types: train, test, cv, predict")
    
