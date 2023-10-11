
import torch
import sys,datetime
from tqdm import tqdm 
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator

def colorful(obj,color="red", display_type="plain"):
    color_dict = {"black":"30", "red":"31", "green":"32", "yellow":"33",
                    "blue":"34", "purple":"35","cyan":"36",  "white":"37"}
    display_type_dict = {"plain":"0","highlight":"1","underline":"4",
                "shine":"5","inverse":"7","invisible":"8"}
    s = str(obj)
    color_code = color_dict.get(color,"")
    display  = display_type_dict.get(display_type,"")
    out = '\033[{};{}m'.format(display,color_code)+s+'\033[0m'
    return out 

class StepRunner:
    def __init__(self, net, loss_fn, accelerator, stage = "train", metrics_dict = None,
                 optimizer = None, lr_scheduler = None
                 ):
        self.net,self.loss_fn,self.metrics_dict,self.stage = net,loss_fn,metrics_dict,stage
        self.optimizer,self.lr_scheduler = optimizer,lr_scheduler
        self.accelerator = accelerator


    def __call__(self, batch):
        # features,labels = batch
        features,labels = batch,batch.y
        #loss
        preds = self.net(features)
        loss = self.loss_fn(preds,labels)

        #backward()
        if self.optimizer is not None and self.stage=="train":
            self.accelerator.backward(loss)
            self.optimizer.step()
            # if self.lr_scheduler is not None:
            #     self.lr_scheduler.step()
            self.optimizer.zero_grad()

        all_preds = self.accelerator.gather(preds)
        all_labels = self.accelerator.gather(labels)
        all_loss = self.accelerator.gather(loss).sum()

        #losses
        step_losses = {self.stage+"_loss":all_loss.item()}

        #metrics
        step_metrics = {self.stage+"_"+name:metric_fn(all_preds, all_labels).item()
                        for name,metric_fn in self.metrics_dict.items()}

        if self.stage=="train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
        return step_losses,step_metrics

class EpochRunner:
    def __init__(self,steprunner,quiet=False):
        self.steprunner = steprunner
        self.stage = steprunner.stage
        self.steprunner.net.train() if self.stage=="train" else self.steprunner.net.eval()
        self.accelerator = self.steprunner.accelerator
        self.quiet = quiet

    def __call__(self,dataloader):
        loop = tqdm(enumerate(dataloader,start=1),
                    total =len(dataloader),
                    file=sys.stdout,
                    disable=not self.accelerator.is_local_main_process or self.quiet,
                    ncols = 100
                   )
        epoch_losses = {}
        for step, batch in loop:
            if self.stage=="train":
                step_losses,step_metrics = self.steprunner(batch)
            else:
                with torch.no_grad():
                    step_losses,step_metrics = self.steprunner(batch)

            step_log = dict(step_losses,**step_metrics)
            for k,v in step_losses.items():
                epoch_losses[k] = epoch_losses.get(k,0.0)+v

            if step!=len(dataloader):
                loop.set_postfix(**step_log)
            else:
                epoch_metrics = step_metrics
                epoch_metrics.update({self.stage+"_"+name:metric_fn.compute().item()
                                 for name,metric_fn in self.steprunner.metrics_dict.items()})
                epoch_losses = {k:v/step for k,v in epoch_losses.items()}
                epoch_log = dict(epoch_losses,**epoch_metrics)
                loop.set_postfix(**epoch_log)
                for name,metric_fn in self.steprunner.metrics_dict.items():
                    metric_fn.reset()
        return epoch_log

class KerasModel(torch.nn.Module):
    
    StepRunner,EpochRunner = StepRunner,EpochRunner
    
    def __init__(self,net,loss_fn,metrics_dict=None,optimizer=None,lr_scheduler = None):
        super().__init__()
        self.net,self.loss_fn,self.metrics_dict = net, loss_fn, torch.nn.ModuleDict(metrics_dict) 
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(
            self.net.parameters(), lr=1e-3)
        self.lr_scheduler = lr_scheduler
        self.from_scratch = True

    def load_ckpt(self, ckpt_path='checkpoint.pt'):
        self.net= torch.load(ckpt_path)
        self.from_scratch = False

    def forward(self, x):
        return self.net.forward(x)
    
    def fit(self, train_data, val_data=None, epochs=10,ckpt_path='checkpoint.pt',
            patience=5, monitor="val_loss", mode="min",
            mixed_precision='no',callbacks = None, plot = True, quiet = True):
        
        self.__dict__.update(locals())
        self.accelerator = Accelerator(mixed_precision=mixed_precision)
        device = str(self.accelerator.device)
        device_type = '🐌'  if 'cpu' in device else '⚡️'
        self.accelerator.print(
            colorful("<<<<<< "+device_type +" "+ device +" is used >>>>>>"))
    
        self.net,self.loss_fn,self.metrics_dict,self.optimizer,self.lr_scheduler= self.accelerator.prepare(
            self.net,self.loss_fn,self.metrics_dict,self.optimizer,self.lr_scheduler)
        
        train_dataloader,val_dataloader = self.accelerator.prepare(train_data,val_data)
        
        self.history = {}
        callbacks = callbacks if callbacks is not None else []
        
        if plot==True: 
            from utils.keras_callbacks import VisProgress
            callbacks.append(VisProgress(self))

        self.callbacks = self.accelerator.prepare(callbacks)
        
        if self.accelerator.is_local_main_process:
            for callback_obj in self.callbacks:
                callback_obj.on_fit_start(model = self)
        
        start_epoch = 1 if self.from_scratch else 0
        for epoch in range(start_epoch,epochs+1):
            # ////
            import time
            start_time = time.time()
            # ///
            should_quiet = False if quiet==False else (quiet==True or epoch>quiet)
            
            if not should_quiet:
                nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.accelerator.print("\n"+"=========="*8 + "%s"%nowtime)
                self.accelerator.print("Epoch {0} / {1}".format(epoch, epochs)+"\n")

            # 1，train -------------------------------------------------  
            train_step_runner = self.StepRunner(
                    net = self.net,
                    loss_fn = self.loss_fn,
                    accelerator = self.accelerator,
                    stage="train",
                    metrics_dict=deepcopy(self.metrics_dict),
                    optimizer = self.optimizer if epoch>0 else None,
                    lr_scheduler = self.lr_scheduler if epoch>0 else None
            )

            train_epoch_runner = self.EpochRunner(train_step_runner,should_quiet)
          
            train_metrics = {'epoch':epoch}
            train_metrics.update(train_epoch_runner(train_dataloader)) 

            for name, metric in train_metrics.items(): 
                self.history[name] = self.history.get(name, []) + [metric]

            if self.accelerator.is_local_main_process:
                for callback_obj in self.callbacks:
                    callback_obj.on_train_epoch_end(model = self)

            # 2，validate -------------------------------------------------
            if val_dataloader:
                val_step_runner = self.StepRunner(
                    net = self.net,
                    loss_fn = self.loss_fn,
                    accelerator = self.accelerator,
                    stage="val",
                    metrics_dict= deepcopy(self.metrics_dict)
                )
                val_epoch_runner = self.EpochRunner(val_step_runner,should_quiet)
                with torch.no_grad():
                    val_metrics = val_epoch_runner(val_dataloader)
                    # ####
                    # self.lr_scheduler.step(val_metrics['val_mae'])
                    #######
                    if self.lr_scheduler.scheduler_type == "ReduceLROnPlateau":
                        self.lr_scheduler.step(
                            metrics=val_metrics['val_mae'])
                    else:
                        self.lr_scheduler.step()
                for name, metric in val_metrics.items(): 
                    self.history[name] = self.history.get(name, []) + [metric]


            # 3，early-stopping -------------------------------------------------
            self.accelerator.wait_for_everyone()
            arr_scores = self.history[monitor]
            self.history['best_val_mae'] = self.history.get('best_val_mae', []) + [ np.min(arr_scores) if mode=="min" else np.max(arr_scores)]  


            best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)

            if best_score_idx==len(arr_scores)-1:
                # net_dict = self.accelerator.get_state_dict(self.net)
                self.accelerator.save(self.net,ckpt_path)
                if not should_quiet:
                    self.accelerator.print(colorful("<<<<<< reach best {0} : {1} >>>>>>".format(
                        monitor,arr_scores[best_score_idx])))

            end_time = time.time()
            self.history['time'] = self.history.get('time', []) + [end_time-start_time]
          
            if len(arr_scores)-best_score_idx>patience:
                self.accelerator.print(colorful(
                    "<<<<<< {} without improvement in {} epoch,""early stopping >>>>>>"
                ).format(monitor,patience))
                break; 
            

            if self.accelerator.is_local_main_process:
                for callback_obj in self.callbacks:
                    callback_obj.on_validation_epoch_end(model = self)
        if self.accelerator.is_local_main_process:   
            dfhistory = pd.DataFrame(self.history) # bug
            self.accelerator.print(dfhistory)
            
            for callback_obj in self.callbacks:
                callback_obj.on_fit_end(model = self)
        
            self.net = self.accelerator.unwrap_model(self.net)
            self.net = torch.load(ckpt_path)
            return dfhistory
    
    @torch.no_grad()
    def evaluate(self, val_data):
        accelerator = Accelerator()
        self.net,self.loss_fn,self.metrics_dict = accelerator.prepare(self.net,self.loss_fn,self.metrics_dict)
        val_data = accelerator.prepare(val_data)
        val_step_runner = self.StepRunner(net = self.net,stage="val",
                    loss_fn = self.loss_fn,metrics_dict=deepcopy(self.metrics_dict),
                    accelerator = accelerator)
        val_epoch_runner = self.EpochRunner(val_step_runner)
        val_metrics = val_epoch_runner(val_data)
        return val_metrics
    @torch.no_grad()
    def predict(self, test_data,ckpt_path, test_out_path='test_out.csv'):
        self.ckpt_path = ckpt_path
        self.load_ckpt(self.ckpt_path)
        self.net.eval()
        targets = [] 
        outputs = []
        id = []
        for data in test_data:
            with torch.no_grad():
                data = data.to(torch.device('cuda'))
                targets.append(data.y.cpu().numpy().tolist())
                output = self.net(data)
                outputs.append(output.cpu().numpy().tolist())
                id += data.structure_id
        targets = sum(targets, [])
        outputs = sum(outputs, [])
        id = sum(sum(id, []),[])
        import csv
        rows = zip(
            id,
            targets,
            outputs
        )
        with open(test_out_path, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            for row in rows:
                writer.writerow(row)
        return outputs
    @torch.no_grad()
    def cubic(self, test_data,ckpt_path, test_out_path='cubic_out.csv'):
        self.ckpt_path = ckpt_path
        self.load_ckpt(self.ckpt_path)
        self.net.eval()
        targets = [] 
        outputs = []
        id = []
        for data in test_data:
            with torch.no_grad():
                data = data.to(torch.device('cuda'))
                targets.append(data.y.cpu().numpy().tolist())
                output = self.net(data)
                outputs.append(output.cpu().numpy().tolist())
                id += data.structure_id
        targets = sum(targets, [])
        outputs = sum(outputs, [])
        id = sum(sum(id, []),[])
        import csv
        rows = zip(
            id,
            targets,
            outputs
        )
        with open(test_out_path, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            for row in rows:
                writer.writerow(row)

    @torch.no_grad()
    def analysis(self, net_name,test_data, ckpt_path, tsne_args,tsne_file_path="tsne_output.png"):
        '''
        Obtains features from graph in a trained model and analysis with tsne
        '''
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        inputs = []
        def hook(module, input, output):
            inputs.append(input)

        self.ckpt_path = ckpt_path
        self.load_ckpt(self.ckpt_path)
        self.net.eval()
        ##Grabs the input of the first linear layer after the GNN
        if net_name in [ "ALIGNN","CLIGNN", "GCPNet"]:
            self.net.fc.register_forward_hook(hook)
        else:
            self.net.post_lin_list[0].register_forward_hook(hook)

        targets = [] # only works for when targets has one index
        for data in test_data:
            with torch.no_grad():
                data = data.to(torch.device('cuda'))
                targets.append(data.y.cpu().numpy().tolist())
                _ = self.net(data)

        targets = sum(targets, [])
        inputs = [i for sub in inputs for i in sub]
        inputs = torch.cat(inputs)
        inputs = inputs.cpu().numpy()
        print("Number of samples: ", inputs.shape[0])
        print("Number of features: ", inputs.shape[1])


        ##Start t-SNE analysis
        tsne = TSNE(**tsne_args)
        tsne_out = tsne.fit_transform(inputs)


        fig, ax = plt.subplots()
        main = plt.scatter(tsne_out[:, 1], tsne_out[:, 0], c=targets, s=3,cmap='coolwarm')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = plt.colorbar(main, ax=ax)
        stdev = np.std(targets)
        cbar.mappable.set_clim(
            np.mean(targets) - 2 * np.std(targets), np.mean(targets) + 2 * np.std(targets)
        )
        plt.savefig(tsne_file_path, format="png", dpi=600)
        plt.show()

    def total_params(self):
        return self.net.total_params()

class LRScheduler:
    """wrapper around torch.optim.lr_scheduler._LRScheduler"""

    def __init__(self, optimizer, scheduler_type, model_parameters):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type

        self.scheduler = getattr(torch.optim.lr_scheduler, self.scheduler_type)(
            optimizer, **model_parameters
        )

        self.lr = self.optimizer.param_groups[0]["lr"]

    @classmethod
    def from_config(cls, optimizer, optim_config):
        scheduler_type = optim_config["scheduler_type"]
        scheduler_args = optim_config["scheduler_args"]
        return cls(optimizer, scheduler_type, **scheduler_args)

    def step(self, metrics=None, epoch=None):
        if self.scheduler_type == "Null":
            return
        if self.scheduler_type == "ReduceLROnPlateau":
            if metrics is None:
                raise Exception("Validation set required for ReduceLROnPlateau.")
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()

        # update the learning rate attribute to current lr
        self.update_lr()

    def update_lr(self):
        for param_group in self.optimizer.param_groups:
            self.lr = param_group["lr"]
