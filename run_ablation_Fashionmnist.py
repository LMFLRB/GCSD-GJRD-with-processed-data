import numpy as np
import torch
import os
import yaml
import time
from easydict import EasyDict as Edict
from typing import Union
from itertools import product

from scipy.io import savemat, loadmat
import pandas as pd
from datasets import LoadDataset

from models import Models
from experiments import (Experiment, 
                        make_model_data_consistent,
                        myTensorboard,
                        transform_to_edict,
                        easydict_to_dict,
                        make_dict_consistent,
                        manual_seed_all,
                        get_version,
                        remove_path,
                        copy_files,
                        copy
                        )
cur_dir = os.path.dirname(__file__)

from run_dataset_loss import pre_train_ae, find_best_seeds

def ablation(data, loss, config:dict={}, seed:Union[dict, list]=None, 
             tensorbord=None, fit_mode:str="montecarlo", 
             restart:bool=False,
             random_seed:bool=False):    
    exp_params  = config.experiment
    model_params= config.model 
    data_params = config.data 
        
    root_dir = copy(exp_params.log_dir)

    # init_ckpt=os.path.join(root_dir, "finetune_cluster", 
    #                        f"version_{config.select_version}", 
    #                        "init.ckpt")
    ckpt_path_sdae = os.path.join(root_dir, "autoencoder",
                            "sdae", "pretrained.ckpt")

    cluster_dir = f"{root_dir}/{fit_mode}_cluster"
    ver_name=f"with_{'random_seed' if random_seed else f'finetune_seed_{config.select_version}'}"
    filename_results=f"{cluster_dir}/{ver_name}.csv"
    is_done=f"{cluster_dir}/{ver_name}-done" 
    
    exp_params.cluster_epochs = 100
    exp_params.save_results = False 


    if restart:
        if os.path.exists(is_done):
            os.remove(is_done)
        if os.path.exists(filename_results):
            os.remove(filename_results)
        if os.path.exists(cluster_dir):
            remove_path(cluster_dir)
        last_version=0
    else:
        if os.path.exists(is_done):
            print("the specified running has been done before")
            return
        else:
            if os.path.exists(filename_results):
                last_version=pd.read_csv(filename_results)['version'].values.tolist()[-1]+1     
            else:
                last_version=0
    os.makedirs(cluster_dir, exist_ok=True)  

    ds_train, ds_val = data
    make_model_data_consistent(ds_train, model_params.autoencoder)
    
    sensitivity_results="{}/{}_cluster/with_{}.csv".format(root_dir, 
        "params_sensitivity",
        'random_seed' if random_seed else f'finetune_seed_{config.select_version}')
    # model = Models[model_params.name](**model_params)
    if fit_mode=='ablation_loss':
        item=pd.read_csv(sensitivity_results).sort_values(by='acc_best', ascending=False)    
        gamma_1s = [item['ddc1'].values[0], 0.0]#[0.05,0.01]   #    
        gamma_2s = [item['ddc2'].values[0], 0.0]#[0.05,0.01]   #
        gamma_3s = [item['ddc3'].values[0], 0.0]#[1.0,0.5,0.1] #    
        entropy_order=[2]
    elif fit_mode=='entorder_sensitivity':
        item=pd.read_csv(sensitivity_results).sort_values(by='acc_best', ascending=False)    
        gamma_1s = [item['ddc1'].values[0]]#[0.05,0.01]   #    
        gamma_2s = [item['ddc2'].values[0]]#[0.05,0.01]   #
        gamma_3s = [item['ddc3'].values[0]]#[1.0,0.5,0.1] #    
        entropy_order=[1.1,2.0,4.0,6.0,8.0,10.0]
    elif fit_mode=='params_sensitivity':  
        gamma_1s = [1.0, 0.2, 0.1, 0.02, 0.01, 0.001]#[1.0,0.5,0.1] #   
        gamma_2s = [0.1, 0.05, 0.01, 0.001, 0.0001]#[0.05,0.01]   #    
        gamma_3s = [0.1, 0.05, 0.01, 0.001, 0.0001]#[0.05,0.01]   #     
        entropy_order=[2]
    
    combinations = list(product(entropy_order, gamma_1s, gamma_2s, gamma_3s)) #grid parameters to generate combinations 
    exp_params.max_version = len(combinations)
    loss_param=copy(exp_params.loss)
    
    experiment = Experiment(exp_params)    
    if not tensorbord.started.is_set() and config.experiment.start_tensorbord:
        tensorbord.start()
        experiment.log_text([f"tensor_board is opened @port={tb.port},id={tb.pid}"])

    for version in range(last_version,len(combinations)):
        # seed=torch.randint(0,10000000,[1,]).tolist()
        exp_params.seed = seed if not random_seed else  torch.randint(0,10000000,[1,]).tolist()[0]
        manual_seed_all(exp_params.seed)    
        experiment.set_version(cluster_dir,version)

        entropy_order, w1, w2, w3 = combinations[version]
        W=Edict(ddc1=w1,ddc2=w2,ddc3=w3,reconst=1.0)
        loss_param.update(Edict(entropy_order=entropy_order, weights=W))
        
        model = Models[model_params.name](**model_params)
        experiment.log_text("loading initiation model weights")
        model.encoder.load_state_dict(torch.load(ckpt_path_sdae)) 
        # model.load_state_dict(torch.load(init_ckpt))
        if exp_params["cuda"]:
            model.cuda()


        experiment.set_loss_function(loss_param)
        
        experiment.log_text([f"setting random seed to {exp_params.seed}",
                             f"reset loss-caller for {loss}",
                             "configurating experiment for {} with {}-{}-{}...".format(
                                    data_params.name,
                                    model_params.name,
                                    model_params.autoencoder.network,
                                    'FC' if model_params.encode_only else 'AE')])  
                
        save_dict=dict(seed=int(exp_params.seed), entropy_order=entropy_order)
        save_dict.update(W)
        experiment.log_text(f"loss_weights: {experiment.loss_caller.weights}")
        experiment.train_cluster(model, ds_train, ds_val)
        for mode in ["last", "acc"]:
            metric_mode=experiment.evaluate(model, ds_val, ckpt_mode=mode, visualize=False, )
            save_dict.update({key if mode=="last" else f"{key}_best": val for key,val in  metric_mode.items()})
        # create a DataFrame and then add it to the end of the file
        data_to_append = pd.DataFrame(save_dict, columns=save_dict.keys(), index=[version])
        data_to_append.to_csv(filename_results, mode='a', index_label="version", 
                              header=True if not os.path.exists(filename_results) else False)  
        
        # experiment.events_to_mat()

        if experiment.save_results:
            with open(f"{experiment.logdir_cluster}/config.yaml", 'w') as f:
                yaml.dump(easydict_to_dict(experiment.params), f)
                yaml.dump({key: val for key,val in model.__dict__.items() if not key.startswith("_")}, f)
            experiment.log_text(f"saving configurations to {experiment.logdir_cluster}/config.yaml. ")
        ###########################################
    
    with open(is_done, 'w') as f:
        f.write('done')
    experiment.log_text(f"{fit_mode} experiment done with total {len(combinations)} runs!")
    
if __name__ == "__main__":    
    with open(f'configs/overall.yaml', 'r') as file:
        config = transform_to_edict(yaml.safe_load(file)) 
    os.makedirs(config.experiment.log_dir, exist_ok=True)    
    try:
        # set tensorboard
        tb = myTensorboard(os.path.join(cur_dir, config.experiment.log_dir), 5)
    except:
        tb = None
        pass
    # 等待3秒钟 网页端口链接成功
    time.sleep(3)    
    # root_dir = copy(config.experiment.log_dir)
    # seeds = {"GJRD": [4430], "GCSD": [7076], "DDC": [3856]}
    root_dir = "runs"
    loss = "GJRD"
    framework="AE"
    dataset = "FashionMNIST"  #"MNIST", "FashionMNIST", "STL10"
    config.file=f"configs/for_gjrd.yaml"
    config.experiment.resnet_type="resnet18"   
    # dataset  
    config.data.name = dataset 
    config.data.cuda =  config.experiment.cuda = torch.cuda.is_available()#  
    data = LoadDataset(config.data, 
                    config.experiment.use_processed_data,
                    config.experiment.feature_type,
                    config.experiment.resnet_type
                    )  #dataset  
    
    with open(config.file, 'r') as file:
        make_dict_consistent(config, transform_to_edict(yaml.safe_load(file)))
    # model
    config.model.encode_only = False
    if config.experiment.use_processed_data or config.experiment.use_resnet:
        config.model.name = 'DDC_resnet'                    
        config.model.use_processed_data=config.experiment.use_processed_data
        config.model.resnet_type=config.experiment.resnet_type
        config.model.feature_type=config.experiment.feature_type 
        config.model.autoencoder.network="MLP" if config.experiment.feature_type=="linear" else "CNN"
    
    config.experiment.log_dir = os.path.join(root_dir,dataset,
                    f"{config.experiment.resnet_type}-{config.experiment.feature_type}",
                    f"{config.model.name}-{config.model.autoencoder.network}-{framework}",
                    f"{loss}-{config.experiment.loss.entropy_order}")   
    finetune_seeds = pd.read_csv(f"{config.experiment.log_dir}/finetune.csv").sort_values(by='acc_best', ascending=False)['seed'].values
    
    # run ablation
    config.select_version = 0
    for fit_mode in ["params_sensitivity", "ablation_loss", "entorder_sensitivity"]:
        ablation(data, loss, config, 
                finetune_seeds[config.select_version], 
                tensorbord=tb, 
                fit_mode=fit_mode,
                restart=False,
                random_seed=False
        )