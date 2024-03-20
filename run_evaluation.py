import numpy as np
import torch
import os
import yaml
import time
from easydict import EasyDict as Edict
from typing import Union

from scipy.io import savemat, loadmat
import pandas as pd
from datasets import LoadDataset

from models import Models
from experiments import (myTensorboard,
                        transform_to_edict,
                        make_dict_consistent,
                        copy
                        )

cur_dir = os.path.dirname(__file__)

from run_dataset_loss import find_best_seeds, fit_with_seeds

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
    root_dir = copy(config.experiment.log_dir)
    # seeds = {"GJRD": [4430], "GCSD": [7076], "DDC": [3856]}
    for dataset in ["MNIST"]:        #"MNIST", "FashionMNIST", "STL10"
    # for dataset in ["MNIST", "FashionMNIST", "STL10"]:   
        config.data.cuda =  config.experiment.cuda = torch.cuda.is_available()#  
        config.data.name = dataset
        for resnet_type in ["resnet18"]: #      "resnet18" 
            config.experiment.resnet_type=resnet_type
            data = LoadDataset(config.data, 
                            config.experiment.use_processed_data,
                            config.experiment.feature_type,
                            config.experiment.resnet_type
                            )  #dataset  
            # for loss in ["GCSD", "GJRD-2", "GJRD-5", "GJRD-10", "DDC"]:#
            for loss in ["GJRD-2"]:#, , "GJRD-10"
                loss_temp = loss.split('-')
                config.file=f"configs/for_{loss_temp[0].lower()}.yaml"        
                with open(config.file, 'r') as file:
                    make_dict_consistent(config, transform_to_edict(yaml.safe_load(file)))
                ###########################################################################################
                for encode_only in [False]:#, True
                    #     enable autoencoder
                    config.model.encode_only = encode_only
                    framework="FC" if config.model.encode_only else "AE"
                    if config.experiment.use_processed_data or config.experiment.use_resnet:
                        config.model.name = 'DDC_resnet'                    
                        config.model.use_processed_data=config.experiment.use_processed_data
                        config.model.resnet_type=config.experiment.resnet_type
                        config.model.feature_type=config.experiment.feature_type 
                        config.model.autoencoder.network="MLP" if config.experiment.feature_type=="linear" else "CNN"
                            
                    if len(loss_temp)>1:
                        config.experiment.loss.entropy_order = float(loss_temp[1])
                    config.experiment.log_dir = os.path.join(root_dir,dataset,
                                    f"{config.experiment.resnet_type}-{config.experiment.feature_type}",
                                    f"{config.model.name}-{config.model.autoencoder.network}-{framework}",
                                    loss)   
                    montecarlo_seeds=torch.randperm(10000).tolist()[:config.experiment.n_montecarlo] 
                    fit_with_seeds(data, loss, config, montecarlo_seeds, tensorbord=tb, fit_mode="montecarlo")
                    

                    finetune_seeds = find_best_seeds(config, finetune_top_k=config.experiment.n_finetune) 
                    fit_with_seeds(data, loss, config, finetune_seeds[:2], tensorbord=tb, fit_mode="eval_only")
                    # 等待TensorBoard进程结束
            
        del data
    
