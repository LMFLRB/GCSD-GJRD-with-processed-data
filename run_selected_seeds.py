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
from experiments import (Experiment, 
                        make_model_data_consistent,
                        myTensorboard,
                        transform_to_edict,
                        easydict_to_dict,
                        make_dict_consistent,
                        manual_seed_all,
                        get_version_numbers,
                        get_version,
                        copy
                        )

cur_dir = os.path.dirname(__file__)

def fit_with_seed(data, config:Edict={}, version=0, tensorbord=None):   
    # make the model repruducible 
    exp_params  = config.experiment
    model_params= config.model 
    data_params = config.data 
    exp_params.max_version = 1
    exp_params.cluster_epochs = 100
    exp_params.save_results = True # save resutls when conducting montecarlo experiments
    
    loss = exp_params.loss.name

    seed=exp_params.seed
    manual_seed_all(seed)      

    ds_train, ds_val = data
    make_model_data_consistent(ds_train, model_params.autoencoder)
    
    # cluster_path="selected_seeds_cluster"
    cluster_path="finetune_cluster"
    cluster_dir = f"{exp_params.log_dir}/{cluster_path}"         

    # version=get_version(cluster_dir, to_confirm=False)
    log_dir_cluster=f"{cluster_dir}/version_{version}"
    exp_params.log_text_path = f"{cluster_path}/version_{version}/textlog_fit_with_seed={seed}"  
    
    if exp_params.save_results:
        os.makedirs(f"{log_dir_cluster}", exist_ok=True)
    
    experiment = Experiment(exp_params)
    experiment.log_text([f"training with seed={seed}",
                        f"configurations loaded for {loss} from {config.file}",
                        f"configurating experiment for {data_params.name} with {model_params.name}-{model_params.autoencoder.network}-{framework}...",])

    model = Models[model_params.name](**model_params)
    
    experiment.log_text(f"configurating model...")          
    if exp_params.enable_pretrain and not model_params.encode_only:
        log_dir_dae=f"{exp_params.log_dir}/autoencoder"
        log_dir_sdae=f"{log_dir_dae}/sdae"                    
        ckpt_path_sdae=f"{log_dir_sdae}/pretrained.ckpt"
        try:
            if not os.path.exists(ckpt_path_sdae):
                autoencoder = model.encoder
                if exp_params.cuda:
                    autoencoder.cuda()
                if not tensorbord.started.is_set() and config.experiment.save_results:
                    tensorbord.start()
                    experiment.log_text([f"tensor_board is opened @port={tensorbord.port},id={tensorbord.pid}"])
                experiment.log_text([f"Autoencoder substacked pretraining stage."])
                experiment.pretrain_dae(autoencoder,ds_train,ds_val,log_dir_dae)
                experiment.log_text("Autoencoder training stage.")
                experiment.train_autoencoder(autoencoder,ds_train,ds_val,log_dir=log_dir_sdae)
                experiment.log_text("Autoencoder training finished.")
            else:
                model.encoder.load_state_dict(torch.load(ckpt_path_sdae))   
                experiment.log_text("Autoencoder loaded from pretrained ckpt.")
        except:
            pass                 
    if exp_params["cuda"]:
        model.cuda()
    
    # experiment.log_text(f"{model.name} clustering with {loss} on {data_params.name} training stage.")
    # experiment.train_cluster(model, ds_train, ds_val, log_dir_cluster)
    
    df_dict=dict(seed=int(seed))
    try:
        df_dict.update(experiment.evaluate(model, ds_val, ckpt_mode="last", log_dir_cluster=log_dir_cluster, visualize=True))
    except:
        pass
    try:
        metric_dict_best = experiment.evaluate(model, ds_val, ckpt_mode="acc", visualize=True)
        df_dict.update({f"{key}_best": val for key,val in  metric_dict_best.items()})
    except:
        pass
    # create a DataFrame and then add it to the end of the file
    
    filename=f"{exp_params.log_dir}/results_selected_seeds.csv"
    data_to_append = pd.DataFrame(df_dict, columns=df_dict.keys(), index=[version])
    data_to_append.to_csv(filename, mode='a', header=True if (version==0 or not os.path.exists(filename)) else False, index_label="version")  

    if os.path.exists(log_dir_cluster):
        with open(f"{log_dir_cluster}/config.yaml", 'w') as f:
            yaml.dump(easydict_to_dict(config), f)
        experiment.log_text(f"saving current configurations to 'config.yaml'. ")

    del model, experiment
    
    
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
    config.data.cuda = config.experiment.cuda = torch.cuda.is_available()#   
    data = LoadDataset(config.data, 
                        config.experiment.use_processed_data,
                        config.experiment.feature_type,
                        config.experiment.resnet_type
                        ) 

    loss = config.experiment.loss.name
    loss_temp = loss.split('-')  
    if len(loss_temp)>1:
        config.experiment.loss.entropy_order = float(loss_temp[1])
    config.file=f"configs/for_{loss_temp[0].lower()}.yaml"        
    with open(config.file, 'r') as file:
        make_dict_consistent(config, transform_to_edict(yaml.safe_load(file)))
    ###########################################################################################

    framework="Enc" if config.model.encode_only else "AE"
    if config.experiment.use_processed_data or config.experiment.use_resnet:
        config.model.name = 'DDC_resnet'                    
        config.model.use_processed_data=config.experiment.use_processed_data
        config.model.resnet_type=config.experiment.resnet_type
        config.model.feature_type=config.experiment.feature_type 
        config.model.autoencoder.network="MLP" if config.experiment.feature_type=="linear" else "CNN"
        
    config.experiment.log_dir = os.path.join(
                    config.experiment.log_dir,
                    config.data.name,
                    f"{config.experiment.resnet_type}-{config.experiment.feature_type}",
                    f"{config.model.name}-{config.model.autoencoder.network}-{framework}",
                    loss)   
    seeds=[1967,6648,8959,8109,2617,1273,5132,2671,6186,1045,]
    for version, seed in enumerate(seeds):
        config.experiment.seed = seed
        fit_with_seed(data, config, version, tensorbord=tb)
