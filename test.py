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
                        get_version,
                        copy
                        )

cur_dir = os.path.dirname(__file__)
     
def pre_train_ae(data, loss, config:dict={}, tensorbord=None):
    if config.model.encode_only or not config.experiment.enable_pretrain:
        return 
    else:
        exp_params  = config.experiment
        model_params= config.model 
        data_params = config.data 
        ds_train, ds_val = data   

        log_dir_dae=f"{exp_params.log_dir}/autoencoder"
        log_dir_sdae=f"{log_dir_dae}/sdae"       
        
        exp_params.save_results=True
        os.makedirs(f"{log_dir_sdae}", exist_ok=True)
        exp_params.log_text_path = f"ae_pre_textlog"
        experiment = Experiment(exp_params)
        experiment.log_text([f"configurations loaded for {loss} from {config.file}",
                            f"configurating experiment for stackVAE on {data_params.name}...",])
        
        make_model_data_consistent(ds_train, model_params.autoencoder)         
        model = Models[model_params.name](**model_params)
        autoencoder = model.encoder
        if exp_params.cuda:
            autoencoder.cuda()
        experiment.log_text([f"Autoencoder substacked pretraining stage."])
        if not tensorbord.started.is_set() and config.experiment.save_results:
            tensorbord.start()
            experiment.log_text([f"tensor_board is opened @port={tensorbord.port},id={tensorbord.pid}"])
        experiment.pretrain_dae(autoencoder,ds_train,ds_val,log_dir_dae)
        experiment.log_text("Autoencoder training stage.")
        experiment.train_autoencoder(autoencoder,ds_train,ds_val,log_dir=log_dir_sdae)

    return f"{log_dir_sdae}/pretrained.ckpt"

def find_best_seeds(config, tag_sel:str="ACC", log_dir: str="", results_file:str="montecarlo.csv", finetune_top_k:int=None):
    # find seeds producing the best performances
    if not tag_sel in ["ACC", "NMI", "ARI"]:
        ValueError("Please input the right tag for performance")
    root_dir = config.experiment.log_dir if len(log_dir)==0 else log_dir
    finetune_top_k=config.experiment.n_finetune if finetune_top_k is None else finetune_top_k
    try:
        Seeds_sel=pd.read_csv(f"{root_dir}/{results_file}").sort_values(by='acc', ascending=False)['seed'].values[:finetune_top_k].tolist()     
    except:
        Seeds_sel=[]
        pass
    return Seeds_sel

def fit_with_seeds(data, loss, config:dict={}, Seeds:Union[dict, list]=None, tensorbord=None, fit_mode:str="montecarlo"):   
    #
    ckpt_path_sdae = pre_train_ae(data, loss, config, tensorbord)
    #
    exp_params  = config.experiment
    model_params= config.model 
    data_params = config.data 
    if fit_mode=="montecarlo":
        exp_params.cluster_epochs = 5
        exp_params.save_results = False # do not save resutls when conducting montecarlo experiments
    elif fit_mode=="finetune":
        exp_params.cluster_epochs = 100
        exp_params.save_results = True # save resutls when conducting montecarlo experiments
    else:
        exp_params.cluster_epochs = 1
        exp_params.save_results = False # do not save resutls when conducting montecarlo experiments
    exp_params.log_text_path = f"{fit_mode}_textlog"
    
    ds_train, ds_val = data
    make_model_data_consistent(ds_train, model_params.autoencoder)
    
    root_dir = exp_params.log_dir

    cluster_dir = f"{root_dir}/{fit_mode}_cluster" 
    seeds=Seeds[loss] if isinstance(Seeds, dict) else Seeds
    exp_params.max_version = len(seeds)
    for version, seed in enumerate(seeds):
        # make the model repruducible
        manual_seed_all(seed)                

        # version=get_version(cluster_dir)
        log_dir_cluster=f"{cluster_dir}/version_{version}"
        # exp_params.log_dir = root_dir
        if exp_params.save_results:
            os.makedirs(f"{log_dir_cluster}", exist_ok=True)
        
        experiment = Experiment(exp_params)
        experiment.log_text([f"generated seed={seed}",
                            f"configurations loaded for {loss} from {config.file}",
                            f"configurating experiment for {data_params.name} with {model_params.name}-{model_params.autoencoder.network}-{framework}...",])
    
        model = Models[model_params.name](**model_params)
        
        experiment.log_text(f"configurating model...")          
        if exp_params.enable_pretrain and not model_params.encode_only:
            model.encoder.load_state_dict(torch.load(ckpt_path_sdae))                    
        if exp_params["cuda"]:
            model.cuda()
        
        if not tensorbord.started.is_set() and config.experiment.save_results:
            tensorbord.start()
            experiment.log_text([f"tensor_board is opened @port={tb.port},id={tb.pid}"])
        experiment.log_text(f"{model.name} clustering with {loss} on {data_params.name} {fit_mode} training stage.")
        
        if fit_mode=="eval_only":
            experiment.evaluate(model, ds_val, ckpt_mode="last", save_results=False, log_dir_cluster=log_dir_cluster)
            return
        
        experiment.train_cluster(model, ds_train, ds_val, log_dir_cluster)
        
        df_dict=dict(seed=int(seed))
        df_dict.update(experiment.evaluate(model, ds_val, ckpt_mode="last", save_results=False))
        if fit_mode=="finetune":
            metric_dict_best = experiment.evaluate(model, ds_val, ckpt_mode="acc")
            df_dict.update({f"{key}_best": val for key,val in  metric_dict_best.items()})
        # create a DataFrame and then add it to the end of the file
        filename=f"{root_dir}/{fit_mode}.csv"
        data_to_append = pd.DataFrame(df_dict, columns=df_dict.keys(), index=[version])
        data_to_append.to_csv(filename, mode='a', header=True if (version==0 or not os.path.exists(filename)) else False, index_label="version")  

        if os.path.exists(log_dir_cluster):
            with open(f"{log_dir_cluster}/config.yaml", 'w') as f:
                yaml.dump(easydict_to_dict(config), f)
            experiment.log_text(f"saving current configurations to 'config.yaml'. ")

        del model, experiment
    
    with open(os.path.join(f"{root_dir}", f'{fit_mode}-done'), 'w') as f:
        f.write('done')
    
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
        config.experiment.cuda = torch.cuda.is_available()#  
        config.data.cuda = False 
        config.data.name = dataset
        if config.experiment.use_resnet:
            config.data.patch_size=[224,224]
        data = LoadDataset(config.data, 
                           config.experiment.use_processed_data,
                           config.experiment.feature_type,
                           config.experiment.resnet_type
                           )  #dataset  
        # for loss in ["GCSD", "GJRD-2", "GJRD-5", "GJRD-10", "DDC"]:#
        for loss in ["GJRD-2", "GJRD-10"]:#, 
            loss_temp = loss.split('-')
            config.file=f"configs/for_{loss_temp[0].lower()}.yaml"        
            with open(config.file, 'r') as file:
                make_dict_consistent(config, transform_to_edict(yaml.safe_load(file)))
            ###########################################################################################
            for encode_only in [True,False]:                
                config.model.encode_only = encode_only#     enable autoencoder
                framework="FC" if config.model.encode_only else "AE"
                for feature_type in ["linear", "conv2d"]:
                    config.experiment.feature_type = feature_type        
                    if config.experiment.use_resnet:
                        config.model.name = 'DDC_resnet'
                ###########################################################################################
                        
                    if len(loss_temp)>1:
                        config.experiment.loss.entropy_order = float(loss_temp[1])
                    config.experiment.log_dir = os.path.join(root_dir,dataset,
                                    f"{config.experiment.resnet_type}-{config.experiment.feature_type}",
                                    f"{config.model.name}-{config.model.autoencoder.network}-{framework}",
                                    loss)           
                    
                    fit_with_seeds(data, loss, config, [0], tensorbord=tb, fit_mode="eval_only")
                    # 等待TensorBoard进程结束
        
        del data
    
