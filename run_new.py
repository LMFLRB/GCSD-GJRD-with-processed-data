import numpy as np
import torch
import os
import yaml
import time
import argparse
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
                        remove_path,
                        copy
                        )

cur_dir = os.path.dirname(__file__)

     
def pre_train_ae(data, model, config:dict={}):
    if not config.experiment.enable_pretrain or config.model.encode_only:
        return 
    else:
        exp_params  = config.experiment
        data_params = config.data 
        ds_train, ds_val = data   

        log_dir_dae=f"{exp_params.log_dir}/autoencoder"
        log_dir_sdae=f"{log_dir_dae}/sdae"                    
        ckpt_path_sdae=f"{log_dir_sdae}/pretrained.ckpt"
        if not os.path.exists(ckpt_path_sdae):              
            exp_params.save_results=True
            os.makedirs(f"{log_dir_sdae}", exist_ok=True)
            exp_params.log_text_path = f"ae_pre_textlog"
            experiment = Experiment(exp_params)
            experiment.log_text([f"configurations loaded for {config.loss} from {config.file}",
                                f"configurating experiment for stackVAE on {data_params.name}...",])  
            if exp_params.cuda:
                model.cuda()            
            experiment.log_text([f"Autoencoder substacked pretraining stage."])
            experiment.pretrain_dae(model,ds_train,ds_val,log_dir_dae)
            experiment.log_text("Autoencoder training stage.")
            experiment.train_autoencoder(model,ds_train,ds_val,log_dir=log_dir_sdae)
        
        return ckpt_path_sdae

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

def fit_with_seeds(data, config:dict={}, Seeds:Union[dict, list]=None, fit_mode:str="montecarlo"):   
    if os.path.exists(os.path.join(config.experiment.log_dir, f'{fit_mode}-done')):
        print(f"{fit_mode} runs have been done before")
    else:
        exp_params  = config.experiment
        model_params= config.model 
        data_params = config.data 
        
        finetune_done=os.path.exists(os.path.join(exp_params.log_dir, "finetune-done"))
        exp_params.eval_only = (fit_mode=="eval_only") and finetune_done 
        if fit_mode=="montecarlo":
            exp_params.cluster_epochs = 5
            exp_params.save_results = False # do not save resutls when conducting montecarlo experiments
        elif fit_mode in ["finetune", "seed_eval"]:
            exp_params.cluster_epochs = 100
            exp_params.save_results = True # save resutls when conducting montecarlo experiments
        elif fit_mode=="eval_only":
            exp_params.cluster_epochs = 100
            exp_params.save_results = True # save resutls when conducting montecarlo experiments
            eval_dir = f"{exp_params.log_dir}/{'finetune' if finetune_done else 'eval_only'}_cluster"

        exp_params.log_text_path = f"{fit_mode}_textlog"
        
        ds_train, ds_val = data
        make_model_data_consistent(ds_train, model_params)
        
        
        root_dir = exp_params.log_dir

        filename_results=f"{root_dir}/{fit_mode}.csv"
        if os.path.exists(filename_results):
            os.remove(filename_results)
        cluster_dir = f"{root_dir}/{fit_mode}_cluster" 
        if os.path.exists(cluster_dir):
            remove_path(cluster_dir)

        try:
            if config.experiment.start_tensorbord:
                tensorbord = myTensorboard(root_dir, 5)
                # 等待3秒钟 网页端口链接成功
                time.sleep(3)    
            else:
                tensorbord = None
        except:
            pass   
    

        exp_params.max_version = len(Seeds)
        for version, seed in enumerate(Seeds):
            # make the model repruducible
            manual_seed_all(seed)                

            # version=get_version(cluster_dir)
            log_dir_cluster=f"{cluster_dir}/version_{version}"
            # exp_params.log_dir = root_dir
            if exp_params.save_results:
                os.makedirs(f"{log_dir_cluster}", exist_ok=True)
            
            experiment = Experiment(exp_params)
            experiment.log_text([f"using seed={seed}",
                                f"configurations loaded for {config.loss} from {config.file}",
                                f"configurating experiment for {data_params.name} with {model_params.name}-{model_params.autoencoder.network}-{'FC' if model_params.encode_only else 'AE'}...",])
        
            model = Models[model_params.name](**model_params)
            
            experiment.log_text(f"configurating model...")          
            if exp_params.enable_pretrain and not model_params.encode_only:                
                ckpt_path_sdae = pre_train_ae(data, model.encoder, config)
                model.encoder.load_state_dict(torch.load(ckpt_path_sdae))                    
            if exp_params["cuda"]:
                model.cuda()
            
            if config.experiment.start_tensorbord and not tensorbord.started.is_set():
                tensorbord.start()
                experiment.log_text([f"tensor_board is opened @port={tensorbord.port},id={tensorbord.pid}"])
            df_dict=dict(seed=int(seed))
            
            if fit_mode in ["montecarlo", "finetune"] or not exp_params.eval_only:
                experiment.log_text(f"{model.name} clustering with {config.loss} on {data_params.name} {fit_mode} training stage.")
                experiment.train_cluster(model, ds_train, ds_val, log_dir_cluster)
            ckpt_path = f"{eval_dir}/version_{version}" if exp_params.eval_only else log_dir_cluster 
            # for mode in ["last","acc"]:
            for mode in ["last" if fit_mode=="montecarlo" else "acc"]:
                metric_mode=experiment.evaluate(model, ds_val, 
                                            ckpt_mode=mode, 
                                            ckpt_path=ckpt_path,
                                            log_dir_cluster=log_dir_cluster,
                                            visualize=False if fit_mode=="montecarlo" else True, )
                df_dict.update({key if mode=="last" else f"{key}_best": val for key,val in  metric_mode.items()})
            # create a DataFrame and then add it to the end of the file
            data_to_append = pd.DataFrame(df_dict, columns=df_dict.keys(), index=[version])
            data_to_append.to_csv(filename_results, mode='a', header=True if (version==0 or not os.path.exists(filename_results)) else False, index_label="version")  

            if os.path.exists(log_dir_cluster):
                with open(f"{log_dir_cluster}/config.yaml", 'w') as f:
                    yaml.dump(easydict_to_dict(config), f)
                experiment.log_text(f"saving current configurations to 'config.yaml'. ")

            del model, experiment
        
        with open(os.path.join(f"{root_dir}", f'{fit_mode}-done'), 'w') as f:
            f.write('done')
        
if __name__ == "__main__":      
    parser = argparse.ArgumentParser(description="data pre-process")
    parser.add_argument('--dataset', nargs='+', type=str, default="REUTERS10K", #"MNIST", #, "STL10", #
                        help='dataset for experiment')
    parser.add_argument('--loss', type=str, default="GJRD", help='loss name')
    parser.add_argument('--entropy_order', type=int, default=2, help='entropy order for GJRD')
    parser.add_argument('--loss_weigths', type=dict, 
                        default=dict(ddc1=1.0,ddc2=0.05,ddc3=0.05,reconst=1.0), 
                        help='entropy order for GJRD')
    parser.add_argument('--model', type=str, default="DDC", help='model name')
    parser.add_argument('--hidden_dims', type=int, default=[500,500,1000], help='model name')
    parser.add_argument('--data_path', type=str, default="G:/Data", help='directory to load data')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--evaluate_batch_size', type=int, default=256, help='evaluate_batch_size')
    parser.add_argument('--root_dir', type=str, default="G:/Logs", help='directory to log results')
    parser.add_argument('--resnet_type', type=str, default="resnet50", help='directory to log results')
    parser.add_argument('--feature_type', type=str, default="linear", help='directory to log results')
    parser.add_argument('--use_processed_data', type=bool, default=True, help='framework choices')
    parser.add_argument('--encode_only', type=bool, default=False, help='framework choices')
    parser.add_argument('--file', type=str, default="configs/basic_configuration.yaml", help='directory to log configuration')
    args = parser.parse_args()

    with open(args.file, 'r') as file:
        config = transform_to_edict(yaml.safe_load(file)) 
    make_dict_consistent(config.data, transform_to_edict(args.__dict__))
    make_dict_consistent(config.model, transform_to_edict(args.__dict__))
    make_dict_consistent(config.experiment, transform_to_edict(args.__dict__))
    config.file = args.file
    config.data.cuda = config.experiment.cuda = torch.cuda.is_available()#  
    config.experiment.data.batch_size=args.batch_size
    config.experiment.data.evaluate_batch_size=args.evaluate_batch_size   
    config.experiment.loss.weights.update(args.loss_weigths)
    config.model.autoencoder.hidden_dims=args.hidden_dims
    config.model.name=args.model   
    config.experiment.loss.name = args.loss
    config.data.name = args.dataset
    config.data.root_dir = args.data_path  
    if args.loss.upper()=="GJRD":
        args.loss = args.loss+f"-{args.entropy_order}"
        config.experiment.loss.entropy_order=args.entropy_order 
    config.loss = args.loss   
    config.model.framework = config.framework = framework = "FC" if args.encode_only else "AE"
    config.model.network = config.network = network = "MLP" if args.feature_type=="linear" else "CNN"
    
    root_dir = os.path.join(args.root_dir, config.experiment.log_dir)  
    if args.dataset.lower()in ["reuters10k"]:
        paths=[root_dir,args.dataset,args.loss] 
    else:
        paths=[root_dir,args.dataset,f"{args.resnet_type}-{args.feature_type}",f"{args.model}-{network}-{framework}",args.loss] 
    config.experiment.log_dir = os.path.join(*paths)     
    data = LoadDataset(config.data)  #dataset

    if not os.path.exists(os.path.join(config.experiment.log_dir, 'montecarlo-done')):
        montecarlo_seeds=torch.randperm(100000).tolist()[:config.experiment.n_montecarlo] 
        fit_with_seeds(data, config, montecarlo_seeds, fit_mode="montecarlo")
    else:
        print("montecarlo-done before")
    if not os.path.exists(os.path.join(config.experiment.log_dir, 'finetue-done')):
        finetune_seeds = find_best_seeds(config, finetune_top_k=config.experiment.n_finetune) 
        fit_with_seeds(data, config, finetune_seeds, fit_mode="finetune")
    else:
        print("finetue-done before")
    
    # fit_with_seeds(data, loss, config, [7782,2854], tensorbord=tb, fit_mode="finetune")
    # 等待TensorBoard进程结束
