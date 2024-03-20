import torch, os
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torch import stack, concat
from os.path import join as join
from scipy.io import savemat, loadmat
import pickle
from .base import *

class ImageProcess(nn.Module):
    name = 'ImageProcessor'
    patch_size=[224,224]
    resnet_dict=dict(MNIST='resnet18',FashionMNIST='resnet34',STL10='resnet50')
    def __init__(self,   
                name:str="",   
                root_dir:str="",           
                resnet_type:str=None,
                feature_type:str="linear",
                **kwargs
                ):
        super(ImageProcess, self).__init__()
        self.use_cuda=torch.cuda.is_available()
        self.root_dir=root_dir
        self.dataset=name
        self.feature_type=feature_type
        if resnet_type is None:
            self.resnet_type=self.resnet_dict[name] if name in ['MNIST','FashionMNIST','STL10'] else 'resnet50'
        else:
            self.resnet_type=resnet_type

        self.model = FeatureResnet(self.resnet_type.upper())
        if feature_type=="linear":
            self.model = nn.Sequential(self.model,
                                       nn.AdaptiveAvgPool2d((1,1)),
                                       Flatten())
        for param in self.model.parameters():
            param.requires_grad = False
        if self.cuda:
            self.model = self.model.cuda()
        self.transform = transforms.Resize(self.patch_size, antialias=True)
        self.flags={}
        self.data_path_proc = join(self.root_dir, 
                                   f"{self.dataset}-processed", 
                                   f'{self.resnet_type}-{self.feature_type}')
        os.makedirs(self.data_path_proc, exist_ok=True) 
    
    def forward(self, datasets_to_process:dict):
        if os.path.exists(os.path.join(self.data_path_proc, 'done')):
            for split, _ in datasets_to_process.items():
                self.flags[split] = True
            print(f"{self.dataset} has beed processed")
            return
        else:   
            for split, dataset in datasets_to_process.items():
                self.process(dataset, split)
            if not False in self.flags.values():
                with open(os.path.join(self.data_path_proc, 'done'), 'w') as f:
                    f.write('done')
    
    def process(self, dataset, split="train", silent:bool=True):
        try:
            features, labels = [], []
            dataset_ = DataLoader(dataset,batch_size=128,shuffle=False,drop_last=False)
            for bn,(feature,label) in tqdm(enumerate(dataset_)):   
                feature = feature.to("cuda" if self.use_cuda else "cpu")
                features.extend(list(self.extract(feature)))
                labels.extend(list(label)) 

            features=stack(features).detach()
            save_dict = dict(features=features.cpu().numpy(),                             
                            labels=stack(labels).numpy())
            if hasattr(dataset, 'clusters'):
                save_dict.update(dict(clusters=dataset.clusters))
            
            try:
                savemat(os.path.join(self.data_path_proc, f"{split}.mat"), 
                        save_dict
                        )
            except:
                with open(os.path.join(self.data_path_proc, f"{split}.pkl"), "wb") as f:
                    pickle.dump(save_dict, f)
            self.flags[split]=True
            
            if not silent:
                print(f"{self.dataset} pre-processing with {split}-set sucessed")
            return        
        except:
            self.flags[split]=False
            Warning(f"{self.dataset} pre-processing with {split}-set failed")

    def extract(self, x):    
        if len(x.shape)==3:
            x = x.unsqeeze(1)
        if x.shape[1]==1:
            x = x.repeat([1,3,1,1])
        if x.shape[-1]!=224:
            x = self.transform(x)
        return self.model(x)
    