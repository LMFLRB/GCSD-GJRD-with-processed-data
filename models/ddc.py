import torch.nn as nn
from torch.optim import Optimizer as Optimizer
from typing import Tuple, Callable, Optional, Union, Any
from torch import Tensor, cat
from numpy import ndarray
from collections import OrderedDict

from .sdae import StackedDenoisingAutoEncoder as AutoEncoder
from .base import Attension, BilateralAttension

      
class DeepDivergenceCluster(nn.Module):
   
    def __init__(self, 
                 name: str="DDC",
                 n_cluster: int=10,
                 encode_only: bool=False,
                 autoencoder: Union[nn.Module, dict]={},        
                 **kwargs):
        """
        DDC clustering module
        args:
            n_cluster: the number if clusters to learn/ classes in data to handle
            autoencoder: an autoencoder of parameters dict for constructing an autoencoder
        keargs: paramweters for autoencoder, not limited to the following
            network: mlp or cnn
            framework: autoencoder or only encoder
            n_latent: dimmension of the latent to learn
            generative: whether to learn a generative VAE for autoencoder
            activation: activating function of the middle layer
            final_activation: activating function of the output layer
        """
        super().__init__()        
        self.name = name
        self.n_cluster = n_cluster        
        self.encode_only = encode_only
        self.n_latent = autoencoder["latent_dim"]
        self.network = autoencoder["network"]
        self.generative = False if autoencoder.get("generative") is None else autoencoder["generative"]
        self.device=None

        self.activation = "ReLU" if kwargs.get("activation") is None else kwargs["activation"]
        
        self.encoder = AutoEncoder(**autoencoder) if isinstance(autoencoder, dict) else autoencoder
        units=[]
        # if self.encoder.network in ["RNN","GRU","LSTM"]:
        #     if autoencoder["bidirectional"]:
        #         units.append(("attension", BilateralAttension(self.n_latent))) 
        #     else:
        #         units.append(("attension", Attension(self.n_latent))) 
        # units=units+[('linear', nn.Linear(self.n_latent, self.n_cluster)),
        #             ('batchnorm', nn.BatchNorm1d(num_features=self.n_cluster)),
        #             ('activation', getattr(nn, self.activation)()),
        #             ('softmax', nn.Softmax(dim=1))]
        # self.assignment = nn.Sequential(OrderedDict(units))
        
        self.assignment  = nn.Sequential(
                                    nn.Linear(self.n_latent, self.n_cluster), 
                                    nn.BatchNorm1d(num_features=self.n_cluster),
                                    getattr(nn, self.activation)(),
                                    nn.Softmax(dim=1))
    def encode(self, x) -> Tensor:
        self.device=x.device
        return self.encoder.encode(x)
    
    def decode(self, embedding: Tensor) -> Tensor:
        return self.encoder.decode(embedding)

    def forward(self, input: Tensor) -> tuple:
        embedding = self.encode(input)        
        return  (self.assignment(embedding),
                embedding,
                input
                ) if self.encode_only else \
                (self.assignment(embedding),
                embedding,
                self.decode(embedding),
                input
                )
    
    def predict(self, inputs: tuple) -> tuple:
        return tuple(output.detach() for output in self.forward(inputs))
    
    def loss_function(self, 
                      results: Union[tuple, Tensor], 
                      loss_caller: Any, 
                      **kwargs) -> dict:
        inputs=dict(assign=results[0], embedding=results[1])
        if not kwargs.get('cluster_only')==True and len(results)>3:
            inputs=dict(inputs, reconst=results[2].flatten(1), origin=results[3].flatten(1)) 
        
        return loss_caller(inputs, **kwargs)    
    
    def metric_function(self, 
                        assign: Union[Tensor, ndarray], 
                        labels: Union[Tensor, ndarray],
                        metric_caller: Any) -> dict:
        if isinstance(assign, Tensor):
            if len(assign.shape)>1 and assign.shape[1]>1:
                preds=assign.max(1)[1].detach().cpu().numpy()  
            else:
                preds=assign.squeeze().detach().cpu().numpy()
        else:
            preds=assign      
        truth=labels.cpu().numpy() if isinstance(labels, Tensor) else labels
            
        return metric_caller(truth, preds)
    