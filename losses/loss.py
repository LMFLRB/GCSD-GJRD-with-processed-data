'''
Author: Mingfei Lu
Description: define the loss functions
Date: 2021-10-22 13:13:53
LastEditTime: 2022-11-02 10:37:07
'''

import torch
import numpy as np
import torch.nn.functional as F
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

from copy import deepcopy as copy
from scipy.linalg import logm
from .utils import *


# def gcsd_hn():

class CrossEntropy(torch.nn.Module):
    def __init__(self, name='cross_entropy', **kwargs) -> None:
        super().__init__()
        self.name = name.upper()
        self.criterion = torch.nn.CrossEntropyLoss()
    def forward(self, *args, **kwargs) -> torch.Tensor:
        loss = self.criterion(args[0], args[1].type(torch.long))
        return loss
    
class myMSE(torch.nn.Module):
    def __init__(self, name='mse', **kwargs) -> None:
        super().__init__()
        self.name = name.upper()
        self.reduction='mean' if kwargs.get('reduction') is None else kwargs.get('reduction')    
        if self.reduction in ['max', 'mean', 'none']:
            self.criterion = torch.nn.MSELoss(reduction=self.reduction)
    def forward(self, *args, **kwargs) -> torch.Tensor:
        reduction = self.reduction
        if len(args)==1 and isinstance(args[0], dict):
            args = list(args[0].values())
        if reduction in ['max', 'mean', 'none']:
            loss = self.criterion(args[0], args[1])
        elif isinstance(reduction, int):
            loss = ((args[0]-args[1])**2)
            for dim in range(len(args[0].shape)-1, reduction, -1):
                loss = loss.mean(dim)
        return loss

class PFIBLoss(torch.nn.Module):
    def __init__(self, name='pfib_loss', **kwargs) -> None:
        super().__init__()
        self.name = name.upper()
        self.beta = kwargs.get('beta')
        self.kernelsize = kwargs.get('kernelsize')
        
    def forward(self, *args) -> torch.Tensor:
        loss = PFIB(args[0], args[1], args[2],
                        beta=self.beta, 
                        kernelsize=self.kernelsize)
        return loss
    
class MILoss(torch.nn.Module):
    def __init__(self, name='mi_loss', **kwargs) -> None:
        super().__init__()
        self.name = name.upper()
        self.kernelsize = kwargs.get('kernelsize')
        self.biasflag = True
        
    def forward(self, *args) -> torch.Tensor:        
        if len(args) == 3:
            inputs, outputs, targets = args[0], args[1], args[2]
            inputs_2d = inputs.view(inputs.shape[0], -1)
            error = outputs - targets    
            loss = mi(inputs_2d, error, self.kernelsize[0], self.kernelsize[1])
        elif len(args) == 2:
            inputs, emdeddings = args[0], args[1]
            loss = mi(inputs, emdeddings, self.kernelsize[0], self.kernelsize[1])
        return loss
   
class MCCLoss(torch.nn.Module):
    def __init__(self, name='mcc_loss', **kwargs) -> None:
        super().__init__()
        self.name = name.upper()
        self.kernelsize = kwargs.get('kernelsize')[0] if isinstance(kwargs.get('kernelsize'), list) else kwargs.get('kernelsize')
        self.reduction='mean' if kwargs.get('reduction') is None else kwargs.get('reduction')    
        
    def forward(self, *args) -> torch.Tensor:
        loss = 1-mcc(args[0]-args[1], sigma=self.kernelsize, reduction=self.reduction)
        return loss

class HSICLoss(torch.nn.Module):
    def __init__(self, name='hsic_loss', **kwargs) -> None:
        super().__init__()
        self.name = name.upper()
        self.kernelsize = kwargs.get('kernelsize')
        self.biasflag = True
        
    def forward(self, *args) -> torch.Tensor:        
        if len(args) == 3:
            inputs, outputs, targets = args[0], args[1], args[2]
            inputs_2d = inputs.view(inputs.shape[0], -1)
            error = outputs - targets    
            loss = hsic(inputs_2d, error, self.kernelsize[0], self.kernelsize[1])
        elif len(args) == 2:
            inputs, emdeddings = args[0], args[1]
            loss = hsic(inputs, emdeddings, self.kernelsize[0], self.kernelsize[1])
        return loss
    
class MultiHSICLoss(torch.nn.Module):
    def __init__(self, name='multi_hsic_loss', **kwargs) -> None:
        super().__init__()
        self.name = name.upper()
        self.kernelsize = kwargs.get('kernelsize')
        self.biasflag = True
        
    def forward(self, *args) -> torch.Tensor:        
        loss = multi_hsic(args[0], self.kernelsize)  
        return loss
    
class MEELoss(torch.nn.Module):
    def __init__(self, name='mee_loss', **kwargs) -> None:
        super().__init__()
        self.name = name.upper()
        self.kernelsize = kwargs.get('kernelsize')[0] if isinstance(kwargs.get('kernelsize'), list) else kwargs.get('kernelsize')
        
    def forward(self, *args) -> torch.Tensor:
        loss = reyi_entropy(args[0]-args[1], sigma=self.kernelsize)
        return loss
    
class MMDLoss(torch.nn.Module):
    def __init__(self, name='mmd_loss', **kwargs) -> None:
        super().__init__()
        self.name = name.upper()
        self.kernelsize = kwargs.get('kernelsize')[0] if isinstance(kwargs.get('kernelsize'), list) else kwargs.get('kernelsize')
        
    def forward(self, *args) -> torch.Tensor:
        inputs, targets = (args, torch.randn_like(args)) if len(args) == 1 else (args[0], args[1])
        loss = mmd(inputs, targets, sigma=self.kernelsize )
        return loss
    
class KLDLoss(torch.nn.Module):
    def __init__(self, name='kld_loss', **kwargs) -> None:
        super().__init__()
        self.name = name.upper()
        self.criter=torch.nn.KLDivLoss(reduction="batchmean")

    def forward(self, *args) -> torch.Tensor:
        if len(args)>1:
            mu, log_var = torch.flatten(args[0],1), torch.flatten(args[1],1)
            loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)  
        else:
            # features=target_distribution(args[0])
            features=F.log_softmax(args[0],1)
            samples= F.softmax(torch.randn_like(features), dim=1)
            # loss = torch.nn.functional.kl_div(features, samples)   
            loss = self.criter(features, samples)
        return loss

class AutoCorrLoss(torch.nn.Module):
    def __init__(self, name='autocorrelation', **kwargs) -> None:
        super().__init__()
        self.name = name.upper()

    def forward(self, *args) -> torch.Tensor:
        loss = 1.0 - autocorrelation(args[0], args[1])
        return loss
    
class InvConsLoss(torch.nn.Module):
    def __init__(self, name='consist_loss', **kwargs) -> None:
        super().__init__()
        self.name = name.upper()

    def forward(self, *args) -> torch.Tensor:
        loss = consist_loss(args[0], args[1])
        return loss
    
class InfNormLoss(torch.nn.Module):
    def __init__(self, name='inf_loss', **kwargs) -> None:
        super().__init__()
        self.name = name.upper()

    def forward(self, *args) -> torch.Tensor:
        dim = [order for order in range(1, len(args[0]))]
        loss = torch.norm(args[0]-args[1], p=float('inf'), dim=dim).mean()
        return loss
    
class SeqCorLoss(torch.nn.Module):
    def __init__(self, name='seq_corr_loss', **kwargs) -> None:
        super().__init__()
        self.name = name.upper()

    def forward(self, *args) -> torch.Tensor:
        loss = seq_decorrelation(args if len(args) == 1 else args[0])
        return loss
     
class CSDLoss(torch.nn.Module):
    def __init__(self, name='CSD', **kwargs) -> None:
        super(CSDLoss, self).__init__()    
        self.name = name.upper()
        self.kernelsize = kwargs.get('kernelsize')[0] if isinstance(kwargs.get('kernelsize'), list) else kwargs.get('kernelsize')
    def forward(self, *args) -> torch.Tensor:
        loss = csd(args[0], args[1], self.kernelsize)
        return loss

class ImageSimilarityLoss(torch.nn.Module):
    def __init__(self, name='image_similarity_loss', kernelsize=[1, 1], size_average=None, 
                    reduce=None, reduction: str = 'mean', **kwargs) -> None:
        super(ImageSimilarityLoss, self).__init__()

        self.reduction = reduction
        self.device = 'cuda:0'    
        self.name = name.upper()
        
        if self.name == 'LPIPS':
            import lpips
            net_type = kwargs['net'] if kwargs.get('net') is not None else "squeeze"
            self.criterion = lpips.LPIPS(net=net_type, ).to(self.device)

        elif self.name == 'SSIM':
            from torchmetrics import SSIM
            self.criterion = SSIM().to(self.device) 

        elif self.name == 'PSNR':
            from torchmetrics import PSNR
            self.criterion = PSNR().to(self.device) 
        
        elif self.name == 'ISC':
            from torchmetrics import IS
            features = kwargs['features'] if kwargs.get('features') is not None else 64
            self.criterion = IS(feature=features).to(self.device) 

        elif self.name == 'FID':
            from torchmetrics import FID
            features = kwargs['features'] if kwargs.get('features') is not None else 64
            self.criterion = FID(feature=features).to(self.device) 

        elif self.name == 'KID':
            from torchmetrics import KID
            subset_size = kwargs['subset_size'] if kwargs.get('subset_size') is not None else 50
            self.criterion = KID(subset_size=subset_size).to(self.device) 
        

    def forward(self, *args, **kwargs) -> torch.Tensor:
        
        if self.name == 'HTW':
            loss = self.criterion.forward(args[0])[0]
        ## the following are fidelity score metrics
        elif self.name == 'ISC':
            self.criterion.update(args[0])
            loss = self.criterion.compute()
        
        elif self.name in ['KID', 'FID']:
            self.criterion.update(args[0], real=True) 
            self.criterion.update(args[1], real=False)
            loss = self.criterion.compute()

        elif self.name in ['LPIPS', 'PSNR', 'SSIM']:
            loss = self.criterion(args[0].float(), args[1].float())

        return loss



############### the remainder for GCSD and GJRD
class CSDLossRobert(torch.nn.Module):
    def forward(self, *args):
        return gcsd_every_2_cluster(args[0], args[1])

class DDC2Loss(torch.nn.Module):
    def forward(self, A, type='mean', statistic_num=100):
        n = A.shape[0]
        M = A @ A.T
        if type=='mean':
            triu_mean = triu(M) * 2/(n*(n-1))
            return  triu_mean
            # return  triu_mean + (1.0-M.trace()/n)

        elif type=='max':
            idx = np.triu_indices(n, 1)
            return torch.sort(M[idx], descending=True)[0][:statistic_num].mean()
      
class DDC3Loss(torch.nn.Module):
    def forward(self, A, G, type='trace', extra:str='gcsd'):   
        if type=='simplex':     
            eye = torch.eye(A.shape[1]).type_as(A)
            M = calculate_gram_mat(A, eye)
            # criterion = CSDLossRobert()
            if extra=='ddc':
                return gcsd_every_2_cluster(M, G) # pairwise csd
            elif extra=='gcsd':
                return gcsd_cluster(M, G) # gcsd
            elif extra=='gjrd':
                return gjrd_cluster(M, G) # gjrd
        
        elif type=='sparse':
            return A.max(1)[0].mean()
        else:
            return (1.0-(A @ A.T).trace()/A.shape[0])

class GCSD(torch.nn.Module):
    def __init__(self, name='GSCD', **kwargs) -> None:
        super(GCSD, self).__init__()    
        self.name = name.upper()        
    def forward(self, *args, call_type='cluster') -> torch.Tensor:  
        div=gcsd_cluster(args[0], args[1])
        return div 
    
class GJRD(torch.nn.Module):
    def __init__(self, name='GJRD', **kwargs) -> None:
        super(GJRD, self).__init__()        
        self.name = name.upper()
        self.entropy_order = 2 if kwargs.get('entropy_order') is None else kwargs.get('entropy_order')
    def forward(self, *args) -> torch.Tensor:  
        return gjrd_cluster(args[0], args[1], self.entropy_order)

class DDCLoss(torch.nn.Module):
    def __init__(self, name='DDC', **kwargs) -> None: 
        super(DDCLoss, self).__init__()           
        self.name = name.upper()
        self.kernelsize = torch.tensor(kwargs.get('kernelsize')[0] if isinstance(kwargs.get('kernelsize'), list) else kwargs.get('kernelsize'), device="cuda:0")
        self.kernelsize_adapt = kwargs["kernelsize_adapt"]
        self.kernelparams = kwargs["kernelsize_search_params"]
        self.epoch_last=int(-1)
        self.weights = copy(kwargs.get('weights'))
        self.generative = False if kwargs.get('generative') is None else kwargs["generative"]
        
        if  name=='DDC' and (self.weights.get('ddc1') is not None) and self.weights['ddc1']>1.0e-10:
            self.ddc1 = CSDLossRobert()
        if (self.weights.get('ddc2') is not None) and self.weights['ddc2']>1.0e-10:
            self.ddc2 = DDC2Loss()        
        if (self.weights.get('ddc3') is not None) and self.weights['ddc3']>1.0e-10:
            self.ddc3 = DDC3Loss()
        if (self.weights.get('reconst') is not None) and self.weights['reconst']>1.0e-10:            
            self.reconst = myMSE()
        if (self.generative and self.weights.get('regular') is not None) and self.weights['regular']>1.0e-10:
            self.regular = KLDLoss()

    def forward(self, args, **kwargs) -> torch.Tensor:
        loss={}
        if args.get("hidden") is None and args.get("embedding") is not None:
            args["hidden"] = args["embedding"]
        if args.get("assign") is not None and args.get("hidden") is not None:
            if self.kernelsize_adapt and kwargs['epoch']!=self.epoch_last and kwargs['is_training']:
            # if self.kernelsize_adapt and kwargs['is_training']:
                from .utils import get_kernelsize
                self.kernelsize = get_kernelsize(args["hidden"], self.kernelparams.param, self.kernelparams.func)
                self.epoch_last = kwargs['epoch']
                  
            A = args["assign"]
            G = calculate_gram_mat(args["hidden"], sigma=self.kernelsize)   
            try:
                csda = self.ddc1(A, G)
                loss = dict(loss,  kernel=self.kernelsize,  ddc1=csda)
                # loss['loss'] = self.weights["ddc1"]*csda
            except:
                pass

            try:
                # eye  = self.ddc2(A, type='mean', enable=kwargs.get('enable'))
                eye  = self.ddc2(A)
                loss = dict(loss, ddc2=eye) 
                # loss['loss'] = loss['loss'] + self.weights["ddc2"]*eye
            except:
                pass
                        
            try:
                csdm = self.ddc3(A, G, type='simplex', extra=self.name.lower()) # sparse
                # csdm = self.ddc3(A, G, type='sparse') # sparse
                # csdm = self.ddc3(A, G) # trace
                loss = dict(loss, ddc3=csdm)
                # loss['loss'] = loss['loss'] + self.weights["ddc3"]*csdm
            except:
                pass
        try:
            if args.get("origin") is not None and args.get("reconst") is not None:
                if kwargs.get("bidirectional"):
                    origin = torch.flatten(args["origin"],2)
                    reconst = torch.flatten(args["reconst"],2)
                    reconst = self.reconst(torch.cat([origin, origin.flip(-1)],-1), reconst)
                else:
                    reconst = self.reconst(torch.flatten(args["origin"],1), torch.flatten(args["reconst"],1))
                # loss['loss'] = loss['loss'] + self.weights["reconst"]*reconst  if loss.get('loss') is not None else self.weights["reconst"]*reconst
                loss = dict(loss, reconst=reconst)
        except:
            pass
        
        try:
            if args.get("mu") is not None and args.get("log_var") is not None:
                regular = self.regular(args["mu"], args["log_var"])
            else:
                regular = self.regular(args["embedding"])
            # loss['loss'] = loss['loss'] + self.weights["regular"]*regular if loss.get('loss') is not None else self.weights["regular"]*regular
            loss = dict(loss, regular=regular)
        except:
            pass
        # if kwargs['epoch']!=self.epoch_last:
        #     print(f"{loss}\n")
        #     self.epoch_last = kwargs['epoch']
        loss['loss'] = sum([value*self.weights[name] for name, value in loss.items() if name in self.weights])
        loss={key: value if (key=="loss" and kwargs['is_training']) else value.item() for key, value in loss.items()}
        return loss

class GCSDLoss(DDCLoss):
    def __init__(self, name='GCSD', **kwargs):
        super(GCSDLoss, self).__init__(name='GCSD', **kwargs)
        self.name = name.upper()
        if (self.weights.get('ddc1') is not None) and self.weights['ddc1']>1.0e-10:
            self.ddc1 = GCSD(**kwargs)
    
class GJRDLoss(DDCLoss):
    def __init__(self, name='GJRD', **kwargs) -> None:
        super(GJRDLoss, self).__init__(name='GJRD', **kwargs)
        self.name = name.upper()
        if (self.weights.get('ddc1') is not None) and self.weights['ddc1']>1.0e-10:
            self.ddc1 = GJRD(**kwargs)

class DECLoss(torch.nn.Module):
    def __init__(self, name='DEC', **kwargs) -> None: 
        super(DECLoss, self).__init__()           
        self.name = name
        self.weights = copy(kwargs.get('weights'))
        self.generative = False if kwargs.get('generative') is None else kwargs["generative"]
        
        # self.dec = torch.nn.KLDivLoss(reduction="batchmean")
        self.dec = torch.nn.KLDivLoss(size_average=False)
        if self.generative and self.weights.get('regular') is not None:
            self.regular = KLDLoss(reduction="batchmean")
        if self.weights.get('reconst') is not None:            
            self.reconst = myMSE()

    def forward(self, args, **kwargs) -> torch.Tensor:
        loss={}
        if args.get("assign") is not None:
            target = target_distribution(args["assign"]).detach()
            loss = dict(loss, dec=self.dec(args["assign"].log(), target)/target.shape[0])
        try:
            if args.get("origin") is not None and args.get("reconst") is not None:
                if kwargs.get("bidirectional"):
                    origin = torch.flatten(args["origin"],2)
                    reconst = torch.flatten(args["reconst"],2)
                    reconst = self.reconst(torch.cat([origin, origin.flip(-1)],-1), reconst)
                else:
                    reconst = self.reconst(torch.flatten(args["origin"],1), torch.flatten(args["reconst"],1))
                # loss['loss'] = loss['loss'] + self.weights["reconst"]*reconst  if loss.get('loss') is not None else self.weights["reconst"]*reconst
                loss = dict(loss, reconst=reconst)
        except:
            pass
        
        try:
            if args.get("mu") is not None and args.get("log_var") is not None:
                regular = self.regular(args["mu"], args["log_var"])
            else:
                regular = self.regular(args["embedding"])
            # loss['loss'] = loss['loss'] + self.weights["regular"]*regular if loss.get('loss') is not None else self.weights["regular"]*regular
            loss = dict(loss, regular=regular)
        except:
            pass

        loss['loss'] = sum([value*self.weights[name] for name, value in loss.items() if name in self.weights])
        loss={key: value if (key=="loss" and kwargs['is_training']) else value.item() for key, value in loss.items()}
        return loss


class VAELoss(torch.nn.Module):
    def __init__(self, name='VAE', **kwargs) -> None: 
        super(VAELoss, self).__init__()           
        self.name = name.upper()
        self.weights = copy(kwargs.get('weights'))
        self.generative = False if kwargs.get('generative') is None else kwargs["generative"]
        
        self.reconst = myMSE()
        if (self.generative or self.weights.get('regular') is not None):
            self.regular = KLDLoss()

    def forward(self, args, **kwargs) -> torch.Tensor:
        loss={}
        if args.get("hidden") is None and args.get("embedding") is not None:
            args["hidden"] = args["embedding"]
        if args.get("embedding") is None and args.get("hidden") is not None:
            args["embedding"] = args["hidden"]
        
        try:
            if args.get("origin") is not None and args.get("reconst") is not None:
                if kwargs.get("bidirectional"):
                    origin = torch.flatten(args["origin"],2)
                    reconst = torch.flatten(args["reconst"],2)
                    reconst = self.reconst(torch.cat([origin, origin.flip(-1)],-1), reconst)
                else:
                    reconst = self.reconst(torch.flatten(args["origin"],1), torch.flatten(args["reconst"],1))                
                loss = dict(loss, reconst=reconst)
        except:
            pass
        
        try:
            if args.get("mu") is not None and args.get("log_var") is not None:
                regular = self.regular(args["mu"], args["log_var"])
            else:
                regular = self.regular(args["embedding"])
            # loss['loss'] = loss['loss'] + self.weights["regular"]*regular if loss.get('loss') is not None else self.weights["regular"]*regular
            loss = dict(loss, regular=regular)
        except:
            pass
        # if kwargs['epoch']!=self.epoch_last:
        #     print(f"{loss}\n")
        #     self.epoch_last = kwargs['epoch']
        loss['loss'] = sum([value*self.weights[name] for name, value in loss.items() if name in self.weights])
        loss={key: value if (key=="loss" and kwargs['is_training']) else value.item() for key, value in loss.items()}
        return loss
