'''
Author: Mingfei Lu
Description: 
Date: 2022-04-08 17:00:00
LastEditTime: 2022-09-15 09:34:37
'''
from .loss import *

Loss = {'Image_loss': ImageSimilarityLoss,
        'MI': MILoss,
        'MCC': MCCLoss,
        'MEE': MEELoss,
        'MSE': myMSE,
        'CE': CrossEntropy,
        'HSIC': HSICLoss,
        'MMD': MMDLoss,
        'KLD': KLDLoss,
        'CSD': CSDLoss,
        'GCSD': GCSDLoss,
        'GJRD': GJRDLoss,
        'DDC': DDCLoss,
        "DEC": DECLoss,
        "VAE": VAELoss,
        "KMeans": KMeans,
        }