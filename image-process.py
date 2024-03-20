from __future__ import print_function
import torch
import yaml
import argparse
from models import ImageProcess
from datasets import LoadDataset
from experiments.utils import transform_to_edict

parser = argparse.ArgumentParser(description="data pre-process")
parser.add_argument('--datasets', nargs='+', type=str, 
                    # default=[
                    #         "DigitsFive", 
                    #         "OfficeCaltech10",
                    #         "Office31", 
                    #         "OfficeHome", 
                    #         "DomainNet", 
                    #         "PACS", 
                    #         "VLCS"
                    #         ], 
                    default=["MNIST", "FashionMNIST"], #, "STL10"
                    help='dataset for experiment')
parser.add_argument('--data_param', type=dict, 
                    default=dict(root_dir="E:/Shared/Data", 
                                 cuda=False,
                                 name="MNIST",), 
                    help='parameters for getting dataloader')
parser.add_argument('--resnet_type', type=str, default='resnet50', help='resnet type')
args = parser.parse_args()


if __name__ == '__main__':
    args.cuda = torch.cuda.is_available()
    # args.cuda = False
    args.data_param.update({'cuda': args.cuda})
    data_param = transform_to_edict(args.data_param)
    
    print('\nglobal configurations:\n', args)

    for dataset in args.datasets:
    # for dataset in ["DigitsFive"]:
        # args.resnet_type = 'resnet18' if dataset=="MNIST" else ('resnet34' if dataset=="FashionMNIST" else 'resnet50')
        data_param.update({"name": dataset})        
        traindata, testdata = LoadDataset(data_param, False)
        DataProcess = ImageProcess(resnet_type="resnet50", **data_param)
        DataProcess(dict(train=traindata, test=testdata))
        