import os 
import sys
import platform

import random
import numpy as np

# from sklearn.metrics import auc
import torch
import torchvision
import torchaudio

from data.dataset import get_dataloader
from data.metrics.metrics import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

saliency_path = r"./dataset"


if __name__ == "__main__":
    task_list = ['domain_increase']
    #  target['salmap'] = torch.ones(8, 1, 112, 112)
    print(f"use {platform.system()}, torch {torch.__version__}, torchvision {torchvision.__version__}, torchaudio {torchaudio.__version__}")
    for task in task_list:
        data_loader = get_dataloader(root=saliency_path, mode='train', task=task)
        print(f"{task= }, {len(data_loader)= }")
        for i,(data, target, vail) in enumerate(data_loader):
            print(i, data['rgb'].shape, data['audio'].shape, target['salmap'].shape, target['binmap'].shape)
            cc = CC( target['salmap'], target['salmap'])
            nss = NSS( target['salmap'], target['binmap'])
            sim = SIM( target['salmap'], target['salmap'])
            auc_judd = AUC_Judd( target['salmap'], target['binmap'])
            auc_borji = AUC_Borji( target['salmap'], target['binmap'])
            auc_shuffled = AUC_shuffled( target['salmap'], target['binmap'],target['binmap'])
            print(f"{cc= }, {nss= }, {sim= }, {auc_judd= }, {auc_borji= }, {auc_shuffled= }")
            break
    
    


