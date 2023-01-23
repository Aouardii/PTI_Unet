import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from monai.utils import first, set_determinism
from monai.transforms import(
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Activations,
)

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import CacheDataset, DataLoader, Dataset

import torch
import matplotlib.pyplot as plt


from glob import glob
import numpy as np
from utilities import dice_metric
from monai.losses import DiceLoss
from Load import prepareTest

#Permet de tester un modele deja entrainé sur une base de donnée

device = torch.device("cuda:0")

#Changer 99 par 11/22/33/44/55/66 selon le modele que l'on souhaite essayer
path = r"H:\TDSI\cc359_preprocessed\Stockage\MRI99\best_metric_model.pth"

#base de donnée sur laquelle on souhaite tester
CibleMRInumber = 3


cible_dir = 'H:\TDSI\cc359_preprocessed\MRI' + str(CibleMRInumber)

loss = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)

test_epoch_loss = 0
test_metric = 0
epoch_metric_test = 0
test_step = 0

test_loader = prepareTest(cible_dir, cache=True)

model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

model.load_state_dict(torch.load(path))

#model = torch.load(path)
model.eval() 
with torch.no_grad():

    


    for test_data in test_loader:

        test_step += 1

        test_volume = test_data["vol"]
        test_label = test_data["seg"]
        test_label = test_label != 0
        test_volume, test_label = (test_volume.to(device), test_label.to(device),)
                                    
        test_outputs = model(test_volume)
                                    
        test_loss = loss(test_outputs, test_label)
        test_epoch_loss += test_loss.item()
        test_metric = dice_metric(test_outputs, test_label)
        epoch_metric_test += test_metric
                                
                            
    test_epoch_loss /= test_step
    print(f'test_loss_epoch: {test_epoch_loss:.4f}')
    epoch_metric_test /= test_step
    print(f'test_dice_epoch: {epoch_metric_test:.4f}')






