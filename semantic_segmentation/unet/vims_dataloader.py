# -*- coding: utf-8 -*-


import os
import torch
import random
import numpy as np
from tqdm import tqdm
# from osgeo import gdal
import rasterio
from os.path import dirname as up
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import pytorch_lightning as pl
import pandas as pd


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# # Pixel-Level class distribution (total sum equals 1.0)
# class_distr = torch.Tensor([0.00452, 0.00203, 0.00254, 0.00168, 0.00766, 0.15206, 0.20232,
#  0.35941, 0.00109, 0.20218, 0.03226, 0.00693, 0.01322, 0.01158, 0.00052])



# These need to be re-defined 
# ********************************************************************************************

# class_distr = torch.Tensor([0.087, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833,
#                            0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833]) # testing with VIMS shoreline data which has 12 categories

# class_distr = torch.Tensor([0.06935144, 0.11771979, 0.1645004 , 0.15852254, 0.16526209,
#        0.16038279, 0.16426094]) # image size 1024 for CUSP

# class_distr = torch.Tensor([0.0867273 , 0.11636207, 0.16303951, 0.15027153, 0.16063503,
#        0.16078242, 0.16218214]) # Image size 256, 7 classes from CUSP

# class_distr = torch.Tensor([0.1052062 , 0.16581804, 0.24249604, 0.23955936, 0.24692036]) # 5 classes

class_distr = torch.Tensor([0.1343008 , 0.21761741, 0.32910008, 0.31898171]) # 4 classes


bands_mean = np.array([0.04783991, 0.04056812, 0.03163572, 0.02972606]).astype('float32') # the mean and std need to re-calculate

bands_std = np.array([0.04725893, 0.04743808, 0.04699043, 0.04967381]).astype('float32')

# ********************************************************************************************



# bands_std = np.array([0.04725893, 0.04743808, 0.04699043, 0.04967381, 0.04946782, 0.06458357,
#  0.07594915, 0.07120246, 0.08251058, 0.05111466, 0.03524419]).astype('float32')

###############################################################
# Pixel-level Semantic Segmentation Data Loader               #
###############################################################

# dataset_path = os.path.join(up(up(up(__file__))), 'data')
dataset_path = os.path.join(up(up(up(up(up(__file__))))), 'VIMS', 'NAIP', 'VA_NAIP_2018_8977') #, '256_images_lesstrain'

# dataset_path = os.path.join(up(up(up(up(up(__file__))))), 'VIMS', 'Shoreline', 'shoreline_data_unet')

class GenDEBRIS(Dataset): # Extend PyTorch's Dataset class

    def __init__(self, mode = 'train', transform=None, standardization=None, path = dataset_path, agg_to_water= False):
        
        if mode=='train':
#             self.ROIs = np.genfromtxt(os.path.join(path, 'splits', 'train_X.txt'),dtype='str')
            self.ROIs = np.array([name for name in os.listdir(os.path.join(path, 'train')) if os.path.isfile(os.path.join(path, 'train', name))])
                
        elif mode=='test':
#             self.ROIs = np.genfromtxt(os.path.join(path, 'splits', 'test_X.txt'),dtype='str')
            self.ROIs = np.array([name for name in os.listdir(os.path.join(path, 'test')) if os.path.isfile(os.path.join(path, 'test', name))])
                
        elif mode=='val':
#             self.ROIs = np.genfromtxt(os.path.join(path, 'splits', 'val_X.txt'),dtype='str')
            self.ROIs = np.array([name for name in os.listdir(os.path.join(path, 'val')) if os.path.isfile(os.path.join(path, 'val', name))])
            
        else:
            raise
        
            
        self.X = []           # Loaded Images
        self.y = []           # Loaded Output masks
            
        for roi in tqdm(self.ROIs, desc = 'Load '+mode+' set to memory'):
            
            # Construct file and folder name from roi
            roi_file = os.path.join(path, mode, roi)
            roi_file_mask = os.path.join(path, 'labels', roi)
            
            # Load Classsification Mask
            ds = rasterio.open(roi_file_mask)
            temp = np.copy(ds.read().astype(np.int64))
            
            # Aggregation

            if agg_to_water:
                temp[temp==6]=0          # Remove class 6 and 7 (quay/wharf and jetty) from detection
                temp[temp==7]=0
                temp[temp==3]=0
                temp[temp==5]=3 # Since remove the class3, so we need to fill it with the old class 5

#                 temp[temp==0]=-1000

            
            # Categories from 1 to 0

#             temp = np.copy(temp - 1)
            ds=None                   # Close file
            
            self.y.append(temp)
            
            # Load Patch
            ds = rasterio.open(roi_file)
            temp = np.copy(ds.read())
            ds=None
            self.X.append(temp)      

        self.impute_nan = np.tile(bands_mean, (temp.shape[1],temp.shape[2],1))
        self.mode = mode
        self.transform = transform
        self.standardization = standardization
        self.length = len(self.y)
        self.path = path
        self.agg_to_water = agg_to_water
        
    def __len__(self):

        return self.length
    
    def getnames(self):
        return self.ROIs    
    
    def __getitem__(self, index):
        
        img = self.X[index]
        target = self.y[index]

        img = np.moveaxis(img, [0, 1, 2], [2, 0, 1]).astype('float32')       # CxWxH to WxHxC
        
        nan_mask = np.isnan(img)
        img[nan_mask] = self.impute_nan[nan_mask]
        
        if self.transform is not None:
#             target = target[:,:,np.newaxis]
            
            target = np.moveaxis(target, [0, 1, 2], [2, 0, 1])    # CxWxH to WxHxC
            
            stack = np.concatenate([img, target], axis=-1).astype('float32') # In order to rotate-transform both mask and image
        
            stack = self.transform(stack)

            img = stack[:-1,:,:]
            target = stack[-1,:,:].long() # Recast target values back to int64 or torch long dtype
            
#             img = stack[:,:,:-1]
#             target = stack[:,:,-1]
        
        if self.standardization is not None:
            img = self.standardization(img)
        
        
        return {'image': img, 'mask': target}
#         return img, target
    
###############################################################
# Transformations                                             #
###############################################################
class RandomRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)
    
###############################################################
# Weighting Function for Semantic Segmentation                #
###############################################################
def gen_weights(class_distribution, c = 1.02):
    return 1/torch.log(c + class_distribution)


# class GeoNAIPDataModule(pl.LightningDataModule):
    
#     def __init__(self, data: pd.DataFrame, batch_size: int):
        
        



