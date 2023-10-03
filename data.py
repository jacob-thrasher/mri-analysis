import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import os
import numpy as np
import pydicom as dicom
import cv2
import torchio as tio

class ADNI_MRI(Dataset):
    def __init__(self, dataroot, csvpath, depth=3, sel_method='center'):
        '''
        MRI Dataset from ADNI data

        Args:
            dataroot(str): path to root of data directory
            csvpath (str): path to csv label file
            depth   (int): number of slices to include in volume. 
            sel_method   : if more slices than depth, method used to select slices
                            [center, first, last]
        '''
        self.root = dataroot
        self.df = pd.read_csv(csvpath)
        self.depth = depth
        self.sel_method = sel_method
        self.rescale = tio.RescaleIntensity((0, 1))

        self.label_dict = {'CN': 0,
                           'MCI': 1,
                           'AD': 2}

        print(np.unique(self.df['dx']))
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dx = row['dx']
        if dx in ['MCI', 'EMCI', 'LMCI', 'SMC']: dx = 'MCI'
        dx = self.label_dict[dx] 

        path = os.path.join(self.root, row['volume_path'])
        slices = self.sort_slices(path)
        volume = torch.from_numpy(self.create_volume(slices))

        if torch.sum(torch.isnan(volume)) > 0: print("has NAN")
        
        volume = self.rescale(volume.unsqueeze(0))
        return volume, dx
    
    def sort_slices(self, volume_path):
        files = os.listdir(volume_path)
        # slice_paths = ['']*len(files)
        slice_paths = []
        indices = []
        for f in files:
            index = int(f.split('_')[-3])
            # slices[index-1] = os.path.join(volume_path, f)
            indices.append(index)
            slice_paths.append(os.path.join(volume_path, f))

    
        return [s for _, s in sorted(zip(indices, slice_paths))]

    def create_volume(self, slices):
        volume = np.empty((self.depth, 256, 256))
        if self.sel_method == 'center':
            center = int(len(slices) / 2) # Caluclate center of list 
            if self.depth % 2 == 1: end_length = int((self.depth - 1) / 2)
            else: end_length = self.depth / 2

            slices = slices[center-end_length:center+end_length]


        for i, s in enumerate(slices):
            img = dicom.dcmread(s).pixel_array
            img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            volume[i] = img

        return volume
