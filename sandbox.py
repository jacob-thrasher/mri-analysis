import pydicom as dicom
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

from data import ADNI_MRI
from utils import *

sample = '/home/jacob/Documents/data/T1-5mm-AXIAL/ADNI/003_S_0908/3_Plane_Localizer/2014-01-06_13_40_49.0/I404530/ADNI_003_S_0908_MR_3_Plane_Localizer__br_raw_20140113094243395_4_S210031_I404530.dcm'
folder = '/home/jacob/Documents/data/T1-5mm-AXIAL/ADNI/003_S_0908/3_Plane_Localizer/2014-01-06_13_40_49.0/I404530/'

root = '/home/jacob/Documents/data/T1-5mm-AXIAL/ADNI'
csvpath = '/home/jacob/Documents/data/cleaned_adni.csv'

# slices = sort_slices(folder)

# print(slices)
# volume = create_volume(slices)

# for s in range(5):
#     plt.imshow(volume[s, :, :])
#     plt.savefig(f'{s}.png')

dataset = ADNI_MRI(dataroot=root, csvpath=csvpath)


for i in tqdm(range(len(dataset))):
    volume, x = dataset[i]