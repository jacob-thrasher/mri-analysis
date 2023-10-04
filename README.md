# mri-analysis

# Non-standard Required libraries
1. PyTorch
2. TorchIO
3. TorchMetrics
4. timm
5. pydicom

# Data
All data collected from the Alzhiemer's Disease Neuroimaging Initiative (ADNI) database. Specific MRIs used are T1 weighted MRIs on the axial plane. The images used were gathered on the 3 plane localizer. 

# Using the dataset class
When using the `ADNI_MRI` dataset class, direct the `dataroot` variable to the folder containing the subject subfolders. `csvpath` will be the path to the label csv found in the Github repo.

`depth` and `sel_method` can be ignored

# Executing the code
To train, execute the file `train.py`. There will be a `model_type` variable directly under the `valid_step()` function. `model_type == inception` will execute the inception backed model. Otherwise, the untrained CNN will be used

Loss figures will be saved as `loss.png` in the project's root directory
