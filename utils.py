import pydicom as dicom
import numpy as np
import os
import pandas as pd
import datetime


def construct_df(root, csvpath):
    '''
    Constructs dataframe from ADNI data

    Args:
        root (str) - path to ADNI folder. Expects root to contain subject folders
    '''

    label_df = pd.read_csv(csvpath)
    df = pd.DataFrame(columns=['subject_id', 'visit', 'dx', 'volume_path'])
    
    for subject in os.listdir(root):
        sub_path = '' # Create a sub path to save to csv (allows more flexibility when moving data)
        sub_path = os.path.join(sub_path, subject, '3_Plane_Localizer')
        
        for visit in os.listdir(os.path.join(root, sub_path)):
            visit_path = os.path.join(sub_path, visit)
            scans = os.listdir(os.path.join(root, visit_path))[0] # Some patients have multiple scans in one visit, just get first scan

            # Path to subject scans to be saved to df
            volume_path = os.path.join(visit_path, scans) 

            # Parse visit date from folder name
            date = visit.split('_')[0]
            date = date.split('-')
            date = '/'.join(date[1:] + date[:1])
            if date[0] == '0': date = date[1:]

            # Use visit date to get DX
            row = label_df[label_df['Subject'] == subject]
            row = row[row['Acq Date'] == date].iloc[0] # If multuple scans in same day, take first one
            dx = row['Group']

            data = [subject, visit, dx, volume_path]
            df.loc[len(df)] = data
            

            # Write to df
    return df

