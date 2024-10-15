# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:57:36 2024

@author: Rita Saraiva

"""

# %% Creating a clean Slate

from IPython import get_ipython
get_ipython().magic('reset -sf')
import os
import shutil
import numpy as np
import pandas as pd
#Directing to the correct working directory

#%%

TrainData_dir = 'C:\\Users\\ritux\\OneDrive - Danmarks Tekniske Universitet\\Skrivebord\\Hackthon\\delightful_lightbulb\\data\\TrainingSet - Copy\\'
 
os.chdir(TrainData_dir)

for Project in os.listdir( ): # Project = 'Project1'
    os.chdir( TrainData_dir + '\\' +  Project )
    
    Objects_df = pd.read_csv(Project+'.csv')
    Objects_df['Ind'] = Objects_df.index
    
    # Source path 
    Mask_Folder =  os.getcwd() + '\\' + Project + '.masks'
    Files = os.listdir(Mask_Folder)
    Valid_Files = []
    Test_Files = []
    
    Labels = Objects_df[Objects_df.columns[1]].unique()
    for Label in Labels:  # Label = Labels[0]
    
        Obj_label_df = Objects_df[Objects_df[Objects_df.columns[1]] == Label]
        
        Idx   = np.array(Obj_label_df['Ind'])
        Valid = np.random.choice(Idx,round(len(Idx)*0.2), replace=False)
        Test  = np.setdiff1d(Idx, Valid)
        
        Valid_Files += [Files[i] for i in Valid]
        Test_Files += [Files[i] for i in Test]
        

    # Create the new folder to copy the files into
    Valid_Mask_Folder = "Valid Data Masks"
    os.makedirs(Valid_Mask_Folder, exist_ok=True)
    destination = os.getcwd() + '\\' + Valid_Mask_Folder

    # Copy each valid file from the source to the destination folder
    for file in Valid_Files:
        shutil.copy(os.path.join(Mask_Folder, file), os.path.join(Valid_Mask_Folder, file))
    
    # Create the new folder to copy the files into
    Test_Mask_Folder = "Test Data Masks"
    os.makedirs(Test_Mask_Folder, exist_ok=True)
    destination = os.getcwd() + '\\' + Test_Mask_Folder

    # Copy each valid file from the source to the destination folder
    for file in Test_Files:
        shutil.copy(os.path.join(Mask_Folder, file), os.path.join(Test_Mask_Folder, file))


        
        

        
        