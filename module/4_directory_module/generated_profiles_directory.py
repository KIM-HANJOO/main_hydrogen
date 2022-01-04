import os
import sys
import getdir4
di = getdir4.get_info()

main_dir = di.main_dir
prep_dir = di.prep_dir
model_dir = di.model_dir
module_dir = di.module_dir
facility_dir = di.facility_dir
plot_dir = di.plot_dir
cluster_dir = di.cluster_dir
facility_df = di.facility_df
facility_dict = di.facility_dict
gp_dir = di.gp_dir
gp_plot = di.gp_plot
print(f'main_dir = {main_dir}\n')

sys.path.append(module_dir)
sys.path.append(os.path.join(module_dir, '4_directory_module'))

import directory_change as dich
import discordlib_pyplot as dlt
import model_library as lib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "/mnt/c/Users/joo09/Documents/Github/fonts/D2Coding.ttf"
font = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font)

import os
from pathlib import Path
import glob
import os.path
import math
import random
from scipy.stats import beta, burr, kde
import scipy.stats
import shutil
import time

'''
note
'''
cwdir = os.getcwd()

def read_excel(excel) :
    df = pd.read_excel(excel)
    if 'Unnamed: 0' in df.columns :
        df.drop('Unnamed: 0', axis = 1, inplace = True)

    if 'Unnamed: 0.1' in df.columns :
        df.drop('Unnamed: 0.1', axis = 1, inplace = True)

    return df

def ave(list1) :
    return sum(list1) / len(list1)

'''
note
'''

subdir_list = ['raw', 'model1', 'model2', 'model3', 'model4']
fc_list = ['교육시설', '문화시설', '숙박시설', '업무시설', '판매시설']
fc_list_2 = ['교육시설', '문화시설', '판매및숙박', '업무시설', '판매시설', '숙박시설']

                
gfc_dir = os.path.join(main_dir, 'GENERATED_PROFILES')
dich.newfolder(gfc_dir)

dich.newfolderlist(gfc_dir, fc_list_2)
for fc in fc_list_2 :
    tempdir = os.path.join(gfc_dir, fc)
    dich.newfolderlist(tempdir, ['group_0', 'group_1'])
    
    for i in range(2) :
        dich.newfolderlist(os.path.join(tempdir, f'group_{i}'), subdir_list)
        
        for sd in subdir_list :
            if (sd == 'raw') | (sd == 'model3'):
                dich.newfolderlist(os.path.join(tempdir, f'group_{i}', sd), ['주중', '주말'])
    
    print(f'{fc} directory all made')
