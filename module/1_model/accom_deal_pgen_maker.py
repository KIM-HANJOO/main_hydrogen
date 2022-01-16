import os
import sys
import get_dir
di = get_dir.get_info()

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

# ----------------------------------------------------------
# find manova data
# ----------------------------------------------------------

facility_list = []
for facility in os.listdir(facility_dir) :
    if 'params' != facility :
        facility_list.append(facility)

ad_dir = os.path.join(main_dir, 'accom_deal_profile_48')
os.chdir(ad_dir)

all_df = pd.DataFrame()

for excel in os.listdir(ad_dir) :
    if 'all' not in excel :
        os.chdir(ad_dir)
        profile_48 = read_excel(excel)
        
        for index in range(profile_48.shape[0]) :
            if 'G' in profile_48.loc[index, 'excel'] :
                profile_48.loc[index, 'excel'] = 'G'
            elif 'I' in profile_48.loc[index, 'excel'] :
                profile_48.loc[index, 'excel'] = 'I'

        all_df = pd.concat([all_df, profile_48], ignore_index = True)
        print(f'{excel} done')

ncols = ['group']

for i in range(1, 49) :
    ncols.append(f'hour_{i}')

all_df.columns = ncols

all_df.to_excel('profile_48_all.xlsx')
print('excel saved')
