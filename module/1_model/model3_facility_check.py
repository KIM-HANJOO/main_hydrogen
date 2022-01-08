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
import model_all_units as mu

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





# ---------------------------------------------------------------
# before / after clustering
# compare model 3
# ---------------------------------------------------------------


def model3_make(profile_48) :
    ave_list = []
    for col in profile_48.columns :
        ave_list.append(profile_48.loc[:, col].mean())

    return ave_list


facility_list = [x for x in os.listdir(facility_dir) if x != 'params']

cols = ['excel']
for i in range(1, 49) :
    cols.append(str(i))

profile_48_all = pd.DataFrame(columns = cols)

fig = plt.figure(figsize = (16, 10))

for number, facility in enumerate(facility_list) :
    for group in [0, 1] :
        tempdir = os.path.join(facility_dir, facility, 'model3', f'group_{group}')

        os.chdir(tempdir)

        profile_48_group = read_excel(f'profile_48_group_{group}.xlsx')
        profile_48_group.columns = profile_48_group.columns.astype(str)
        profile_48_all = pd.concat([profile_48_all, profile_48_group], ignore_index = True)



        profile_48_group = profile_48_group.loc[:, '1' : '48']
        
        globals()[f'ave_list_{group}'] = model3_make(profile_48_group)

    profile_48_all = profile_48_all.loc[:, '1' : '48']
    ave_list_all = model3_make(profile_48_all)

    ax = fig.add_subplot(2, 2, number + 1)

    hours = []
    for i in range(1, 49) :
        hours.append(i)

    ax.plot(hours, ave_list_0, 'g--', label = 'group 0')
    ax.plot(hours, ave_list_1, 'g--', label = 'group 1')
    ax.plot(hours, ave_list_all, 'r', linewidth = 3, label = 'ungrouped(all)')

    ax.legend()
    ax.set_xlabel('hours')
    ax.set_title(f'{facility}')
    ax.set_xlim([1, 48])
    ax.set_xticks([1, 12, 24, 36, 48], rotation = 90)
    #ax.grid()
    os.chdir(cwdir)

dlt.savefig(cwdir, 'model3_compare.png', 400)


    
