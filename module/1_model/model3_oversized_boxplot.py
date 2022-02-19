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

# --------------------------------------------------------------------------
# Oversize check
# --------------------------------------------------------------------------


# load merged_profile_48, add org_group column

profile_dir = os.path.join(main_dir, 'accom_deal_profile_48', 'all_profile_48')
os.chdir(profile_dir)
profile_48 = read_excel('merged_profile_48.xlsx')

profile_48['org_group'] = None
for index in range(profile_48.shape[0]) : 
    profile_48.loc[index, 'org_group'] = int(profile_48.loc[index, 'group'] // 2)
print(profile_48['org_group'])

# check oversize for hours

fig = plt.figure(figsize = (30, 20))
xvalues = []
for i in range(1, 49) :
    xvalues.append(i)

facility_list = ['문화시설', '교육시설', '업무시설', '판매및숙박']

for group in range(4) :
    df_now = profile_48[profile_48['org_group'] == group]
    df_now.reset_index(drop = True, inplace = True)
    print(facility_list[group])
    print(df_now.shape[0])

    df_0 = df_now[df_now['group'] == 2 * group + 0]
    df_1 = df_now[df_now['group'] == 2 * group + 1]
    df_0.reset_index(drop = True, inplace = True)
    df_1.reset_index(drop = True, inplace = True)

    profile_all = df_now.loc[:, 'hours_1' : 'hours_48'].mean().tolist()
    group_0 = pd.DataFrame(columns = xvalues)
    group_1 = pd.DataFrame(columns = xvalues)

    for index in range(df_0.shape[0]) :
        for i in range(1, 49) :
            group_0.loc[index, f'hours_{i}'] = df_0.loc[index, f'hours_{i}'] - profile_all[i - 1]

    for index in range(df_1.shape[0]) :
        for i in range(1, 49) :
            group_1.loc[index, f'hours_{i}'] = df_1.loc[index, f'hours_{i}'] - profile_all[i - 1]


    ax = None
    ax = fig.add_subplot(2, 2, group + 1)

    for i in range(1, 49) :
        ax.boxplot(group_0.loc[:, f'hours_{i}'], positions = [i], showfliers = False, whis = 0)
        ax.boxplot(group_1.loc[:, f'hours_{i}'], positions = [i], showfliers = False, whis = 0)

    ax.plot([0, 49], [0, 0], c = 'red')
        

    ax.set_xticks(xvalues, rotation = 90)
    ax.set_xlim(1, 48)
    ax.set_xlabel('hours')
    ax.set_title(f'{facility_list[group]}')

dlt.savefig(profile_dir, 'watch_boxplot.png', 400)

    


