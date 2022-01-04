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
import ClusterAnalysis as ca

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


# --------------------------------------------------------
# Merge all facility's model 3
# --------------------------------------------------------

cols = ['excel']
for i in range(1, 49) :
    cols.append(str(i))
big_df = pd.DataFrame(columns = cols)

concat_number = 0
for subdir in os.listdir(facility_dir) :
    model3_dir = os.path.join(facility_dir, subdir, 'model3')
    for group in range(2) :
        tempdir = os.path.join(model3_dir, f'group_{group}')

        os.chdir(tempdir)
        df = read_excel(f'profile_48_group_{group}.xlsx')

        big_df = pd.concat([big_df, df], ignore_index = True)

        concat_number += 1
        print(f'{concat_number} of {8} | {subdir}, {group}, len = {df.shape[0]}')

print(big_df.shape[0]) 

num_clusters = []
for i in range(2, 20) :
    num_clusters.append(i)

cluster_df, all_score_df = ca.all_score(big_df.loc[:, '1' : ], num_clusters)
df_optimal = ca.all_score_plot(big_df.loc[:, '1' :], all_score_df, 1)
dlt.savefig(os.path.join(gp_plot, 'temp'), 'cluster.png', 400)
os.chdir(os.path.join(gp_plot, 'temp'))
df_optimal.to_excel('scores_optimal_K.xlsx')
print(all_score_df)
print(cluster_df)
print(df_optimal)
