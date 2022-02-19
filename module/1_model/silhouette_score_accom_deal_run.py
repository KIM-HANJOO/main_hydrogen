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

# ------------------------------------------------------------
# run silhouette_score_accom_deal.py
# ------------------------------------------------------------
import silhouette_score_accom_deal as ss

profile_dir = os.path.join(main_dir, 'accom_deal_profile_48')
save_dir = os.path.join(profile_dir, 'saved')
dich.newfolder(save_dir)

os.chdir(profile_dir)
# profile_48 = read_excel('profile_48_all.xlsx')
profile_48 = read_excel('profile_48_all_realname.xlsx')


# load original dataframe

profile_0 = read_excel('profile_48_group_0.xlsx')
profile_1 = read_excel('profile_48_group_1.xlsx')

df_0 = pd.DataFrame()
df_1 = pd.DataFrame()

df_0 = pd.concat([df_0, profile_0['excel']], axis = 0)
df_1 = pd.concat([df_1, profile_1['excel']], axis = 0)

df_0.columns = ['excel']
df_1.columns = ['excel']

df_0.loc[:, 'group'] = 0
df_1.loc[:, 'group'] = 1


df = pd.concat([df_0, df_1], axis = 0, ignore_index = True)

print(df_0.head())
print(df_1.head())
print(df.head())

# make merged_cluster_df

ncols = ['excel']
manova_cols = ['excel']
for i in range(1, 49) :
    ncols.append(str(i))
    manova_cols.append(f'hours_{i}')
profile_48.columns = ncols
manova_cols.append('group')

group_cols = ncols
group_cols.append('group')

facility = '판매및숙박'

excel_name = pd.DataFrame(profile_48['excel'])
all_labels = ss.silhouette_score_make(facility, profile_48, save_dir)

index_G = []
index_I = []

for index in range(profile_48.shape[0]) :
    if 'G' in profile_48.loc[index, 'excel'] :
        index_G.append(index)
    elif 'I' in profile_48.loc[index, 'excel'] :
        index_I.append(index)

all_labels.colums = group_cols

print(all_labels.head())


profile_G = profile_48.loc[index_G, :].copy()
profile_I = profile_48.loc[index_I, :].copy()

facility_G = '판매시설'
facility_I = '숙박시설'

G_labels = ss.silhouette_score_make(facility_G, profile_G, save_dir)
I_labels = ss.silhouette_score_make(facility_I, profile_I, save_dir)

G_labels.columns = group_cols
I_labels.columns = group_cols


print(G_labels.head())
print(I_labels.head())

G_labels.reset_index(drop = True, inplace = True)
I_labels.reset_index(drop = True, inplace = True)
for index in range(I_labels.shape[0]) :
    if I_labels.loc[index, 'group'] == 0 :
        I_labels.loc[index, 'group'] = 2
    if I_labels.loc[index, 'group'] == 1 :
        I_labels.loc[index, 'group'] = 3
print(I_labels.head())


merged_labels = pd.concat([G_labels, I_labels], axis = 0, ignore_index = True)
merged_labels.columns = manova_cols
# merged_labels = merged_labels.loc[:, 'hours_1' : 'group']

# match whole-set clusters & splitted clusters

merged_labels['org_group'] = None

group_0_excel = df_0.loc[:, 'excel'].tolist()
group_1_excel = df_1.loc[:, 'excel'].tolist()
print(group_0_excel)

for index in range(merged_labels.shape[0]) :
    if merged_labels.loc[index, 'excel'] in group_0_excel :
        merged_labels.loc[index, 'org_group'] = 0
    elif merged_labels.loc[index, 'excel'] in group_1_excel :
        merged_labels.loc[index, 'org_group'] = 1

    else :
        print(merged_labels.loc[index, 'excel'])



print(merged_labels.head())
merged_labels.to_excel('merged_labels_groups_added.xlsx')
 











