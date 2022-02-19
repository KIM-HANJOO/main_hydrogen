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


# -----------------------------------------------------------------
# compare clusters
# -----------------------------------------------------------------

load_dir = os.path.join(main_dir, 'accom_deal_profile_48', 'saved')
os.chdir(load_dir)

df = read_excel('merged_labels_groups_added.xlsx')
df.reset_index(drop = True, inplace = True)

df_0 = df[df['org_group'] == 0]
df_1 = df[df['org_group'] == 1]
df_0.reset_index(drop = True, inplace = True)
df_1.reset_index(drop = True, inplace = True)

numbers_df = pd.DataFrame(columns = ['group_0', 'group_1'], index = ['accom', 'deal'])
numbers_df.loc[:, :] = 0

for index in range(df.shape[0]) :
    now_type = None
    now_group = None

    if 'G' in df.loc[index, 'excel'] :
        now_type = 'deal'
    elif 'I' in df.loc[index, 'excel'] :
        now_type = 'accom'

    now_group = int(df.loc[index, 'org_group'])


    numbers_df.loc[now_type, f'group_{now_group}'] += 1

numbers_df.to_excel('accom_deal_group_ratio.xlsx') 




#
#df = df[['group', 'org_group']]
#
#labels = ['Group 0', 'Group 1']
#
#ngroup_0 = []
#ngroup_1 = []
#ngroup_2 = []
#ngroup_3 = []
#
## have new groups of num 0 to 3
#
#check_number = pd.DataFrame(columns = ['0', '1', '2', '3'], index = ['0', '1'])
#check_number.loc[:, :] = 0
#
#for index in range(df.shape[0]) :
#    org_group = df.loc[index, 'org_group']
#    new_group = df.loc[index, 'group']
#
#    check_number.loc[str(org_group), str(new_group)] += 1
#
#print(check_number)
#
#ngroup_0 = check_number.loc[:, '0']
#ngroup_1 = check_number.loc[:, '1']
#ngroup_2 = check_number.loc[:, '2']
#ngroup_3 = check_number.loc[:, '3']
#
#fig = plt.figure(figsize = (12, 5))
#
#check_number.columns = ['split_0', 'split_1', 'split_2', 'split_3']
#check_number.index = ['merged_0', 'merged_1']
#
#green_main = 'forestgreen'
#red_main = 'firebrick'
#
#color = [green_main, red_main]
#check_number = check_number.transpose()
#
#check_number.plot(kind = 'bar', color = color, stacked = True)
#plt.xticks(rotation = 0)
#
#dlt.savefig(load_dir, 'stacked_barplot_accom_deal_split_merge.png', 400)
#
#plt.clf()
#plt.cla()
#plt.close()
#
# -----------------------------------------------------------------
# print all ave_profiles
# -----------------------------------------------------------------
#
#org_group_0 = df[df['org_group'] == 0].loc[:, 'hours_1' : 'hours_48'].mean(axis = 0).tolist()
#org_group_1 = df[df['org_group'] == 1].loc[:, 'hours_1' : 'hours_48'].mean(axis = 0).tolist()
#
#new_group_0 = df[df['group'] == 0].loc[:, 'hours_1' : 'hours_48'].mean(axis = 0).tolist()
#new_group_1 = df[df['group'] == 1].loc[:, 'hours_1' : 'hours_48'].mean(axis = 0).tolist()
#new_group_2 = df[df['group'] == 2].loc[:, 'hours_1' : 'hours_48'].mean(axis = 0).tolist()
#new_group_3 = df[df['group'] == 3].loc[:, 'hours_1' : 'hours_48'].mean(axis = 0).tolist()
#
#
#fig = plt.figure(figsize = (12, 5))
#
#xvalues = []
#for i in range(48) :
#    xvalues.append(i)
#
#green_main = 'forestgreen'
#green_sub = 'darkseagreen'
#
#red_main = 'firebrick'
#red_sub = 'salmon'
#
#plt.plot(xvalues, org_group_0, c = 'forestgreen', linewidth = 3, label = 'merged_group_0')
#plt.plot(xvalues, org_group_1, c = 'firebrick', linewidth = 3, label = 'merged_group_1')
#
#plt.plot(xvalues, new_group_0, c = 'darkseagreen', label = 'splitted_group_0')
#plt.plot(xvalues, new_group_2, c = 'darkseagreen', label = 'splitted_group_2')
#plt.plot(xvalues, new_group_1, c = 'salmon', label = 'splitted_group_1')
#plt.plot(xvalues, new_group_3, c = 'salmon', label = 'splitted_group_3')
#
#plt.legend()
#plt.xticks(xvalues, rotation = 90)
#plt.xlabel('hours')
#plt.title('Hotel & Restaurant | Retail Compare')
#plt.xlim(0, 47)
#
#dlt.savefig(load_dir, 'hotel_restaurant_retail_compare.png', 400)
#
#
#
