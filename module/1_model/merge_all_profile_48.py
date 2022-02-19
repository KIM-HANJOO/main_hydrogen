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

# --------------------------------------------------------------------
# merge all profile_48
# --------------------------------------------------------------------

org_facility_dir = facility_dir

accom_deal_facility_dir = os.path.join(main_dir, 'accom_deal_profile_48', 'FACILITEIS')

fc_group_df = pd.DataFrame(columns = ['label', 'facility', 'group'])
df_num = 0

fc_list_org = ['문화시설', '교육시설', '업무시설', '판매및숙박']
#fc_list_ad = ['판매시설', '숙박시설']

cols = ['excel', 'group']
ncols = ['excel', 'group']
for i in range(1, 49) :
    cols.append(str(i))
    ncols.append(f'hours_{i}')

all_profiles = pd.DataFrame(columns = cols)

fc_number = 0

for facility in fc_list_org :
    fc_dir = os.path.join(org_facility_dir, facility)
    model3_dir = os.path.join(fc_dir, 'model3')
    
    for group in [0, 1] :
        group_dir = os.path.join(model3_dir, f'group_{group}')
        os.chdir(group_dir)

        profile_48 = read_excel(f'profile_48_group_{group}.xlsx')
        group_number = fc_number * 2 + group
        profile_48['group'] = group_number 


        all_profiles = pd.concat([all_profiles, profile_48], axis = 0, ignore_index = True)
        all_profiles.columns = cols

        fc_group_df.loc[df_num, :] = [group_number, facility, group]
        df_num += 1

        print(f'{facility} | {group}')

    
    fc_number += 1

#for facility in fc_list_ad :
#    fc_dir = os.path.join(accom_deal_facility_dir, facility)
#    model3_dir = os.path.join(fc_dir, 'model3')
#    
#    for group in [0, 1] :
#        group_dir = os.path.join(model3_dir, f'group_{group}')
#        os.chdir(group_dir)
#
#        profile_48 = read_excel(f'profile_48_group_{group}.xlsx')
#        group_number = fc_number * 2 + group
#        profile_48['group'] = group_number 
#
#
#        all_profiles = pd.concat([all_profiles, profile_48], axis = 0, ignore_index = True)
#        all_profiles.columns = cols
#
#        fc_number += 1
#        print(f'{facility} | {group}')
#

all_profiles.columns = ncols

print(all_profiles)
os.chdir(os.path.join(main_dir, 'accom_deal_profile_48', 'all_profile_48'))
all_profiles.drop(['excel'], axis = 1, inplace = True)
all_profiles.to_excel('merged_profile_48.xlsx')

fc_group_df.to_excel('merged_profiel_48_fc_group_name.xlsx')

