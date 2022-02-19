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

# -----------------------------------------------------
# MODEL 3 get 90%, 10%
# -----------------------------------------------------

# merged version for accomodation & dealership facilities

# set top_percentile, middle_percentile, bottom_percentile
top_percentile = 90
middle_percentile = None # middle percentile : average
bottom_percentile = 10

facility_name = []
for subdir in os.listdir(facility_dir) :
    if 'params' != subdir :
        facility_name.append(subdir)

ncols = []
for i in range(1, 49) :
    ncols.append(str(i))

facility_dir = os.path.join(main_dir, 'accom_deal_profile_48', 'FACILITIES')
facility_name = ['판매시설', '숙박시설']


nindex = []
for facility in facility_name :
    for group in [0, 1] :   
        for which in ['top', 'middle', 'bottom'] :
            nindex.append(f'{facility}_{group}_{which}')

boundaries = pd.DataFrame(columns = ncols, index = nindex)
facility_dir = os.path.join(main_dir, 'accom_deal_profile_48', 'GENERATED_PROFILES')
facility_name = ['판매시설', '숙박시설']

for facility in facility_name :
    #model3_dir = os.path.join(facility_dir, facility, 'model3')

    for group in [0, 1] :
        #group_dir = os.path.join(model3_dir, f'group_{group}')
        group_dir = os.path.join(facility_dir, facility, f'group_{group}', 'model3')

        os.chdir(group_dir)
        #profile_48 = read_excel(f'profile_48_group_{group}.xlsx').loc[:, '1' : '48']
        profile_48 = read_excel('profile_48.xlsx')

        for i in range(1, 49) :
            temp_list = sorted(profile_48.loc[:, str(i)].tolist())

            now_index_name = f'{facility}_{group}'
            top_index_name = f'{now_index_name}_top'
            middle_index_name = f'{now_index_name}_middle'
            bottom_index_name = f'{now_index_name}_bottom'
            boundaries.loc[top_index_name, str(i)] = \
                    np.percentile(temp_list, top_percentile)
            boundaries.loc[bottom_index_name, str(i)] = \
                    np.percentile(temp_list, bottom_percentile)
            boundaries.loc[middle_index_name, str(i)] = \
                    ave(temp_list)

        print(f'{facility}, group_{group} done')


print(boundaries)
os.chdir(cwdir)
boundaries.to_excel('accom_deal_model3_boundaries_90%_10%.xlsx')
#dlt.shoot_file(cwdir, 'model3_boundaries_90%_10%.xlsx')
# 
#
#xvalues = []
#for i in range(1, 49) :
#    xvalues.append(i)
#
#for facility in facility_name :
#    for group in [0, 1] :
#        if 'top' in target_index : 
#            plt.plot(xvalues, boundaries.loc[target_index, '1' : '48'], 'b--', linewidth = 2, label = 'upper')
#        elif 'bottom' in target_index :
#            plt.plot(xvalues, boundaries.loc[target_index, '1' : '48'], 'b--', linewidth = 2, label = 'bottom')
#        else :
#            plt.plot(xvalues, boundaries.loc[target_index, '1' : '48'], 'r', linewidth = 3, label = 'average')
#
#
#
#        for target_index in boundaries.index.tolist() :
#
#
#
#
#
#            if facility in target_index :
#
#                if '0' in target_index :
#                    if 'top' in target_index : 
#                        plt.plot(xvalues, boundaries.loc[target_index, '1' : '48'], 'b--', linewidth = 2, label = 'upper')
#                    elif 'bottom' in target_index :
#                        plt.plot(xvalues, boundaries.loc[target_index, '1' : '48'], 'b--', linewidth = 2, label = 'bottom')
#                    else :
#                        plt.plot(xvalues, boundaries.loc[target_index, '1' : '48'], 'r', linewidth = 3, label = 'average')
#
#        for index in range(target_profile_48.shape[0]) :
#            plt.plot(xvalues, target_profile_48.loc[index, '1' : '48'], 'black', alpha = 0.2)
#
#
#        for facility in facility_name :
#            model3_dir = os.path.join(facility_dir, facility, 'model3')
#
#            for group in [0, 1] :
#                group_dir = os.path.join(model3_dir, f'group_{group}')
#                os.chdir(group_dir)
#                profile_48 = read_excel(f'profile_48_group_{group}.xlsx').loc[:, '1' : '48']
#
#        plt.xlim(1, 48)
#        plt.xlabel('hours')
#        plt.legend()
#        plt.title('교육시설, 80% boundaries')
#        plt.xticks(xvalues, ncols, rotation = 90)
#        plt.grid()
#
#        plt.savefig('model3_boundaries_90%_10%.png', dpi = 400)
#        dlt.savefig(cwdir,'model3_boundaries_90%_10%.png', dpi = 400)
#
#
#
