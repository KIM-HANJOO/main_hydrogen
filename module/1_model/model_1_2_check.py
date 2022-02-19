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

def get_iqr(list1) :
    Q1 = np.percentile(list1, 25)
    Q3 = np.percentile(list1, 75)
    return Q3 - Q1



# ----------------------------------------------------------------------
# check model1 and model2, connection btween  daily average and daily fraction
# ----------------------------------------------------------------------

facility_list = ['숙박시설', '판매시설']

main_dir = os.path.join(main_dir, 'accom_deal_profile_48')
facility_dir = os.path.join(main_dir, 'FACILITIES')
plot_dir = os.path.join(main_dir, 'temp_plot')

for facility in facility_list :
    raw_dir = os.path.join(facility_dir, facility, 'preprocessed')
    wd_dir = os.path.join(raw_dir, '주중')
    we_dir = os.path.join(raw_dir, '주말')

    df = pd.DataFrame(columns = ['ave', 'fraction'])
    df_num = 0
    df_iqr = pd.DataFrame(columns = ['ave', 'iqr'])
    iqr_num = 0

    fig = plt.figure(figsize = [20, 20])

    for de_num, de in enumerate(['주중', '주말']) :
        temp_dir = os.path.join(raw_dir, de)
        os.chdir(temp_dir)

        for num, excel in enumerate(os.listdir(temp_dir)) :
            if num > -1 :
                fraction = []
                temp = read_excel(excel)
                ave_day = temp.sum().sum() / temp.shape[0]

                for index in range(temp.shape[0]) :
                    ft = temp.loc[index, :].sum() / ave_day
                    df.loc[df_num, :] = [ave_day, ft]
                    fraction.append(ft)
                    df_num += 1
                iqr = get_iqr(fraction)
                df_iqr.loc[iqr_num, :] = [ave_day, iqr]
                iqr_num += 1
                print(f'{excel} done', end = '\r')


        ax1 = fig.add_subplot(2, 2, 1 + 2 * de_num) 
        ax2 = fig.add_subplot(2, 2, 2 + 2 * de_num)

        title = f'{facility}, {de}'
        ax1.scatter(df.loc[:, 'ave'], df.loc[:, 'fraction'])
        ax1.set_xlabel('ave')
        ax1.set_ylabel('fraction')
        ax1.set_title(f'{title}, fraction')


        ax2.scatter(df_iqr.loc[:, 'ave'], df_iqr.loc[:, 'iqr'])
        ax2.set_xlabel('ave')
        ax2.set_ylabel('iqr')
        ax2.set_title(f'{title}, iqr')

    os.chdir(plot_dir)
    plt.savefig(f'{title}.png', dpi = 400)
    plt.clf()
            
    print(title, 'saved')

