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
from pathlib import Path

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

# ------------------------------------------------
# check model2 excel, area with usage
# ------------------------------------------------

facility_dir = os.path.join(main_dir, 'accom_deal_profile_48', 'FACILITIES')
facility_list = ['숙박시설', '판매시설']

fig = plt.figure(figsize = [15, 10])

for number, facility in enumerate(facility_list) :
    model1_dir = os.path.join(facility_dir, facility, 'model1')
    model2_dir = os.path.join(facility_dir, facility, 'model2')

    os.chdir(model1_dir)
    model1 = read_excel(f'모델1_{facility}.xlsx')
    
    os.chdir(model2_dir)
    model2 = read_excel('Model2_daily fractoin.xlsx')

    for column in model2.columns :
        if facility[ : 2] in column :
            if '주중' in column :
                print('weekday', column)
                model2_day_tgt = model2[column].tolist()

            elif '주말' in column :
                model2_end_tgt = model2[column].tolist()

    tgt_length = len(model2_day_tgt) + len(model2_end_tgt)

    result = model1.iloc[:, 0].tolist()
    result.append(model1.columns[0])

    source = model1.iloc[:, 2].tolist()
    source.append(model1.columns[2])

    ax = fig.add_subplot(1, 2, number + 1)

#    print(min(source), max(source))
#    print(min(result), max(result))

    devide = []

    print(len(result), tgt_length, tgt_length/ len(result))
    
    for index in range(model1.shape[0]) :
        #print(f'{round(result[index], 3)}\t{round(source[index], 3)}\n{round(source[index] / result[index], 3)}')
        devide.append(source[index] / result[index])

    #print(min(devide), ave(devide), max(devide))
    ax.boxplot(devide)
    ax.set_title(f'{facility} model2 devided range')
    


cwdir = Path(cwdir)
module_dir = cwdir.parent
tmp_dir = os.path.join(module_dir, 'tmp')

dlt.savefig(tmp_dir, 'model2_range.png', 400)
    


    
