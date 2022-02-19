from itertools import *
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


def mean_of_specimen(list1, num) :
    combinated = list(combinations(list1, num))

    col = []
    for i in range(num) :
        col.append(str(num + 1))
    col.append('mean')

    df = pd.DataFrame(columns = col)
    df_num = 0

    ave1 = []
    for number in range(len(combinated)) :
        temp_list = list(combinated[number])
        ave1.append(ave(temp_list))

    return ave1


# -------------------------------------
# check the affect on safety factor
# daily average : daily average variation (beta distribution)
# check how safety factor in building energy consumption / design is defined
# -------------------------------------

facility_list = ['교육시설', '문화시설', '판매시설', '숙박시설', '업무시설']

facility_dir = os.path.join(main_dir, 'FACILITIES')
save_dir = os.path.join(main_dir, 'temp', 'model1_variation')

for facility in facility_list :
    if (facility != '판매시설') & (facility != '숙박시설') :
        model1_dir = os.path.join(facility_dir, facility, 'model1')
    else :
        model1_dir = os.path.join(facility_dir, '판매및숙박', 'model1')

    os.chdir(model1_dir)
    model1 = read_excel(f'모델1_{facility}.xlsx')
    model1_list = model1.iloc[:, 0].tolist()
    model1_list.append(model1.columns[0])

    fig = plt.figure(figsize = (7, 7))

    for num in [1, 3] :
        plt.hist(mean_of_specimen(model1_list, num), density = True, label = f'sample num = {num}')
        print(f'{num} done', end = '\r')
    plt.legend()
    plt.grid()
    os.chdir(save_dir)
    plt.title(f'{facility}\nbeta random distribution')
    plt.savefig(f'{facility}.png', dpi = 400)
    print(os.listdir(model1_dir))
    print(facility)
    print(model1.shape)
    print(model1.head())
