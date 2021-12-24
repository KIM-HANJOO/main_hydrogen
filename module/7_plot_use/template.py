import os
import sys
import get_dir
di = get_dir.get_info()

main_dir = di.main_dir
prep_dir = di.prep_dir
model_dir = di.model_dir
module_dir = di.module_dir
facility_dir = di.facility_dir
nfc_dir = di.nfc_dir
plot_dir = di.plot_dir
cluster_dir = di.cluster_dir
facility_df = di.facility_df

os.chdir(module_dir)
sys.path.append(module_dir)
sys.path.append(module_dir + '//4_directory_module')
import directory_change as dich
import model_library as lib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgunbd.TTF"
font_path = "/mnt/c/Windows/Fonts/malgunbd.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
matplotlib.rcParams['axes.unicode_minus'] = False
import os
import glob
import os.path
import math
import random
from scipy.stats import beta, kde
import scipy.stats
import time

'''
note
'''

def read_excel(excel) :
    df = pd.read_excel(excel)
    if 'Unnamed: 0' in df.columns :
        df.drop('Unnamed: 0', axis = 1, inplace = True)

    if 'Unnamed: 0.1' in df.columns :
        df.drop('Unnamed: 0.1', axis = 1, inplace = True)

    return df

def ave(list1) :
	return sum(list1) / len(list1)

print(facility_dir)

for fc in os.listdir(facility_dir) :
    fcdir = os.path.join(facility_dir, fc)
    model1_dir = os.path.join(fcdir, 'model1')
    model2_dir = os.path.join(fcdir, 'model2')
    model3_dir = os.path.join(fcdir, 'model3')
    model4_dir = os.path.join(fcdir, 'model4')
    pre_dir = os.path.join(fcdir, 'preprocessed')

    for de in os.listdir(pre_dir) : #[weekday, weekend]
        print(f'working on {fc}, {de}')
        tempdir = os.path.join(pre_dir, de)
        os.chdir(tempdir)
        for excel in os.listdir(tempdir) :
            temp = read_excel(excel)

            if de == '주말' :
                end_list = []
                for index in range(temp.shape[0]) :
                    tempsum = temp.loc[index, :].sum()
                    end_list.append(tempsum)

                end_ave = ave(end_list)
                    
            if de == '주중' :
                day_list = []
                for index in range(temp.shape[0]) :
                    tempsum = temp.loc[index, :].sum()
                    day_list.append(tempsum)

                day_ave = ave(day_list)

            print(f'{excel} calc', end = '\r')
            
    print(f'{fc}, {de}, weekend = {round(end_ave, 3)}, weekday = {round(day_ave, 3)}')
