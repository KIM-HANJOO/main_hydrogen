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
nfc_dir = di.nfc_dir
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
# make model1 to model4
# -----------------------------------------------------
'''
note

use 'model_all_units.py' as one and only module
'''
import model_all_units as mu

facility_list = ['업무시설', '판매시설', '숙박시설']

for facility in facility_list : #os.listdir(gp_dir) :
    if 'params' != facility :
        for group in range(2) :
            check = 0 
                
#            if (facility == '업무시설') | (facility == '판매및숙박') : #& (group == 0):
#                if facility == '업무시설' :
#                    if group == 1 :
#                        check = 1
#                else :
#                    check = 1
            if facility != '업무시설' : 
                check = 1

            if check == 1 :

                tdir = os.path.join(gp_dir, facility)
                print(facility, '\t', 'group number', group)
                nmaindir = os.path.join(tdir, f'group_{group}')
                model1_dir = os.path.join(nmaindir, 'model1')
                model2_dir = os.path.join(nmaindir, 'model2')
                model3_dir = os.path.join(nmaindir, 'model3')
                model4_dir = os.path.join(nmaindir, 'model4')
                final_dir = os.path.join(nmaindir, 'raw')
                plot_dir = os.path.join(gp_plot, facility)
                dich.newfolder(plot_dir)




                print('\nmodel_1 running')
                mu.model_1(facility, final_dir, model1_dir)
                print('\nmodel_2 running')
                mu.model_2(facility, final_dir, model2_dir)
                print('\nmodel_3 running')
                mu.model_3(final_dir, model3_dir)
                print('\nmodel_4 running')
                mu.model_4(model3_dir, model4_dir)
               
                plot_solo = os.path.join(plot_dir, 'solo')
                dich.newfolder(plot_solo)
                
                print('\nmodel_1 plotting')
                mu.model1_plot(facility, group, model1_dir, plot_solo)
                print('\nmodel_2 plotting')
                mu.model2_plot(facility, group, model2_dir, plot_solo)
                print('\nmodel_3 plotting')
                mu.model3_plot(facility, group, model3_dir, plot_solo)
                print('\nmodel_4 plotting')
                mu.model4_plot(facility, group, model4_dir, plot_solo)
                
                nplot_dir = os.path.join(plot_dir, 'compare')
                dich.newfolder(nplot_dir)
                
#                print('\nmodel_1 compare')
#                mu.model1_compare(facility, group, model1_dir, nplot_dir, nfc_dir)
##                print('\nmodel_2_all')
##                mu.model2_all(facility, model2_dir, plot_dir, nfc_dir)
#                print('\nmodel_2 compare')
#                mu.model2_compare(facility, group, model2_dir, nplot_dir, nfc_dir)
#                print('\nmodel_3 compare')
#                mu.model3_compare(facility, group, model3_dir, nplot_dir, nfc_dir)
#                print('\nmodel_4 compare')
#                mu.model4_compare(facility, group, model4_dir, nplot_dir, nfc_dir)
