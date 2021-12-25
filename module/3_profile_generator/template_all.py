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

sys.path.append(module_dir)
sys.path.append(module_dir + '//4_directory_module')
import directory_change as dich
import model_library as lib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager, rc
#font_path = "C:/Windows/Fonts/malgunbd.TTF"
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

print(facility_dir)

fc_list = []
for fc in os.listdir(facility_dir) :
    fc_list.append(fc)

df = pd.DataFrame(columns = fc_list, index = ['weekday', 'weekend'])

for facility in os.listdir(facility_dir) :
    if facility == '판매및숙박' :
        fcdir = os.path.join(facility_dir, facility)
        model1_dir = os.path.join(fcdir, 'model1')
        model2_dir = os.path.join(fcdir, 'model2')
        model3_dir = os.path.join(fcdir, 'model3')
        model4_dir = os.path.join(fcdir, 'model4')
        pre_dir = os.path.join(fcdir, 'preprocessed')

        df = pd.DataFrame(columns = ['group_0', 'group_1'])
        for group in range(2) :
            df_num = 0
            tdir = model3_dir + f'//group_{group}//group_{group}_preprocessed'
            
            for de in os.listdir(tdir) : #[weekday, weekend]
                print(f'working on {fc}, group_{group}, {de}')
                tempdir = os.path.join(tdir, de)
                os.chdir(tempdir)
                for excel in os.listdir(tempdir) :
                    temp = read_excel(excel)
                    templist = []
                    for index in range(temp.shape[0]) :
                        tempsum = temp.loc[index, :].sum()
                        templist.append(tempsum)

                    temp_ave = ave(templist)
                    df.loc[df_num, f'group_{group}'] = temp_ave
                    df_num += 1
            
        os.chdir(cwdir)
        print(df)
        df.to_excel('deal_accom_ami.xlsx')


for facility in os.listdir(facility_dir) :
    if facility == '판매및숙박' :
        fcdir = os.path.join(facility_dir, facility)
        model1_dir = os.path.join(fcdir, 'model1')
        model2_dir = os.path.join(fcdir, 'model2')
        model3_dir = os.path.join(fcdir, 'model3')
        model4_dir = os.path.join(fcdir, 'model4')
        pre_dir = os.path.join(fcdir, 'preprocessed')

