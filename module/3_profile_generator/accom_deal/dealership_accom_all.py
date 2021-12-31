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

def pjoin(p1, p2) :
    return os.path.join(p1, p2) 

def pjoin(plist) :
    tempdir = plist[0]
    for i in range(len(plist)) :
        if i != 0 :
            tempdir = os.path.join(tempdir, plist[i])

    return tempdir

print(facility_dir)


df_all = pd.DataFrame(columns = ['m1_accom', 'm1_deal', 'm3_group0', 'm3_group1'])

# model3 | get daily average value with AMI Data

for facility in os.listdir(facility_dir) :
    if facility == '판매및숙박' :
        fcdir = os.path.join(facility_dir, facility)
        model1_dir = os.path.join(fcdir, 'model1')
        model2_dir = os.path.join(fcdir, 'model2')
        model3_dir = os.path.join(fcdir, 'model3')
        model4_dir = os.path.join(fcdir, 'model4')
        pre_dir = os.path.join(fcdir, 'preprocessed')

        for group in range(2) :
            df_num = 0
            tdir = model3_dir + f'//group_{group}//group_{group}_preprocessed'
            
            for de in os.listdir(tdir) : #[weekday, weekend]
                print(f'working on {facility}, group_{group}, {de}')
                tempdir = os.path.join(tdir, de)
                os.chdir(tempdir)
                for excel in os.listdir(tempdir) :
                    temp = read_excel(excel)
                    templist = []
                    allsum = temp.sum().sum()
                    allnum = temp.shape[0]

                    df_all.loc[df_num, f'm3_group{group}'] = allsum / allnum 
                    df_num += 1

#                    for index in range(temp.shape[0]) :
#                        tempsum = temp.loc[index, :].sum()
#                        templist.append(tempsum)
#
#                    temp_ave = ave(templist)
#                    df_all.loc[df_num, f'm3_group{group}'] = temp_ave
#                    df_num += 1
            
        #print(df)
        #df.to_excel('deal_accom_ami.xlsx')


#model1 | daily average value of SEUM Data

for facility in os.listdir(facility_dir) :
    if facility == '판매및숙박' :
        fcdir = os.path.join(facility_dir, facility)
        model1_dir = os.path.join(fcdir, 'model1')
        model2_dir = os.path.join(fcdir, 'model2')
        model3_dir = os.path.join(fcdir, 'model3')
        model4_dir = os.path.join(fcdir, 'model4')
        pre_dir = os.path.join(fcdir, 'preprocessed')

        os.chdir(model1_dir)
        temp_accom = read_excel("모델1_숙박시설.xlsx")
        temp_deal = read_excel("모델1_판매시설.xlsx")

        accom_list = temp_accom.iloc[:, 2].tolist()
        deal_list = temp_deal.iloc[:, 2].tolist()        
        accom_list.append(float(temp_accom.columns[2]))
        deal_list.append(float(temp_accom.columns[2]))


df_num = 0
for i in range(len(accom_list)) :
    df_all.loc[df_num, 'm1_accom'] = accom_list[i]
    df_num += 1


df_num = 0
for i in range(len(deal_list)) :
    df_all.loc[df_num, 'm1_deal'] = deal_list[i]
    df_num += 1


print(df_all)
os.chdir(cwdir)
df_all.to_excel('accom_deal_m1_m3.xlsx')

