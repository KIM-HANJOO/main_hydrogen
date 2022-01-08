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
from pyclustertend import hopkins
from statsmodels.multivariate.manova import MANOVA

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

'''
hopkins test(statistic) : check the cluster tendency
'''
os.chdir(cwdir)
smpl = read_excel('sample(manova).xlsx')
smpl = smpl.loc[:, 'Sepal_Length' : 'Petal_Width']
print('sepal fucking smthing')
print(hopkins(smpl, smpl.shape[0]))

facility_list = [x for x in os.listdir(facility_dir) if x != 'params']

tendency_all = pd.DataFrame(columns = facility_list)
tendency_group = pd.DataFrame(columns = facility_list, index = ['group_0', 'group_1'])

fc_group_list = []
for fc in facility_list :
    for group in [0, 1] :
        fc_group_list.append(f'{fc}_{group}')



for facility in os.listdir(facility_dir) :
    if 'params' not in facility :
        print(facility)
        fc_dir = os.path.join(facility_dir, facility)

        cols = ['excel']
        for i in range(1, 49) :
            cols.append(str(i)) 

        profile_48_all = pd.DataFrame(columns = cols)

        for group in [0, 1] :
            print(f'group_{group}')
            group_dir = os.path.join(fc_dir, 'model3', f'group_{group}')
            os.chdir(group_dir)
            profile_48_group = read_excel(f'profile_48_group_{group}.xlsx')
            
            profile_48_all = pd.concat([profile_48_all, profile_48_group], ignore_index = True)

            tendency_group.loc[f'group_{group}', facility] = hopkins(profile_48_group.loc[:, '1' :], profile_48_group.shape[0])

        tendency_all.loc[0, facility] = hopkins(profile_48_all.loc[:, '1' :], profile_48_all.shape[0])


print(tendency_all)
print(tendency_group)

os.chdir(cwdir)
tendency_all.to_excel('cluster_tendency_facilities.xlsx')
tendency_group.to_excel('cluster_tendency_group.xlsx')

model2_col = []
for fc in facility_list :
    for de in ['주중', '주말'] :
        model2_col.append(f'{fc}_{de}')

model4_col = []
for fc in facility_list :
    for group in [0, 1] :
        for de in ['주중', '주말'] :
            model4_col.append(f'{fc}_{group}_{de}')

model1_hopkins = pd.DataFrame(columns = facility_list)
model2_hopkins = pd.DataFrame(columns = model2_col)
model4_hopkins = pd.DataFrame(columns = model4_col)

cols = ['group']
for i in range(1, 49) :
    cols.append(str(i))

manova_model4_day = pd.DataFrame(columns = cols)
manova_model4_end = pd.DataFrame(columns = cols)

for fc in facility_list :

    model1_file = f'모델1_{fc}.xlsx'
    model2_file = 'Model2_daily fraction.xlsx'
    model3_file = 'profile_48_group_0.xlsx'
    model4_weekdays_file = 'model4_weekdays.xlsx'
    model4_weekends_file = 'model4_weekends.xlsx'


    tempdir = os.path.join(facility_dir, fc)
    model1_dir = os.path.join(tempdir, 'model1')
    model2_dir = os.path.join(tempdir, 'model2')
    model3_dir = []
    for excel in os.listdir(model2_dir) :
        if 'Model2_daily' in excel :
            model2_file = excel
    
#    # model 1 hopkins
#    os.chdir(model1_dir)
#    temp = read_excel(model1_file)
#    temp = pd.DataFrame(temp.iloc[:, 0].tolist(), columns = ['target'])
#
#    model1_hopkins.loc[0, fc] = hopkins(temp, temp.shape[0])
#
#    # model 2 hopkins
#    os.chdir(model2_dir)
#    temp = read_excel(model2_file)
#    for de in ['주중', '주말'] :
#        column_now = f'{fc}_{de}'
#        temp = temp.loc[:, column_now]
#        model2_hopkins.loc[0, column_now] = hopkins(temp, temp.shape[0])

    # model 4 hopkins

    for group in [0, 1] :
        model3_dir.append(os.path.join(tempdir, 'model3', f'group_{group}'))
        pass

    model4_dir = os.path.join(tempdir, 'model4')
    for group in [0, 1] :
        model4_group_dir = os.path.join(model4_dir, f'group_{group}_model4')
        os.chdir(model4_group_dir)
        temp_day = read_excel(model4_weekdays_file)
        temp_end = read_excel(model4_weekends_file)

        temp_day.drop(['excel'], axis = 1, inplace = True)
        temp_end.drop(['excel'], axis = 1, inplace = True)

        temp_day = temp_day.loc[:, '1' :]
        model4_hopkins.loc[0, f'{fc}_{group}_주중'] = hopkins(temp_day, temp_day.shape[0])
        model4_hopkins.loc[0, f'{fc}_{group}_주말'] = hopkins(temp_end, temp_end.shape[0])

#        temp_day['group'] = f'{fc}_{group}_주중'
#        temp_end['group'] = f'{fc}_{group}_주말'

#        print(temp_day)
        
    # model 4 MANOVA
#        manova_model4_day = pd.concat([manova_model4_day, temp_day])
#        manova_model4_end = pd.concat([manova_model4_end, temp_end])

os.chdir(cwdir)
#print(model4_hopkins)
#model1_hopkins.to_excel('model1_hopkins')
#model2_hopkins.to_excel('model2_hopkins')
model4_hopkins.to_excel('model4_hopkins')

#manova_model4_day.to_excel('manova_model4_day.xlsx')
#manova_model4_end.to_excel('manova_model4_end.xlsx')



