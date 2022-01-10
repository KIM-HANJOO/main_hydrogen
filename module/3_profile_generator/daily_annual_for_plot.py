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



# ---------------------------------------------------------------
# Make facility lists
# ---------------------------------------------------------------

facility_list = []
for facility in os.listdir(facility_dir) :
    if 'params' != facility :
        facility_list.append(facility)
daily_cols = []
for facility in facility_list :
    for group in [0, 1] :
        daily_cols.append(f'{facility}_{group}_주중')
        daily_cols.append(f'{facility}_{group}_주말')
        daily_cols.append(f'{facility}_{group}_주중+주말')

annual_cols = []
for facility in facility_list :
    for group in [0, 1] :
        annual_cols.append(f'{facility}_{group}')



# make dataframe to log daily, anual datas
#actual_daily = pd.DataFrame(columns = daily_cols)
#actual_annual = pd.DataFrame(columns = annual_cols)
#
#gen_daily = pd.DataFrame(columns = daily_cols)
#gen_annual = pd.DataFrame(columns = annual_cols)
#
actual_daily = pd.DataFrame()
actual_annual = pd.DataFrame()

gen_daily = pd.DataFrame()
gen_annual = pd.DataFrame()

# ---------------------------------------------------------------
# facility_dir (actual profiles)
# ---------------------------------------------------------------
# facility_dir  //  교육시설    //  model3  //  group_0 //  group_0_preproccessed
# //  '주중', '주말'  //  []_0_주말_봄가을.xlsx ( : -11)

now_dir = facility_dir

ncols_daily = []
ncols_annual = []

for facility in facility_list :
    daily_list = []
    annual_list = []
    for group in [0, 1] :
        
        group_dir = os.path.join(now_dir, facility, 'model3', f'group_{group}', f'group_{group}_preprocessed')
        
        weekday_dir = os.path.join(group_dir, '주중')
        weekend_dir = os.path.join(group_dir, '주말')

        # weekday daily, weekend daily 넣을 리스트
        weekday_daily = []
        weekend_daily = []

        # weekday annual, weekend annual 넣을 리스트
        temp_annual = []
        for excel in os.listdir(weekday_dir) :

            # 엑셀 파일 불러오기
            os.chdir(weekday_dir)
            weekday_excel = read_excel(excel)
            weekday_excel.columns = weekday_excel.columns.astype(str) 

            os.chdir(weekend_dir)
            weekend_excel = read_excel(f'{excel[ : -11]}주말_봄가을.xlsx')

            # weekday, weekend 날짜 개수 각각
            weekday_day = weekday_excel.shape[0]
            weekend_day = weekend_excel.shape[0]
            weekend_excel.columns = weekend_excel.columns.astype(str) 


            # calculate daily sum
            for index in range(weekday_day) :
                weekday_daily.append(weekday_excel.loc[index, '1' : '24'].sum())

            for index in range(weekend_day) :
                weekend_daily.append(weekend_excel.loc[index, '1' : '24'].sum())

            # calculate annual sum
            temp_annual.append(sum(weekday_daily) * (261 / weekday_day) + sum(weekend_daily) * (104 / weekend_day))

            print(f'{facility}, {excel} ended', end = '\r')

        now_cols_daily_day= f'{facility}_{group}_주중'
        now_cols_daily_end= f'{facility}_{group}_주말'
        now_cols_daily_all= f'{facility}_{group}_주중+주말'
        now_cols_annual = f'{facility}_{group}'

        ncols_daily.append(now_cols_daily_day)
        ncols_daily.append(now_cols_daily_end)
        ncols_daily.append(now_cols_daily_all)

        ncols_annual.append(now_cols_annual)
#
#        weekday_daily = np.array(weekday_daily).reshape(-1, 1)
#        temp_annual = np.array(temp_annual).reshape(-1, 1)


        actual_daily = pd.concat([actual_daily, pd.DataFrame(weekday_daily)], axis = 1, ignore_index = True)
        actual_daily = pd.concat([actual_daily, pd.DataFrame(weekend_daily)], axis = 1, ignore_index = True)
        day_end_daily = weekday_daily + weekend_daily
        actual_daily = pd.concat([actual_daily, pd.DataFrame(day_end_daily)], axis = 1, ignore_index = True)

        actual_annual = pd.concat([actual_annual, pd.DataFrame(temp_annual)], axis = 1, ignore_index = True)

        print(actual_daily.shape)
        print(actual_annual.shape)

        actual_daily.columns = ncols_daily
        actual_annual.columns = ncols_annual

        print('excel now')
        print(actual_daily.head())
        print(actual_annual.head())




print(actual_daily)
print(actual_annual)

os.chdir(cwdir)
actual_daily.to_excel('actual_daily.xlsx')
actual_annual.to_excel('actual_annual.xlsx')





# ---------------------------------------------------------------
# gp_dir (generated profiles)
# ---------------------------------------------------------------
print(facility_list)
# GENERATED_PROFILES    //  교육시설    //  group_0 //  raw //  '주중', '주말'  //  1번째 세대 전력프로필(주말).xlsx ( : -9)

now_dir = gp_dir

ncols_daily = []
ncols_annual = []

for facility in facility_list :
    if '판매및숙박' != facility :
        daily_list = []
        annual_list = []
        for group in [0, 1] :
            
            group_dir = os.path.join(now_dir, facility, f'group_{group}', 'raw')
            
            weekday_dir = os.path.join(group_dir, '주중')
            weekend_dir = os.path.join(group_dir, '주말')

            # weekday daily, weekend daily 넣을 리스트
            weekday_daily = []
            weekend_daily = []

            # weekday annual, weekend annual 넣을 리스트
            temp_annual = []
            for excel in os.listdir(weekday_dir) :

                # 엑셀 파일 불러오기
                os.chdir(weekday_dir)
                weekday_excel = read_excel(excel)
                weekday_excel.columns = weekday_excel.columns.astype(str) 

                os.chdir(weekend_dir)
                weekend_excel = read_excel(f'{excel[ : -9]}(주말).xlsx')

                # weekday, weekend 날짜 개수 각각
                weekday_day = weekday_excel.shape[0]
                weekend_day = weekend_excel.shape[0]
                weekend_excel.columns = weekend_excel.columns.astype(str) 


                # calculate daily sum
                for index in range(weekday_day) :
                    weekday_daily.append(weekday_excel.loc[index, '1' : '24'].sum())

                for index in range(weekend_day) :
                    weekend_daily.append(weekend_excel.loc[index, '1' : '24'].sum())

                # calculate annual sum
                temp_annual.append(sum(weekday_daily) * (261 / weekday_day) + sum(weekend_daily) * (104 / weekend_day))

                print(f'{facility}, {excel} ended', end = '\r')

            now_cols_daily_day= f'{facility}_{group}_주중'
            now_cols_daily_end= f'{facility}_{group}_주말'
            now_cols_daily_all= f'{facility}_{group}_주중+주말'
            now_cols_annual = f'{facility}_{group}'

            ncols_daily.append(now_cols_daily_day)
            ncols_daily.append(now_cols_daily_end)
            ncols_daily.append(now_cols_daily_all)

            ncols_annual.append(now_cols_annual)
#
#        weekday_daily = np.array(weekday_daily).reshape(-1, 1)
#        temp_annual = np.array(temp_annual).reshape(-1, 1)


            actual_daily = pd.concat([actual_daily, pd.DataFrame(weekday_daily)], axis = 1, ignore_index = True)
            actual_daily = pd.concat([actual_daily, pd.DataFrame(weekend_daily)], axis = 1, ignore_index = True)
            day_end_daily = weekday_daily + weekend_daily
            actual_daily = pd.concat([actual_daily, pd.DataFrame(day_end_daily)], axis = 1, ignore_index = True)

            actual_annual = pd.concat([actual_annual, pd.DataFrame(temp_annual)], axis = 1, ignore_index = True)

            print(actual_daily.shape)
            print(actual_annual.shape)

            actual_daily.columns = ncols_daily
            actual_annual.columns = ncols_annual

            print('excel now')
            print(actual_daily.head())
            print(actual_annual.head())




print(actual_daily)
print(actual_annual)

os.chdir(cwdir)
actual_daily.to_excel('generated_daily.xlsx')
actual_annual.to_excel('generated_annual.xlsx')


