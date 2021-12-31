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
facility_dict = di.facility_dict
gp_dir = di.gp_dir
gp_plot = di.gp_plot
params = di.params
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
import warnings

'''
note
'''
warnings.filterwarnings(action = 'ignore')
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
note

before running,
run 'temp_new_facility_folder.py'
run 'generated_profiles_directory.py'

'''


subdir_list = ['raw', 'model1', 'model2', 'model3', 'model4']
fc_list = ['교육시설', '문화시설', '숙박시설', '업무시설', '판매시설', 'params']
fc_list_2 = ['교육시설', '문화시설', '판매및숙박', '업무시설', 'params']

                
#gfc_dir = main_dir + '//GENERATED_PROFILES'
gfc_dir = os.path.join(main_dir, 'GENERATED_PROFILES')
dich.newfolder(gfc_dir)

dich.newfolderlist(gfc_dir, fc_list_2)
for fc in fc_list_2 :
    tempdir = os.path.join(gfc_dir, fc)

    if fc != 'params' : 
        dich.newfolderlist(tempdir, ['group_0', 'group_1'])
        
        for i in range(2) :
            dich.newfolderlist(os.path.join(tempdir, f'group_{i}'), subdir_list)
            
            for sd in subdir_list :
#            if (sd == 'raw') | (sd == 'model3'):

                dich.newfolderlist(os.path.join(tempdir, f'group_{i}' ,f'{sd}'), ['주중', '주말'])
        
    print(f'{fc} directory all made')


os.chdir(params)
os.system('cp ../../GENERATED_PLOTS/model2_fitted/model2_burr_fitted.xlsx ./')
print(os.listdir(params))

'''
note
'''


facility_list = ['교육시설', '문화시설', '업무시설', '판매및숙박'] #profile_num_maker에서 쓰이며, model1, model2 파일 사전작업에 사용되었음

def profile_num_maker(nfc_dir) :
    
    facility_list = ['교육시설', '문화시설', '업무시설', '판매및숙박']
            
    maker_df = pd.DataFrame(columns = ['facility', 'group', 'number', 'percentage'])
    maker_num = 0
    
    for num, fc in enumerate(facility_list) :
        for i in range(2) : # group_number
            tempdir = os.path.join(nfc_dir, fc, 'model3', f'group_{i}', f'group_{i}_model3', '주중')
#            tempdir = nfc_dir + f'//{fc}//model3//group_{i}//group_{i}_model3//주중'
            op_num = len(os.listdir(tempdir))
            
            maker_df.loc[maker_num, 'facility'] = fc
            maker_df.loc[maker_num, 'group'] = i
            maker_df.loc[maker_num, 'number'] = op_num
            maker_num += 1
    
    all_profile_numbers = sum(maker_df.loc[:, 'number'].tolist())
    
    for i in range(maker_df.shape[0]) :
        maker_df.loc[i, 'percentage'] = round(maker_df.loc[i, 'number'] / all_profile_numbers * 100, 2)
    
    return maker_df
            
    
    
def generate_fc(nfc_dir, facility, group) :    
# need
# facility, save_dir, profile_num, model1_file, \
# model2_file, model3_file, model4_file_day, model4_file_end


# directories
    wd = os.path.join(nfc_dir, facility)
    model1_dir = os.path.join(wd, 'model1')
    model2_dir = os.path.join(wd, 'model2')
    model3_dir = os.path.join(wd, 'model3')
    model4_dir = os.path.join(wd, 'model4')


#    wd = nfc_dir + f'//{facility}'
#    model1_dir = wd + '//model1'
#    model2_dir = wd + '//model2'
#    model3_dir = wd + '//model3' # + '//group_0', '//group_1' -> 'profile_48_group_0.xlsx'
#    model4_dir = wd + '//model4' # + '//group_0_model4'

# excel names
    m1_xlna = 'model_1.xlsx'
    m2_xlna = 'model_2.xlsx'
    m2_xlna = 'model2_burr_fitted.xlsx'

    m3_0_xlna = 'profile_48_group_0.xlsx'
    m3_1_xlna = 'profile_48_group_1.xlsx'

    m4_day_xlna = 'model4_weekdays.xlsx'
    m4_end_xlna = 'model4_weekends.xlsx'

# import files
    print(f'\nworking on {facility}')
    os.chdir(model1_dir)
    model1_file = dich.read_excel(m1_xlna)
    print('model1 loaded')

#    os.chdir(model2_dir)
#    model2_file = dich.read_excel(m2_xlna)
#    print('model2 loaded')
    os.chdir(params)
    model2_file = dich.read_excel(m2_xlna)
    print('model2 loaded')

    os.chdir(os.path.join(model3_dir, f'group_{group}'))
    model3_file = dich.read_excel(locals()[f'm3_{group}_xlna'])
    print('model3 loaded')

    os.chdir(os.path.join(model4_dir, f'group_{group}_model4'))
    model4_file_day = dich.read_excel(m4_day_xlna)
    print('model4 weekdays loaded')
    model4_file_end = dich.read_excel(m4_end_xlna)
    print('model4 weekends loaded')

    main_dir = str(Path(facility_dir).parent.absolute())
    save_dir = os.path.join(gp_dir, facility, f'group_{group}', 'raw')
    save_dir_day = os.path.join(save_dir, '주중')
    save_dir_end = os.path.join(save_dir, '주말')
    os.chdir(os.path.join(main_dir, 'module', '7_plot_use'))
    dayend_perc = dich.read_excel('weekday+end_perc.xlsx')


    file_dict = dict()

    key_list = ['facility', 'save_dir', 'save_dir_day', 'save_dir_end', \
    'model1_file', 'model2_file', 'model3_file', 'model4_file_day', 'model4_file_end', 'dayend_perc']

    for key in key_list :
        file_dict[key] = locals()[f'{key}']

    print('files all loaded')

    return key_list, file_dict

    

# 원하는 세대 수와 세대 특징 입력된 파일 읽기
def profile_generator(profile_num, key_list, file_dict) :
    
    for key in key_list :
        globals()[f'{key}'] = file_dict[key]
        
    file_dict = None
        
    hours = []
    for i in range(1, 25) :
        hours.append(str(i))
    dayend_perc.index = ['weekday', 'weekend']
    ave_day = dayend_perc.loc['weekday', facility]    
    ave_end = dayend_perc.loc['weekend', facility]    

    model3_file_day = model3_file.loc[:, '1' : '24']
    model3_file_end = model3_file.loc[:, '25' : '48']
    model3_file_end.columns = model3_file_day.columns
    
    all_ave = []
    all_st = []

    for profile_num_now in range(profile_num):
        '''
        모델 1
        '''
        model1_file.index = ['a', 'b', 'loc', 'scale']
        if facility == '판매및숙박' :

            break

        else :
            for col in model1_file.columns :
                print(col, '\t', facility)
                if facility in col :
                    a = model1_file.loc['a', col]
                    b = model1_file.loc['b', col]
                    loc = model1_file.loc['loc', col]
                    scale = model1_file.loc['scale', col]

        ave_week_1day = beta.rvs(a, b, loc = loc, scale = scale, size = 1)
        
        C = (365 * ave_week_1day) / (104 * ave_end + 261 * ave_end)

        ave_weekday_1day = ave_day * C
        ave_weekend_1day = ave_end * C
        '''
        모델2
        '''
        model2_file.index = ['c', 'd', 'loc', 'scale']
    
        # seperated columns already
        # ~ model2_file.columns = ['교육주중', '교육주말', '판매숙박주중', '판매숙박주말', '업무주중', '업무주말', \
                                # ~ '문화주중', '문화주말']
                                
        for col in model2_file.columns :
            if facility in col :
                if '주중' in col :
                    c2 = model2_file.loc['c', col]
                    d2 = model2_file.loc['d', col]
                    loc2 = model2_file.loc['loc', col]
                    scale2 = model2_file.loc['scale', col]
        
        st_week = burr.rvs(c2, d2, loc = loc2, scale = scale2, size = 1)
        while st_week < 0 :
            print('negative value for st_weekend')
            st_week = burr.rvs(c2, d2, loc = loc2, scale = scale2, size = 1)
            
        for col in model2_file.columns :
            if facility in col :
                if '주말' in col :
                    print(col)
                    c3 = model2_file.loc['c', col]
                    d3 = model2_file.loc['d', col]
                    loc3 = model2_file.loc['loc', col]
                    scale3 = model2_file.loc['scale', col]
        
        st_weekend = burr.rvs(c3, d3, loc = loc3, scale = scale3, size = 1)
        while st_weekend < 0 :
            print('negative value for st_weekend')
            st_weekend = burr.rvs(c3, d3, loc = loc3, scale = scale3, size = 1)

        all_st.append(st_weekend)
        all_st.append(st_week)
    
    
        '''
        모델3
        '''
        
        # 다변량 이용하여 고정 프로필 1개 만들기(평일)
        sample = model3_file_day.loc[:, '1' : '24']
        var_1 = sample.cov()
        mean_1 = sample.mean()
    
        fixed_profile_week = pd.DataFrame()
    
        X = np.random.multivariate_normal(mean_1, var_1)
        while 1:
            for x in X:
                if x < 0:
                    X = np.random.multivariate_normal(mean_1, var_1)
                    a = 0
                    break
                a = 1
            if a is 0:
                continue
            break
        fixed_profile_week['fixed'] = X
    
        # 다변량 이용하여 고정 프로필 1개 만들기(주말)
        sample = model3_file_end.loc[:, '1' : '24']
#        sample = sample.transpose()
#        sample = sample.transpose()
        var_1 = sample.cov()
        mean_1 = sample.mean()
    
    
        fixed_profile_weekend = pd.DataFrame()
    
        X = np.random.multivariate_normal(mean_1, var_1)
        while 1:
            for x in X:
                if x < 0:
                    X = np.random.multivariate_normal(mean_1, var_1)
                    a = 0
                    break
                a = 1
            if a is 0:
                continue
            break
        fixed_profile_weekend['fixed'] = X
    
    
    
        '''
        모델4
        '''        
        
        
        # 일마다 변하는 프로필 만들기(평일)
        
        test = model4_file_day.loc[:, '1' : '24']
        ncols = []
        for i in range(24) :
            ncols.append(i)
        test.columns = ncols
        
        var_1 = test.cov()
        mean_1 = test.mean()
        changed_profile_week = pd.DataFrame()
        
        for t in range(261):
            # 평균과 표준편차를 이용하여 1일 사용량 도출
            while 1:
                #week_1day = np.random.normal(ave_weekday_1day, st_week)
                week_1day = ave_weekday_1day * st_week
                if week_1day > 0:
                    break
            all_ave.append(week_1day)
            print(week_1day)
            print("day sum / ave_day")
            #weekend_1day = np.random.normal(ave_weekend_1day, st_weekend)
            # 1일 프로필에 변화를 주는 프로필 생산
            Y = []
            X = np.random.multivariate_normal(mean_1, var_1, check_valid='ignore')
            
            # check if model3 + model4 < 0 ; if < 0, convert to 0
            for k in range(24):
                x = fixed_profile_week.iloc[k,0] + X[k]
                if x < 0 :
                    X[k] = 0

                        
            for f in range(24):
                y = (fixed_profile_week.iloc[f,0] + X[f])
                Y.append(y)


            for f in range(24) :
                if Y[f] < 0 :
                    print('HHHHHHHHHHHHHHHHEEEEEEEEEEEEEEEEEEEEEEELLLLLLLLLLLLLL')
            Y = Y/sum(Y)
            Y = Y * week_1day
            changed_profile_week['{}일'.format(t+1)] = Y
    
        # 일마다 변하는 프로필 만들기(주말)
    
        test = model4_file_end.loc[:, '1' : '24']
        ncols = []
        for i in range(24) :
            ncols.append(i)
        test.columns = ncols
        var_1 = test.cov()
        mean_1 = test.mean()
        changed_profile_weekend = pd.DataFrame()
    
        for t in range(104):
            # 평균과 표준편차를 이용하여 1일 사용량 도출
            #week_1day = np.random.normal(ave_week_1day, st_week)
            while 1:
                weekend_1day = ave_weekend_1day * st_weekend
                #weekend_1day = np.random.normal(ave_weekend_1day, st_weekend)
                if weekend_1day > 0:
                    break
            all_ave.append(weekend_1day)
            # 1일 프로필에 변화를 주는 프로필 생산
            Y = []
            X = np.random.multivariate_normal(mean_1, var_1, check_valid='ignore')
            # check if model3 + model4 < 0 ; if < 0, convert to 0
            for k in range(24):
                x = fixed_profile_weekend.iloc[k,0] + X[k]
                if x < 0 :
                    X[k] = 0
                        
            for f in range(24):
                y = (fixed_profile_weekend.iloc[f,0] + X[f])
                Y.append(y)
            Y = Y/sum(Y)
            Y = Y * weekend_1day
            changed_profile_weekend['{}일'.format(t+1)] = Y
        
        changed_profile_week = changed_profile_week.transpose()
        changed_profile_weekend = changed_profile_weekend.transpose()
        
        changed_profile_week.reset_index(drop = True, inplace = True)
        changed_profile_week.columns = hours
        
        changed_profile_weekend.reset_index(drop = True, inplace = True)
        changed_profile_weekend.columns = hours
            
        print('{}번째전력프로필 완성\n'.format(profile_num_now + 1))
        
        
        
        os.chdir(save_dir)
        
        # 완성된 파일 저장
        os.chdir(save_dir_day)
        changed_profile_week.to_excel('{}번째 세대 전력프로필(주중).xlsx'.format(profile_num_now + 1))
        
        os.chdir(save_dir_end)
        changed_profile_weekend.to_excel('{}번째 세대 전력프로필(주말).xlsx'.format(profile_num_now + 1))
        

    print(all_st)
    print(all_ave)


    print('#####################')
    print('model2 max - min :')
    print(max(all_st) - min(all_st))
    print('ave max - min :')
    print(max(all_ave) - min(all_ave))
    print('#####################')
print('\n\n######################################################\n\n')

facility_list = ['교육시설', '문화시설', '업무시설', '판매및숙박'] 
all_profiles_perc = 100 #%

maker_df = profile_num_maker(nfc_dir)
for i in range(maker_df.shape[0]) :
    maker_df.loc[i, 'number'] = round(maker_df.loc[i, 'number'] * all_profiles_perc / 100)

tempsum = sum(maker_df.loc[:, 'number'].tolist())

for i in range(maker_df.shape[0]) :
    maker_df.loc[i, 'percentage'] = round(maker_df.loc[i, 'number'] / tempsum * 100, 2)

for i in range(maker_df.shape[0]) :
    print(f"{maker_df.loc[i, 'facility']}, group_{maker_df.loc[i, 'group']}, generate n = {maker_df.loc[i, 'number']}\n")
# ~ for i in range(maker_df.shape[0]) :
    # ~ if maker_df.loc[i, 'number'] > 100 :
        # ~ maker_df.loc[i, 'number'] = 100
        
print(maker_df)


for facility in facility_list :
    for group in range(2) : # group_number
        check = 1
        if facility == '판매및숙박' :
            check == 0
            print(f'{facility}, {group} X')
#        if (facility == '문화시설') | (facility == '교육시설') | (facility == '업무시설') :
#            check = 0
#            print(f'{facility}, {group} X')
#        # ~ if (facility == '교육시설') & (group == 0) :
#            # ~ check = 0
#            # ~ print(f'{facility}, {group} X')
        if check == 1 :
            for i in range(maker_df.shape[0]) :
                if (maker_df.loc[i, 'facility'] == facility) & (int(maker_df.loc[i, 'group']) == group) :
                    profile_num = maker_df.loc[i, 'number']
            print(f"{facility}, group_{group}, n = {profile_num}")
            key_list, file_dict = generate_fc(nfc_dir, facility, group)
            profile_generator(profile_num, key_list, file_dict)

    
    
    
    

# ~ os.chdir(main_dir + '//temp')
# ~ model1_file = lib.read_excel('model1_beta_fitting.xlsx')
# ~ model2_file = lib.read_excel('model2_beta_fitting.xlsx')

# ~ facility = '교육'
# ~ excel_name = '[P.교육 서비스업(85)]_0_'
# ~ model3_file = lib.read_excel('profile_48_group_0.xlsx')


# ~ model4_file_day = lib.read_excel('model4_weekdays.xlsx')
# ~ model4_file_end = lib.read_excel('model4_weekends.xlsx')

# ~ save_dir = main_dir + '//temp'
# ~ profile_num = 5


