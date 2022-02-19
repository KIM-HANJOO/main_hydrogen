import os
import sys

# placed on subdirectory
from pathlib import Path
cwdir = os.getcwd()
profile_generator_dir = str(Path(cwdir).parent.absolute())
sys.path.append(profile_generator_dir)
import get_dir
di = get_dir.get_info()

main_dir = di.main_dir
prep_dir = di.prep_dir
model_dir = di.model_dir
module_dir = di.module_dir
facility_dir = di.facility_dir
facility_dir = di.facility_dir
plot_dir = di.plot_dir
cluster_dir = di.cluster_dir
facility_df = di.facility_df
facility_dict = di.facility_dict
gp_dir = di.gp_dir
gp_plot = di.gp_plot
params = di.params

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

# ---------------------------------------------
# copy datas from facility to 'main_hydrogen/accom_deal_profile_48/'
# ---------------------------------------------



# ---------------------------------------------
# make new directories
# ---------------------------------------------

ad_dir = os.path.join(facility_dir, '판매및숙박')

model1_dir = os.path.join(ad_dir, 'model1')
model2_dir = os.path.join(ad_dir, 'model2')
model3_dir = os.path.join(ad_dir, 'model3')
model4_dir = os.path.join(ad_dir, 'model4')
preprocessed_dir = os.path.join(ad_dir, 'preprocessed')

# inside accom_deal_profile_48

# make accomodation, dealership directory under accom_deal_profile_48

accom_deal_dir = os.path.join(main_dir, 'accom_deal_profile_48')
main_dir = accom_deal_dir
facility_dir = os.path.join(accom_deal_dir, 'FACILITIES')
gp_dir = os.path.join(accom_deal_dir, 'GENERATED_PROFILES')
gp_plot = os.path.join(accom_deal_dir, 'GENERATED_PLOTS')

print(f'main_dir = {main_dir}\n')




# ---------------------------------------------
# copy files from org_facility_dir to new directories
# ---------------------------------------------

# inside facility_dir 



#dich.newfolderlist(gp_dir, ['숙박시설', '판매시설'])
#
#accom_dir = os.path.join(facility_dir, '숙박시설')
#deal_dir = os.path.join(facility_dir, '판매시설')
#
#nfolderlist = ['model1', 'model2', 'model3', 'model4']
#group_list = ['group_0', 'group_1']
#
#dich.remove_inside_folder(accom_dir)
#dich.remove_inside_folder(deal_dir)
#
#print('118. line')
#dich.newfolderlist(accom_dir, nfolderlist)
#dich.newfolderlist(deal_dir, nfolderlist)
#
#
#accom_model1 = os.path.join(accom_dir, 'model1')
#accom_model2 = os.path.join(accom_dir, 'model2')
#accom_model3 = os.path.join(accom_dir, 'model3')
#accom_model4 = os.path.join(accom_dir, 'model4')
#accom_preprocessed = os.path.join(accom_dir, 'preprocessed')
#
#deal_model1 = os.path.join(deal_dir, 'model1')
#deal_model2 = os.path.join(deal_dir, 'model2')
#deal_model3 = os.path.join(deal_dir, 'model3')
#deal_model4 = os.path.join(deal_dir, 'model4')
#deal_preprocessed = os.path.join(deal_dir, 'preprocessed')
#
## model 3 
#dich.newfolderlist(accom_model3, group_list)
#dich.newfolderlist(deal_model3, group_list)
#
## model 4
#dich.newfolderlist(accom_model4, group_list)
#dich.newfolderlist(deal_model4, group_list)
#
#for group_dir in group_list :
#    add_subdir = f'{group_dir}_model4'
#    tempdir_accom = os.path.join(accom_model3, add_subdir)
#    tempdir_deal = os.path.join(deal_model3, add_subdir)
#
## add weekend / weekdays to 'preprocessed' dir
#dich.newfolderlist(accom_preprocessed, ['주중', '주말'])
#dich.newfolderlist(deal_preprocessed, ['주중', '주말'])
#
#
#
#### copy files fromt accom+deal to sep directories
#
## copy 'model1' from accom+deal to sep directories
#for excel in os.listdir(model1_dir) :
#    if '숙박시설' in excel :
#        dich.copyfile(model1_dir, accom_model1, excel)
#    elif '판매시설' in excel :
#        dich.copyfile(model1_dir, deal_model1, excel)
#    else :
#        dich.copyfile(model1_dir, deal_model1, excel)
#        dich.copyfile(model1_dir, accom_model1, excel)
#
#
## copy 'model2' from accom+deal to sep directories
#dich.copyfile(model2_dir, accom_model2, 'Model2_daily fractoin.xlsx')
#dich.copyfile(model2_dir, deal_model2, 'Model2_daily fractoin.xlsx')
#
#
## copy 'model3' from accom+deal to sep directories
#
#dich.copydir_f(model3_dir, accom_model3)
#dich.copydir_f(model3_dir, deal_model3)
#
## copy 'model4' from accom+deal to sep directories
#dich.copydir_f(model4_dir, accom_model4)
#dich.copydir_f(model4_dir, deal_model4)
#
## copy 'preprocessed' from accom+deal to sep directories
#dich.copydir_f(preprocessed_dir, accom_preprocessed)
#dich.copydir_f(preprocessed_dir, deal_preprocessed)
#
#print('coping direcotries done')


# ---------------------------------------------
# make profiles
# notion !
# ratio : by each building type from AMI
# ---------------------------------------------


# ---------------------------------------------

# 1. make directory variables

# 2. def profile_num_maker // 사전에 시설군들의 비중을 확인하기 위해 사용

# 3. generate_fc // 디렉토리 위치, model 1 ~ 4 파일 정보를 변수로 저장해서 profile_generator 에 넘겨줌

# 4. profile_generator // generate_fc 에서 받은 데이터프레임, 정보를 이용해 주어진 저장 위치에 프로필 생성
#       model 1 : beta, model 2 : burr, model 3 + 4 : multivariate distribution

# ---------------------------------------------


# make directories
'''
facility_dir : '판매시설', '숙박시설' 포함된 디렉토리
gp_dir : 생성된 프로필
gp_plot : 그래프 

'''

# lists
subdir_list = ['raw', 'model1', 'model2', 'model3', 'model4']
fc_list = ['교육시설', '문화시설', '숙박시설', '업무시설', '판매시설', 'params']
fc_list_2 = ['숙박시설', '판매시설']

#make gp_dir
dich.newfolder(gp_dir)

for fc in fc_list_2 :
    if 'params' != fc :
        tempdir = os.path.join(gp_dir, fc)
        dich.newfolder(tempdir)
        for ssub in ['compare', 'solo'] :
            ttempdir = os.path.join(tempdir, ssub)
            dich.newfolder(ttempdir)

dich.newfolder(os.path.join(gp_plot, 'model2_fitted'))

# copy 'model2_fitted' folder
model2_fitted_dir = os.path.join(di.main_dir, 'temp', 'model2_fitted')

dst_dir = gp_plot
dst_dir = os.path.join(gp_plot, 'model2_fitted')
dich.newfolder(dst_dir)
dich.copydir_f(model2_fitted_dir, dst_dir)
check = 0

print(dst_dir)
print(os.listdir(dst_dir))
for f in os.listdir(dst_dir) :
    if 'model2_fitted' == f :
        check = 1

if check == 1:
    print('model2_fitted folder copied to GENERATED_PLOTS dir')
else :
    print('33333333333333333333333warning33333333333333333333\n')

                
#gp_dir = main_dir + '//GENERATED_PROFILES'
dich.newfolder(gp_dir)

dich.newfolderlist(gp_dir, fc_list_2)
for fc in fc_list_2 :
    tempdir = os.path.join(gp_dir, fc)

    if fc != 'params' : 
        dich.newfolderlist(tempdir, ['group_0', 'group_1'])
        
        for i in range(2) :
            dich.newfolderlist(os.path.join(tempdir, f'group_{i}'), subdir_list)
            
            for sd in subdir_list :
                if (sd != 'model1') | (sd != 'model2') :
#            if (sd == 'raw') | (sd == 'model3'):
                    dich.newfolderlist(os.path.join(tempdir, f'group_{i}' ,f'{sd}'), ['주중', '주말'])
        
    print(f'{fc} directory all made')


os.chdir(params)
dich.copyfile(os.path.join(di.gp_plot, 'model2_fitted'), params, 'model2_burr_fitted.xlsx')
print(os.listdir(params))

'''
note
'''

print('starts, 276. line')

facility_list = ['판매시설', '숙박시설'] #profile_num_maker에서 쓰이며, model1, model2 파일 사전작업에 사용되었음


# ---------------------------------------------
# make functions
# ---------------------------------------------


# profile_num_maker 은 사전에 전체 시설군에 대한 건물 비중을 확인하기 위해 사용
# 실제로 프로필 생성에 영향을 주는 함수는 아님
def profile_num_maker(facility_dir) :
    
    facility_list = ['판매시설', '숙박시설']
            
    maker_df = pd.DataFrame(columns = ['facility', 'group', 'number', 'percentage'])
    maker_num = 0
    
    for num, fc in enumerate(facility_list) :
        for i in range(2) : # group_number
            tempdir = os.path.join(facility_dir, fc, 'model3', f'group_{i}', f'group_{i}_model3', '주중')
#            tempdir = facility_dir + f'//{fc}//model3//group_{i}//group_{i}_model3//주중'
            op_num = len(os.listdir(tempdir))
            
            maker_df.loc[maker_num, 'facility'] = fc
            maker_df.loc[maker_num, 'group'] = i
            maker_df.loc[maker_num, 'number'] = op_num
            maker_num += 1
    
    all_profile_numbers = sum(maker_df.loc[:, 'number'].tolist())
    
    for i in range(maker_df.shape[0]) :
        maker_df.loc[i, 'percentage'] = round(maker_df.loc[i, 'number'] / all_profile_numbers * 100, 2)
    
    return maker_df
            
    
    
def generate_fc(facility_dir, facility, group, facility_add) :    
# need
# facility, save_dir, profile_num, model1_file, \
# model2_file, model3_file, model4_file_day, model4_file_end


# directories
    key_list = ['facility', 'save_dir', 'save_dir_day', 'save_dir_end', 'facility_add', 'model1_file', \
    'model2_file', 'model3_file', 'model4_file_day', 'model4_file_end', 'dayend_perc']

    print(f'starting from {facility_dir} . . .')
    wd = os.path.join(facility_dir, facility)
    model1_dir = os.path.join(wd, 'model1')
    model2_dir = os.path.join(wd, 'model2')
    model3_dir = os.path.join(wd, 'model3')
    model4_dir = os.path.join(wd, 'model4')


#    wd = facility_dir + f'//{facility}'
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
    model1_file = model1_file[[f'{facility}']]
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

# set directories
    dich.newfolder(gp_dir)
    dich.newfolder(gp_plot)
    print(f'generated profiles to {gp_dir}')
    print(f'generated plots to {gp_plot}')

    main_dir = str(Path(facility_dir).parent.absolute())
    print(f'main_dir is {main_dir}')
    save_dir = os.path.join(gp_dir, facility, f'group_{group}', 'raw')
    
    dich.newfolder(save_dir)

    save_dir_day = os.path.join(save_dir, '주중')
    save_dir_end = os.path.join(save_dir, '주말')

    os.chdir(os.path.join(di.main_dir, 'module', '7_plot_use'))
    dayend_perc = dich.read_excel('weekday+end_perc.xlsx')

    if facility == '판매및숙박' :
        save_dir = os.path.join(gp_dir, facility_add, f'group_{group}', 'raw')
        save_dir_day = os.path.join(save_dir, '주중')
        save_dir_end = os.path.join(save_dir, '주말')

        dich.newfolder(save_dir)
        dich.newfolderlist(save_dir, ['주중', '주말'])




    file_dict = dict()
    key_list = None

    print('files all loaded')
    print(len(file_dict.keys()))

    return key_list, file_dict, model1_file, facility, save_dir, save_dir_day, save_dir_end, facility_add, model2_file, model3_file, model4_file_day, model4_file_end, dayend_perc

    
def profile_generator(profile_num, key_list, file_dict, model1_file, facility, save_dir, save_dir_day, save_dir_end, facility_add, model2_file, model3_file, model4_file_day, model4_file_end, dayend_perc) :
#    for key in file_dict.keys() :
#        globals()[f'{key}'] = file_dict[key]
#    file_dict = None
        
    hours = []
    for i in range(1, 25) :
        hours.append(str(i))
    dayend_perc.index = ['weekday', 'weekend']
    dayend_perc = dayend_perc[['판매및숙박']]

    dayend_perc.columns = [f'{facility}']
    ave_day = dayend_perc.loc['weekday', facility]    
    ave_end = dayend_perc.loc['weekend', facility]    
    print('percentage for weekday, weekend')
    model3_file_day = model3_file.loc[:, '1' : '24']
    model3_file_end = model3_file.loc[:, '25' : '48']
    model3_file_end.columns = model3_file_day.columns

    for profile_num_now in range(profile_num):
        '''
        모델 1
        '''
        model1_file.index = ['a', 'b', 'loc', 'scale']

        for col in model1_file.columns :
            print(col, '\t', facility)
            if facility in col :
                a = model1_file.loc['a', col]
                b = model1_file.loc['b', col]
                loc = model1_file.loc['loc', col]
                scale = model1_file.loc['scale', col]
#
        ave_day_beta = beta.rvs(a, b, loc = loc, scale = scale, size = 1)
        
        constant_weekday = ave_day * (365 / (261 * ave_day + 104 * ave_end))
        constant_weekend = ave_end * (365 / (261 * ave_day + 104 * ave_end))

        ave_weekday_1day = ave_day_beta * constant_weekday
        ave_weekend_1day = ave_day_beta * constant_weekend

        '''
        모델2
        '''
        model2_file.index = ['c', 'd', 'loc', 'scale']
    
        # seperated columns already
        # ~ model2_file.columns = ['교육주중', '교육주말', '판매숙박주중', '판매숙박주말', '업무주중', '업무주말', \
                                # ~ '문화주중', '문화주말']
                                
        for col in model2_file.columns :
            if '판매및숙박' in col :
                if '주중' in col :
                    c2 = model2_file.loc['c', col]
                    d2 = model2_file.loc['d', col]
                    loc2 = model2_file.loc['loc', col]
                    scale2 = model2_file.loc['scale', col]
        
        st_week = burr.rvs(c2, d2, loc = loc2, scale = scale2, size = 261)
        st_week = st_week.tolist()
        for i in range(len(st_week)) :
            while st_week[i] <= 0 :
                print('negative value for st_weekend')
                st_week[i] = burr.rvs(c2, d2, loc = loc2, scale = scale2, size = 1)
            
            
        for col in model2_file.columns :
            if '판매및숙박' in col :
                if '주말' in col :
                    c3 = model2_file.loc['c', col]
                    d3 = model2_file.loc['d', col]
                    loc3 = model2_file.loc['loc', col]
                    scale3 = model2_file.loc['scale', col]
        
        st_weekend = burr.rvs(c3, d3, loc = loc3, scale = scale3, size = 104)
        st_weekend = st_weekend.tolist()
        for i in range(len(st_weekend)) :
            while st_weekend[i] <= 0 :
                print('negative value for st_weekend')
                st_weekend[i] = burr.rvs(c3, d3, loc = loc3, scale = scale3, size = 1)
    
    
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

        '''
        프로필 생성
        '''
        
        for t in range(261):
            # 평균과 표준편차를 이용하여 1일 사용량 도출
            while 1:
                #week_1day = np.random.normal(ave_weekday_1day, st_week)
                '''
                '''
                week_1day = ave_weekday_1day * st_week[t]
                if week_1day > 0:
                    break

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
                '''
                '''
                weekend_1day = ave_weekend_1day * st_weekend[t]
                #weekend_1day = np.random.normal(ave_weekend_1day, st_weekend)
                if weekend_1day > 0:
                    break
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
        


print('\n\n######################################################\n\n')

# facility_list = ['교육시설', '문화시설', '업무시설', '판매및숙박'] 
facility_list = ['판매시설', '숙박시설']
all_profiles_perc = 100 #%

maker_df_col = ['facility', 'group', 'number', 'percentage']

maker_df = pd.DataFrame(columns = maker_df_col)
df_num = 0

for facility in facility_list :
    fc_dir = os.path.join(facility_dir, facility)
    for group in [0, 1] :
        model3_dir = os.path.join(fc_dir, 'model3', f'group_{group}', f'group_{group}_preprocessed', '주말')

        excel_num = 0
        for excel in os.listdir(model3_dir) :
            if facility == '판매시설' :
                if 'G' in excel :
                    excel_num += 1
            elif facility == '숙박시설' :
                if 'I' in excel :
                    excel_num += 1

        maker_df.loc[df_num, 'facility'] = facility
        maker_df.loc[df_num, 'group'] = group
        maker_df.loc[df_num, 'number'] = excel_num
        
        df_num += 1

all_sum = maker_df.loc[:, 'number'].sum()
for index in range(maker_df.shape[0]) :
    maker_df.loc[index, 'percentage'] = round(maker_df.loc[index, 'number'] / all_sum * 100, 2)

print(maker_df)


for facility in facility_list :
    for group in range(2) : # group_number
        check = 1

        if check == 1 :
            for i in range(maker_df.shape[0]) :
                if (maker_df.loc[i, 'facility'] == facility) & (int(maker_df.loc[i, 'group']) == group) :
                    profile_num = maker_df.loc[i, 'number']
            if facility == '판매및숙박' :
                for facility_add in ['판매시설', '숙박시설'] :
                    print(f"{facility}, group_{group}, n = {profile_num}")
                    key_list, file_dict,model1_file, facility, save_dir, save_dir_day, save_dir_end, facility_add, model2_file, model3_file, model4_file_day, model4_file_end, dayend_perc  = generate_fc(facility_dir, facility, group, facility_add)
                    profile_generator(profile_num, key_list, file_dict, model1_file, facility, save_dir, save_dir_day, save_dir_end, facility_add, model2_file, model3_file, model4_file_day, model4_file_end, dayend_perc)

            else :
                facility_add = None
                print(f"{facility}, group_{group}, n = {profile_num}")
                key_list, file_dict, model1_file, facility, save_dir, save_dir_day, save_dir_end, facility_add, model2_file, model3_file, model4_file_day, model4_file_end, dayend_perc = generate_fc(facility_dir, facility, group, facility_add)
                profile_generator(profile_num, key_list, file_dict, model1_file, facility, save_dir, save_dir_day, save_dir_end, facility_add, model2_file, model3_file, model4_file_day, model4_file_end, dayend_perc)

    
    
    
    

