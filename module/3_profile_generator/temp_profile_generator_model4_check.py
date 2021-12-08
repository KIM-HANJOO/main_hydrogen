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

sys.path.append(module_dir)
sys.path.append(module_dir + '\\4_directory_module')
import directory_change as dich
import model_library as lib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgunbd.TTF"
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
import shutil
import pathlib
from pathlib import Path

'''
note

before running,
run 'temp_new_facility_folder.py'
run 'generated_profiles_directory.py'

'''

facility_list = ['교육시설', '문화시설', '업무시설', '판매및숙박'] #profile_num_maker에서 쓰이며, model1, model2 파일 사전작업에 사용되었음

def profile_num_maker(nfc_dir) :
	
	facility_list = ['교육시설', '문화시설', '업무시설', '판매및숙박']
			
	maker_df = pd.DataFrame(columns = ['facility', 'group', 'number', 'percentage'])
	maker_num = 0
	
	for num, fc in enumerate(facility_list) :
		for i in range(2) : # group_number
			tempdir = nfc_dir + f'\\{fc}\\model3\\group_{i}\\group_{i}_model3\\주중'
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
	wd = nfc_dir + f'\\{facility}'
	model1_dir = wd + '\\model1'
	model2_dir = wd + '\\model2'
	model3_dir = wd + '\\model3' # + '\\group_0', '\\group_1' -> 'profile_48_group_0.xlsx'
	model4_dir = wd + '\\model4' # + '\\group_0_model4'
	
	# excel names
	m1_xlna = 'model_1.xlsx'
	m2_xlna = 'model_2.xlsx'
	
	m3_0_xlna = 'profile_48_group_0.xlsx'
	m3_1_xlna = 'profile_48_group_1.xlsx'
	
	m4_day_xlna = 'model4_weekdays.xlsx'
	m4_end_xlna = 'model4_weekends.xlsx'
	
	# import files
	print(f'\nworking on {facility}')
	os.chdir(model1_dir)
	model1_file = dich.read_excel(m1_xlna)
	print('model1 loaded')
	
	os.chdir(model2_dir)
	model2_file = dich.read_excel(m2_xlna)
	print('model2 loaded')

	os.chdir(model3_dir + f'\\group_{group}')
	model3_file = dich.read_excel(locals()[f'm3_{group}_xlna'])
	print('model3 loaded')
	
	os.chdir(model4_dir + f'\\group_{group}_model4')
	model4_file_day = dich.read_excel(m4_day_xlna)
	model4_file_end = dich.read_excel(m4_end_xlna)
	print('model4 weekdays loaded')
	print('model4 weekends loaded')
	
	main_dir = str(Path(facility_dir).parent.absolute())
	save_dir = main_dir + f'\\GENERATED_PROFILES\\{facility}\\group_{group}\\raw'
	save_dir_day = save_dir + '\\주중'
	save_dir_end = save_dir + '\\주말'
	
	file_dict = dict()

	key_list = ['facility', 'save_dir', 'save_dir_day', 'save_dir_end', \
	'model1_file', 'model2_file', 'model3_file', 'model4_file_day', 'model4_file_end']
	
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
	
	model3_file_day = model3_file.loc[:, '1' : '24']
	model3_file_end = model3_file.loc[:, '25' : '48']
	model3_file_end.columns = model3_file_day.columns
	
	for profile_num_now in range(profile_num):
		'''
		모델 1
		'''
		model1_file.index = ['a', 'b', 'loc', 'scale']
		for col in model1_file.columns :
			if facility in col :
				a = model1_file.loc['a', col]
				b = model1_file.loc['b', col]
				loc = model1_file.loc['loc', col]
				scale = model1_file.loc['scale', col]
		
		ave_week_1day = beta.rvs(a, b, loc = loc, scale = scale, size = 1)
		ave_weekend_1day = ave_week_1day
		
		'''
		모델2
		'''
		model2_file.index = ['a', 'b', 'loc', 'scale']
	
		# seperated columns already
		# ~ model2_file.columns = ['교육주중', '교육주말', '판매숙박주중', '판매숙박주말', '업무주중', '업무주말', \
								# ~ '문화주중', '문화주말']
								
		for col in model2_file.columns :
			if facility in col :
				if '주중' in col :
					print(col)
					a = model2_file.loc['a', col]
					b = model2_file.loc['b', col]
					loc = model2_file.loc['loc', col]
					scale = model2_file.loc['scale', col]
		
		st_week = beta.rvs(a, b, loc = loc, scale = scale, size = 1)
		
		for col in model2_file.columns :
			if facility in col :
				if '주말' in col :
					print(col)
					a = model2_file.loc['a', col]
					b = model2_file.loc['b', col]
					loc = model2_file.loc['loc', col]
					scale = model2_file.loc['scale', col]
		
		st_weekend = beta.rvs(a, b, loc = loc, scale = scale, size = 1)
	

	
	
	
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
		sample = sample.transpose()
		sample = sample.transpose()
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
				week_1day = np.random.normal(ave_week_1day, st_week) # ave_week_1day : model 1 # st_week : model 2
				if week_1day > 0:
					break
			#weekend_1day = np.random.normal(ave_weekend_1day, st_weekend)
			# 1일 프로필에 변화를 주는 프로필 생산
			Y = []
			X = np.random.multivariate_normal(mean_1, var_1, check_valid='ignore') # mean_1, var_1 : model 4
			
			
			count_neg = 0
			hours = []
			for i in range(1, 25) :
				hours.append(str(i))
				
			all_model4 = pd.DataFrame(columns = hours)
			
			while 1:
				for k in range(24):
					x = fixed_profile_week.iloc[k,0] + X[k] # fixed_profile_week : 24시간짜리 model 3
					if x < 0:
						X = np.random.multivariate_normal(mean_1, var_1, check_valid='ignore')
						a = 0
						
						# count under 0
						count_neg += 1
						break
					a = 1
				if a is 0:
					continue
				break
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
				weekend_1day = np.random.normal(ave_weekend_1day, st_weekend)
				if weekend_1day > 0:
					break
			# 1일 프로필에 변화를 주는 프로필 생산
			Y = []
			X = np.random.multivariate_normal(mean_1, var_1, check_valid='ignore')
			while 1:
				for k in range(24):
					x = fixed_profile_weekend.iloc[k,0] + X[k]
					if x < 0:
						X = np.random.multivariate_normal(mean_1, var_1, check_valid='ignore')
						a = 0
						break
					a = 1
				if a is 0:
					continue
				break
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
			
		print('{}번째전력프로필 완성'.format(profile_num_now + 1))
		
		

		
		
print('\n\n######################################################\n\n')


facility_list = ['교육시설', '문화시설', '업무시설', '판매및숙박'] 
all_profiles_perc = 50 #%

maker_df = profile_num_maker(nfc_dir)
for i in range(maker_df.shape[0]) :
	maker_df.loc[i, 'number'] = round(maker_df.loc[i, 'number'] * all_profiles_perc / 100)

tempsum = sum(maker_df.loc[:, 'number'].tolist())

for i in range(maker_df.shape[0]) :
	maker_df.loc[i, 'percentage'] = round(maker_df.loc[i, 'number'] / tempsum * 100, 2)
	
for i in range(maker_df.shape[0]) :
	if maker_df.loc[i, 'number'] > 100 :
		maker_df.loc[i, 'number'] = 100
		
print(maker_df)


for facility in facility_list :
	for group in range(2) : # group_number
		if (facility != '교육시설') & (facility != '문화시설') :
			for i in range(maker_df.shape[0]) :
				if (maker_df.loc[i, 'facility'] == facility) & (int(maker_df.loc[i, 'group']) == group) :
					profile_num = maker_df.loc[i, 'number']
			
			key_list, file_dict = generate_fc(nfc_dir, facility, group)
			profile_generator(profile_num, key_list, file_dict)


# ~ os.chdir(main_dir + '\\temp')
# ~ model1_file = lib.read_excel('model1_beta_fitting.xlsx')
# ~ model2_file = lib.read_excel('model2_beta_fitting.xlsx')

# ~ facility = '교육'
# ~ excel_name = '[P.교육 서비스업(85)]_0_'
# ~ model3_file = lib.read_excel('profile_48_group_0.xlsx')


# ~ model4_file_day = lib.read_excel('model4_weekdays.xlsx')
# ~ model4_file_end = lib.read_excel('model4_weekends.xlsx')

# ~ save_dir = main_dir + '\\temp'
# ~ profile_num = 5


