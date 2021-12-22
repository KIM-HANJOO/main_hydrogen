'''
note :

'''

###################################################################################
import os
import dir_info
di = dir_info.Info()
main_dir = di.main_dir
prep_dir = di.prep_dir
model_dir = di.model_dir
module_dir = di.module_dir
facility_dir = di.facility_dir
plot_dir = di.plot_dir
cluster_dir = di.cluster_dir

import sys
sys.path.append(module_dir)
import model_library as lib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import os.path
import math
import random
from scipy.stats import beta, kde
import scipy.stats

# 원하는 세대 수와 세대 특징 입력된 파일 읽기
def profile_generator(facility, save_dir, profile_num, model1_file, \
				model2_file, model3_file, model4_file_day, model4_file_end) :
	hours = []
	for i in range(1, 25) :
		hours.append(str(i))
	
	model3_file_day = model3_file.loc[:, '1' : '24']
	model3_file_end = model3_file.loc[:, '25' : '48']
	model3_file_end.columns = model3_file_day.columns
	
	for i in range(profile_num):
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
	
		model2_file.columns = ['교육주중', '교육주말', '판매숙박주중', '판매숙박주말', '업무주중', '업무주말', \
								'문화주중', '문화주말']
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
				week_1day = np.random.normal(ave_week_1day, st_week)
				if week_1day > 0:
					break
			#weekend_1day = np.random.normal(ave_weekend_1day, st_weekend)
			# 1일 프로필에 변화를 주는 프로필 생산
			Y = []
			X = np.random.multivariate_normal(mean_1, var_1, check_valid='ignore')
			while 1:
				for k in range(24):
					x = fixed_profile_week.iloc[k,0] + X[k]
					if x < 0:
						X = np.random.multivariate_normal(mean_1, var_1, check_valid='ignore')
						a = 0
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
			
		print('{}번째전력프로필 완성'.format(i+1))
		
		
		
		os.chdir(save_dir)
		
		# 완성된 파일 저장
		changed_profile_week.to_excel('{}번째 세대 전력프로필(평일).xlsx'.format(i+1))
		changed_profile_weekend.to_excel('{}번째 세대 전력프로필(주말).xlsx'.format(i+1))
		
		break
	
	
	
def generate_fc(facility, group) :
	# excel names
	m1_xlna = 'model1_beta_fitting.xlsx'
	m2_xlna = 'model2_beta_fitting.xlsx'
	m3_xlna = 'profile_48_group_0.xlsx'
	m4_xlna = 'model4_weekdays.xlsx'
	
	# main
	m1 = facility_dir + '\\model1'
	m2 = facility_dir + '\\model2'
	m3 = facility_dir + '\\model3_cluster'
	m4 = facility_dir + '\\model4'
	
	os.chdir(m1)
	m1 = lib.read_excel(m1_xlna)
	
	m2 = lib.read_excel(m2_xlna)
	
	
	
	
	
	
	if group == 'all' :
		pass
		
	else :
	
	
	
	
	
	
	
	
	

os.chdir(main_dir + '\\temp')
model1_file = lib.read_excel('model1_beta_fitting.xlsx')
model2_file = lib.read_excel('model2_beta_fitting.xlsx')

facility = '교육'
excel_name = '[P.교육 서비스업(85)]_0_'
model3_file = lib.read_excel('profile_48_group_0.xlsx')


model4_file_day = lib.read_excel('model4_weekdays.xlsx')
model4_file_end = lib.read_excel('model4_weekends.xlsx')

save_dir = main_dir + '\\temp'
profile_num = 5

profile_generator(facility, save_dir, profile_num, model1_file, \
				model2_file, model3_file, model4_file_day, model4_file_end)
