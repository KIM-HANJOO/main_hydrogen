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
sys.path.append(module_dir + '\\4_directory_module')
import directory_change as dich
import model_library as lib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
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


def model_1(facility_name, finaldir, model1dir) :
	all_model_1 = pd.DataFrame(columns = ['excel', 'var'])
	df_num = 0
	
	all_model_12 = pd.DataFrame(columns = ['excel', 'var'])
	df_num2 = 0
	
	hours = []
	for i in range(1, 25) :
		hours.append(i)
		
	hours_extend = []
	for i in range(1, 49) :
		hours_extend.append(i)
		
	folderlist = ['주중', '주말']
	folderdays = '주중'
	folderends = '주말'
	
	osdays = finaldir + '\\주중'
	osends = finaldir + '\\주말'
	
	for de in folderlist :
		osnow = finaldir + '\\' + de
		if de == folderdays :
			osnow = osdays
			string = '주중'
		else :
			osnow = osends
			string = '주말'
			
		for excel in os.listdir(osnow) :
			os.chdir(osnow)
			temp = read_excel(excel)
			all_sum = []
			for i in range(temp.shape[0]) :
				all_sum.append(temp.loc[i, :].sum())
			all_model_1.loc[df_num, 'excel'] = excel + f'_{string}'
			all_model_1.loc[df_num, 'var'] = ave(all_sum)
			df_num += 1
			
			if string == '주말' :
				all_model_12.loc[df_num2, 'excel'] = excel + f'_{string}'
				all_model_12.loc[df_num2, 'var'] = ave(all_sum)
				df_num2 += 1
			
			
		# 저장
		
		if string == '주중' :
			os.chdir(model1dir)
			all_model_1.to_excel(f'model_1_var_{string}.xlsx')
			
		else :
			os.chdir(model1dir)
			all_model_12.to_excel(f'model_1_var_{string}.xlsx')
	
	os.chdir(model1dir)
	all_model1_1.to_excel('model_1_var.xlsx')
	
	m1 = all_model1_1.loc[:, 'var'].tolist()
	
	a, b, s1, s2 = scipy.stats.beta.fit(m1)
	m1_df = pd.DataFrame(columns = f'{facility_name}', index = ['a', 'b', 'loc', 'scale'])
	m1_df.loc['a', f'{facility_name}'] = a
	m1_df.loc['b', f'{facility_name}'] = b
	m1_df.loc['loc', f'{facility_name}'] = s1
	m1_df.loc['scale', f'{facility_name}'] = s2
	
	m1_df.to_excel('model_1_beta.xlsx')
	
	pass


def model_2(facility_name, finaldir, model2dir) :
	
	ndf = pd.DataFrame(columns = [f'{facility_name}_주중', f'{facility_name}_주말'], index = ['a', 'b', 'loc', 'scale']]
	
	
	hours = []
	for i in range(1, 25) :
		hours.append(i)
		
	hours_extend = []
	for i in range(1, 49) :
		hours_extend.append(i)
		
	folderlist = ['주중', '주말']
	folderdays = '주중'
	folderends = '주말'
	
	osdays = finaldir + '\\주중'
	osends = finaldir + '\\주말'
	
	for de in folderlist :
		osnow = finaldir + '\\' + de
		if de == folderdays :
			osnow = osdays
		else :
			osnow = osends
			
		for excel in os.listdir(osnow) :
			os.chdir(osnow)
			temp = read_excel(excel)
			for i in range(temp.shape[0]) :
				
	##
	
	for folder in os.listdir(finaldir) :
	tempdir = finaldir + '\\' + folder
	if '주중' in folder :
		df = pd.DataFrame(columns = ['excel', 'std'])
		df_num = 0

		for excel in os.listdir(tempdir) :
			os.chdir(tempdir)
			temp = lib.read_excel(excel)
			alllist = []
			for i in range(temp.shape[0]) :
				tempsum = temp.loc[i, :].sum()
				alllist.append(tempsum)

			temp_std = np.std(alllist)
			df.loc[df_num, 'excel'] = excel
			df.loc[df_num, 'std'] = temp_std
			df_num += 1

		os.chdir(model2dir)
		df.to_excel('model2_weekdays_std.xlsx')
		print('{} done'.format(excel), end = '\r')
		
		m2_1 = df.loc[:, 'std'].tolist()
		a, b, s1, s2 = scipy.stats.beta.fit(m2_1)
		
		ndf.loc['a', f'{facility_name}_주중'] = a
		ndf.loc['b', f'{facility_name}_주중'] = b
		ndf.loc['loc', f'{facility_name}_주중'] = s1
		ndf.loc['scale', f'{facility_name}_주중'] = s2
		
		
		
	if '주말' in folder :
		df = pd.DataFrame(columns = ['excel', 'std'])
		df_num = 0

		for excel in os.listdir(tempdir) :
			os.chdir(tempdir)
			temp = lib.read_excel(excel)
			alllist = []
			for i in range(temp.shape[0]) :
				tempsum = temp.loc[i, :].sum()
				alllist.append(tempsum)

			temp_std = np.std(alllist)
			df.loc[df_num, 'excel'] = excel
			df.loc[df_num, 'std'] = temp_std
			df_num += 1
		os.chdir(model2dir)
		df.to_excel('model2_weekends_std.xlsx')
		print('{} done'.format(excel), end = '\r')
		
		m2_2 = df.loc[:, 'std'].tolist()
		a, b, s1, s2 = scipy.stats.beta.fit(m2_2)
		
		ndf.loc['a', f'{facility_name}_주중'] = a
		ndf.loc['b', f'{facility_name}_주중'] = b
		ndf.loc['loc', f'{facility_name}_주중'] = s1
		ndf.loc['scale', f'{facility_name}_주중'] = s2
	
	
	ndf.to_excel('model_2_beta.xlsx')
	print('model_2_beta.xlsx saved')
		


def model_3(finaldir, model3dir) :
	# finaldir 은 '봄가을' 전처리 완료 폴더
	# finaldir 속의 ['주중', '주말'] 폴더 속 엑셀파일을 불러와
	# model3dir 의 ['주중', '주말'] 폴더에 model3을 저장하고
	# model3dir 에 profile_48.xlsx 저장함
	
	hours = []
	for i in range(1, 25) :
		hours.append(i)
		
	hours_extend = []
	for i in range(1, 49) :
		hours_extend.append(i)
		
	folderlist = ['주중', '주말']
	folderdays = '주중'
	folderends = '주말'
	
	osdays = finaldir + '\\주중'
	osends = finaldir + '\\주말'
	
	for de in folderlist :
		osnow = finaldir + '\\' + de
		if de == folderdays :
			osnow = osdays
		else :
			osnow = osends
			
		for excel in os.listdir(osnow) :
			os.chdir(osnow)
			temp = read_excel(excel)
			temp_m2 = pd.DataFrame(columns = temp.columns)
			temp.reset_index(drop = True, inplace = True)
			
			check = 0
			for index in range(temp.shape[0]) :
				total_index = temp.loc[index, :].sum()
				if total_index != 0 :
					for cat in temp.columns :
						temp_m2.loc[index, cat] = temp.loc[index, cat] / total_index
						
			os.chdir(model3dir + '\\' + de)
			temp_m2.to_excel(excel)
			print('{} saved'.format(excel), end = '\r')
				
	df = pd.DataFrame(columns = ['excel'] + hours_extend)
	df.columns = df.columns.astype(str)
	df_num = 0

	for folder in os.listdir(model3dir) :
		tempdir = model3dir + '\\' + folder
		if '주중' in folder :
			folderdays = folder
			model3days = tempdir
		elif '주말' in folder :
			folderends = folder
			model3ends = tempdir

	for excel in os.listdir(model3days) :
		print('working on {}'.format(excel), end = '\r')
		os.chdir(model3days)
		temp = lib.read_excel(excel)
		for excel2 in os.listdir(model3ends) :
			os.chdir(model3ends)
			if excel[ : -11] == excel2[ : -11] :
				excel_match = excel2
				temp2 = lib.read_excel(excel_match)
			
		temp.columns = temp.columns.astype(str)
		temp2.columns = temp2.columns.astype(str)
		temp.reset_index(drop = True, inplace = True)
		temp2.reset_index(drop = True, inplace = True)

		df.loc[df_num, 'excel'] = excel[: -11]

		temp_ave = pd.DataFrame(columns = temp.columns)
		temp_ave2 = pd.DataFrame(columns = temp2.columns)

		ave_num = 0

		for cat in temp_ave.columns :
			temp_ave.loc[ave_num, cat] = temp.loc[ : , cat].mean()

		for cat in temp_ave2.columns :
			temp_ave2.loc[ave_num, cat] = temp2.loc[ : , cat].mean() 



		df.loc[df_num, '1' : '48'] = temp_ave.loc[ave_num, '1' : '24'].tolist() \
									+ temp_ave2.loc[ave_num, '1' : '24'].tolist()
		df_num += 1
		
		print('~{}, ~{} is matched'.format(excel[-15 : -5], excel_match[-15 : -5]), end = '\r')

		

	os.chdir(model3dir)
	df.to_excel('profile_48.xlsx')
	self.profile_48  = df
	print('48 hours profile made, shape : {}'.format(df.shape))
	print(self.profile_48)


def model_4(model3dir, model4dir) :
	# folder_dir 에 \\model3 파일이 있어야 하고,\\model3의 ['주말', '주중'] 속 엑셀 사용
	# 2개의 엑셀 파일을 model4dir 에 저장
	
	hours = []
	for i in range(1, 25) :
		hours.append(i)

	for folder in os.listdir(model3dir) :
		# folder = ['주중', '주말']
		tempdir = model3dir + '\\' + folder
		if os.path.isdir(tempdir) :
			if '주말' in folder :
				df = pd.DataFrame(columns = ['excel'] + hours)
				df_num = 0
				df.columns = df.columns.astype(str)
				os.chdir(tempdir)
				
				for excel in os.listdir(tempdir) :
					temp = read_excel(excel)
					temp.columns = temp.columns.astype(str)
					
					profile = pd.DataFrame(columns = hours)
					profile.columns = profile.columns.astype(str)
					for cat in temp.columns :
						profile.loc[0, cat] = temp.loc[:, cat].mean()
						
					for i in range(temp.shape[0]):
						df.loc[df_num, 'excel'] = excel + '_{}'.format(i)
							
						for cat in temp.columns :
							df.loc[df_num, str(cat)] = temp.loc[i, str(cat)] - profile.loc[0, str(cat)]
						df_num += 1
					print('weekends,{} done'.format(excel), end = '\r')
					
				os.chdir(model4dir)
				df.to_excel('model4_weekends.xlsx')
				
			elif '주중' in folder :
				df = pd.DataFrame(columns = ['excel'] + hours)
				df_num = 0
				df.columns = df.columns.astype(str)
				os.chdir(tempdir)
				
				for excel in os.listdir(tempdir) :
					temp = read_excel(excel)
					temp.columns = temp.columns.astype(str)
					
					profile = pd.DataFrame(columns = hours)
					profile.columns = profile.columns.astype(str)
					for cat in temp.columns :
						profile.loc[0, cat] = temp.loc[:, cat].mean()
						
					for i in range(temp.shape[0]):
						df.loc[df_num, 'excel'] = excel + '_{}'.format(i)
							
						for cat in temp.columns :
							df.loc[df_num, str(cat)] = temp.loc[i, str(cat)] - profile.loc[0, str(cat)]
						df_num += 1
					print('weekdays, {} done'.format(excel), end = '\r')
					
				os.chdir(model4dir)
				df.to_excel('model4_weekdays.xlsx')


def model1_plot(model1_dir) :
	os.chdir(model1_dir)
	temp = read_excel('model_1_beta.xlsx')
	for col in temp.columns :
		if 
	plt.figure(figsize = (8, 8))
	plt.title('{}\nBeta Distribution(a = {}, b = {})'.format(facility, round(a, 3), round(b, 3)))
	plt.xlabel('model 1')
	plt.ylabel('density')
	
	r = beta.rvs(a, b, loc = s1, scale = s2, size = 10000)
	density_real = kde.gaussian_kde(m2)
	density_sample = kde.gaussian_kde(r)
	x = np.linspace(min(r), max(r), 300)
	
	y_real = density_real(x)
	y_sample = density_sample(x)
	
	ax.grid()
	ax.plot(x, y_sample, 'b--', label = 'random variates')
	ax.plot(x, y_real, 'r', label = 'real value')
	ax.legend()
	pass

def model2_plot(model3_dir) :
	pass

def model3_plot(model3_dir) :
	pass
	
def model4_plot(model4_dir) :
	pass
