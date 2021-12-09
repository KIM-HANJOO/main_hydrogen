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


def model_1(finaldir, model1dir) :
	
	
	pass


def model_2(finaldir, model2dir) :
	
	
	pass


def model_3(finaldir, model3dir) :
	# finaldir 은 '봄가을' 전처리 완료 폴더
	# finaldir 속의 ['주중', '주말'] 폴더 속 엑셀파일을 불러와
	# model3dir 의 ['주중', '주말'] 폴더에 model3을 저장하고
	# model3dir 에 profile_48.xlsx 저장함
	
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

def model1_plot(model3_dir) :
	pass

def model2_plot(model3_dir) :
	pass

def model3_plot(model3_dir) :
	pass
	
def model4_plot(model4_dir) :
	pass
