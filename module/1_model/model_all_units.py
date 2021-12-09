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

def ave(list1) :
	return sum(list1) / len(list1)
	
def model_1(facility_name, final_dir, model1_dir) :
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
	
	osdays = final_dir + '\\주중'
	osends = final_dir + '\\주말'
	
	for de in folderlist :
		osnow = final_dir + '\\' + de
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
			
			print(f'{excel} done', end = '\r')
			
			
		# 저장
		
		if string == '주중' :
			os.chdir(model1_dir)
			all_model_1.to_excel(f'model_1_var_{string}.xlsx')
			
		else :
			os.chdir(model1_dir)
			all_model_12.to_excel(f'model_1_var_{string}.xlsx')
			
		
	
	os.chdir(model1_dir)
	all_model_1.to_excel('model_1_var.xlsx')
	
	m1 = all_model_1.loc[:, 'var'].tolist()
	
	a, b, s1, s2 = scipy.stats.beta.fit(m1)
	m1_df = pd.DataFrame(columns = [f'{facility_name}'], index = ['a', 'b', 'loc', 'scale'])
	m1_df.loc['a', f'{facility_name}'] = a
	m1_df.loc['b', f'{facility_name}'] = b
	m1_df.loc['loc', f'{facility_name}'] = s1
	m1_df.loc['scale', f'{facility_name}'] = s2
	
	m1_df.to_excel('model_1_beta.xlsx')
	
	pass


def model_2(facility_name, final_dir, model2_dir) :
	
	ndf = pd.DataFrame(columns = [f'{facility_name}_주중', f'{facility_name}_주말'], index = ['a', 'b', 'loc', 'scale'])
	
	
	hours = []
	for i in range(1, 25) :
		hours.append(i)
		
	hours_extend = []
	for i in range(1, 49) :
		hours_extend.append(i)

	
	for folder in os.listdir(final_dir) :
		tempdir = final_dir + '\\' + folder
			
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
	
			os.chdir(model2_dir)
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
			os.chdir(model2_dir)
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
		


def model_3(final_dir, model3_dir) :
	# final_dir 은 '봄가을' 전처리 완료 폴더
	# final_dir 속의 ['주중', '주말'] 폴더 속 엑셀파일을 불러와
	# model3_dir 의 ['주중', '주말'] 폴더에 model3을 저장하고
	# model3_dir 에 profile_48.xlsx 저장함
	
	hours = []
	for i in range(1, 25) :
		hours.append(i)
		
	hours_extend = []
	for i in range(1, 49) :
		hours_extend.append(i)
		
	folderlist = ['주중', '주말']
	folderdays = '주중'
	folderends = '주말'
	
	osdays = final_dir + '\\주중'
	osends = final_dir + '\\주말'
	
	for de in folderlist :
		osnow = final_dir + '\\' + de
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
						
			os.chdir(model3_dir + '\\' + de)
			temp_m2.to_excel(excel)
			print('{} saved'.format(excel), end = '\r')
				
	df = pd.DataFrame(columns = ['excel'] + hours_extend)
	df.columns = df.columns.astype(str)
	df_num = 0

	for folder in os.listdir(model3_dir) :
		tempdir = model3_dir + '\\' + folder
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


	os.chdir(model3_dir)
	df.to_excel('profile_48.xlsx')
	
	print('48 hours profile made, shape : {}'.format(df.shape))
	


def model_4(model3_dir, model4_dir) :
	# folder_dir 에 \\model3 파일이 있어야 하고,\\model3의 ['주말', '주중'] 속 엑셀 사용
	# 2개의 엑셀 파일을 model4_dir 에 저장
	
	hours = []
	for i in range(1, 25) :
		hours.append(i)

	for folder in os.listdir(model3_dir) :
		# folder = ['주중', '주말']
		tempdir = model3_dir + '\\' + folder
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
					
				os.chdir(model4_dir)
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
					
				os.chdir(model4_dir)
				df.to_excel('model4_weekdays.xlsx')


def model1_plot(facility_name, group, model1_dir, plot_dir) :
	os.chdir(model1_dir)
	temp = read_excel('model_1_var.xlsx')
	info = read_excel('model_1_beta.xlsx')
	info.index = ['a', 'b', 'loc', 'scale']
	
	a = info.loc['a', facility_name]
	b = info.loc['b', facility_name]
	loc = info.loc['loc', facility_name]
	scale = info.loc['scale', facility_name]
	
	m1 = temp.loc[:, 'var'].tolist()
	
	plt.figure(figsize = (8, 8))
	plt.title('{}\nBeta Distribution(a = {}, b = {})'.format(facility_name, round(a, 3), round(b, 3)))
	plt.xlabel('model 1')
	plt.ylabel('density')
	
	r = beta.rvs(a, b, loc = loc, scale = scale, size = 10000)
	density = kde.gaussian_kde(m1)
	
	x = np.linspace(min(m1), max(m1), 300)
	y_real = density(x)
	
	plt.grid()
	# ~ plt.plot(x, y_sample, 'b--', label = 'random variates')
	plt.plot(x, y_real, 'r', label = 'real value')
	plt.legend()
	os.chdir(plot_dir)
	plt.savefig(f'model1_{facility_name}_{group}.png', dpi = 400)
	plt.clf()
	plt.cla()
	plt.close()
	pass

def model2_plot(facility_name, group, model2_dir, plot_dir) :
	os.chdir(model2_dir)
	temp_day = read_excel('model2_weekdays_std.xlsx')
	temp_end = read_excel('model2_weekends_std.xlsx')
	m2_day = temp_day.loc[:, 'std'].tolist()
	m2_end = temp_end.loc[:, 'std'].tolist()
	info = read_excel('model_2_beta.xlsx')
	info.index = ['a', 'b', 'loc', 'scale']
	
	min_all = min([min(m2_day), min(m2_end)])
	max_all = max([max(m2_day), max(m2_end)])
	fig = plt.figure(figsize = (16, 10))
	
	for i in range(2) :
		ax = fig.add_subplot(2, 1, 1 + i)
		
		facility = info.columns[i]
		
		a = info.loc['a', facility]
		b = info.loc['b', facility]
		loc = info.loc['loc', facility]
		scale = info.loc['scale', facility]
		# ~ r = beta.rvs(a, b, loc = loc, scale = scale, size = 10000)
		
		# ~ ax.figure(figsize = (8, 8))
		ax.set_title('{}\nBeta Distribution(a = {}, b = {})'.format(facility, round(a, 3), round(b, 3)))
		ax.set_xlabel('model 2')
		ax.set_ylabel('density')
		
		if i == 0 :
			density_real = kde.gaussian_kde(m2_day)
			
		if i == 1 :
			density_real = kde.gaussian_kde(m2_end)
			
		x = np.linspace(min_all, max_all, 300)
		y_real = density_real(x)
		
		ax.grid()
		ax.plot(x, y_real, 'r', label = 'real value')
		ax.legend()
		
	os.chdir(plot_dir)
	plt.savefig(f'model2_{facility_name}_{group}_all.png', dpi = 400)
	plt.clf()
	plt.cla()
	plt.close()
	pass


'''
os.chdir(model2_dir)
	temp = read_excel('model_2_beta.xlsx')
	
	fig = plt.figure(figsize = (16, 10))
	
	for i in range(2) :
		ax = fig.add_subplot(2, 1, 1 + i)
		facility = temp.columns[i]
		
		a = temp.loc['a', facility]
		b = temp.loc['b', facility]
		s1 = temp.loc['loc', facility]
		s2 = temp.loc['scale', facility]
		
		ax.figure(figsize = (8, 8))
		ax.set_title('{}\nBeta Distribution(a = {}, b = {})'.format(facility, round(a, 3), round(b, 3)))
		ax.set_xlabel('model 1')
		ax.set_ylabel('density')
		
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
		
	os.chdir(plot_dir)
	plt.savefit(f'model1_{facility}_solo.png', dpi = 400)
	plt.show()
'''
def model3_plot(facility_name, group, model3_dir, plot_dir) :
	os.chdir(model3_dir)
	pf_48 = read_excel('profile_48.xlsx')
	
	fig = plt.figure(figsize = (14, 7))
	ax = fig.add_subplot(1, 1, 1)
	
	xticks_6 = [6, 12, 18, 24, 30, 36, 42, 48]
	xticks_48 = []
	hours_ex = []
	for i in range(1, 49) :
		xticks_48.append(i)
		hours_ex.append(i)
		
	line_1 = 22.5
	line_2 = 45.5

	print('\tplot start')
		# model 3 group_0

	for i in range(pf_48.shape[0]) :
		ax.plot(hours_ex, pf_48.loc[i, '1' : '48'], alpha = 1)	
	
	ax.plot([24, 24], [-2, 2], color = 'dimgrey') 
	ax.plot([48, 48], [-2, 2], color = 'dimgrey')
	ax.text(line_1, 0.17, 'weekdays', rotation = 90, color = 'dimgrey')
	ax.text(line_2 + 1, 0.17, 'weekends', rotation = 90, color = 'dimgrey')
	
	ax.set_xlim([1, 48])
	ax.set_ylim([0, 0.2])
	ax.set_title('{}_{}_model3'.format(facility_name, group))
	ax.set_xlabel('hours')
	ax.set_xticks(xticks_6)
	ax.grid()
	
	os.chdir(plot_dir)
	plt.savefig('{}_{}_model3.png'.format(facility_name, group), dpi = 400)
	plt.clf()
	plt.cla()
	plt.close()
	pass
	
def model4_plot(facility_name, group, model4_dir, plot_dir) :
	fig = plt.figure(figsize = (10, 5))
	ax = fig.add_subplot(1, 1, 1)
	
	flierprops = dict(marker='o', markerfacecolor='g', markersize= 2 ,\
					linestyle='none', markeredgecolor='dimgrey')
	xticks_48 = []
	hours_ex = []
	for i in range(1, 49) :
		xticks_48.append(i)
		hours_ex.append(i)
		
	line_1 = 22.5
	line_2 = 45.5
				
	os.chdir(model4_dir)
	m4_day = read_excel('model4_weekdays.xlsx')
	m4_end = read_excel('model4_weekends.xlsx')
	
	for i in range(1, 49) :
		if i < 25 : # plot from weekday df
			temp = ax.boxplot(m4_day.loc[:, f'{i}'], positions = [i], flierprops = flierprops)
		else :
			temp = ax.boxplot(m4_end.loc[:, f'{i - 24}'], positions = [i], flierprops = flierprops)
			
	ax.set_xlabel('hours')
	ax.set_title('{}_{}_model4'.format(facility_name, group))
	
	ax.plot([24, 24], [-2, 2], color = 'dimgrey') 

	ax.text(line_1, 0.8, 'weekdays', rotation = 90, color = 'dimgrey')
	ax.text(line_2 + 2, 0.8, 'weekends', rotation = 90, color = 'dimgrey')
	ax.set_xticklabels(xticks_48, rotation = 90)
	
	ax.set_xlim([0, 49])
	ax.set_ylim([-0.2, 1])
	ax.grid()

	os.chdir(plot_dir)
	plt.savefig('{}_{}_model4.png'.format(facility_name, group), dpi = 400)
	plt.clf()
	plt.cla()
	plt.close()
	pass


def model1_compare(facility_name, group, model1_dir, plot_dir, nfc_dir) :
	
	os.chdir(nfc_dir + f'\\{facility_name}\\model1')
	
	smpl = read_excel('model_1.xlsx')
	smpl.index = ['a', 'b', 'loc', 'scale']
	
	if facility_name == '판매및숙박' :
		smvar1 = read_excel('모델1_숙박시설.xlsx')
		smvar2 = read_excel('모델1_판매시설.xlsx')
		smvar1 = smvar1.iloc[:, 0]
		smvar2 = smvar2.iloc[:, 0]
		smvar1.columns = ['var']
		smvar2.columns = ['var']
		
		smvar = pd.concat([smvar1, smvar2])
		smvar.reset_index(drop = True, inplace = True)
		smvar = smvar.loc[:, 'var'].tolist()
		
	else :
		smvar = read_excel(f'모델1_{facility_name}.xlsx')
		smvar = smvar.iloc[:, 0].tolist()
		
		
		
	ass = smpl.loc['a', facility_name]
	bs = smpl.loc['b', facility_name]
	locs = smpl.loc['loc', facility_name]
	scales = smpl.loc['scale', facility_name]
	
	os.chdir(model1_dir)
	temp = read_excel('model_1_var.xlsx')
	info = read_excel('model_1_beta.xlsx')
	info.index = ['a', 'b', 'loc', 'scale']
	
	a = info.loc['a', facility_name]
	b = info.loc['b', facility_name]
	loc = info.loc['loc', facility_name]
	scale = info.loc['scale', facility_name]
	
	m1 = temp.loc[:, 'var'].tolist()
	
	plt.figure(figsize = (8, 8))
	plt.title('{}\nBeta Distribution(a = {}, b = {}) \n(smpl : a = {}, b = {})'.format(facility_name, round(a, 3), round(b, 3), round(ass, 3), round(bs, 3)))
	plt.xlabel('model 1')
	plt.ylabel('density')
	
	r = beta.rvs(a, b, loc = loc, scale = scale, size = 10000)
	density = kde.gaussian_kde(m1)
	density_smpl = kde.gaussian_kde(smvar)
	
	x_smpl = np.linspace(min(smvar), max(smvar), 300)
	y_smpl = density_smpl(x_smpl)
	
	x = np.linspace(min(m1), max(m1), 300)
	y_real = density(x)
	
	plt.grid()
	# ~ plt.plot(x, y_sample, 'b--', label = 'random variates')
	plt.plot(x_smpl, y_smpl, 'b--', label = 'sample')
	plt.plot(x, y_real, 'r', label = 'generated profiles')
	plt.legend()
	os.chdir(plot_dir)
	plt.savefig(f'model1_{facility_name}_{group}_compare.png', dpi = 400)
	plt.clf()
	plt.cla()
	plt.close()
	pass
