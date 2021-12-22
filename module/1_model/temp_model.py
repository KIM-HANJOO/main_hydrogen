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
gp_dir = di.gp_dir

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
import time
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
	
def denan_list(lst) :
	return [x for x in lst if str(x) != 'nan'] 


'''
model1 : sample : FACILITES/fc/model1/model_1_var.xlsx
	gprofile : GENERATED_PROFILES/fc/group_0/model1_model_1_var.xlsx

model2 : sample : FACILITES/fc/
	gprofile : GENERATED_PROFILES/fc/model2/model_2_weekdays_std.xlsx
				model2/model_2_weekdays_std.xlsx

'''


def model_1_merge(facility_name, group, sample_model1_dir, gp_model1_dir, save_dir) :
	for excel in os.listdir(sample_model1_dir) :
		if '모델1' in excel :
			os.chdir(sample_model1_dir)
			sample_m1 = read_excel(excel)
		# columns : none, iloc[:, 0]
	print('sample : model1 loaded')

	if 'model_1_var.xlsx' in os.listdir(gp_model1_dir) :
		os.chdir(gp_model1_dir)
		gp_m1 = read_excel('model_1_var.xlsx')
		
		# columns : ['excel', 'var']
	print('generated profile : model1 loaded')
	
	fitem = sample_m1.columns[0]
	elist = sample_m1.iloc[:, 0].tolist()
	elist.append(fitem)

	df = pd.DataFrame(columns = ['sample', 'gprofile'])
	df_num = 0

	print(len(elist))
	print(len(gp_m1.loc[:, 'var'].tolist()))
	if len(elist) >= len(gp_m1.loc[:, 'var'].tolist()) :
		
		df.loc[:, 'sample'] = elist
		df.reset_index(drop = True, inplace = True)
		df.loc[0 : len(gp_m1.loc[:, 'var'].tolist()) - 1, 'gprofile'] = gp_m1.loc[:, 'var'].tolist()
		
	else :
		df.loc[:, 'gprofile'] = gp_m1.loc[:, 'var'].tolist()
		df.reset_index(drop = True, inplace = True)
		df.loc[0 : len(elist) - 1, 'sample'] = elist

	os.chdir(save_dir)
	df.to_excel(f'model_1_merge_{facility_name}_group{group}.xlsx')
	print(f'model_1_merge_{facility_name}_group{group}.xlsx saved')
	
def mmerge(facility_name, group, save_dir) :
	for i in ['0', '1'] :
		
		globals()[f'temp_{i}'] = read_excel(f'model_1_merge_{facility_name}_group{i}.xlsx')
		
	print('group_0, 1 loaded')
	
	temp_all = pd.DataFrame(columns = ['sample', 'gprofile'])
	
	temp_all = pd.concat([temp_all, temp_0])
	temp_all = pd.concat([temp_all, temp_1])
	temp_all.reset_index(drop = True, inplace = True)
	
	for col in temp_all.columns :	
		denan = [x for x in temp_all.loc[:, col].tolist() if str(x) != 'nan']
		for i in range(temp_all.shape[0]) :
			temp_all.loc[i, col] = None
			
		temp_all.loc[0 : len(denan) - 1, col] = denan
	
	temp_all.to_excel(f'model_1_merge_{facility_name}_all.xlsx')
	print(f'model_1_merge_{facility_name}_all.xlsx saved')
	
		


def model_2_merge(facility_name, sample_model2_dir, gp_model2_dir, save_dir) :
	for excel in os.listdir(sample_model2_dir) :
		if 'daily' in excel :
			os.chdir(sample_model2_dir)
			sample_m2 = read_excel(excel)
			
	for col in sample_m2.columns :
		if facility_name in col :
			if '주중' in col :
				sm_day = denan_list(sample_m2.loc[:, col].tolist())
				
			elif '주말' in col :
				sm_end = denan_list(sample_m2.loc[:, col].tolist())
	
	sample_m2 = None
	
	print('sample loaded')
	
	os.chdir(gp_model2_dir)
	if 'model2_weekdays_std.xlsx' in os.listdir(gp_model2_dir) :
		temp_day = read_excel('model2_weekdays_std.xlsx')

	else :
		temp_day = pd.DataFrame(columns = ['excel', 'std'])
		for excel in os.listdir(gp_model2_dir) :
			if 'model2_weekdays_std_' in excel :
				print(f'{excel} loading')
				start = time.time()
				os.chdir(gp_model2_dir)
				temp = read_excel(excel)
				temp_day = pd.concat([temp_day, temp])

				end = time.time()
				print(f'{excel} concatenated, elapsed : {round(end - start, 2)}')

	
	if 'model2_weekends_std.xlsx' in os.listdir(gp_model2_dir) :
		temp_end = read_excel('model2_weekends_std.xlsx')

	else :
		temp_end = pd.DataFrame(columns = ['excel', 'std'])
		for excel in os.listdir(gp_model2_dir) :
			if 'model2_weekends_std_' in excel :
				print(f'{excel} loading')
				start = time.time()
				os.chdir(gp_model2_dir)
				temp = read_excel(excel)
				temp_end = pd.concat([temp_end, temp])

				end = time.time()
				print(f'{excel} concatenated, elapsed : {round(end - start, 2)}')
	
	print('gprofiles loaded')
	
	acol = ['id']
	ncol = []
	for i in range(20) :
		acol.append(f'gprofile_{i}')
		ncol.append(f'gprofile_{i}')
	
	
	# ~ merge_day = pd.DataFrame(columns = acol)	
	# ~ merge_end = pd.DataFrame(columns = acol)	
	# ~ merge_all = pd.DataFrame(columns = acol)
	tday = np.array(temp_day.loc[:, 'std']).tolist()
	tend = np.array(temp_end.loc[:, 'std']).tolist()
	
	for i in range(20 - len(tday) % 20) :
		tday.append('nan')
		
	for i in range(20 - len(tend) % 20) :
		tend.append('nan')
		
	print(len(tday), len(tend))
	tarray_day = pd.DataFrame(np.reshape(tday, (-1, 20)))
	tarray_end = pd.DataFrame(np.reshape(tend, (-1, 20)))
	tarray_day.columns = ncol
	tarray_end.columns = ncol
	
	tarray_day['id'] = 0
	tarray_end['id'] = 0
	
	# ~ temp_all = temp_day + temp_end
	temp_day = None
	temp_end = None
	# ~ tarray_all = np.array(temp_all).reshape(-1, 20)
	
	# ~ merge_day.loc[0 : tarray_day.shape[0] - 1, 'gprofile_0' : 'gprofile_19'] = tarray_day
	# ~ merge_end.loc[0 : tarray_end.shape[0] - 1, 'gprofile_0' : 'gprofile_19'] = tarray_end
	# ~ merge_all.loc[:, 'gprofile_0' : 'gprofile_19'] = tarray_all
	# ~ print(merge_day.shape, len(sm_day))
	
	df_smday = pd.DataFrame(columns = ['id', 'sample'])	
	df_smday.loc[:, 'sample'] = sm_day
	
	df_smend = pd.DataFrame(columns = ['id', 'sample'])	
	df_smend.loc[:, 'sample'] = sm_end
	
	tarray_day.reset_index(drop = True, inplace = True)
	tarray_end.reset_index(drop = True, inplace = True)
	df_smday.reset_index(drop = True, inplace = True)
	df_smend.reset_index(drop = True, inplace = True)
	
	for i in range(tarray_day.shape[0]) :
		tarray_day.loc[i, 'id'] = i
		
	for i in range(tarray_end.shape[0]) :
		tarray_end.loc[i, 'id'] = i
	
	for i in range(df_smday.shape[0]) :
		df_smday.loc[i, 'id'] = i
		
	for i in range(df_smend.shape[0]) :
		df_smend.loc[i, 'id'] = i
		
	
	if df_smday.shape[0] > tarray_day.shape[0] :
		di = 'left'
	else :
		di = 'right'
		
	merge_day = pd.merge(df_smday, tarray_day, how = di, on = 'id')
	merge_end = pd.merge(df_smend, tarray_end, how = di, on = 'id')
	merge_day = merge_day.drop(['id'], axis = 'columns')
	merge_end = merge_end.drop(['id'], axis = 'columns')
	
	# ~ merge_day = pd.merge([df_smday, tarray_day])
	# ~ merge_end = pd.mrege([df_smend, tarray_end])
	
	
	# ~ merge_day.loc[0 : len(sm_day) - 1, 'sample'] = sm_day
	# ~ merge_end.loc[0 : len(sm_end) - 1, 'sample'] = sm_end
	# ~ merge_all.loc[0 : len(sm_end) + len(sm_day) - 1, 'sample'] = sm_day + sm_end
	
	print('merging complete')
	print(merge_day)
	os.chdir(save_dir)
	merge_day.to_excel(f'model_2_merged_{facility_name}_group{group}_weekday.xlsx')
	merge_end.to_excel(f'model_2_merged_{facility_name}_group{group}_weekend.xlsx')
	print(f'all saved, {facility_name}, group_{group}')
	

for facility in os.listdir(gp_dir) :
	save_dir = main_dir + '\\merged'
	
	for group in range(2) :
		check = 0
			
		if (facility != '판매및숙박'):
			check = 1

		if check == 1 :
			smdir = f'{facility_dir}\\{facility}'
			sm1dir = smdir + '\\model1'
			sm2dir = smdir + '\\model2'
			
			tdir = gp_dir + '\\' + facility
			print(facility, '\t', 'group number', group)
			nmaindir = tdir + f'\\group_{group}'
			model1_dir = nmaindir + '\\model1'
			model2_dir = nmaindir + '\\model2'
			model3_dir = nmaindir + '\\model3'
			model4_dir = nmaindir + '\\model4'
			final_dir = nmaindir + '\\raw'
			plot_dir = main_dir + '\\GENERATED_PLOTS\\' + facility
			
			
			
			# ~ model_1_merge(facility, group, sm1dir, model1_dir, save_dir)
			model_2_merge(facility, sm2dir, model2_dir, save_dir)

# ~ for facility in os.listdir(gp_dir) :
	# ~ if (facility != '판매및숙박') :
		# ~ mmerge(facility, group, save_dir)



