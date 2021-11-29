'''
note :

'''

###################################################################################
import os
import dir_info
di = dir_info.Info()
main_dir = di.main_dir
prep_dir = di.prep_dir 					# \1_preprocessing
model_dir = di.model_dir 				# \2_model
module_dir = di.module_dir 				# \module
facility_dir = di.facility_dir			# \facility
plot_dir = di.plot_dir					# \plot
cluster_dir = di.cluster_dir 			# \0_temp_dir(clustering)
facility_list = di.facility_list_merge		#['업무시설', '판매및숙박', '문화시설', '교육시설']

import model_library as lib
import pandas as pd
import numpy as np
import scipy.stats as stats

def IQR(dist):
    return np.percentile(dist, 75) - np.percentile(dist, 25)
    
def sorted_find(t_list, find_var) :
	max_var = len(t_list)
	found_index = []
	
	for i in range(len(t_list)) :
		if t_list[i] in name :
			found_index.append(i)
			max_var = i
		else :
			if i > max_var :
				break
				
	return found_index
			
			
model3_dir = facility_dir + '\\model3_cluster'
model4_dir = facility_dir + '\\model4'

hours_str = []
for i in range(1, 49) :
	hours_str.append(str(i))
	
for num, fc in enumerate(facility_list) :
	print(f'{fc} working ...')
	
	fc_dir = model3_dir + f'\\{fc}'
	os.chdir(fc_dir)
	pf48_g0 = lib.read_excel('profile_48_group_0.xlsx')
	pf48_g1 = lib.read_excel('profile_48_group_1.xlsx')
	
	pf48_g0.columns = pf48_g0.columns.astype(str)
	pf48_g1.columns = pf48_g1.columns.astype(str)
		
	os.chdir(model4_dir + f'\\{fc}' + '\\group_0_model4')
	g0_m4_day = lib.read_excel('model4_weekdays.xlsx')
	g0_m4_end = lib.read_excel('model4_weekends.xlsx')
	g0_m4_day.columns = g0_m4_day.columns.astype(str)
	g0_m4_end.columns = g0_m4_end.columns.astype(str)
	
	os.chdir(model4_dir + f'\\{fc}' + '\\group_1_model4')
	g1_m4_day = lib.read_excel('model4_weekdays.xlsx')
	g1_m4_end = lib.read_excel('model4_weekends.xlsx')
	g1_m4_day.columns = g1_m4_day.columns.astype(str)
	g1_m4_end.columns = g1_m4_end.columns.astype(str)
	
	print('\tall loaded')
	
	check = 0 # group num 'un'matches for model3 and model4
	for names in g0_m4_day.loc[:, 'excel'] :
		if pf48_g0.loc[0, 'excel'] in names :
			check = 1 # group num matches for model3 and model4

	if check == 0 : # switch model4 0th with 1st
		temp = g0_m4_day.copy()
		g0_m4_day = g1_m4_day
		g1_m4_day = temp
		
		temp = g0_m4_end.copy()
		g0_m4_end = g1_m4_end
		g1_m4_end = temp
		temp = None
		print('\t0th -> 1st')	
	else :
		print('\tmatched')	
		
	# compare with scatterplot (model3 var - model4 size of box)
	
	for i in [0, 1] : # for groups
		# for current group, 
			
		name = f'{fc}_group{i}'
		t_model3 = locals()[f'pf48_g{i}']
		t_model4_day = locals()[f'g{i}_m4_day']
		t_model4_end = locals()[f'g{i}_m4_end']
		
		t_model4_day = t_model4_day.sort_values(by = ['excel'])
		t_model4_end = t_model4_end.sort_values(by = ['excel'])
		
		t_model4_day.reset_index(drop = True, inplace = True)
		t_model4_end.reset_index(drop = True, inplace = True)
			
		print(f'\tfound current group as {fc}_group{i}')

		# make 48 hours model 4
		
		ncols = ['excel'] + hours_str
		t_model4 = pd.DataFrame(columns = ncols)
		num_t = 0
		
		for j in range(t_model3.shape[0]) :
			
			print(t_model3.loc[j, 'excel'])
			print(t_model4_day.loc[:, 'excel'].tolist())
			
			day_index = sorted_find(t_model4_day.loc[:, 'excel'].tolist(), t_model3.loc[j, 'excel'])
			end_index = sorted_find(t_model4_end.loc[:, 'excel'].tolist(), t_model3.loc[j, 'excel'])
			
			print(day_index)
			print(end_index)
			
			t_m3 = t_model3.loc[j, :].tolist()
			t_day = t_model4_day.loc[day_index, :].copy()
			t_end = t_model4_end.loc[end_index, :].copy()
			t_day.reset_index(drop = True, inplace = True)
			t_end.reset_index(drop = True, inplace = True)
			
			all_y = []
			print(f't_day = {t_day}\n')
			print(f't_end = {t_end}\n')
			
			for k in range(1, 25) :
				temp_list = t_day.loc[:, f'{k}'].tolist()
				iqr_temp = IQR(temp_list)
				all_y.append(iqr_temp)
			
			for k in range(1, 25) :
				temp_list = t_end.loc[:, f'{k}'].tolist()
				iqr_temp = IQR(temp_list)
				all_y.append(iqr_temp)
				
			plt.scatter(t_m3, all_y)
			plt.show()
			
		
			# t_m3 과 all_y 를 서로 correlation analysis로 확인(pearson 상관계)
		
		
		수
		
		# compare (model3 var - model4 iqr)
		
		for j in range(t_model3.shape[0]) :
			temp_m3 = t_model3.loc[j, :]					   # make 1 builidng's model 3
			temp_m3.reset_index(drop = True, inplace = True)
			
			temp_m4 = pd.DataFrame(columns = t_model4.columns) # make 1 building's model 4
			temp_num = 0
			
			for k in range(t_model4.shape[0]) :
				if temp_m3.loc[0, 'excel'] in t_model4.loc[k, 'excel'] :
					temp_m4.loc[temp_num, :] = t_model4.loc[k, :]
					temp_num += 1
			
					
			all_x = temp_m3.tolist()
			all_y = []	
			
			for k in range(1, 49) :
				temp_list = temp_m4.loc[:, f'{k}'].tolist()
				iqr_temp = IQR(temp_list)
				all_y.append(iqr_temp)
				
			print(f'all_x = {all_x}\n')
			print(f'all_y = {all_y}\n')
			
			a, p = stats.pearson(all_x, all_y)
			
			print(f'p-value = {p}')
			if p < 0.05 :
				print('상관관계가 있다')
			else :
				print('상관관계가 없다')
				
			plt.scatter(all_x, all_y)
			plt.show()
				
				
					
		
			
		
