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
import os
import glob
import os.path
import math
import random
from scipy.stats import beta, kde
import scipy.stats
import shutil

'''
note
'''

subdir_list = ['preprocessed', 'model1', 'model2', 'model3', 'model4']
m1_col_list = ['교육시설', '문화시설', '숙박시설', '업무시설', '판매시설']
m2_col_list = ['교육시설_주중', '교육시설_주말', '판매및숙박_주중', '판매및숙박_주말',\
				'업무시설_주중', '업무시설_주말', '문화시설_주중', '문화시설_주말']
				
nfc_dir = main_dir + '\\FACILITIES'
dich.newfolder(nfc_dir)

for key in facility_dict :
	check = 0
	if key == '판매및숙박' :
		check = 1
	
	if check == 1 :
		print(f'\n\n{key}\n\n')
		temp_dir = nfc_dir + f'\\{key}'
		dich.newfolder(temp_dir)
		dich.newfolderlist(temp_dir, subdir_list)
		
		ndir_m1 = temp_dir + '\\model1'
		ndir_m2 = temp_dir + '\\model2'
		ndir_m3 = temp_dir + '\\model3'
		ndir_m4 = temp_dir + '\\model4'
		ndir_pre = temp_dir + '\\preprocessed'
		
		odir_m1 = facility_dir + '\\model1'
		odir_m2 = facility_dir + '\\model2'
		odir_m3 = facility_dir + '\\model3_cluster\\{}'.format(key)
		odir_m4 = facility_dir + '\\model4\\{}'.format(key)
		odir_pre = facility_dir + '\\봄가을\\{}'.format(key)
		
		os.chdir(odir_m1)
		model1_df = dich.read_excel('model1_beta_fitting.xlsx')
		model1_df.columns = m1_col_list
		
		if key == '판매및숙박' :
			new_model1 = model1_df[['판매시설', '숙박시설']]
			new_model1.reset_index(drop = True, inplace = True)
			os.chdir(ndir_m1)
			new_model1.to_excel('model_1.xlsx')
			
			pass
			
		else :
			new_model1 = pd.DataFrame(model1_df[f'{key}'])
			new_model1.reset_index(drop = True, inplace = True)
			os.chdir(ndir_m1)
			new_model1.to_excel('model_1.xlsx')
		
		model1_df = None
		new_model1 = None
		print('model 1 ended')
		
		os.chdir(odir_m2)
		model2_df = dich.read_excel('model2_beta_fitting.xlsx')
		model2_df.columns = m2_col_list
		
		if key == '판매및숙박' :
			new_model2 = model2_df[['판매및숙박_주중', '판매및숙박_주말']]
			new_model2.reset_index(drop = True, inplace = True)
			os.chdir(ndir_m2)
			new_model2.to_excel('model_2.xlsx')
			
			pass
			
		else :
			col_all = []
			for col in model2_df.columns :
				if key in col :
					col_all.append(col)
					
			new_model2 = pd.DataFrame(model2_df[col_all])
			new_model2.reset_index(drop = True, inplace = True)
			os.chdir(ndir_m2)
			new_model2.to_excel('model_2.xlsx')
		
		model2_df = None
		new_model2 = None
		print('model 2 ended')
		
	
		dich.copydir_f(odir_m3, ndir_m3)
		
		os.chdir(ndir_m3)
		if 'group_0_preprocessed' not in os.listdir(ndir_m3) :
			os.rename(ndir_m3 + '\\group_0', ndir_m3 + '\\group_0_preprocessed')
			
		if 'group_1_preprocessed' not in os.listdir(ndir_m3) :
			os.rename(ndir_m3 + '\\group_1', ndir_m3 + '\\group_1_preprocessed')
		
		if os.path.isdir(ndir_m3 + '\\group_0') :
			dich.remove(ndir_m3 + '\\group_0')
		
		if os.path.isdir(ndir_m3 + '\\group_1') :
			dich.remove(ndir_m3 + '\\group_1')
			
		dich.newfolder(ndir_m3 + '\\group_0')
		dich.newfolder(ndir_m3 + '\\group_1')
	
		
			
		
		for direc in os.listdir(ndir_m3) :
			
			if 'group_0' in direc :
				if direc != 'group_0' :
					src = ndir_m3 + '\\' + direc
					dst = ndir_m3 + '\\group_0\\' + direc
					
					dich.move(src, dst)
				
			elif 'group_1' in direc :
				if direc != 'group_1' :
					src = ndir_m3 + '\\' + direc
					dst = ndir_m3 + '\\group_1\\' + direc
					
					dich.move(src, dst)
			else : pass
			
		dich.copydir_f(odir_m4, ndir_m4)
		
		if key == '판매및숙박' :
			
			dich.newfolderlist(ndir_pre, ['주중', '주말'])
			odir_pre_1 = facility_dir + '\\봄가을\\숙박시설'
			odir_pre_2 = facility_dir + '\\봄가을\\판매시설'
			
			for ed in ['주중', '주말'] :
				dich.copydir_f(odir_pre_1 + f'\\{ed}', ndir_pre + f'\\{ed}')
				dich.copydir_f(odir_pre_2 + f'\\{ed}', ndir_pre + f'\\{ed}')
				
		else :
			dich.copydir_f(odir_pre, ndir_pre)
			
		
		if key == '판매및숙박' :
			for gp in [0, 1] :
				gp_pre = ndir_m3 + f'\\group_{gp}\\group_{gp}_preprocessed'
				
				dich.remove(gp_pre)
				dich.newfolder(gp_pre)
				dich.newfolderlist(gp_pre, ['주중', '주말'])
				
				gdir = ndir_m3 + f'\\group_{gp}\\group_{gp}_model3'
				pdir = ndir_pre
				
				for sd in ['주중', '주말'] :
					temp_src = pdir + '\\' + sd
					temp_compare = gdir + '\\' + sd
					temp_dst = gp_pre + '\\' + sd
					
					for excel in os.listdir(temp_compare) :
						dich.copyfile(temp_src, temp_dst, excel)
						
					print(f'{excel} copied to {temp_dst}')
			
			
			
	
