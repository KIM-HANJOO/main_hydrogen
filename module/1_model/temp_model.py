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
	



'''
model1 : sample : FACILITES/fc/model1/model_1_var.xlsx
	gprofile : GENERATED_PROFILES/fc/group_0/model1_model_1_var.xlsx

model2 : sample : FACILITES/fc/
	gprofile : GENERATED_PROFILES/fc/model2/model_2_weekdays_std.xlsx
				model2/model_2_weekdays_std.xlsx

'''


def model_1_merge(sample_model1_dir, gp_model1_dir. save_dir) :
	for excel in os.listdir(sample_model1_dir) :
		if '모델1' in excel :
			os.chdir(sample_model1_dir)
			sample_m1 = read_excel(excel)
		# columns : none, iloc[:, 0]

	if 'model_1_weekdays_std.xlsx' in os.listdir(gp_model1_dir) :
		os.chdir(gp_model1_dir)
		gp_m1 = read_excel('model_1_var.xlsx')
		# columns : ['excel', 'var']

	fitem = sample_m1.columns[0]
	elist = sample_m1.iloc[:, 0].tolist()
	elist.append(fitem)

	df = pd.DataFrame(columns = ['sample', 'gprofile'])
	df_num = 0

	df.loc[:, 'sample'] = elist
	df.loc[:, 'gprofile'] = gp_m1.loc[:, 'var'].tolist()

	os.chdir(save_dir)
	df.to_excel('model_1_merge.xlsx')

def model_2_merge(sample_model2_dir, gp_model2_dir, save_dir) :
	for excel in os.listdir(sample_model2_dir) :
		if 'daily' in excel :
			sample_m2 = read_excel(excel)

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
				temp_day = pd.concat([temp_day.temp])

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
				temp_end = pd.concat([temp_end.temp])

				end = time.time()
				print(f'{excel} concatenated, elapsed : {round(end - start, 2)}')

	

	


