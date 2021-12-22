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

'''
note

use 'model_all_units.py' as one and only module
'''
import model_all_units_판숙 as mu

for facility in os.listdir(gp_dir) :
	if facility == '판매및숙박' :
		tdir = gp_dir + '\\' + facility
		for group in range(2) :
			print(facility, '\t', 'group number', group)
			nmaindir = tdir + f'\\group_{group}'
			model1_dir = nmaindir + '\\model1'
			model2_dir = nmaindir + '\\model2'
			model3_dir = nmaindir + '\\model3'
			model4_dir = nmaindir + '\\model4'
			final_dir = nmaindir + '\\raw'
			plot_dir = main_dir + '\\GENERATED_PLOTS\\' + facility
			dich.newfolder(plot_dir)
			tempdir_0 = 'C:\\Users\\김한주\\Documents\\GitHub\\main_hydrogen\\temp_model3\\group_0'
			tempdir_1 = 'C:\\Users\\김한주\\Documents\\GitHub\\main_hydrogen\\temp_model3\\group_1'
			
			srcdir_0 = 'C:\\Users\\김한주\\Documents\\GitHub\\main_hydrogen\\FACILITIES\\판매및숙박\\model3\\group_0\\group_0_preprocessed'
			srcdir_1 = 'C:\\Users\\김한주\\Documents\\GitHub\\main_hydrogen\\FACILITIES\\판매및숙박\\model3\\group_1\\group_1_preprocessed'
			print('\nmodel_1 running')
			
			# ~ mu.model_1(facility, srcdir_0, tempdir_0)
			# ~ mu.model_1(facility, srcdir_1, tempdir_1)
			
			print('1')
			mu.model1_compare(facility, group, tempdir_0, main_dir, nfc_dir)
			mu.model1_compare(facility, group, tempdir_1, main_dir, nfc_dir)
			# ~ print('\nmodel_2 running')
			# ~ mu.model_2(facility, final_dir, model2_dir)
			# ~ print('\nmodel_3 running')
			# ~ mu.model_3(final_dir, model3_dir)
			# ~ print('\nmodel_4 running')
			# ~ mu.model_4(model3_dir, model4_dir)
			
			# ~ print('\nmodel_1 plotting')
			# ~ mu.model1_plot(facility, group, model1_dir, plot_dir)
			# ~ print('\nmodel_2 plotting')
			# ~ mu.model2_plot(facility, group, model2_dir, plot_dir)
			# ~ print('\nmodel_3 plotting')
			# ~ mu.model3_plot(facility, group, model3_dir, plot_dir)
			# ~ print('\nmodel_4 plotting')
			# ~ mu.model4_plot(facility, group, model4_dir, plot_dir)
			
