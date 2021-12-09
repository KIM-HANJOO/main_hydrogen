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
sys.path.append(module_dir + '\\4_directory_moduel')
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

subdir_list = ['raw', 'model1', 'model2', 'model3', 'model4']
fc_list = ['교육시설', '문화시설', '숙박시설', '업무시설', '판매시설']
fc_list_2 = ['교육시설', '문화시설', '판매및숙박', '업무시설']

				
gfc_dir = main_dir + '\\GENERATED_PROFILES'
dich.newfolder(gfc_dir)

dich.newfolderlist(gfc_dir, fc_list_2)
for fc in fc_list_2 :
	tempdir = gfc_dir + f'\\{fc}'
	dich.newfolderlist(tempdir, ['group_0', 'group_1'])
	
	for i in range(2) :
		dich.newfolderlist(tempdir + f'\\group_{i}', subdir_list)
		
		for sd in subdir_list :
			if (sd == 'raw') | (sd == 'model3'):
				dich.newfolderlist(tempdir + f'\\group_{i}\\{sd}', ['주중', '주말'])
	
	print(f'{fc} directory all made')
