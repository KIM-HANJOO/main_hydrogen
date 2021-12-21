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


def model_1_merge(sample_model1_dir, gp_model1_dir) :
	if 'model_1_var.xlsx' in os.listdir(sample_model1_dir) :
		sample_m1 = read_excel('model_1.xlsx')

	if 'model_1_weekdays_std.xlsx' in os.listdir(gp_model1_dir) :
		gp_m1 = read_excel('model_1_var.xlsx')

	
	
	
