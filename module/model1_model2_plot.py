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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgunbd.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
matplotlib.rcParams['axes.unicode_minus'] = False
import os



