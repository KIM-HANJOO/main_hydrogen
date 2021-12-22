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
import shutil

'''
note
'''



facility_list = ['교육시설', '문화시설', '업무시설', '판매및숙박'] #profile_num_maker에서 쓰이며, model1, model2 파일 사전작업에 사용되었음

def profile_num_maker(nfc_dir) :
	
	facility_list = ['교육시설', '문화시설', '업무시설', '판매및숙박']
			
	maker_df = pd.DataFrame(columns = ['facility', 'group', 'number', 'percentage'])
	maker_num = 0
	
	for num, fc in enumerate(facility_list) :
		for i in range(2) : # group_number
			tempdir = nfc_dir + f'\\{fc}\\model3\\group_{i}\\group_{i}_model3\\주중'
			op_num = len(os.listdir(tempdir))
			
			maker_df.loc[maker_num, 'facility'] = fc
			maker_df.loc[maker_num, 'group'] = i
			maker_df.loc[maker_num, 'number'] = op_num
			maker_num += 1
	
	all_profile_numbers = sum(maker_df.loc[:, 'number'].tolist())
	
	for i in range(maker_df.shape[0]) :
		maker_df.loc[i, 'percentage'] = round(maker_df.loc[i, 'number'] / all_profile_numbers * 100, 2)
	
	return maker_df

maker_df = profile_num_maker(nfc_dir)

print(maker_df)


ratio = maker_df.loc[:, 'percentage'].tolist()
labels = []
for i in range(maker_df.shape[0]) :
	if maker_df.loc[i, 'facility'] == '문화시설' :
		if maker_df.loc[i, 'group'] == 1 :
			name = ''
			
		else :
			name = f"{ratio[i]}%, {ratio[i + 1]}%\n문화시설(0,1)"
			
			
	elif maker_df.loc[i, 'facility'] == '교육시설' :
		if maker_df.loc[i, 'group'] == 1 :
			name = f"\n\n{ratio[i]}%\n{maker_df.loc[i, 'facility']}({maker_df.loc[i, 'group']})"
		else :
			name = f"\n\n\n{ratio[i]}%\n{maker_df.loc[i, 'facility']}({maker_df.loc[i, 'group']})"
	else :
		name = f"{ratio[i]}%\n{maker_df.loc[i, 'facility']}({maker_df.loc[i, 'group']})"
		
	labels.append(name)
	
explode = []
for i in range(len(labels)) :
	explode.append(0.05)


plt.figure(figsize = (8, 8))
colors = ['dimgrey', 'grey', 'dimgrey', 'grey', 'brown', 'brown', 'indianred', 'indianred']
plt.pie(ratio, labels = labels, startangle = 260, counterclock = False, explode = explode, shadow = False, colors = colors)
plt.title('percentages')
os.chdir('C:\\Users\\joo09\\Documents\\GitHub\\main_hydrogen\\module\\3_profile_generator\\temp_plot')
plt.savefig('group_percentage_piechart.png')
plt.show()
