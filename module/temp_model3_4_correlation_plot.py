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
import matplotlib.pyplot as plt
import scipy.stats as stats


profile_num_all = 20

def IQR(dist):
    return np.percentile(dist, 75) - np.percentile(dist, 25)
# subplot numbers
num_0 = [1, 2, 5, 6]
num_1 = [3, 4, 7, 8]
num_2 = [9, 10, 13, 14]
num_3 = [11, 12, 15, 16]
	
hours_ex = []
for i in range(1, 49) :
	hours_ex.append(i)

line_1 = 22.5
line_2 = 45.5
	
# make model3_model4 plot

model3_dir = facility_dir + '\\model3_cluster'
model4_dir = facility_dir + '\\model4'

# ~ fig = plt.figure(figsize = (15, 20))

facility_list = ['교육시설']
for profile_num in range(profile_num_all) :
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
		
		
		##
		ncols = ['excel']
		for i in range(1, 25) :
			ncols.append(str(i))
			
		temp_df_day = pd.DataFrame(columns = ncols)
		temp_df_end = pd.DataFrame(columns = ncols)
		day_num = 0
		end_num = 0
		
		aseries = pf48_g0.loc[profile_num, :].copy()
		
		for i in range(g0_m4_day.shape[0]) :
			if pf48_g0.loc[profile_num, 'excel'] in g0_m4_day.loc[i, 'excel'] :
				temp_df_day.loc[day_num, '1' : '24'] = g0_m4_day.loc[i, '1' : '24']
				day_num += 1
				
		for i in range(g0_m4_end.shape[0]) :
			if pf48_g0.loc[profile_num, 'excel'] in g0_m4_end.loc[i, 'excel'] :
				temp_df_end.loc[end_num, '1' : '24'] = g0_m4_end.loc[i, '1' : '24']
				end_num += 1
				
		break
			
	'''
		# subplot add
		ax0_m3 = fig.add_subplot(2, 1, 1)
		ax0_m4 = fig.add_subplot(2, 1, 2)
		
		
		# ~ flierprops = dict(marker='o', markerfacecolor='g', markersize= 2 ,\
						# ~ linestyle='none', markeredgecolor='dimgrey')
						
		xticks_6 = [6, 12, 18, 24, 30, 36, 42, 48]
		xticks_48 = []
		for i in range(1, 49) :
			xticks_48.append(i)
			
		print('\tplot start')
			# model 3 group_0
		
		ax0_m3.plot(hours_ex, pf48_g0.loc[0, '1' : '48'], alpha = 1, c = 'grey', marker = 'o')
	
		ax0_m3.plot([24, 24], [-2, 2], color = 'dimgrey') 
		ax0_m3.plot([48, 48], [-2, 2], color = 'dimgrey')
		# ~ ax0_m3.text(line_1, 0.17, 'weekdays', rotation = 90, color = 'dimgrey')
		# ~ ax0_m3.text(line_2 + 1, 0.17, 'weekends', rotation = 90, color = 'dimgrey')
		
	
		ax0_m3.set_xlim([1, 48])
		ax0_m3.set_ylim([0, 0.2])
		ax0_m3.set_title('{}_{}_model3'.format(fc, '0'))
		ax0_m3.set_xlabel('hours')
		ax0_m3.set_xticks(xticks_6)
		ax0_m3.grid()
		
		print('\tmodel3 plot end')
		
		
			# model 4 group_0
		for i in range(1, 25) :
			temp = ax0_m4.boxplot(temp_df_day.loc[:, f'{i}'], positions = [i]) #, flierprops = flierprops)
			
		for i in range(1, 25) :
			temp = ax0_m4.boxplot(temp_df_end.loc[:, f'{i}'], positions = [i + 24]) #, flierprops = flierprops)
				
		ax0_m4.set_xlabel('hours')
		ax0_m4.set_title('{}_{}_model4'.format(fc, '0'))
		
		ax0_m4.plot([24, 24], [-2, 2], color = 'dimgrey') 
	# 	ax0_m4.plot([48, 48], [-2, 2], color = 'dimgrey')
		# ~ ax0_m4.text(line_1, 0.8, 'weekdays', rotation = 90, color = 'dimgrey')
		# ~ ax0_m4.text(line_2 + 2, 0.8, 'weekends', rotation = 90, color = 'dimgrey')
		ax0_m4.set_xticklabels(xticks_48, rotation = 90)
		
		ax0_m4.set_xlim([0, 49])
		ax0_m4.set_ylim([-0.2, 1])
		ax0_m4.grid()
	
	
		print('\tplot end')
	
	'''
	
	
	x_m3 = pf48_g0.loc[profile_num, '1' : '48'].tolist()
	y_m4 = []
	for col in temp_df_day.columns :
		if 'excel' not in col :
			temp_list = temp_df_day.loc[:, col].tolist()
			iqr_temp = IQR(temp_list)
			y_m4.append(iqr_temp)
		
	for col in temp_df_end.columns :
		if 'excel' not in col :
			temp_list = temp_df_end.loc[:, col].tolist()
			iqr_temp = IQR(temp_list)
			y_m4.append(iqr_temp)
		
	a, p = stats.pearsonr(x_m3, y_m4)
	p = round(p, 5)
	
	plt.scatter(x_m3, y_m4, c = 'b')
	plt.xlabel('model3 | usage')
	plt.ylabel('model4 | IQR(box width)')
	plt.title('p = {}'.format(p))
	
	os.chdir(plot_dir)
	# ~ fig.subplots_adjust(hspace = 0.5, wspace = 0.5)
	# ~ fig.suptitle("Model 3 & Model 4 scatter", fontsize = 12)
	plt.savefig('model_3_4_compare_scatter_{}.png'.format(profile_num), dpi = 400)
	plt.show()


