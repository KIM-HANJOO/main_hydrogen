import model_library as lib
import ClusterAnalysis as ca

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
cwdir = os.getcwd()
import sys
sys.path.append(cwdir + '\\module')
import table as tb

import math
import shutil
from sklearn.cluster import KMeans
from sklearn.metrics import *
from distutils.dir_util import copy_tree

from sklearn.cluster import KMeans
from sklearn.cluster import k_means


def read_excel(excel) :
	df = pd.read_excel(excel)
	if 'Unnamed: 0' in df.columns :
		df.drop('Unnamed: 0', axis = 1, inplace = True)

	if 'Unnamed: 0.1' in df.columns :
		df.drop('Unnamed: 0.1', axis = 1, inplace = True)

	return df
		
def newfolder(directory):
	try:
		if not os.path.exists(directory):
			os.makedirs(directory)
	except OSError:
		print ('Error: Creating directory. ' +  directory)


def newfolderlist(directory, folderlist):
	for i, names in enumerate(folderlist):
		directory_temp = directory + '\\' + names
		try:
			if not os.path.exists(directory_temp):
				os.makedirs(directory_temp)
		except OSError:
			print ('Error: Creating directory. ' +  directory_temp)

def copyfile(src_dir, dst_dir, src_file) :
	src = src_dir + '\\' + src_file
	dst = dst_dir
	shutil.copyfile(src, dst)

def remove(path):
	# """ param <path> could either be relative or absolute. """
	if os.path.isfile(path) or os.path.islink(path):
		os.remove(path)  # remove the file

	elif os.path.isdir(path):
		shutil.rmtree(path)  # remove dir and all contains
		
	else:
		raise ValueError("file {} is not a file or dir.".format(path))
		
class model() :
	def __init__(self, main_dir) :
		pre_dir = main_dir + '\\전체 업태'
		model_dir = main_dir + '\\model'
		try :
			if not os.path.exists(model_dir) :
				os.makedirs(model_dir)
		except OSError :
			print('{} already made'.format(model_dir))
			
		folder_list = []
		for folder in os.listdir(pre_dir) :
			if os.path.isdir(pre_dir + '\\' + folder) :
				folder_list.append(folder)
		print(folder_list)
		
		for num, folder in enumerate(folder_list) :
			temp_dir = model_dir + '\\' + folder
			try :
				if not os.path.exists(temp_dir) :
					os.makedirs(temp_dir)
			except OSError :
				print('{} subfolders already made'.format(temp_dir))
				
		model_list = ['preprocessed', 'model1', 'model2', 'model3', \
						'model3_cluster', 'model4', 'plot']
						
		dayends = ['주중', '주말']
		
		for folder in os.listdir(model_dir) :
			for model_name in model_list :
				if 'model4' not in model_name :
					if 'plot' not in model_name :
						for de in dayends :
							adddir = model_dir + '\\' + folder + '\\' + model_name + '\\' + de
							newfolder(adddir)
		
		for folder in os.listdir(pre_dir) :
			tempdir = pre_dir + '\\' + folder
			weekend_src = tempdir + '\\7_봄가을\\주말'
			weekday_src = tempdir + '\\7_봄가을\\주중'
			
			weekend_dst = model_dir + '\\' + folder + '\\preprocessed\\주말'
			weekday_dst = model_dir + '\\' + folder + '\\preprocessed\\주중'
			
			for f in os.listdir(weekend_src) :
				shutil.copyfile(weekend_src + '\\' + f, weekend_dst + '\\' + f)
			for f in os.listdir(weekday_src) :
				shutil.copyfile(weekday_src + '\\' + f, weekday_dst + '\\' + f)
			
		print('files all pasted...')
		print('ready!')
		
		hours = []
		hours_str = []
		for i in range(1, 25) :
			hours.append(i)
			hours_str.append(str(i))
		hours_ex = []
		hours_ex_str = []
		for i in range(1, 49) :
			hours_ex.append(i)
			hours_ex_str.append(str(i))
			
		condition_list = []
		for folder in os.listdir(model_dir) :
			if os.path.isdir(model_dir + '\\' + folder) :
				condition_list.append(folder)
				
		self.condition_list = condition_list
		self.hours = hours
		self.hours_str = hours_str
		self.hours_ex = hours_ex
		self.hours_ex_str = hours_ex_str
		
		
		self.main_dir = main_dir
		self.src_dir = pre_dir
		self.model_dir = model_dir
		
		
	def reset(self) :
		for everything in os.listdir(self.model_dir) :
			remove(self.model_dir + '\\' + everything)
		self.__init___(self.main_dir)
		
	def model3(self) :
		
		# finaldir -> self.model_dir + '\\' + 업태 + '\\preprocessed\\주중'  
		# model3 정규화 프로필, 평균 프로필
		# model3 완
		
		## orgdir = '\\업태별'
		## maindir = '\\업태별\\0_전체 업태로 모델 만들기2'
		## finaldir = maindir + '\\전체업태전처리'
		## model2dir / model3dir / model3cluster / model4dir / plotdir성
		
		hours = self.hours
		hours_str = self.hours_str
		hours_extend = self.hours_ex
		hours_ex_str = self.hours_ex_str
		
		for condition in self.condition_list :
			srcdir = self.src_dir + '\\' + condition
			cwdir = self.model_dir + '\\' + condition 
			finaldir = cwdir + '\\preprocessed'
			model3dir = cwdir + '\\model3'
			model3cluster = cwdir + '\\model3_cluster'
			model4dir = cwdir + '\\model4'
			
			folderlist = ['주중', '주말']
			folderdays = '주중'
			folderends = '주말'
			
			osdays = finaldir + '\\주중'
			osends = finaldir + '\\주말'
			
			for de in folderlist :
				osnow = finaldir + '\\' + de
				if de == folderdays :
					osnow = osdays
				else :
					osnow = osends
					
				for excel in os.listdir(osnow) :
					os.chdir(osnow)
					temp = read_excel(excel)
					temp_m2 = pd.DataFrame(columns = temp.columns)
					temp.reset_index(drop = True, inplace = True)
					
					check = 0
					for index in range(temp.shape[0]) :
						total_index = temp.loc[index, :].sum()
						if total_index != 0 :
							for cat in temp.columns :
								temp_m2.loc[index, cat] = temp.loc[index, cat] / total_index
								
					os.chdir(model3dir + '\\' + de)
					temp_m2.to_excel(excel)
					print('{} saved'.format(excel), end = '\r')
						
			df = pd.DataFrame(columns = ['excel'] + hours_extend)
			df.columns = df.columns.astype(str)
			df_num = 0

			for folder in os.listdir(model3dir) :
				tempdir = model3dir + '\\' + folder
				if '주중' in folder :
					folderdays = folder
					model3days = tempdir
				elif '주말' in folder :
					folderends = folder
					model3ends = tempdir

			for excel in os.listdir(model3days) :
				print('working on {}'.format(excel), end = '\r')
				os.chdir(model3days)
				temp = lib.read_excel(excel)
				for excel2 in os.listdir(model3ends) :
					os.chdir(model3ends)
					if excel[ : -11] == excel2[ : -11] :
						excel_match = excel2
						temp2 = lib.read_excel(excel_match)
					
				temp.columns = temp.columns.astype(str)
				temp2.columns = temp2.columns.astype(str)
				temp.reset_index(drop = True, inplace = True)
				temp2.reset_index(drop = True, inplace = True)

				df.loc[df_num, 'excel'] = excel[: -11]

				temp_ave = pd.DataFrame(columns = temp.columns)
				temp_ave2 = pd.DataFrame(columns = temp2.columns)

				ave_num = 0

				for cat in temp_ave.columns :
					temp_ave.loc[ave_num, cat] = temp.loc[ : , cat].mean()

				for cat in temp_ave2.columns :
					temp_ave2.loc[ave_num, cat] = temp2.loc[ : , cat].mean() 



				df.loc[df_num, '1' : '48'] = temp_ave.loc[ave_num, '1' : '24'].tolist() \
											+ temp_ave2.loc[ave_num, '1' : '24'].tolist()
				df_num += 1
				
				print('~{}, ~{} is matched'.format(excel[-15 : -5], excel_match[-15 : -5]), end = '\r')

				

			os.chdir(model3dir)
			df.to_excel('profile_48.xlsx')
			self.profile_48  = df
			print('48 hours profile made, shape : {}'.format(df.shape))
			print(self.profile_48)
						
	def model3_cluster(self, condition_set) :
		hours = self.hours
		hours_str = self.hours_str
		hours_ex = self.hours_ex
		hours_ex_str = self.hours_ex_str
		
		check = 0
	
		for condition in self.condition_list :
			if condition_set in condition :
				srcdir = self.src_dir + '\\' + condition
				cwdir = self.model_dir + '\\' + condition 
				finaldir = cwdir + '\\preprocessed'
				model3dir = cwdir + '\\model3'
				model3cluster = cwdir + '\\model3_cluster'
				model4dir = cwdir + '\\model4'
				
				check = 1
		if check == 1 :
			os.chdir(model3dir)
			profile = read_excel('profile_48.xlsx')
			
			K = ca.chs_find(profile, 1)
			locals()['info_{}'.format(condition)], locals()['centers_{}'.format(condition)] = \
			ca.cluster_new(profile, K)
			print('optimal K = {}, based on "calinski-harabasz score"'.format(K))
			K = int(K)
			_, cluster_labels, _ = k_means(profile.loc[:, '1' : '48'], n_clusters = K)

			for i in range(locals()['centers_{}'.format(condition)].shape[0]) :
				print('{}th group profile'.format(i))
				ca.wanted_centers_plot(locals()['info_{}'.format(condition)], [i])
				
			for group_num in range(K) :
				locals()['group_{}'.format(group_num)] = []
				
				for i in range(len(cluster_labels)) :
					if cluster_labels[i] == group_num :
						locals()['group_{}'.format(group_num)].append(i)
			
			file_list = []
			for group_num in range(K) :
				file_list.append('group_{}'.format(group_num))
			
			for group_num in range(K) :
				locals()['group_{}'.format(group_num)] = \
				profile.loc[locals()['group_{}'.format(group_num)], :]
				
				locals()['group_{}'.format(group_num)].reset_index(drop = True, inplace = True)
			
				os.chdir(model3cluster)
				locals()['group_{}'.format(group_num)].to_excel('profile_48_group_{}.xlsx'.format(group_num))
			
				group_info = locals()['group_{}'.format(group_num)].loc[:, 'excel'].tolist()
				print('{}th group = {}, size = {}'.format(group_num, group_info[ : 5], len(group_info)))
				
				# 새로운 폴더 만들기
				newfolderlist(model3cluster, file_list)
				for f in file_list :
					newfolderlist(model3cluster + '\\' + f, ['주중', '주말'])
				
				# finaldir에서 옮기
				for excel_name in group_info :
					for folder in os.listdir(finaldir) : # folder = ['주중', '주말']
						tempdir = finaldir + '\\' + folder
						for real_excel in os.listdir(tempdir) : # ex - '주중' 폴더에
							if excel_name in real_excel :
								src = tempdir
								dst = model3cluster + '\\group_{}\\'.format(group_num) + folder
								shutil.copyfile(src + '\\' + real_excel, dst + '\\' + real_excel)
							
							
		
			
		# model3 48시간 프로필 만들기
		# model3 클러스터링
		# model3 클러스터 완성
		# model3 MANOVA성
		# model3 클러스터링 완
		
	def normalize_cluster(self, condition) :
		
		for c in self.condition_list :
			if condition in c :
				condition = c
				
		
		print('normalizing...1')
		cluster_all = self.model_dir + '\\' + condition + '\\model3_cluster'
		print('cwd is {}'.format(cluster_all))
		
		folder_list = []

		for folder in os.listdir(cluster_all) :
			if os.path.isdir(cluster_all + '\\' + folder) :
				if 'group_' in folder :
					folder_list.append('normalized_g_{}'.format(folder[-1 :]))
					print(folder, folder[-1 :], 'normalized_g_{}'.format(folder[-1 :]))

		lib.newfolderlist(cluster_all, folder_list)
		for folder in folder_list :
			lib.newfolderlist(cluster_all + '\\' + folder, ['주중', '주말'])
	
		print('normalizing...2')
		
		for folder in os.listdir(cluster_all) :
			if 'group_' in folder :
				group_num = folder[-1 :]
				tempdir = cluster_all + '\\' + folder
				if os.path.isdir(tempdir) :
					for de in os.listdir(tempdir) :
						ttempdir = tempdir + '\\' + de
						dst = cluster_all + '\\normalized_g_{}\\'.format(group_num) + de
			
						for excel in os.listdir(ttempdir) :
							os.chdir(ttempdir)
							temp = lib.read_excel(excel)
							
							sum = temp.sum().sum()
							sum = sum / temp.shape[0]

							for i in range(temp.shape[0]) :
								for col in temp.columns :
									temp.loc[i, col] = temp.loc[i, col] / sum

							os.chdir(dst)
							temp.to_excel('{}_정규화.xlsx'.format(excel[ : -5]))
							print('{}_정규화.xlsx saved'.format(excel[ : -5]), end = '\r')
		print('done')
		
	def model4(self) :
		
		# model3으로 model4 만들기
		# model4 완료

		hours = self.hours
		hours_str = self.hours_str
		hours_ex = self.hours_ex
		hours_ex_str = self.hours_ex_str
		
		for condition in self.condition_list :
			srcdir = self.src_dir + '\\' + condition
			cwdir = self.model_dir + '\\' + condition 
			finaldir = cwdir + '\\preprocessed'
			model3dir = cwdir + '\\model3'
			model3cluster = cwdir + '\\model3_cluster'
			model4dir = cwdir + '\\model4'
			
			folderlist = ['주중', '주말']
			folderdays = '주중'
			folderends = '주말'
			
			
			print('cwd is {}'.format(condition))
			for folder in os.listdir(model3dir) :
				# folder = ['주중', '주말']
				tempdir = model3dir + '\\' + folder
				if os.path.isdir(tempdir) :
					if '주말' in folder :
						df = pd.DataFrame(columns = ['excel'] + hours)
						df_num = 0
						df.columns = df.columns.astype(str)
						os.chdir(tempdir)
						
						for excel in os.listdir(tempdir) :
							temp = read_excel(excel)
							temp.columns = temp.columns.astype(str)
							
							profile = pd.DataFrame(columns = hours)
							profile.columns = profile.columns.astype(str)
							for cat in temp.columns :
								profile.loc[0, cat] = temp.loc[:, cat].mean()
								
							for i in range(temp.shape[0]):
								df.loc[df_num, 'excel'] = excel + '_{}'.format(i)
									
								for cat in temp.columns :
									df.loc[df_num, str(cat)] = temp.loc[i, str(cat)] - profile.loc[0, str(cat)]
								df_num += 1
							print('weekends,{} done'.format(excel), end = '\r')
							
						os.chdir(model4dir)
						df.to_excel('model4_weekends.xlsx')
						
					elif '주중' in folder :
						df = pd.DataFrame(columns = ['excel'] + hours)
						df_num = 0
						df.columns = df.columns.astype(str)
						os.chdir(tempdir)
						
						for excel in os.listdir(tempdir) :
							temp = read_excel(excel)
							temp.columns = temp.columns.astype(str)
							
							profile = pd.DataFrame(columns = hours)
							profile.columns = profile.columns.astype(str)
							for cat in temp.columns :
								profile.loc[0, cat] = temp.loc[:, cat].mean()
								
							for i in range(temp.shape[0]):
								df.loc[df_num, 'excel'] = excel + '_{}'.format(i)
									
								for cat in temp.columns :
									df.loc[df_num, str(cat)] = temp.loc[i, str(cat)] - profile.loc[0, str(cat)]
								df_num += 1
							print('weekdays, {} done'.format(excel), end = '\r')
							
						os.chdir(model4dir)
						df.to_excel('model4_weekdays.xlsx')
						
	def unite(self) :
		main_dir = self.main_dir
		model_dir = self.model_dir
		
		f_dir = main_dir + '\\시설군별\\봄가을'
		newfolder(f_dir)
		
		f_list = ['업무시설', '판매시설', '숙박시설', '문화시설', '교육시설']
		
		unite_df = pd.DataFrame(columns = f_list)
		temp1 = ['출판 영상 방송통신 및 정보서비스업', '금융 및 보험업', '부동산업 및 임대업', \
				'전문 과학 및 기술 서비스업', '사업시설관리 및 사업지원 서비스업']
		temp2 = ['도매 및 소매업', '숙박 및 음식점업\\model3_cluster\\group_0']
		temp3 = ['숙박 및 음식점업\\model3_cluster\\group_1']
		temp4 = ['예술 스포츠 및 여가관련 서비스업']
		temp5 = ['교육 서비스업']
		
		c_list = ['업무시설', '판매시설', '숙박시설', '문화시설', '교육시설']
		
		max_len = 0
		for num in range(len(c_list)) :
			if max_len < len(locals()['temp{}'.format(num + 1)]) :
				max_len = len(locals()['temp{}'.format(num + 1)])
				
		
			
		for i, name in enumerate(c_list) :
			for j in range(max_len) :
				length = len(locals()['temp{}'.format(i + 1)])
				if j < length :
					unite_df.loc[j, name] = locals()['temp{}'.format(i + 1)][j]
				else :
					unite_df.loc[j, name] = 'empty'
			
		
		for i in range(unite_df.shape[0]) :
			for cat in unite_df.columns :
				if 'model3_cluster' not in str(unite_df.loc[i, cat]) :
					if str(unite_df.loc[i, cat]) != 'empty' :
						unite_df.loc[i, cat] = str(unite_df.loc[i, cat]) + '\\preprocessed'
		
		#unite_table = tb.Table(unite_df, '시설')
		#unite_string = unite_table.make()
		#print(unite_string)
		
		for name in temp1 :
			if name not in os.listdir(self.model_dir) :
				print('{} 업태의 이름을 확인해주세요.'.format(name))
				check = 0
			else : 
				pass
		
		#check_i = input('continue ? (y/n)')
		#if check_i == 'Y' or check_i == 'y' :
		#	check = 1
		#else :
		#	check = 0	
		
		check = 1
		
		if check == 0 :
			pass
		else :
			newfolderlist(f_dir, f_list)
			for folder in os.listdir(f_dir) :
				tempdir = f_dir + '\\' + folder
				newfolderlist(tempdir, ['주중', '주말'])
				c_list = unite_df.loc[:, folder].tolist()
				nc_list = []
				for item in c_list :
					if item not in nc_list :
						if 'empty' not in item :
							nc_list.append(item)
				c_list = nc_list
				nc_list = None

				# 파일 이동
				print(c_list)
				for src in c_list :
					#print(model_dir + '\\' + src + '\\' + 'excel.xlsx')
					for excel in os.listdir(model_dir + '\\' + src + '\\주중') :
						shutil.copyfile(model_dir + '\\' + src + '\\주중\\' + excel, tempdir + '\\주중\\' + excel)
					for excel in os.listdir(model_dir + '\\' + src + '\\주말') :
						shutil.copyfile(model_dir + '\\' + src + '\\주말\\' + excel, tempdir + '\\주말\\' + excel)
					print('{} copy done'.format(src))
			print('{} copy done'.format(folder))
		
		print('done')
				
			
	def unite_model3(self) :
		
		hours = self.hours
		hours_str = self.hours_str
		hours_ex = self.hours_ex
		hours_ex_str = self.hours_ex_str
		
		check = 0
	
		for condition in self.condition_list :
			srcdir = self.src_dir + '\\' + condition
			cwdir = self.model_dir + '\\' + condition 
			finaldir = cwdir + '\\preprocessed'
			model3dir = cwdir + '\\model3'
			model3cluster = cwdir + '\\model3_cluster'
			model4dir = cwdir + '\\model4'
				
		print('done')
		

main_dir  = 'D:\\project\\봄가을 전처리\\보낼파일\\봄가을 자동 완성'
make = model(main_dir)
#make.model4()
make.unite()
