import model_library as lib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import math
import shutil
from sklearn.cluster import KMeans
from sklearn.metrics import *
from distutils.dir_util import copy_tree

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
								check += 1						
					
					
					if check > 5 :
						name = 'error_{}'.format(excel)
					else :
						name = excel
					os.chdir(model3dir + '\\' + de)
					temp_m2.to_excel(name)
					print('{} saved'.format(name))
						
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
				print('working on {}'.format(excel))
				os.chdir(model3days)
				temp = lib.read_excel(excel)
				for excel2 in os.listdir(model3ends) :
					os.chdir(model3ends)
					if excel[ : -11] == excel2[ : -11] :
						excel_match = excel2
						temp2 = lib.read_excel(excel_match)
						print(excel[ : -11], excel2[ : -11])
					
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
				
				print('{}. {} is matched'.format(excel, excel_match))
				print('df num = {}'.format(df_num))

				

			os.chdir(model3dir)
			df.to_excel('profile_48.xlsx')
			self.profile_48  = df
			print('48 hours profile made, shape : {}'.format(df.shape))
			print(self.profile_48)
						
	def model3_cluster(self) :
		
		# model3 48시간 프로필 만들기
		# model3 클러스터링
		# model3 클러스터 완성
		# model3 MANOVA성
		# model3 클러스터링 완
		
		df = self.profile_48
		max_iter = 30
		
		
		pass
		
	def model4(self) :
		
		# model3으로 model4 만들기
		# model4 완료
		
		pass
		

main_dir  = 'D:\\project\\봄가을 전처리\\보낼파일\\봄가을 자동 완성'
make = model(main_dir)
make.model3()
