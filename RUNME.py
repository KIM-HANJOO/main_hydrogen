'''
note :

'''

###################################################################################
import os
import dir_info
di = dir_info.Info()
main_dir = di.main_dir
prep_dir = di.prep_dir
model_dir = di.model_dir
module_dir = di.module_dir
facility_dir = di.facility_dir
plot_dir = di.plot_dir
cluster_dir = di.cluster_dir

import os
import sys
cwdir = os.getcwd()
sys.path.append(cwdir + '\\module')

########################################################################

import model_library as lib
import ClusterAnalysis as ca
import table as tb
import branch as brn

########################################################################

from distutils.dir_util import copy_tree
import shutil
import math


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgunbd.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

########################################################################

from sklearn.metrics import *
from sklearn.cluster import *
from validclust import dunn

from statsmodels.multivariate.manova import MANOVA

########################################################################







########################################################################

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

########################################################################

					
def subplot_num(count) :
	subplot_df = pd.DataFrame(columns = ['1', '2', '3'])
	fig_size = pd.DataFrame(columns = ['1', '2'])
	
	if count == 1 :
		subplot_df.loc[0, :] = [1, 1, 1]
		fig_size.loc[0, :] = [12, 5]
		
	elif count == 2 :
		subplot_df.loc[0,  :] = [2, 1, 1]
		subplot_df.loc[1,  :] = [2, 1, 2]
		fig_size.loc[0, :] = [12, 10]
		
	elif count == 3 :
		subplot_df.loc[0, :] = [3, 1, 1]
		subplot_df.loc[1, :] = [3, 1, 2]
		subplot_df.loc[2, :] = [3, 1, 3]
		fig_size.loc[0, :] = [12, 15]
		
	elif count == 4 :
		subplot_df.loc[0, :] = [4, 1, 1]
		subplot_df.loc[1, :] = [4, 2, 1]
		subplot_df.loc[2, :] = [4, 1, 2]
		subplot_df.loc[3, :] = [4, 2, 2]
		fig_size.loc[0, :] = [24, 10]
	
	return subplot_df, fig_size
	
########################################################################

def all_score(X, num_clusters) :
	
	X_num = []
	for i in range(len(X)) :
		X_num.append(str(i + 1))
	
	cluster_df = pd.DataFrame(columns = ['num_clusters', 'labels', 'dist'])
	cluster_num = 0
	
	for num in num_clusters :
		_, cluster_labels, _ = k_means(X, n_clusters = num)
		dist = pairwise_distances(X)
#		 cluster = KMeans(n_clusters = num)
#		 cluster_labels = cluster.fit_predict(X)
		
		cluster_df.loc[cluster_num, 'num_clusters'] = num
		cluster_df.loc[cluster_num, 'labels'] = cluster_labels
		cluster_df.loc[cluster_num, 'dist'] = dist
		cluster_num += 1
	
	cluster_df.reset_index(drop = True, inplace = True)
	
	all_score_df = pd.DataFrame(columns = ['num_clusters', 'silhouette_score', 'dunn index', \
										   'calinski_harabasz_score','davies_bouldin_score'])
	all_score_num = 0
	dist = pairwise_distances(X)
	
	for i in range(len(num_clusters)) :
		all_score_df.loc[all_score_num, 'num_clusters'] = cluster_df.loc[i, 'num_clusters']
		temp_label = cluster_df.loc[i, 'labels']
		temp_dist = cluster_df.loc[i, 'dist']
	
		all_score_df.loc[all_score_num, 'silhouette_score'] = silhouette_score(X, temp_label)
		all_score_df.loc[all_score_num, 'dunn index'] = dunn(temp_dist, temp_label)
		all_score_df.loc[all_score_num, 'calinski_harabasz_score'] = \
												metrics.calinski_harabasz_score(X, temp_label)
		all_score_df.loc[all_score_num, 'davies_bouldin_score'] = \
												davies_bouldin_score(X, temp_label)
		
		all_score_num += 1
		
	all_score_df.reset_index(drop = True, inplace = True)
		
	
	return cluster_df, all_score_df


def all_score_plot(X, all_score_df, plot) :
	
	max_silh = max(all_score_df.loc[:, 'silhouette_score'])
	max_dunn = max(all_score_df.loc[:, 'dunn index'])
	max_chscore = max(all_score_df.loc[:, 'calinski_harabasz_score'])
	max_dbi = max(all_score_df.loc[:, 'davies_bouldin_score'])
	min_dbi = min(all_score_df.loc[:, 'davies_bouldin_score'])
	
	mp_silh = 0
	mp_dunn = 0
	mp_chscore = 0
	mp_dbi = 0
	minp_dbi = 0
	
	for i in range(all_score_df.shape[0]) :
		if all_score_df.loc[i, 'silhouette_score'] == max_silh :
			mp_silh = all_score_df.loc[i, 'num_clusters']
			
		if all_score_df.loc[i, 'dunn index'] == max_dunn :
			mp_dunn = all_score_df.loc[i, 'num_clusters']
			
		if all_score_df.loc[i, 'calinski_harabasz_score'] == max_chscore :
			mp_chscore = all_score_df.loc[i, 'num_clusters']
			
		if all_score_df.loc[i, 'davies_bouldin_score'] == max_dbi :
			mp_dbi = all_score_df.loc[i, 'num_clusters']
		
		if all_score_df.loc[i, 'davies_bouldin_score'] == min_dbi :
			minp_dbi = all_score_df.loc[i, 'num_clusters']
	
	
	for i in range(all_score_df.shape[0]) :
		all_score_df.loc[i, 'silhouette_score'] = all_score_df.loc[i, 'silhouette_score'] / max_silh
		all_score_df.loc[i, 'dunn index'] = all_score_df.loc[i, 'dunn index'] / max_dunn
		all_score_df.loc[i, 'calinski_harabasz_score'] = all_score_df.loc[i, 'calinski_harabasz_score'] / max_chscore
		all_score_df.loc[i, 'davies_bouldin_score'] = all_score_df.loc[i, 'davies_bouldin_score'] / max_dbi
	
	
	if plot == 1 :
		
		# plot Profiles
		plt.rcParams["figure.figsize"] = (12, 5)
		if X.shape[1] == 2 :
			for i in range(X.shape[0]) :
				plt.plot(X[i][0], X[i][1], c = 'r', marker = 'o')
			plt.title('samples')
			plt.show()
			
		else :
			hours_ex = []
			for i in range(1, X.shape[1] + 1) :
				hours_ex.append(i)
			
			for i in range(X.shape[0]) :
				plt.plot(hours_ex, X[i])
			
			plt.xlim(1, X.shape[1])
			plt.xlabel('hours')
			plt.ylabel('KWh')
			plt.title('random profiles')
			plt.show()

		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1)
		plt.rcParams["figure.figsize"] = (12, 5)
		xvalues = all_score_df.loc[:, 'num_clusters'].astype(int)

		y_silh = all_score_df.loc[:, 'silhouette_score']
		y_dunn = all_score_df.loc[:, 'dunn index']
		y_chscore = all_score_df.loc[:, 'calinski_harabasz_score']
		y_dbi = all_score_df.loc[:, 'davies_bouldin_score']

		plt.plot(xvalues, y_silh, label = 'Silhouette score', c = 'b')
		plt.plot(xvalues, y_dunn, label = "Dunn's index", c = 'orange')
		plt.plot(xvalues, y_chscore, label = 'Calinski Harabasz score', c = 'r')
		plt.plot(xvalues, y_dbi, label = 'Davies Bouldin score', c = 'green')

	#	 plt.axvline(x = mp_silh, c = 'b', alpha = 0.5)
	#	 plt.axvline(x = mp_dunn, c = 'orange', alpha = 0.5)
	#	 plt.axvline(x = mp_chscore, c = 'r', alpha = 0.5)
	#	 plt.axvline(x = mp_dbi, c = 'green', alpha = 0.5)

		plt.xlim(min(all_score_df.loc[:, 'num_clusters']), max(all_score_df.loc[:, 'num_clusters']))
		major_ticks = xvalues.astype(int)
		ax.set_xticks(major_ticks)

		plt.xlabel('number of clusters')
		plt.ylabel('score (normalized)')
		plt.grid(True, axis='x', color='black', alpha = 0.5, linestyle='--')
		plt.legend()
		plt.show()
	
	
	df_optimal = pd.DataFrame(columns = ['silhouette_score', 'dunn index', 'calinski_harabasz_score', 'davies_bouldin_score'])
	df_optimal.loc[0, :] = [mp_silh, mp_dunn, mp_chscore, minp_dbi]
	df_optimal.loc[1, :] = [max_silh, max_dunn, max_chscore, min_dbi]
	df_optimal.index = ['optimal_K', 'score']
	
	
	return df_optimal

def all_score_set(X, num_clusters, plot) :
	cluster_df, all_score_df = all_score(X, num_clusters)
	df_maxval = all_score_plot(X, all_score_df, plot)
	
	return cluster_df, all_score_df, df_maxval
	

def chs_find(profile, plot) :
	if profile.shape[0] < 13 :
		num_clusters = range(2, profile.shape[0] - 1)
	else :
		num_clusters = range(2, 14)

	X = np.array(profile.loc[:, '1' : '48'].copy())
	
	
	clsuter_df, all_score_df, df_optimal = all_score_set(X, num_clusters, plot)
	K = df_optimal.loc['optimal_K', 'calinski_harabasz_score']
	return K
   

def chs_find(profile, plot) :
	if profile.shape[0] < 13 :
		num_clusters = range(2, profile.shape[0] - 1)
	else :
		num_clusters = range(2, 14)

	X = np.array(profile.loc[:, '1' : '48'].copy())
	
	
	clsuter_df, all_score_df, df_optimal = all_score_set(X, num_clusters, plot)
	K = df_optimal.loc['optimal_K', 'calinski_harabasz_score']
	return K


def chs_find_big(profile, plot) :

	num_clusters = range(2, profile.shape[0])

	X = np.array(profile.loc[:, '1' : '48'].copy())
	
	
	clsuter_df, all_score_df, df_optimal = all_score_set(X, num_clusters, plot)
	K = df_optimal.loc['optimal_K', 'calinski_harabasz_score']
	return K
	
def wanted_centers_plot(info, wanted) :
	for i in range(info.shape[0]) :
		if info.loc[i, 'label'] in wanted :
			plt.plot(range(1, 49), info.loc[i, '1' : '48'])
	plt.show() 

def big_centers_plot(centers) :
	for i in range(centers.shape[0]) :
		if centers.loc[i, 'count'] > 5 :
			alpha_temp = 1
		else :
			alpha_temp = 0
		plt.plot(centers.loc[i, 'center'], label = 'i', alpha = alpha_temp)
	
	plt.show()

def centers_plot(centers) :

	if min(centers.loc[:, 'count'].tolist()) > 5 :
		alpha_rth = 0
	else :
		alpha_rth = 1
	
	if alpha_rth == 0 :
		for i in range(centers.shape[0]) :
			plt.plot(centers.loc[i, 'center'], label = 'i', alpha = 1)

	else :
		for i in range(centers.shape[0]) :
			if centers.loc[i, 'count'] > 5 :
				alpha_temp = 0.3
			else :
				alpha_temp = 1
			plt.plot(centers.loc[i, 'center'], label = 'i', alpha = alpha_temp)
	
	plt.show()
		
########################################################################

