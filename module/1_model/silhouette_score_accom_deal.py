import os
import sys
import get_dir
di = get_dir.get_info()

main_dir = di.main_dir
prep_dir = di.prep_dir
model_dir = di.model_dir
module_dir = di.module_dir
facility_dir = di.facility_dir
plot_dir = di.plot_dir
cluster_dir = di.cluster_dir
facility_df = di.facility_df
facility_dict = di.facility_dict
gp_dir = di.gp_dir
gp_plot = di.gp_plot
print(f'main_dir = {main_dir}\n')

sys.path.append(module_dir)
sys.path.append(os.path.join(module_dir, '4_directory_module'))

import directory_change as dich
import discordlib_pyplot as dlt
import model_library as lib

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "/mnt/c/Users/joo09/Documents/Github/fonts/D2Coding.ttf"
font = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font)

import os
from pathlib import Path
import glob
import os.path
import math
import random
from scipy.stats import beta, burr, kde
import scipy.stats
import shutil
import time

'''
note
imports for kmeans & silhouette method
'''
from sklearn.cluster import k_means, KMeans
from sklearn.metrics import silhouette_score, silhouette_samples


cwdir = os.getcwd()

def read_excel(excel) :
    df = pd.read_excel(excel)
    if 'Unnamed: 0' in df.columns :
        df.drop('Unnamed: 0', axis = 1, inplace = True)

    if 'Unnamed: 0.1' in df.columns :
        df.drop('Unnamed: 0.1', axis = 1, inplace = True)

    return df

def ave(list1) :
	return sum(list1) / len(list1)


# --------------------------------------------------------
# Silhouette score for each facilities
# --------------------------------------------------------

def silhouette_score_make(facility, profile_48, save_dir) :
    xvalues = []
    xvalues_str = []

    for i in range(1, 49) :
        xvalues.append(i)
        xvalues_str.append(str(i))

    # DataFrame X points the dataframe 'profile_48', 
    # changes in values of X will affect on the dataframe 'profile_48'
    X = profile_48.loc[:, '1' : '48']

    # find optimal K values with silhouette score
    #   plot clusters on ax2 subplot
    range_n_clusters = []
    for i in range(2, 14) :
        range_n_clusters.append(i)

    silhouette_scores = []
    silhouette_num = []
    for index_cluster, number_clusters in enumerate(range_n_clusters) :
        # make 'clusterer' object with KMeans
        clusterer = KMeans(n_clusters = number_clusters, random_state = 10)
        cluster_labels = clusterer.fit_predict(X)
        centers = clusterer.cluster_centers_

        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_num.append(number_clusters)
        silhouette_scores.append(silhouette_avg)

        sample_silhouette_values = silhouette_samples(X, cluster_labels)

    max_silhouette_score = max(silhouette_scores)
    for index in range(len(silhouette_scores)) :
        if silhouette_scores[index] == max_silhouette_score :
            optimal_k = silhouette_num[index]

        
    # make silhouette (datas)
    optimal_cluster = KMeans(n_clusters = optimal_k, random_state = 10)
    optimal_labels = optimal_cluster.fit_predict(X)
    optimal_centers = optimal_cluster.cluster_centers_
    optimal_labels = [int(item) for item in optimal_labels]
    profile_48['group'] = optimal_labels
#    profile_labels = pd.concat([profile_48, pd.DataFrame(optimal_labels, columns = ['group'])], axis = 1, ignore_index = True)
    print(f'{facility}, optimal K = {optimal_k}')
    print(optimal_labels)
    print(profile_48.head())
    #print(profile_labels.head())
            

    # add subplots (save plots for each facilities)
    fig = plt.figure(figsize = (18, 7))
#    ax1 = fig.add_subplot(1, 2, 1)
#    ax2 = fig.add_subplot(1, 2, 2)
    spec = matplotlib.gridspec.GridSpec(ncols = 2, nrows = 1, width_ratios = [1, 2])
    ax1 = fig.add_subplot(spec[0])
    ax2 = fig.add_subplot(spec[1])
    

    # DataFrame X points the dataframe 'profile_48', 
    # changes in values of X will affect on the dataframe 'profile_48'
    X = profile_48.loc[:, '1' : '48']
    X.reset_index(drop = True, inplace = True)

    number_clusters = optimal_k
    clusterer = KMeans(n_clusters = number_clusters, random_state = 10)
    cluster_labels = clusterer.fit_predict(X)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    cluster_centers = clusterer.cluster_centers_

    
    # make dataframe for each clusters (for plotting purpose, ax2)
    for i in range(number_clusters) :
        globals()[f'cluster_{i}'] = []
#    cluster_0 = []
#    cluster_1 = []

    for index in range(cluster_labels.shape[0]) :
        globals()[f'cluster_{cluster_labels[index]}'].append(index)
#        if cluster_labels[index] == 0 :
#            cluster_0.append(index)
#        else :
#            cluster_1.append(index)

    print(X.shape[0])
    print(len(cluster_0))
    print(len(cluster_1))

    for i in range(number_clusters) :
        globals()[f'dataframe_{i}th'] = X.loc[globals()[f'cluster_{i}'], :]
        globals()[f'dataframe_{i}th'].reset_index(drop = True, inplace = True)

        globals()[f'upper_{i}th'] = pd.DataFrame(columns = X.columns)
        globals()[f'lower_{i}th'] = pd.DataFrame(columns = X.columns)
        globals()[f'ave_{i}th'] = pd.DataFrame(columns = X.columns)

#    dataframe_0th = X.loc[cluster_0, :]
#    dataframe_1th = X.loc[cluster_1, :]
#    dataframe_0th.reset_index(drop = True, inplace = True)
#    dataframe_1th.reset_index(drop = True, inplace = True)
#

#    upper_0th = pd.DataFrame(columns = X.columns)
#    lower_0th = pd.DataFrame(columns = X.columns)
#    upper_1th = pd.DataFrame(columns = X.columns)
#    lower_1th = pd.DataFrame(columns = X.columns)
#    ave_0th = pd.DataFrame(columns = X.columns)
#    ave_1th = pd.DataFrame(columns = X.columns)


    for i in range(1, 49) :
        for c in range(number_clusters) :
            globals()[f'temp_list_{c}th'] = sorted(globals()[f'dataframe_{c}th'].loc[:, str(i)].tolist())

            globals()[f'upper_{c}th'].loc[0, str(i)] = np.percentile(globals()[f'temp_list_{c}th'], 90)
            globals()[f'lower_{c}th'].loc[0, str(i)] = np.percentile(globals()[f'temp_list_{c}th'], 10)
            globals()[f'ave_{c}th'].loc[0, str(i)] = ave(globals()[f'temp_list_{c}th'])


#        # upper, lower boundaries for 0th cluster
#        temp_list_0th = sorted(dataframe_0th.loc[:, str(i)].tolist())
#
#        upper_0th.loc[0, str(i)] = np.percentile(temp_list_0th, 90)
#        lower_0th.loc[0, str(i)] = np.percentile(temp_list_0th, 10)
#        ave_0th.loc[0, str(i)] = ave(temp_list_0th)
#
#        # upper, lower boundaries for 1th cluster
#        temp_list_1th = sorted(dataframe_1th.loc[:, str(i)].tolist())
#
#        upper_1th.loc[0, str(i)] = np.percentile(temp_list_1th, 90)
#        lower_1th.loc[0, str(i)] = np.percentile(temp_list_1th, 10)
#        ave_1th.loc[0, str(i)] = ave(temp_list_1th)



    # for range of 2 (each groups), plot silhouette values

    # initial point for y-axis range
    y_lower = 10
    
    df_silhouette = pd.DataFrame(columns = ['cluster', 'silhouette_score'])
    start_num = 0

    for i in range(number_clusters) :
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        print(ith_cluster_silhouette_values.shape)


        # concat silhouette scores for all buildings to df_silhouette
        temp = ith_cluster_silhouette_values.tolist()
        temp_df =  pd.DataFrame(columns = ['cluster', 'silhouette_score'])

        temp_df.loc[: , 'silhouette_score'] = temp
        temp_df.loc[0 : len(temp) -1, 'cluster'] = i

        df_silhouette = pd.concat([df_silhouette, temp_df], ignore_index = True)


        # make the range of y-axis for each cluster (y_lower + size of cluster)
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        # set colors for clusters silhouette score bars
        if i == 0 :
            color = 'red'
        else :
            color = 'blue'
        
        ax1.fill_betweenx(
                np.arange(y_lower, y_upper), # range in y-axis
                0,
                ith_cluster_silhouette_values, # silhouette scores for each meters
                facecolor = color, 
                edgecolor = color,
                alpha = 1, #0.7,
        )

        # label the silhouette plots with cluster numbers at the middle
        ax1.text(-0.15, y_lower + 0.5 * size_cluster_i, f'group {i}')

        # reset y_lower for the next cluster
        # gap between the clusters is 10
        y_lower = y_upper + 10


    if (facility == '업무시설') | (facility == '판매및숙박') :
        alpha = 0.1
    else :
        alpha = 0.3

    for index in range(X.shape[0]) :
        ax2.plot(xvalues, X.loc[index, :], 'black', alpha = alpha)
    
    for i in range(number_clusters) :
        if i == 0 :
            egg = 'r'
        elif i == 1 :
            egg = 'b'
        else :
            egg = 'c'

        ax2.plot(xvalues, globals()[f'upper_{i}th'].loc[0, :], f'{egg}--', linewidth = 3, label = f'{i}th_upper')
        ax2.plot(xvalues, globals()[f'lower_{i}th'].loc[0, :], f'{egg}--', linewidth = 3, label = f'{i}th_lower')
        ax2.plot(xvalues, globals()[f'ave_{i}th'].loc[0, :], f'{egg}--', linewidth = 3, label = f'{i}th_center')
#    ax2.plot(xvalues, upper_0th.loc[0, :], 'r--', linewidth = 3, label = '0th_upper')
#    ax2.plot(xvalues, lower_0th.loc[0, :], 'r--', linewidth = 3, label = '0th_lower')
#    ax2.plot(xvalues, ave_0th.loc[0, :], 'r', linewidth = 5, label = '0th_center')
#
#    ax2.plot(xvalues, upper_1th.loc[0, :], 'b--', linewidth = 3, label = '1th_upper')
#    ax2.plot(xvalues, lower_1th.loc[0, :], 'b--', linewidth = 3, label = '1th_lower')
#    ax2.plot(xvalues, ave_1th.loc[0, :], 'b', linewidth = 5, label = '1th_center')
#
            
    ax2.set_xticks(xvalues, xvalues_str, rotation = 90)
    ax2.set_xlim(1, 48)
    #ax2.legend()

    ax1.set_title('silhouette scores')
    ax1.set_xlabel('silhouette scores')
    ax1.set_ylabel('sample data')

    ax2.set_title('model3')
    ax2.set_xlabel('hours')

    fig.suptitle(f'{facility}, 2 groups\nSilhouette Method', fontsize = 15)

    os.chdir(save_dir)
    plt.savefig(f'{facility}_silhouette_plot.png', dpi = 400)
    dlt.savefig(save_dir, f'{facility}_silhouette_plot.png', dpi = 400)

    df_silhouette.to_excel(f'{facility}_silhouette_scores.xlsx')
    dlt.shoot_file(save_dir, f'{facility}_silhouette_scores.xlsx')

    plt.cla()
    plt.clf()

    return profile_48 #profile_labels



    


        
##########################################
#
#        range_n_clusters = [2, 3, 4, 5, 6, 7]
#        hours = []
#        for i in range(1, 49, 1) :
#            hours.append(str(i))
#
#        for folder in os.listdir(model3dir) :
#            tempdir = model3dir + '\\' + folder
#            os.chdir(tempdir)
#            temp = read_excel('profile_48.xlsx')
#            temp.columns = temp.columns.astype(str)
#            temp.reset_index(drop = True, inplace = True)
#            X = temp.loc[:, '1' : '48'].copy()
#            fig = plt.figure(figsize = (34, 5 * 3))
#            gs = gridspec.GridSpec(nrows = 3, ncols = 4, height_ratios = [5, 5, 5], width_ratios = [5, 12, 5, 12])
#            
#            silhouette_num = []
#            silhouette_scores = []
#            
#            for i_cluster, n_clusters in enumerate(range_n_clusters):
#                locals()['ax{}'.format(2 * n_clusters)] = plt.subplot(gs[2 * i_cluster])
#                locals()['ax{}'.format(2 * n_clusters + 1)] = plt.subplot(gs[2 * i_cluster + 1])
#                #ax1 = plt.subplot(gs[0])
#                #ax2 = plt.subplot(gs[1])
#
#                locals()['ax{}'.format(2 * n_clusters)].set_xlim([-0.1, 1])
#                # The (n_clusters+1)*10 is for inserting blank space between silhouette
#                # plots of individual clusters, to demarcate them clearly.
#                locals()['ax{}'.format(2 * n_clusters)].set_ylim([0, X.shape[0] + (n_clusters + 1) * 10])
#
#                # Initialize the clusterer with n_clusters value and a random generator
#                # seed of 10 for reproducibility.
#                clusterer = KMeans(n_clusters=n_clusters, random_state=10)
#                cluster_labels = clusterer.fit_predict(X)
#
#                # The silhouette_score gives the average value for all the samples.
#                # This gives a perspective into the density and separation of the formed
#                # clusters
#                silhouette_avg = silhouette_score(X, cluster_labels)
#                silhouette_num.append(n_clusters)
#                silhouette_scores.append(silhouette_avg)
#                print(
#                    "For n_clusters =",
#                    n_clusters,
#                    "The average silhouette_score is :",
#                    silhouette_avg,
#                )
#
#                # Compute the silhouette scores for each sample
#                sample_silhouette_values = silhouette_samples(X, cluster_labels)
#                print('ax{}'.format(2 * n_clusters), '\tax{}'.format(2 * n_clusters + 1))
#                y_lower = 10
#                for i in range(n_clusters):
#                    # Aggregate the silhouette scores for samples belonging to
#                    # cluster i, and sort them
#                    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
#
#                    ith_cluster_silhouette_values.sort()
#
#                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
#                    y_upper = y_lower + size_cluster_i
#
#                    color = cm.nipy_spectral(float(i) / n_clusters)
#                    locals()['ax{}'.format(2 * n_clusters)].fill_betweenx(
#                        np.arange(y_lower, y_upper),
#                        0,
#                        ith_cluster_silhouette_values,
#                        facecolor=color,
#                        edgecolor=color,
#                        alpha=0.7,
#                    )
#
#                    # Label the silhouette plots with their cluster numbers at the middle
#                    locals()['ax{}'.format(2 * n_clusters)].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
#
#                    # Compute the new y_lower for next plot
#                    y_lower = y_upper + 10  # 10 for the 0 samples
#
#                locals()['ax{}'.format(2 * n_clusters)].set_title("The silhouette plot for the various clusters.")
#                locals()['ax{}'.format(2 * n_clusters)].set_xlabel("The silhouette coefficient values")
#                locals()['ax{}'.format(2 * n_clusters)].set_ylabel("Cluster label")
#
#                # The vertical line for average silhouette score of all the values
#                locals()['ax{}'.format(2 * n_clusters)].axvline(x=silhouette_avg, color="red", linestyle="--")
#
#                locals()['ax{}'.format(2 * n_clusters)].set_yticks([])  # Clear the yaxis labels / ticks
#                locals()['ax{}'.format(2 * n_clusters)].set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
#
#                # 2nd Plot showing the actual clusters formed
#                colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
#        #         for i_meters in range(X.shape[0]) :
#        #             ax2.plot(hours, X.loc[i_meters, :], alpha = 0.7, c = 'lightgrey')
#
#                # Labeling the clusters
#                centers = clusterer.cluster_centers_
#                centers_df = pd.DataFrame(centers)
#                for i_meters in range(centers.shape[0]) :
#                    locals()['ax{}'.format(2 * n_clusters + 1)].plot(hours, centers_df.loc[i_meters, :], alpha = 1, label = '{}th cluster'.format(i_meters))
#                locals()['ax{}'.format(2 * n_clusters + 1)].plot([24, 24], [-2, 2], color = 'dimgrey') 
#                locals()['ax{}'.format(2 * n_clusters + 1)].plot([48, 48], [-2, 2], color = 'dimgrey')
#                locals()['ax{}'.format(2 * n_clusters + 1)].text(23, 0.12, 'weekdays', rotation = 90, color = 'dimgrey')
#                locals()['ax{}'.format(2 * n_clusters + 1)].text(46, 0.12, 'weekends', rotation = 90, color = 'dimgrey')
#                xvalues_str = []
#                for i in range(1, 49) :
#                    xvalues_str.append(str(i))
#                    
#                #locals()['ax{}'.format(2 * n_clusters + 1)].xticks(np.arange(1, 49, 1), xvalues_str)
#                #locals()['ax{}'.format(2 * n_clusters + 1)].grid(True)
#                # Draw white circles at cluster centers
#                locals()['ax{}'.format(2 * n_clusters + 1)].set_title("The visualization of the clustered data.")
#                locals()['ax{}'.format(2 * n_clusters + 1)].set_xlabel("hours")
#                locals()['ax{}'.format(2 * n_clusters + 1)].set_ylabel(" ")
#                locals()['ax{}'.format(2 * n_clusters + 1)].legend()
#                locals()['ax{}'.format(2 * n_clusters + 1)].set_xlim([0, 47])
#                locals()['ax{}'.format(2 * n_clusters + 1)].set_ylim([0, 0.175])
#                locals()['ax{}'.format(2 * n_clusters + 1)].plot()
#            
#            max_score = max(silhouette_scores)
#            for i in range(len(silhouette_scores)) :
#                if silhouette_scores[i] == max_score :
#                    max_num = silhouette_num[i]
#                    
#            plt.suptitle(
#                    "{}\nOptimal K by score = {}".format(folder, max_num),
#                    ha = 'center',
#                    va = 'top',
#                    fontsize=14,
#                    fontweight="bold",
#            )
#            #plt.subplots_adjust(hspace= 0.25)
#            os.chdir(model3_cluster + '\\' + folder)
#            plt.savefig('{}_silhouette_method.png'.format(folder), bbox_inches = 'tight', dpi=800)
#            #plt.show()
##########################################
#




