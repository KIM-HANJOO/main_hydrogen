from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from validclust import dunn
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import k_means
from sklearn import metrics
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#########################################################################################
#########################################################################################
##                                                                                     ##
##                               #### HOW TO IMPORT ####                               ##
##                                                                                     ##
## home desktop | import ClusterAnalysis                                               ##
##                                                                                     ##
## import sys
## sys.path.append('C:\\Users\\joo09\\Documents\\GitHub\\LIBRARY')
## import ClusterAnalysis as ca
##                                                                                     ##
##                                                                                     ##
##                                                                                     ##
##                           #### HOW TO RE - IMPORT ####                              ##
##                                                                                     ##
## from imp import reload
## reload(ca)
##                                                                                     ##
#########################################################################################
#########################################################################################


def all_score(X, num_clusters) :
    
    X_num = []
    for i in range(len(X)) :
        X_num.append(str(i + 1))
    
    cluster_df = pd.DataFrame(columns = ['num_clusters', 'labels', 'dist'])
    cluster_num = 0
    
    for num in num_clusters :
        _, cluster_labels, _ = k_means(X, n_clusters = num)
        dist = pairwise_distances(X)
#         cluster = KMeans(n_clusters = num)
#         cluster_labels = cluster.fit_predict(X)
        
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

    #     plt.axvline(x = mp_silh, c = 'b', alpha = 0.5)
    #     plt.axvline(x = mp_dunn, c = 'orange', alpha = 0.5)
    #     plt.axvline(x = mp_chscore, c = 'r', alpha = 0.5)
    #     plt.axvline(x = mp_dbi, c = 'green', alpha = 0.5)

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


def chs_find_big(profile, plot) :

    num_clusters = range(2, profile.shape[0])

    X = np.array(profile.loc[:, '1' : '48'].copy())
    
    
    clsuter_df, all_score_df, df_optimal = all_score_set(X, num_clusters, plot)
    K = df_optimal.loc['optimal_K', 'calinski_harabasz_score']
    return K

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

def big_centers_plot(centers) :
    for i in range(centers.shape[0]) :
        if centers.loc[i, 'count'] > 5 :
            alpha_temp = 1
        else :
            alpha_temp = 0
        plt.plot(centers.loc[i, 'center'], label = 'i', alpha = alpha_temp)
    
    plt.show()

def wanted_centers_plot(info, wanted) :
    for i in range(info.shape[0]) :
        if info.loc[i, 'label'] in wanted :
            plt.plot(range(1, 49), info.loc[i, '1' : '48'])
    plt.show()
##########################################
def name_find(name) :
    num = name[ : -2].find('-')
    return name[ : num - 2]

def cluster_new(profile, K) :
    obj = profile.loc[:, '1' : '48'].copy()
    
    hours_ex = []
    for i in range(1, 49) :
        hours_ex.append(str(i))
    obj.columns = hours_ex
    obj.reset_index(drop = True, inplace = True)
    profile.reset_index(drop = True, inplace = True)
    
    kmeans = KMeans(n_clusters = K, random_state = 0).fit(obj)
    info_clu = pd.DataFrame(columns = ['excel', 'label', 'center'] + hours_ex)
    
    for i in range(profile.shape[0]) :   
        info_clu.loc[i, 'excel'] = profile.loc[i, 'excel']
        info_clu.loc[i, 'label'] = kmeans.labels_[i]
        info_clu.loc[i, 'center'] = kmeans.cluster_centers_[kmeans.labels_[i]]
        info_clu.loc[i, '1' : '48'] = obj.loc[i, '1' : '48']
    
    only_centers = pd.DataFrame(columns = ['cat', 'label', 'count', 'center'])
    for i in range(K) :
        only_centers.loc[i, 'cat'] = name_find(profile.loc[0, 'excel'])
        only_centers.loc[i, 'label'] = i
        only_centers.loc[i, 'center'] = kmeans.cluster_centers_[i]
        
    only_centers.reset_index(drop = True, inplace = True)

    for i in range(only_centers.shape[0]) :
        only_centers.loc[i, 'count'] = info_clu.loc[:, 'label'].tolist().count(i)
    info_clu = info_clu.sort_values(by = 'label')
    return info_clu, only_centers

# def cluster_number(info_clu, K, cat) :

#     numbers = pd.DataFrame(columns = ['cat'] + range(K))
#     numbers.columns = numbers.columns.astype(str)

#     for i in range(info_clu.shape[0]) :
#         for j in range(K) :
            
def wanted_label_to_profile(info, wanted_label) :
    hours_ex = []
    for i in range(1, 49) :
        hours_ex.append(i)
    
    df = pd.DataFrame(columns = ['excel'] + hours_ex)
    df_num = 0

    info.columns = info.columns.astype(str)
    df.columns = df.columns.astype(str)

    info.reset_index(drop = True, inplace = True)
    
    for i in range(info.shape[0]) :
        for j in range(len(wanted_label)) :
            if info.loc[i, 'label'] == wanted_label[j] :
                df.loc[df_num, 'excel'] = info.loc[i, 'excel']
                df.loc[df_num, '1' : '48'] = info.loc[i, '1' : '48'].tolist()
                df_num += 1
            
    return df

##########################################


def variation_scores(iteration, num_features, size, plot) :
    num_cluster = []
    for i in range(2, 20) :
        num_cluster.append(i)
    
    k_list_small = [2, 3, 4]
    k_list_middle = [5, 6, 7, 8]
    k_list_big = [9, 10, 11, 12]
    k_list_all = k_list_small + k_list_middle + k_list_big
    
    # default size
    k_list = k_list_middle
    
    # user size
    if type(size) == str :
        if size == 'small' :
            k_list = k_list_small
            
        elif size == 'middle' :
            k_list = k_list_middle
            
        elif size == 'big' :
            k_list = k_list_big
            
        elif size == 'all' :
            k_list = k_list_all
    else :
        k_list = size

    

    vari_columns = ['num_clusters', 'silhouette_score', 'dunn index',\
                       'calinski_harabasz_score', 'davies_bouldin_score']

    vari_df = pd.DataFrame(columns = vari_columns)
    vari_num = 0
    
    for k_num in k_list :
        for i in range(0, iteration) :
#             temp_k = random.choice(k_list)

            temp_k = k_num
            X, Y = make_blobs(centers = temp_k, n_features = num_features)

            cluster_df, all_score_df, df_optimal = all_score_set(X, num_cluster, plot)

            vari_df.loc[vari_num, 'num_clusters'] = temp_k
            vari_df.loc[vari_num, 'silhouette_score'] = df_optimal.loc['optimal_K', 'silhouette_score']
            vari_df.loc[vari_num, 'dunn index'] = df_optimal.loc['optimal_K', 'dunn index']
            vari_df.loc[vari_num, 'calinski_harabasz_score'] = df_optimal.loc['optimal_K', 'calinski_harabasz_score']
            vari_df.loc[vari_num, 'davies_bouldin_score'] = df_optimal.loc['optimal_K', 'davies_bouldin_score']

            vari_num += 1
            print('percentage = {}, now k = {}...'.format(round(i / iteration * 100, 3), k_num), end = '\r')
    print('done', end = '\r')
        
    return vari_df


def vari_df_plot(vari_df, num_clusters, show_silh, show_dunn, show_chscore, show_dbi) :

    for i in range(vari_df.shape[0]) :
        x_num = vari_df.loc[i, 'num_clusters']
        y_silh = vari_df.loc[i, 'silhouette_score']
        y_dunn = vari_df.loc[i, 'dunn index']
        y_chscore = vari_df.loc[i, 'calinski_harabasz_score']
        y_dbi = vari_df.loc[i, 'davies_bouldin_score']

        if show_silh == 1 :
            plt.plot(x_num, y_silh, label = 'Silhouette score', c = 'b', marker = 'o')
            plt.title('Silhouette score')
        if show_dunn == 1 :
            plt.plot(x_num, y_dunn, label = "Dunn's index", c = 'orange', marker = 'o')
            plt.title("Dunn's index")
        if show_chscore == 1 :
            plt.plot(x_num, y_chscore, label = 'Calinski Harabasz score', c = 'r', marker = 'o')
            plt.title('Calinski Harabasz score')
        if show_dbi == 1 :
            plt.plot(x_num, y_dbi, label = 'Davies Bouldin score', c = 'green', marker = 'o')
            plt.title('Davies Bouldin score')


    plt.plot([-1, max(num_clusters) + 3], [-1, max(num_clusters) + 3], color = 'dimgrey')
    plt.xlim([min(num_clusters), max(num_clusters)])
    plt.xlabel('number of clusters')
    plt.ylabel('optimal number of clusters, by score')
    plt.grid(alpha = 0.5)
    plt.show()
    
    

def var_applied_scores(cluster_df, all_score_df) :
   
    var_df = pd.DataFrame(columns = ['num_clusters', 'var', 'num_members'])
    var_num = 0
    
    for i in range(cluster_df.shape[0]) :
        var_df.loc[var_num, 'num_clusters'] = cluster_df.loc[i, 'num_clusters']
        
        temp_list = cluster_df.loc[i, 'labels'].tolist()
        temp_num = []
        
        for i in range(len(temp_list)) :
            if temp_list.count(i) != 0 :
                temp_num.append(temp_list.count(i))
                
        var_df.loc[var_num, 'var'] = np.var(temp_num)
        var_df.loc[var_num, 'num_members'] = temp_num
        var_num += 1

        
    return var_df



def all_score_plot_var(X, all_score_df, var_df) :
    
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
    
    
    

#     ax2 = fig.add_subplot(1, 1, 1)
    
    plt.rcParams["figure.figsize"] = (12, 5)
    
    hours_ex = []
    for i in range(1, 49) :
        hours_ex.append(i)
    
    for i in range(X.shape[0]) :
        plt.plot(hours_ex, X[i])
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
    
#     plt.axvline(x = mp_silh, c = 'b', alpha = 0.5)
#     plt.axvline(x = mp_dunn, c = 'orange', alpha = 0.5)
#     plt.axvline(x = mp_chscore, c = 'r', alpha = 0.5)
#     plt.axvline(x = mp_dbi, c = 'green', alpha = 0.5)
    
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


def rmse_cat(vari_df, cat) :
    org_list = vari_df.loc[:, 'num_clusters'].tolist()
    obj_list = vari_df.loc[:, cat].tolist()
    print(org_list)
    print(obj_list)
    diff_list = []
    for i in range(len(org_list)) :
        diff = org_list[i] - obj_list[i]
        diff_list.append(math.pow(diff, 2))
    
#     print(diff_list)
    return math.sqrt(sum(diff_list) / len(diff_list))


#####################


def elbow(X) :
    # plot elbow
    inertia_err = []
    k_range = range(2, 15)

    df_inertia = pd.DataFrame(columns = ['num_clusters', 'inertia'])
    num_iner = 0

    for k in k_range :
        Kmeans = KMeans(n_clusters = k)
        Kmeans.fit(X)
        inertia = Kmeans.inertia_
        df_inertia.loc[num_iner, 'num_clusters'] = k
        df_inertia.loc[num_iner, 'inertia'] = inertia
        num_iner += 1

    plt.plot(df_inertia.loc[:, 'num_clusters'], df_inertia.loc[:, 'inertia'])
    plt.xlabel('number of clusters')
    plt.ylabel('inertia')
    plt.title('elbow method')
    plt.show()