import pandas as pd
import numpy as np
import os
import shutil
import glob
import time
import math
import matplotlib.pyplot as plt

#########################################################################################
#########################################################################################
##                                                                                     ##
##                               #### HOW TO IMPORT ####                               ##
##                                                                                     ##
## home desktop | import model_lib                                                     ##
##                                                                                     ##
## import sys
## sys.path.append('C:\\Users\\joo09\\Documents\\GitHub\\LIBRARY')
## import model_lib as lib
##                                                                                     ##
##                                                                                     ##
##                                                                                     ##
##                           #### HOW TO RE - IMPORT ####                              ##
##                                                                                     ##
## from imp import reload
## reload(lib)
##                                                                                     ##
#########################################################################################
#########################################################################################

                                #    < LIST >
                                
                                # .                                       
                                # ├─ 1.preprocessing                      
                                # │  ├─nan values                         
                                # │  ├─outliers      
                                # │  │  ├─check1per                        
                                # │  │  ├─zscore (x)                         
                                # │  │  ├─1 percent                       
                                # │  │  └─tukeys fences                   
                                # │  └─interpotation                      
                                # │                                       
                                # ├─ 2.model                              
                                # │  ├─model3                             
                                # │  ├─model2                             
                                # │  └─model4                             
                                # │                                       
                                # ├─ 3.plotting                           
                                # │  ├─boxplot                            
                                # │  │  ├─boxplot_df                      
                                # │  │  └─boxplot_list                    
                                # │  └─barplot                            
                                # │     ├─barplot                         
                                # │     └─barplot_st                      
                                # │                                       
                                # └─ 3.etc                                
                                #    ├─read_excel                         
                                #    ├─pernull                            
                                #    └─os                                 
                                #       ├─copyfolderlist
                                #       ├─newfolderlist                   
                                #       ├─deletestr                       
                                #       └─delun   


################################################################################
############################# < PREPROCESSING > ################################

# nan values

## 1- 1) outliers
#check1per
def check1per(load_directory, save_directory, whis) :
    for excel in os.listdir(load_directory) :
        os.chdir(load_directory)
        df = read_excel(excel)

        df_flatten = df.to_numpy().flatten()
        df_flatten = df_flatten[~np.isnan(df_flatten)]
        Q1 = np.nanpercentile(df_flatten, 25, interpolation = 'midpoint') #df_flatten.quantile(interpolation='nearest', q = 0.25)
        Q3 = np.nanpercentile(df_flatten, 75, interpolation = 'midpoint') #df_flatten.quantile(interpolation='nearest', q = 0.75)
        IQR = Q3 - Q1

        delper_temp = []
        total_num = df.shape[0] * df.shape[1]
        for cat in df.columns :
            for i in range(df[cat].shape[0]) :
                if (df[cat][i] < Q1 - whis * IQR) | (df[cat][i] > Q3 + whis * IQR) :
                    delper_temp.append(df[cat][i])
        
        outlier_num = len(delper_temp)
        outlier_per = outlier_num / total_num

        if outlier_per < 0.01 :
            df_outlier = alltukey(df, whis)
            os.chdir(save_directory)
            df_outlier.to_excel('{}'.format(excel))

            print('{} done with tukeys fences'.format(excel), end = '\r')

        else :
            df_outlier = top_one_per(df)
            os.chdir(save_directory)
            df_outlier.to_excel('{}'.format(excel))
        
            print('{} done with top_one_per'.format(excel), end = '\r')
    print('all done')




def interp_dropnum(df) :
    df = delun(df)
    df.reset_index(drop = True, inplace = True)
   
    dropindex = []
    dropnum = 0

    for index in range(df.shape[0]) :
        templist = np.array(df.loc[index, :])
        nanpoint = []
        interp_true = 1

        for i in range(len(templist)) :
            if np.isnan(templist[i]) :
                nanpoint.append(i)

        if (0 in nanpoint) | (len(templist) - 1 in nanpoint) :
            interp_true = 0

        for i in range(len(nanpoint)) :
            if (i != 0) & (i != len(nanpoint) - 1) :
                if nanpoint[i] - nanpoint[i - 1] == 1 :
                    if nanpoint[i + 1] - nanpoint[i] == 1 :
                        interp_true = 0
                        
        if interp_true == 0 :
            dropnum += 1

    return dropindex

def howmany_oneper(directory, whis) :

    temp_df = pd.DataFrame(columns = ['standard_days', 'save_%', 'drop_%'])
    temp_num = 0

    print('starts')
    for i in range(0, 101) :
        savenum = 0
        dropnum = 0

        for j, excel in enumerate(os.listdir(directory)) :
            os.chdir(directory)
            
            dir_num = len(os.listdir(directory))
            df = read_excel(excel)
            dropindex = interp_dropnum(df)

            df_flatten = df.to_numpy().flatten()
            df_flatten = df_flatten[~np.isnan(df_flatten)]
            Q1 = np.nanpercentile(df_flatten, 25, interpolation = 'midpoint') #df_flatten.quantile(interpolation='nearest', q = 0.25)
            Q3 = np.nanpercentile(df_flatten, 75, interpolation = 'midpoint') #df_flatten.quantile(interpolation='nearest', q = 0.75)
            IQR = Q3 - Q1

            delper_temp = []
            total_num = df.shape[0] * df.shape[1]
            for cat in df.columns :
                for i in range(df[cat].shape[0]) :
                    if (df[cat][i] < Q1 - whis * IQR) | (df[cat][i] > Q3 + whis * IQR) :
                        df.loc[i, cat] = np.nan

            
            dropindex = 0
            dropnum = 0

            for index in range(df.shape[0]) :
                templist = np.array(df.loc[index, :])
                nanpoint = []
                interp_true = 1

                for i in range(len(templist)) :
                    if np.isnan(templist[i]) :
                        nanpoint.append(i)

                if (0 in nanpoint) | (len(templist) - 1 in nanpoint) :
                    interp_true = 0

                for i in range(len(nanpoint)) :
                    if (i != 0) & (i != len(nanpoint) - 1) :
                        if nanpoint[i] - nanpoint[i - 1] == 1 :
                            if nanpoint[i + 1] - nanpoint[i] == 1 :
                                interp_true = 0
                                
                if interp_true == 0 :
                    dropindex += 1

            if dropindex <= i:
                savenum += 1
            else :
                dropnum += 1
            
            print('{}% proceeding, + {}%, excel = {}'.format(round(i / 101 * 100, 3), \
                round(j / dir_num * 100, 3), excel), end = '\r')

        temp_df.loc[temp_num, 'standard_days'] = i
        temp_df.loc[temp_num, 'save_%'] = savenum / (savenum + dropnum) * 100
        temp_df.loc[temp_num, 'drop_%'] = dropnum / (savenum + dropnum) * 100
        temp_num += 1

    temp_df.reset_index(drop = True, inplace = True)
    
    return temp_df

def outlier_df_check(outlier_df, whis, standard) :

    oneper_list = []
    tukey_list = []

    for i in range(outlier_df.shape[0]) :
        if outlier_df.loc[i, '{}'.format(whis)] > standard :
            oneper_list.append(outlier_df.loc[i, 'excel'])

        else :
            tukey_list.append(outlier_df.loc[i, 'excel'])

    return oneper_list, tukey_list
# zscore

# 1 percent
def top_one_per(df_org) :
    df = df_org.copy()
    df = delun(df)
    df.reset_index(drop = True, inplace = True)
        
    values = []

    for i in range(df.shape[0]) :
        for cat in df.columns :
            values.append(df.loc[i, cat])

    oneper = round(len(values) / 100)
    values_oneper = sorted(values)[ -oneper :]

    for i in range(df.shape[0]) :
        for cat in df.columns :
            if df.loc[i, cat] >= min(values_oneper) :
                df.loc[i, cat] = np.nan

    return df


# tukeys fences
def alltukey(df_org, coe) :
    df = df_org.copy()
    if 'Unnamed: 0' in df.columns :
        df.drop('Unnamed: 0', axis = 1, inplace = True)


    df_flatten = df.to_numpy().flatten()
    df_flatten = df_flatten[~np.isnan(df_flatten)]
    Q1 = np.nanpercentile(df_flatten, 25, interpolation = 'midpoint') #df_flatten.quantile(interpolation='nearest', q = 0.25)
    Q3 = np.nanpercentile(df_flatten, 75, interpolation = 'midpoint') #df_flatten.quantile(interpolation='nearest', q = 0.75)
    IQR = Q3 - Q1

    for cat in df.columns :
        for i in range(df[cat].shape[0]) :
            if (df.loc[i, cat] < Q1 - coe * IQR) | (df.loc[i, cat] > Q3 + coe * IQR) :
                df.loc[i, cat] = np.nan
    
    return df

def alltukey_cat(df_org, coe, cat) :
    df = df_org.copy()
    if 'Unnamed: 0' in df.columns :
        df.drop('Unnamed: 0', axis = 1, inplace = True)
    df.reset_index(drop = True, inplace = True)

    df_flatten = df.loc[:, cat].to_numpy().flatten()
    
    
    Q1 = np.nanpercentile(df_flatten, 25, interpolation = 'midpoint')
    Q3 = np.nanpercentile(df_flatten, 75, interpolation = 'midpoint') 
    IQR = Q3 - Q1
    

    dropindex_func = []
    for i in range(df[cat].shape[0]) :
        if (df.loc[i, cat] < Q1 - coe * IQR) | (df.loc[i, cat] > Q3 + coe * IQR) :
            df.loc[i, cat] = np.nan
            
            dropindex_func.append(i)

    df = df.drop(dropindex_func)
    df.reset_index(drop = True, inplace = True)

    return df


def showdelper_tukey(df_org, coe1, coe2, coe3) :
    df = df_org.copy()
    if 'Unnamed: 0' in df.columns :
        df.drop('Unnamed: 0', axis = 1, inplace = True)

    df_flatten = df.to_numpy().flatten()
    df_flatten = df_flatten[~np.isnan(df_flatten)]
    Q1 = np.nanpercentile(df_flatten, 25, interpolation = 'midpoint') #df_flatten.quantile(interpolation='nearest', q = 0.25)
    Q3 = np.nanpercentile(df_flatten, 75, interpolation = 'midpoint') #df_flatten.quantile(interpolation='nearest', q = 0.75)
    IQR = Q3 - Q1
    # for cat in df.columns :
    #     Q1 = df[cat].quantile(interpolation='nearest', q = 0.25)
    #     Q3 = df[cat].quantile(interpolation='nearest', q = 0.75)
    #     IQR = Q3 - Q1

    delper_Q1_temp = []
    delper_Q2_temp = []
    delper_Q3_temp = []
    total_num = df.shape[0] * df.shape[1]
    for cat in df.columns :
        for i in range(df[cat].shape[0]) :
            if (df[cat][i] < Q1 - coe1 * IQR) | (df[cat][i] > Q3 + coe1 * IQR) :
                delper_Q1_temp.append(df[cat][i])
                if (df[cat][i] < Q1 - coe2 * IQR) | (df[cat][i] > Q3 + coe2 * IQR) :
                    delper_Q2_temp.append(df[cat][i])
                    if (df[cat][i] < Q1 - coe3 * IQR) | (df[cat][i] > Q3 + coe3 * IQR) :
                        delper_Q3_temp.append(df[cat][i])

    coe_df = pd.DataFrame(columns = ['org_num', '{}'.format(coe1), '{}'.format(coe2), '{}'.format(coe3), \
                                    '{}_nan'.format(coe1), '{}_nan'.format(coe2), '{}_nan'.format(coe3), ]) 
    num_coe_df = 0

    coe_df.loc[num_coe_df] = None

    coe_df.loc[num_coe_df, 'org_num'] = total_num

    # if try to find value of coe_df.loc[0, 'delper_Q1'], add .iloc[0] at the end for pure value
    coe_df.loc[num_coe_df, '{}'.format(coe1)] = len(delper_Q1_temp) / total_num * 100
    coe_df.loc[num_coe_df, '{}'.format(coe2)] = len(delper_Q2_temp) / total_num * 100
    coe_df.loc[num_coe_df, '{}'.format(coe3)] = len(delper_Q3_temp) / total_num * 100
    coe_df.loc[num_coe_df, '{}_nan'.format(coe1)] = len(delper_Q1_temp) / total_num * 100 + pernull(df)
    coe_df.loc[num_coe_df, '{}_nan'.format(coe2)] = len(delper_Q2_temp) / total_num * 100 + pernull(df)
    coe_df.loc[num_coe_df, '{}_nan'.format(coe3)] = len(delper_Q3_temp) / total_num * 100 + pernull(df)



    return coe_df

# whisker값 선택
def outlier_df_maker(loaddir, savedir) :
    outlier_df = pd.DataFrame(columns = ['excel', 'original', '1.5', '3.0', '4.5', '1.5_nan', '3.0_nan', '4.5_nan'])
    num_out = 0

    os.chdir(loaddir)
    for folder in os.listdir(loaddir) :
        tempdir = loaddir + '\\' + folder
        for excel in os.listdir(tempdir) :
            os.chdir(tempdir)
            
            temp = read_excel(excel)
            info_temp = showdelper_tukey(temp, 1.5, 3.0, 4.5)
            
            outlier_df.loc[num_out, 'excel'] = excel
            outlier_df.loc[num_out, 'original'] = pernull(temp)
            outlier_df.loc[num_out, '1.5'] = info_temp.loc[0, '1.5']
            outlier_df.loc[num_out, '3.0'] = info_temp.loc[0, '3.0']
            outlier_df.loc[num_out, '4.5'] = info_temp.loc[0, '4.5']
            outlier_df.loc[num_out, '1.5_nan'] = info_temp.loc[0, '1.5_nan']
            outlier_df.loc[num_out, '3.0_nan'] = info_temp.loc[0, '3.0_nan']
            outlier_df.loc[num_out, '4.5_nan'] = info_temp.loc[0, '4.5_nan']
            num_out += 1
            print('{} done'.format(excel), end = '\r')

    print('\n', '\n', '평균 결측값 퍼센트 = {}%'.format(round(ave(outlier_df.loc[:, 'original'].tolist()), 3)))
    print('whisker = 1.5일 때 평균 결측값 퍼센트 = {}%'.format(round(ave(outlier_df.loc[:, '1.5_nan'].tolist()), 3)))
    print('whisker = 3.0일 때 평균 결측값 퍼센트 = {}%'.format(round(ave(outlier_df.loc[:, '3.0_nan'].tolist()), 3)))
    print('whisker = 4.5일 때 평균 결측값 퍼센트 = {}%'.format(round(ave(outlier_df.loc[:, '4.5_nan'].tolist()), 3)))

    print('whisker = 1.5일 때 추가 결측치 퍼센트 평균 = {}%'.format(round(ave(outlier_df.loc[:, '1.5'].tolist()), 3)))
    print('whisker = 3.0일 때 추가 결측치 퍼센트 평균 = {}%'.format(round(ave(outlier_df.loc[:, '3.0'].tolist()), 3)))
    print('whisker = 4.5일 때 추가 결측치 퍼센트 평균 = {}%'.format(round(ave(outlier_df.loc[:, '4.5'].tolist()), 3)))

    os.chdir(savedir)
    outlier_df.to_excel('세대 및 whisker별 이상치 퍼센트.xlsx')

    return outlier_df



# outlier_df 결과 플로팅
def outlier_df_plot(temp, savedir, stmd, category_ami) :

    totallist_days = []  
    totallist_ends = []
    outliers_days = []
    outliers_ends = []


    for i in range(temp.shape[0]) :
        if '주말' in temp.loc[i, 'excel'] :
            end_ends = i
            
    ############ list ###############
            
            
    # 주중
    totallist_days.append(temp.loc[end_ends + 1 : , 'original'])
    totallist_days.append(temp.loc[end_ends + 1 : , '1.5_nan'])
    totallist_days.append(temp.loc[end_ends + 1 : , '3.0_nan'])
    totallist_days.append(temp.loc[end_ends + 1 : , '4.5_nan'])

    outliers_days.append(temp.loc[end_ends + 1 : , '1.5'])
    outliers_days.append(temp.loc[end_ends + 1 : , '3.0'])
    outliers_days.append(temp.loc[end_ends + 1 : , '4.5'])

    # 주말
    totallist_ends.append(temp.loc[: end_ends, 'original'])
    totallist_ends.append(temp.loc[: end_ends, '1.5_nan'])
    totallist_ends.append(temp.loc[: end_ends, '3.0_nan'])
    totallist_ends.append(temp.loc[: end_ends, '4.5_nan'])

    outliers_ends.append(temp.loc[: end_ends, '1.5'])
    outliers_ends.append(temp.loc[: end_ends, '3.0'])
    outliers_ends.append(temp.loc[: end_ends, '4.5'])


    ########### plot #################
    st = stmd
    newfolder(savedir)
    os.chdir(savedir)
    

    ############### weekdays
    title = 'outliers_{}_days'.format(category_ami)
    plt.rcParams["figure.figsize"] = (7,14)
    plt.boxplot(totallist_days)
    #         plt.xticks(rotation=60)
    plt.xticks([1, 2, 3, 4], ['original < {}%'.format(st), '1.5_nan', '3.0_nan', '4.5_nan'])
    plt.ylabel('total nan percentage(%)')
    plt.title("{}".format(title))
    plt.savefig('주중_이상치퍼센트_박스플롯.png', dpi=400)
    plt.show()

    ############### weekends
    title = 'outliers_{}_ends'.format(category_ami)
    plt.rcParams["figure.figsize"] = (7,14)
    plt.boxplot(totallist_ends)
    #         plt.xticks(rotation=60)
    plt.xticks([1, 2, 3, 4], ['original < {}%'.format(st), '1.5_nan', '3.0_nan', '4.5_nan'])
    plt.ylabel('total nan percentage(%)')
    plt.title("{}".format(title))
    plt.savefig('주말_결측값퍼센트_박스플롯.png', dpi=400)
    plt.show()

    ############### only outliers, days

    title = 'only_{}_days'.format(category_ami)
    plt.rcParams["figure.figsize"] = (7,14)
    plt.boxplot(outliers_days)
    #         plt.xticks(rotation=60)
    plt.xticks([1, 2, 3], ['1.5', '3.0', '4.5'])
    plt.ylabel('outliers / total(%)')
    plt.title("{}".format(title))
    plt.savefig('주중_이상치만_박스플롯.png', dpi=400)
    plt.show()

    ############### only outliers, ends

    title = 'only_{}_ends'.format(category_ami)
    plt.rcParams["figure.figsize"] = (7,14)
    plt.boxplot(outliers_ends)
    #         plt.xticks(rotation=60)
    plt.xticks([1, 2, 3], ['1.5', '3.0', '4.5'])
    plt.ylabel('outliers / total(%)')
    plt.title("{}".format(title))
    plt.savefig('주말_이상치만_박스플롯.png', dpi=400)
    plt.show()



def interp(df_org) :
    df_org = delun(df_org)
    df = df_org.copy()
    df.reset_index(drop = True, inplace = True)

    dropindex = []

    for index in range(df.shape[0]) :
        # templist = np.array(temp) # np.array(df.loc[i, :])
        templist = np.array(df.loc[index, :])
        nanpoint = []
        interp_true = 1

        for i in range(len(templist)) :
            if np.isnan(templist[i]) :
                nanpoint.append(i)

        if (0 in nanpoint) | (len(templist) - 1 in nanpoint) :
            interp_true = 0

        for i in range(len(nanpoint)) :
            if (i != 0) & (i != len(nanpoint) - 1) :
                if nanpoint[i] - nanpoint[i - 1] == 1 :
                    if nanpoint[i + 1] - nanpoint[i] == 1 :
                        interp_true = 0
                        
        if interp_true == 0 :
            dropindex.append(index)
        if interp_true == 1 :
            templist = interp_list(templist)

        df.loc[index, :] = templist.tolist()
    
    
    df = pd.DataFrame(df)
    df = df.drop(dropindex)
    df.reset_index(drop = True, inplace = True)

    return df, dropindex

def interp_list(list) :
    for i in range(list.shape[0]) :
        if np.isnan(list[i]) :
            if ~np.isnan(list[i + 1]) :
                list[i] = (list[i - 1] + list[i + 1]) / 2
            if np.isnan(list[i + 1]) :
                leftend = list[i - 1]
                rightend = list[i +  2]
                list[i] = list[i - 1] + (list[i + 2] - list[i - 1]) / 3 * 1
                list[i + 1] = list[i - 1] + (list[i + 2] - list[i - 1]) / 3 * 2
    return list

def interp_percentage(maindir, interpolate_day_std) :

    dropnum = 0
    savenum = 0

    os.chdir(maindir)
    for folder in os.listdir(maindir) :
        tempdir = maindir + '\\' + folder
        if os.path.isdir(tempdir) :
            for excel in os.listdir(tempdir) :
                os.chdir(tempdir)
                temp = read_excel(excel)

                temp, dropindex = interp(temp)
                if len(dropindex) < interpolate_day_std:
                    savenum += 1
                    print('{} saved'.format(excel), end = '\r')

                else :
                    dropnum += 1
                    print('{} droped, {}th'.format(excel, dropnum), end = '\r')

    temp_df = pd.DataFrame(columns = ['all', 'saved', 'dropped'])
    temp_df.loc[0, 'saved'] = savenum
    temp_df.loc[0, 'dropped'] = dropnum
    temp_df.loc[0, 'all'] = savenum + dropnum

    temp_df.loc[1, 'all'] = 100
    temp_df.loc[1, 'saved'] = savenum / (savenum + dropnum) * 100
    temp_df.loc[1, 'dropped'] = dropnum / (savenum + dropnum) * 100

    temp_df.index = ['num', 'percentage']

    return temp_df




################################################################################
################################# < MODEL > ####################################

# Model 3

# Model 2

# Model 4


################################################################################
################################ < PLOTTING > ##################################


def boxplot_list(list, title, filename, save) :
    array = np.array(list)
    plt.rcParams["figure.figsize"] = (4,14)
    plt.boxplot(array, whis = 1.5)
    plt.title(title)
    if save == 1 :
        plt.savefig('{}.png'.format(filename), dpi=400)
    plt.show()



def boxplot(df_org, title) : 
    df = df_org.copy()
    delun(df)
    df = df.to_numpy().flatten().transpose()
    
    plt.rcParams["figure.figsize"] = (4,14)
    plt.boxplot(df.transpose(), whis = 1.5)
    plt.title(title)
    #plt.savefig('{}.png'.format(title), dpi=400)
    plt.show()

def boxplot_nan(df_org, title) :
    df = df_org.copy()
    delun(df)
    df = df.to_numpy().flatten().transpose()
    df = df[~np.isnan(df)]
    
    plt.rcParams["figure.figsize"] = (4,14)
    plt.boxplot(df.transpose(), whis = 1.5)
    plt.title(title)
    #plt.savefig('{}.png'.format(title), dpi=400)
    plt.show()

def boxplot_save(df_org, title, filename) :
    df = df_org.copy()
    delun(df)
    df = df.to_numpy().flatten().transpose()
    df = df[~np.isnan(df)]
    
    plt.rcParams["figure.figsize"] = (4,14)
    plt.boxplot(df.transpose(), whis = 1.5)
    plt.title(title)
    plt.savefig('{}.png'.format(filename), dpi=400)
    plt.show()

def boxplot_save_3(df_org, title, filename) :
    df = df_org.copy()
    delun(df)
    df = df.to_numpy().flatten().transpose()
    df = df[~np.isnan(df)]
    
    plt.rcParams["figure.figsize"] = (4,14)
    plt.boxplot(df.transpose(), whis = 3.0)
    plt.title(title)
    plt.savefig('{}.png'.format(filename), dpi=400)
    plt.show()

def boxplot_save_coe(df_org, coe, title, filename) :
    df = df_org.copy()
    delun(df)
    df = df.to_numpy().flatten().transpose()
    df = df[~np.isnan(df)]
    
    plt.rcParams["figure.figsize"] = (4,14)
    plt.boxplot(df.transpose(), whis = coe)
    plt.title(title)
    plt.savefig('{}.png'.format(filename), dpi=400)
    plt.show()

def savefig(name, dpi_self) :
    plt.savefig(name, dpi = dpi_self)
    plt.show

def barplot_info(df_org, category, interval, xlabel, ylabel, title, xylim, figsize, save, save_name) :
    df = df_org.copy()
    yvalues = df.loc[:, category].tolist()
    xvalues = np.arange(len(yvalues))

    plt.rcParams["figure.figsize"] = (figsize[0], figsize[1])
    for i in range(len(xvalues)) :
        if xvalues[i] % interval == 0 :
            plt.bar(xvalues[i], yvalues[i], color = 'red')
        else :
            plt.bar(xvalues[i], yvalues[i], color = 'grey')

    plt.xlabel('{}'.format(xlabel))
    plt.ylabel('{}'.format(ylabel))
    plt.axis([xylim[0], xylim[1], xylim[2], xylim[3]])
    plt.title('{}'.format(title))
    
    x = []
    y = []

    for i in range(interval, df.shape[0], interval) :
        x.append(i)
        y.append(yvalues[i])
    
    for a, b in zip(x, y): 
        if b > 10 :
            plt.text(a, b + 1, str(round(b, 1)) + '%', color = 'red')
        else :
            plt.text(a, b + 5, str(round(b, 1)) + '%', color = 'red')
        # plt.text(a + 1, b + 10, 'missing data percentage')


    plt.rcParams.update({'font.size': 15})
    if save == 1 :
        plt.savefig('{}.png'.format(save_name), dpi=400)

    plt.show()

def barplot_st(df_org, category, xlabel, ylabel, title, xylim, figsize, save, save_name) :
    df = df_org.copy()
    yvalues = df.loc[:, category].tolist()
    xvalues = np.arange(len(yvalues))

    plt.rcParams["figure.figsize"] = (figsize[0], figsize[1])
    plt.bar(xvalues, yvalues)
    plt.xlabel('{}'.format(xlabel))
    plt.ylabel('{}'.format(ylabel))
    plt.axis([xylim[0], xylim[1], xylim[2], xylim[3]])
    plt.title('{}'.format(title))
    if save == 1 :
        plt.savefig('{}.png'.format(save_name), dpi=400)

    plt.show()


def barplot(df_org, category, xlabel, ylabel, title, xylim, figsize, cf, save, save_name) :
    df = df_org.copy()
    yvalues = df.loc[:, category].tolist()
    xvalues = np.arange(len(yvalues))

    plt.rcParams["figure.figsize"] = (figsize[0], figsize[1])
    plt.bar(xvalues, yvalues)
    plt.xlabel('{}'.format(xlabel))
    plt.ylabel('{}'.format(ylabel))
    plt.axis([xylim[0], xylim[1], xylim[2], xylim[3]])
    # plt.axis([0, len(xvalues), min(yvalues), max(yvalues)])
    plt.title('{}'.format(title))
    plt.plot([0, len(xvalues)], [cf, cf], 'r-', label = 'red line : {}%'.format(cf))
    plt.legend()

    lowvals = []
    for i in range(len(xvalues)) :
        temp = df.loc[i, category]
        if temp < cf :
            lowvals.append(temp)
    
    lowvals_per = len(lowvals) / len(xvalues) * 100

    plt.text(-40, cf, 'below {}%'.format(str(cf)), color = 'red')
    plt.text(-40, cf - 3, str(round(lowvals_per, 2)) + '%', color = 'red')
    plt.text(-40, cf - 6, '{} / {}'.format(len(lowvals), len(xvalues)) , color = 'red')
    plt.rcParams.update({'font.size': 15})

    if save == 1 :
        plt.savefig('{}.png'.format(save_name), dpi=400)

    plt.show()

    

    return lowvals_per

def find_nearest(list, value) :
    array = np.array(list)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def barplot_findcf(df_org, category, xlabel, ylabel, title, xylim, figsize, find_cf, save, save_name) :
    df = df_org.copy()
    yvalues = df.loc[:, category].tolist()
    xvalues = np.arange(len(yvalues))

    plt.rcParams["figure.figsize"] = (figsize[0], figsize[1])
    plt.bar(xvalues, yvalues)
    plt.xlabel('{}'.format(xlabel))
    plt.ylabel('{}'.format(ylabel))
    plt.axis([0, len(xvalues), min(yvalues), max(yvalues)])
    plt.axis([xylim[0], xylim[1], xylim[2], xylim[3]])
    plt.title('{}'.format(title))

    findcf_list = []
    findcf_list_perc = []

    for i in range(int(min(yvalues)), int(max(yvalues) + 1)) :
        lowvals = 0

        for j in range(len(xvalues)) :
            temp = df.loc[j, category]
            if temp < i :
                lowvals += 1
                
        lowvals_perc = lowvals / len(xvalues) * 100
        findcf_list.append(i)
        findcf_list_perc.append(lowvals_perc)

    nearest_val = find_nearest(findcf_list_perc, find_cf)
    find_index = findcf_list_perc.index(nearest_val)
    cf = findcf_list[find_index]

    plt.plot([0, len(xvalues)], [cf, cf], 'r-', label = 'red line : {}%'.format(cf))
    plt.legend()
    if save == 1 :
        plt.savefig('{}.png'.format(save_name), dpi=400)
    plt.show()


    return cf



################################################################################
################################## < ETC > #####################################

                                # └─ 3.etc                                
                                #    ├─read_excel                         
                                #    ├─pernull                            
                                #    └─os                                 
                                #       ├─copyfolderlist
                                #       ├─newfolderlist                   
                                #       ├─deletestr             
                                #       ├─delun             
                                #       └─ave    



def read_excel(excel) :
    df = pd.read_excel(excel)
    if 'Unnamed: 0' in df.columns :
        df.drop('Unnamed: 0', axis = 1, inplace = True)

    if 'Unnamed: 0.1' in df.columns :
        df.drop('Unnamed: 0.1', axis = 1, inplace = True)

    return df



def pernull(df_org) :
    df = df_org.copy()
    if 'Unnamed: 0' in df.columns :
        df.drop('Unnamed: 0', axis = 1, inplace = True)
        
    nullnum = df.isnull().sum().sum()
    notnullnum = df.notnull().sum().sum()
    percentage = nullnum / (nullnum + notnullnum) * 100
    return percentage



def copyfolderlist(directory1, directory2) :
    folderlist = []
    for folder in os.listdir(directory1) :
        folderlist.append(folder)
    newfolderlist(directory2, folderlist)



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



def deletestr(characters, string) :
    for x in range(len(characters)) :
        string = string.replace(characters[x], "")

    return string



def delun(df) :
    if 'Unnamed: 0' in df.columns :
        df.drop('Unnamed: 0', axis = 1, inplace = True)

    if 'Unnamed: 0.1' in df.columns :
        df.drop('Unnamed: 0.1', axis = 1, inplace = True)

    return df



def ave(lst) :
    return sum(lst) / len(lst)


def remove(path):
    # """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file

    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
        
    else:
        raise ValueError("file {} is not a file or dir.".format(path))


def copyfile(src_dir, dst_dir, src_file) :
    src = src_dir + '\\' + src_file
    dst = dst_dir
    shutil.copyfile(src, dst)


#################### 날짜 찾기 함수 ###################


def make_md() :
    md = pd.DataFrame(columns = ['month', 'day', 'all_start', 'all_end'])
    month = range(1, 13)

    md.loc[:, 'month'] = month

    md.loc[0, 'day'] = 31
    md.loc[1, 'day'] = 28
    md.loc[2, 'day'] = 31
    md.loc[3, 'day'] = 30
    md.loc[4, 'day'] = 31
    md.loc[5, 'day'] = 30
    md.loc[6, 'day'] = 31
    md.loc[7, 'day'] = 31
    md.loc[8, 'day'] = 30
    md.loc[9, 'day'] = 31
    md.loc[10, 'day'] = 30
    md.loc[11, 'day'] = 31

    sum = 0
    for i in range(md.shape[0]) :
        md.loc[i, 'all_start'] = sum
        sum = sum + md.loc[i, 'day'] - 1

        md.loc[i, 'all_end'] = sum
        sum = sum + 1
        
    return md

def find_position(month_, day_) :
    
    md = pd.DataFrame(columns = ['month', 'day', 'all_start', 'all_end'])
    month = range(1, 13)

    md.loc[:, 'month'] = month

    md.loc[0, 'day'] = 31
    md.loc[1, 'day'] = 28
    md.loc[2, 'day'] = 31
    md.loc[3, 'day'] = 30
    md.loc[4, 'day'] = 31
    md.loc[5, 'day'] = 30
    md.loc[6, 'day'] = 31
    md.loc[7, 'day'] = 31
    md.loc[8, 'day'] = 30
    md.loc[9, 'day'] = 31
    md.loc[10, 'day'] = 30
    md.loc[11, 'day'] = 31

    sum = 0
    for i in range(md.shape[0]) :
        md.loc[i, 'all_start'] = sum
        sum = sum + md.loc[i, 'day'] - 1

        md.loc[i, 'all_end'] = sum
        sum = sum + 1

    curr_md = md[md['month'] == month_]
    curr_md.reset_index(drop = True, inplace = True)
    start = curr_md.loc[0, 'all_start']
    position = start + day_ - 1

    return position

def find_date(num) :
    df = pd.DataFrame(columns = ['num', 'month', 'day'])
    
    num_list = []
    for i in range(365) :
        num_list.append(i)
        
    df.loc[:, 'num'] = num_list
    
    md = make_md()
    
    for i in range(md.shape[0]) :
        if (num >= md.loc[i, 'all_start']) & (num <= md.loc[i, 'all_end']) :
            now_month = md.loc[i, 'month']
            now_day = num - md.loc[i, 'all_start'] + 1
            
    return [now_month, now_day]


def para() :
    first_sat_sun = [5, 6]
    weekends = []

    for x in first_sat_sun :
        x = x - 1
        while x < 365 :
            weekends.append(x)
            x = x + 7

    weekends = sorted(weekends)
    weekdays = []
    for i in range(0, 365) :
        if i not in weekends :
            weekdays.append(i)
    
    day_list = []
    day_num = []
    for i in range(261) :
        day_list.append('day')
        day_num.append(i)
        
    end_list = []
    end_num = []
    for i in range(104) :
        end_num.append(i)
        end_list.append('end')
    all_list = day_list + end_list

    ult_para = pd.DataFrame(columns = ['weekday/end', 'num', 'actual', 'month', 'day', 'drop'])    
    ult_para.loc[:, 'weekday/end'] = all_list
    ult_para.loc[0 : 0 + 261 - 1, 'num'] = day_num
    ult_para.loc[261 : 261 + 104 - 1, 'num'] = end_num
    ult_para.loc[0 : 0 + 261 - 1, 'actual'] = weekdays
    ult_para.loc[261 : 261 + 104 - 1, 'actual'] = weekends
    
    for i in range(ult_para.shape[0]) :
        temp = ult_para.loc[i, 'actual']
        month_day = find_date(temp)
        ult_para.loc[i, 'month'] = month_day[0]
        ult_para.loc[i, 'day'] = month_day[1]
        
    start_summer = find_position(6, 22)
    end_summer = find_position(9, 23)

    start_winter_2 = find_position(12, 22)
    end_winter_1 = find_position(3, 21)

    start_winter_1 = 0
    end_winter_2 = 364

    for i in range(ult_para.shape[0]) :
        act = ult_para.loc[i, 'actual']
        if (start_summer <= act <= end_summer) | \
            (start_winter_1 <= act <= end_winter_1) | (start_winter_2 <= act <= end_winter_2) :
            ult_para.loc[i, 'drop'] = 'drop'
        else :
            ult_para.loc[i, 'drop'] = 'save'
            
        
    return ult_para

def change_para(ult_para) :
    change_log = pd.DataFrame(columns = ['weekday/end', '1_split', '2_missing', '3_outliers', '4_linear', \
                                        '5_semi', '6_done', '7_sf'])
    
    change_log.loc[:, 'weekday/end'] = ult_para.loc[:, 'weekday/end'].tolist()
    change_log.loc[:, '1_split'] = ult_para.loc[:, 'num'].tolist()
    
    change_log.loc[:, '2_missing'] = change_log.loc[:, '1_split']
    change_log.loc[:, '3_outliers'] = change_log.loc[:, '1_split']

    change_log

    return change_log
