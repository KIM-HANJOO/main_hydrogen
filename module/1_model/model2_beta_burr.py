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
gp_plot = di.gp_plot
print(f'main_dir = {main_dir}\n')

sys.path.append(module_dir)
sys.path.append(os.path.join(module_dir, '4_directory_module'))

import directory_change as dich
import discordlib_pyplot as dlt
import model_library as lib

import numpy as np
import pandas as pd
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

'''
note
'''
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



# -----------------------------------------------------
# fitting beta, burr distribution
# -----------------------------------------------------


fig = plt.figure(figsize = (50, 15))

beta_index = ['a', 'b', 'loc', 'scale']
burr_index = ['c', 'd', 'loc', 'scale']

df_beta = pd.DataFrame(columns = [], index = beta_index)
df_burr = pd.DataFrame(columns = [], index = burr_index)

df_burr_barplot = pd.DataFrame()
facility_list = [x for x in os.listdir(facility_dir) if x != 'params']

ncols = []
for fc in facility_list :
    ncols.append(f'{fc}_weekdays')
    ncols.append(f'{fc}_weekends')

df_density = pd.DataFrame(columns = ncols)

for fac_num, subdir in enumerate(facility_list) :
    df_num = 0
    print('excel loading')

    model2_dir = os.path.join(facility_dir, subdir, 'model2')
    os.chdir(model2_dir)
    for excel in os.listdir(model2_dir) :
        if 'Model2_daily' in excel :
            model2 = read_excel(excel)
    #model2 = read_excel('Model2_daily fractoin.xlsx')
    print('excel is loaded')
    ncols = []
    for col in model2.columns :
        if subdir in col :
            ncols.append(col)

    model2 = model2[ncols]


    for de_num, col in enumerate(model2.columns) :
        print(f'{col} started')
        df_beta[col] = None
        df_burr[col] = None

        if '주중' in col  :
            de = 'weekdays'
        else :
            de = 'weekends'

        tmp = model2.loc[:, col].tolist()
        tmp = [x for x in tmp if str(x) != 'nan']
#        a, b, s1, s2 = scipy.stats.beta.fit(tmp)
#        print('fitted')
#
#        df_beta.loc['a', col] = a
#        df_beta.loc['b', col] = b
#        df_beta.loc['loc', col] = s1
#        df_beta.loc['scale', col] = s2

        # r = beta.rvs(a, b, loc = s1, scale = s2, size = n)

        c, d, s3, s4 = scipy.stats.burr.fit(tmp)
        print('burr fitted')

        df_burr.loc['c', col] = c
        df_burr.loc['d', col] = d
        df_burr.loc['loc', col] = s3
        df_burr.loc['scale', col] = s4
        
        # r2 = burr.rvs(c, d, loc = s3, scale = s4, size = n)

        #r_beta = beta.rvs(a, b, loc = s1, scale = s2, size = len(tmp))
        r_burr = burr.rvs(c, d, loc = s3, scale = s4, size = len(tmp))
        print('r_burr made')

# -----------------------------------------------------
# plotting original density
# -----------------------------------------------------

#        x = np.linspace(min(tmp), max(tmp), len(tmp))
#
#        tmp_smpl = kde.gaussian_kde(tmp)
#        tmp_beta = kde.gaussian_kde(r_beta)
#        tmp_burr = kde.gaussian_kde(r_burr)
#        
#        density_smpl = tmp_smpl(x)
#        density_beta = tmp_beta(x)
#        density_burr = tmp_burr(x)

        print('procedure before writing ended')

        df_num = 0
        for index in range(len(r_burr.tolist())) :
            if '주중' in col :
                df_density.loc[df_num, f'{subdir}_weekdays'] = r_burr.tolist()[index]
                df_num += 1
                print(f'{subdir}, 주중', df_num / len(r_burr.tolist()))
            else :
                df_density.loc[df_num, f'{subdir}_weekends'] = r_burr.tolist()[index]
                df_num += 1
                print(f'{subdir}, 주말', df_num / len(r_burr.tolist()))
        print(f'ended for {subdir}')

#        #df_density.loc[:, subdir] = np.array(density_burr).transpose()
#
#        print(len(density_smpl))
#        print(len(density_beta))
#        print(len(density_burr))
##        all_min = min(min(density_smpl), min(density_beta), min(density_burr))
##        all_max = max(max(density_smpl), max(density_beta), max(densityu_burr))
#        print(tmp_burr)
#        print(density_burr)
#        print(np.array(tmp_burr))
#        print(np.array(density_burr))
#        
## -----------------------------------------------------
## plotting beta distribution
## -----------------------------------------------------
#        
#        tnum = 2 * fac_num + de_num + 1
#        print(f'plotting beta on {tnum}')
#
#        locals()[f'ax_{tnum}'] = fig.add_subplot(2, 8, tnum)
#        locals()[f'ax_{tnum}'].plot(x, density_smpl, 'r', label = 'sample')
#        locals()[f'ax_{tnum}'].plot(x, density_beta, 'b--', label = 'beta fitted')
#        locals()[f'ax_{tnum}'].grid()
#        locals()[f'ax_{tnum}'].legend()
#        locals()[f'ax_{tnum}'].set_title(f'{subdir}, {de}\nbeta fitted')
#
#
## -----------------------------------------------------
## plotting burr distribution 
## -----------------------------------------------------
#
#        tnum = 8 + 2 * fac_num + de_num + 1
#        print(f'plotting burr on {tnum}')
#
#        locals()[f'ax_{tnum}'] = fig.add_subplot(2, 8, tnum)
#        locals()[f'ax_{tnum}'].plot(x, density_smpl, 'r', label = 'sample')
#        locals()[f'ax_{tnum}'].plot(x, density_burr, 'b--', label = 'burr fitted')
#        locals()[f'ax_{tnum}'].grid()
#        locals()[f'ax_{tnum}'].legend()
#        locals()[f'ax_{tnum}'].set_title(f'{subdir}, {de}\nburr fitted')
#
#        print(f'{subdir}, {de} done !')



#save_dir = os.path.join(gp_plot, 'model2_fitted')
#dich.newfolder(save_dir)
#os.chdir(save_dir)
#df_beta.to_excel('model2_beta_fitted.xlsx')
#df_burr.to_excel('model2_burr_fitted.xlsx')
#
#dlt.savefig(save_dir, 'model2_beta_burr.png', 400)
#print(df_beta)
#print(df_burr)
os.chdir(cwdir)
df_density.to_excel('burr_hist.xlsx')
