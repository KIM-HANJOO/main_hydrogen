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

def model2_plot(facility_name, group, model2_dir, plot_dir) :
    os.chdir(model2_dir)
    
    if 'model2_weekdays_std.xlsx' in os.listdir(model2_dir) :
        temp_day = read_excel('model2_weekdays_std.xlsx')
    else :
        temp_day = pd.DataFrame(columns = ['excel', 'std'])
        for excel in os.listdir(model2_dir) :
            if 'model2_weekdays_std_' in excel :
                print(f'{excel} loading')
                start = time.time()
                os.chdir(model2_dir)
                temp = read_excel(excel)
                temp_day = pd.concat([temp_day, temp])
                
                end = time.time()
                print(f'{excel} concatenated, elapsed : {round(end - start, 2)}')
        temp = None
        
    
    if 'model2_weekends_std.xlsx' in os.listdir(model2_dir) :
        temp_end = read_excel('model2_weekends_std.xlsx')
    else :
        temp_end = pd.DataFrame(columns = ['excel', 'std'])
        for excel in os.listdir(model2_dir) :
            if 'model2_weekends_std' in excel :
                print(f'{excel} loading')
                start = time.time()
                os.chdir(model2_dir)
                temp = read_excel(excel)
                temp_end = pd.concat([temp_end, temp])
                
                end = time.time()
                print(f'{excel} concatenated, elapsed : {round(end - start, 2)}')
        temp = None
        
                
    m2_day = temp_day.loc[:, 'std'].tolist()
    m2_end = temp_end.loc[:, 'std'].tolist()
    
#    print(m2_day)
#    print(m2_end)
    print(min(m2_day))
    print(min(m2_end))
    min_all = min([min(m2_day), min(m2_end)])
    max_all = max([max(m2_day), max(m2_end)])
    fig = plt.figure(figsize = (16, 10))
    
    os.chdir(os.path.join(gp_dir, 'params'))
    model2_burr = read_excel('model2_burr_fitted.xlsx')
    model2_burr.index = ['c', 'd', 'loc', 'scale']


    for i in range(2) :
        ax = fig.add_subplot(2, 1, 1 + i)

        for col in model2_burr.columns :
            if (facility_name == '판매시설') | (facility_name == '숙박시설') :
                if '판매및숙박' in col :
                    if i == 0 :
                        if '주중' in col :
                            ncol = col
                            c = model2_burr.loc['c', ncol]
                            d = model2_burr.loc['d', ncol]
                            loc = model2_burr.loc['loc', ncol]
                            scale = model2_burr.loc['scale', ncol]
                    elif i == 1 :
                        if '주말' in col :
                            ncol = col
                            c = model2_burr.loc['c', ncol]
                            d = model2_burr.loc['d', ncol]
                            loc = model2_burr.loc['loc', ncol]
                            scale = model2_burr.loc['scale', ncol]

            else :
                if facility_name in col :
                    if i == 0 :
                        if '주중' in col :
                            c = model2_burr.loc['c', col]
                            d = model2_burr.loc['d', col]
                            loc = model2_burr.loc['loc', col]
                            scale = model2_burr.loc['scale', col]
                    elif i == 1 :
                        if '주말' in col :
                            c = model2_burr.loc['c', col]
                            d = model2_burr.loc['d', col]
                            loc = model2_burr.loc['loc', col]
                            scale = model2_burr.loc['scale', col]
                
        # ~ r = beta.rvs(a, b, loc = loc, scale = scale, size = 10000)
        
        # ~ ax.figure(figsize = (8, 8))
        ax.set_title('{}\nBurr Distribution(c = {}, d = {})'.format(facility_name, round(c, 3), round(d, 3)))
        ax.set_xlabel('model 2')
        ax.set_ylabel('density')
        
        if i == 0 :
            density_real = kde.gaussian_kde(m2_day)
            
        elif i == 1 :
            density_real = kde.gaussian_kde(m2_end)
            
        x = np.linspace(min_all, max_all, 300)
        y_real = density_real(x)
        
        ax.grid()
        ax.plot(x, y_real, 'r', label = 'real value')
        ax.legend()
        
    plt.subplots_adjust(hspace = 0.35)
    os.chdir(plot_dir)
    plt.savefig(f'model2_{facility_name}_{group}_all.png', dpi = 400)
    dlt.savefig(plot_dir, f'model2_{facility_name}_{group}_all.png', 400)

    plt.clf()
    plt.cla()
    plt.close()
        
    
    m2_day = None
    m2_end = None
    pass



# -----------------------------------------------------
# beta fitted line to 10000 list
# -----------------------------------------------------

facility_list = ['bus', 'cult', 'edu', 'accom', 'deal']
facility_name_list = ['업무시설', '문화시설', '교육시설', '숙박시설', '판매시설']

model1_dir = os.path.join(main_dir, 'temp', 'model1_plot')
model1_plot = os.path.join(model1_dir, 'plot')

plot_num = 10000

fig = plt.figure(figsize = (30, 20))
for num, facility in enumerate(facility_list) :
    os.chdir(model1_dir)
    for files in os.listdir(model1_dir) :
        if 'model_1.xlsx' in files :
            if facility in files :
                smpl = read_excel(files)
        if 'real.xlsx' in files :
            if facility in files :
                real = read_excel(files)

    facility_name = facility_name_list[num]


#    print(smpl.columns)
#    print(facility_name)
#    smpl_list = smpl.loc[:, facility_name].tolist()
#    smpl = pd.DataFrame(columns = [facility_name])
#    smpl.loc[:, facility_name] = smpl_list
#    print(smpl)
    print(smpl.columns)
    print(smpl.loc[:, facility_name])
    smpl = pd.DataFrame(smpl.loc[:, facility_name])


    a = smpl.iloc[0, 0]
    b = smpl.iloc[1, 0]
    loc = smpl.iloc[2, 0]
    scale = smpl.iloc[3, 0]

    real_list = real.iloc[:, 0].tolist()
    real_list.append(real.columns[0])
    print(len(real_list))

    #a, b, loc, scale = beta.fit(real_list)

    number = 10000
    r_fitted = beta.rvs(a, b, loc = loc, scale = scale, size = number)

    ax = fig.add_subplot(2, 3, num + 1)

    ax.set_title(f'{facility_name}\na = {round(a, 3)}, b = {round(b, 3)}\nloc = {round(loc, 3)}, scale = {round(scale, 3)}')

    x = np.linspace(min(real_list), max(real_list), plot_num)
    y = beta(a, b, loc, scale).pdf(x)
    print(beta.pdf(x, a, b, loc = loc, scale = scale))
    #y = beta(a, b).pdf(x)

    ax.hist(real_list, label = 'sample', alpha = 0.4, bins = 10, density = True)
    ax.hist(r_fitted, label = 'rvs', alpha = 0.4, bins = 10, density = True)
    ax.plot(x, y, label = 'beta fitted line')
    ax.set_xlim(min(real_list) - 0.2, max(real_list) + 0.2)
    ax.set_ylim(0, 3)

    ax.grid()
    ax.legend()
    print(f'{facility_name} done')

os.chdir(model1_plot)
plt.savefig('all_model1_fitted_line.png', dpi = 400)



    
    






