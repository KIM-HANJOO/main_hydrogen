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

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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

def get_iqr(list1) :
    Q1 = np.percentile(list1, 25)
    Q3 = np.percentile(list1, 75)
    IQR = Q3 - Q1

    return IQR

# ---------------------------------------------------------
# relation between model3, model4
# ---------------------------------------------------------


facility_list = ['업무시설', '교육시설', '문화시설', '판매및숙박']
save_dir = os.path.join(main_dir, 'temp', 'model3_model4_rel')

hours_int = []
for i in range(1, 25) :
    hours_int.append(i)

hours = []
for i in range(1, 25) :
    hours.append(str(i))
    
hours_ex = []
for i in range(1, 49) :
    hours_ex.append(str(i))

df = pd.DataFrame(columns = ['ave', 'iqr'])
#
#for num, facility in enumerate(facility_list) :
#    fig = plt.figure(figsize = (20, 15))
#
#    fc_dir = os.path.join(main_dir, 'FACILITIES', facility)
#    model3_dir = os.path.join(fc_dir, 'model3')
#    model4_dir = os.path.join(fc_dir, 'model4')
#
#    for group in [0, 1] :
#        temp_df = pd.DataFrame(columns = ['ave', 'iqr'])
#        temp_num = 0
#
#        subplot_num = group * 2
#        ax1 = fig.add_subplot(2, 2, subplot_num + 1)
#        ax2 = fig.add_subplot(2, 2, subplot_num + 2)
#
#        tdir3 = os.path.join(model3_dir, f'group_{group}')
#        os.chdir(tdir3)
#        model3 = read_excel(f'profile_48_group_{group}.xlsx')
#        model3.columns = model3.columns.astype(str)
#
#        tdir4 = os.path.join(model4_dir, f'group_{group}_model4')
#        os.chdir(tdir4)
#        model4_day = read_excel('model4_weekdays.xlsx')
#        model4_end = read_excel('model4_weekends.xlsx')
#        model4_day.columns = model4_day.columns.astype(str)
#        model4_end.columns = model4_end.columns.astype(str)
#        
#        model4_iqr = []
#        for hour in hours :
#            model4_iqr.append(get_iqr(model4_day.loc[:, hour]))
#        for hour in hours :
#            model4_iqr.append(get_iqr(model4_end.loc[:, hour]))
#
#        model3 = model3[hours_ex]
#        ave = (model3.sum() / model3.shape[0]).tolist()
#
#        print(f'{facility}, {group} loaded')
#        
#        # plot
#        for hour in range(1, 49) :
#            ax1.scatter(ave[hour - 1], model4_iqr[hour - 1])
#        print(f'{facility}, {group}, ax1 done')
#
#        ax2.set_title(f'{facility}, group = {group}\n model3 to model4')
#        ax2.grid()
#        for hour in range(1, 49) :
#            if hour < 25 :
#                # extended ave with model4 weekday
#                temp_model4 = model4_day.loc[:, str(hour)].tolist()
#
#                temp_ave = []
#                for i in range(len(temp_model4)) :
#                    temp_ave.append(ave[hour - 1])
#
#                ax2.scatter(temp_ave, temp_model4)
##                for index in range(model4_day.shape[0]) :
##                    ax2.scatter(ave[hour - 1], model4_day.loc[index, str(hour)])
#
#            else :
#                # extended ave with model4 weekend
#                hour_temp = hour - 24
#                temp_model4 = model4_end.loc[:, str(hour_temp)].tolist()
#
#                temp_ave = []
#                for i in range(len(temp_model4)) :
#                    temp_ave.append(ave[hour - 1])
#
#                ax2.scatter(temp_ave, temp_model4)
##                for index in range(model4_end.shape[0]) :
##                    ax2.scatter(ave[hour - 1], model4_end.loc[index, str(hour_temp)])
#            print(f'{hour} done', end = '\r')
#
#        # Linear Regression with iqr (ax1)
#        # LR, x : ave, y : model4_iqr
#        line_fitter = LinearRegression()
#        line_fitter.fit(np.array(ave).reshape(-1, 1), model4_iqr)
#
#        m = round(line_fitter.coef_[0], 3)
#        c = round(line_fitter.intercept_, 3)
#        print(f'm : {line_fitter.coef_}\nc : {line_fitter.intercept_}')
#        ax1.plot(ave, line_fitter.predict(np.array(ave).reshape(-1, 1)))
#        ax1.set_title(f'{facility}, group = {group}\n model3 to iqr\ny = {m}x + {c}')
#
#        # Linear Regression with intercept forced to be Zero
#        ave_fz = np.array(ave)[:, np.newaxis]
#        a1, _, _, _ = np.linalg.lstsq(ave_fz, model4_iqr)
#
#        ax1.plot(ave, a1 * ave, 'r-', label = f'coef : {round(a1[0], 3)}')
#        ax1.legend()
#        ax1.grid()
#        #ax1.set_aspect('equal', adjustable = 'box')
#
#
#        print(f'{facility}, {group}, ax2 done')
#
#        # R squared
#        for i in range(len(ave)) :
#            temp_df.loc[temp_num, :] = [ave[i], model4_iqr[i]]
#            temp_num += 1
#        df = pd.concat([df, temp_df], axis = 0, ignore_index = True)
#
#        predicted = line_fitter.predict(np.array(ave).reshape(-1, 1))
#        predicted_zero = ave * a1
#
#        r2 = r2_score(temp_df.loc[:, 'iqr'], predicted)
#        r2_zero = r2_score(temp_df.loc[:, 'iqr'], predicted_zero)
#            
#        print(f'{facility}, {group}, R2 : {round(r2, 4)}')
#        print(f'{facility}, {group}, R2_zero : {round(r2_zero, 4)}')
#
#
#    os.chdir(save_dir)
#    plt.savefig(f'{facility}.png', dpi = 400)
#    plt.clf()
#
#
#os.chdir(save_dir)
#df.to_excel('iqr.xlsx')

os.chdir(save_dir)
df = read_excel('iqr.xlsx')
fig = plt.figure(figsize = [10, 10])
plt.scatter(df.loc[:, 'ave'], df.loc[:, 'iqr'])
# Linear Regression
line_fitter = LinearRegression()
line_fitter.fit(np.array(df.loc[:, 'ave']).reshape(-1, 1), df.loc[:, 'iqr'].tolist())

m1 = round(line_fitter.coef_[0], 3)
c1 = round(line_fitter.intercept_, 3)

plt.plot(df.loc[:, 'ave'], line_fitter.predict(np.array(df.loc[:, 'ave']).reshape(-1, 1)))


# Linear Regression, with intercept forced to be Zero
ave_fz_1 = np.array(df.loc[:, 'ave'])[:, np.newaxis]
a2, _, _, _ = np.linalg.lstsq(ave_fz_1, df.loc[:, 'iqr'].tolist())

plt.plot(df.loc[:, 'ave'], a2 * df.loc[:, 'ave'], 'r-', label = f'coef : {round(a2[0], 3)}')
#plt.legend()
plt.grid()
predicted = line_fitter.predict(np.array(df.loc[:, 'ave']).reshape(-1, 1))
predicted_zero = df.loc[:, 'ave'].tolist() * a2

r2 = r2_score(df.loc[:, 'iqr'], predicted)
r2_zero = r2_score(df.loc[:, 'iqr'], predicted_zero)

plt.title(f'all facilities\ny = {m1}x + {c1}, R squared : {round(r2, 4)}\n y = {round(a2[0], 3)}x, R squared: {round(r2_zero, 4)}')
plt.xlabel('model3')
plt.ylabel('model4 iqr')
plt.savefig(f'all.png', dpi = 400)


print(round(r2, 4))
print(round(r2_zero, 4))
