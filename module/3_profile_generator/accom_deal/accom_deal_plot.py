import os
import sys
import get_dir
di = get_dir.get_info()

main_dir = di.main_dir
prep_dir = di.prep_dir
model_dir = di.model_dir
module_dir = di.module_dir
facility_dir = di.facility_dir
nfc_dir = di.nfc_dir
plot_dir = di.plot_dir
cluster_dir = di.cluster_dir
facility_df = di.facility_df

sys.path.append(module_dir)
sys.path.append(module_dir + '//4_directory_module')
import directory_change as dich
import model_library as lib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager, rc
#font_path = "C:/Windows/Fonts/malgunbd.TTF"
font_path = "/mnt/c/Windows/Fonts/malgunbd.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
matplotlib.rcParams['axes.unicode_minus'] = False
import os
import glob
import os.path
import math
import random
from scipy.stats import beta, kde
import scipy.stats
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

def pjoin(p1, p2) :
    return os.path.join(p1, p2) 

def pjoin(plist) :
    tempdir = plist[0]
    for i in range(len(plist)) :
        if i != 0 :
            tempdir = os.path.join(tempdir, plist[i])

    return tempdir


info = read_excel('accom_deal_m1_m3.xlsx')
fig = plt.figure(figsize = (16, 10))
ax = fig.add_subplot(1, 1, 1)

for i, col in enumerate(info.columns) :
    temp = info.loc[:, col].tolist()
    temp = [x for x in temp if str(x) != 'nan']

    box = ax.boxplot(temp, positions = [i])

ax.set_title('accomodation, dealership | model1 & model3 daily ave compare')
ax.set_label('cats')
ax.set_xticklabels(info.columns, rotation = 90)
ax.set_ylim([-100, 5000])
ax.grid()
plt.savefig('accom_deal_daily_ave_compare_plot.png', dpi = 400)
