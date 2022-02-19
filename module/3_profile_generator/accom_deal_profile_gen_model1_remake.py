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

# ----------------------------------------------------------
# make model_1.xlsx in accomodation & dealership folder
# ----------------------------------------------------------
# import scipy.stats
from scipy.stats import beta

# re-declare directories

accom_deal_profile_48 = os.path.join(main_dir, 'accom_deal_profile_48')
facility_list = ['판매시설', '숙박시설']
facility_dir = os.path.join(accom_deal_profile_48, 'FACILITIES')

model1_df = pd.DataFrame(columns = facility_list, index = ['a', 'b', 'loc', 'scale'])
model1_iloc = pd.DataFrame(columns = facility_list, index = ['a', 'b', 'loc', 'scale'])

for facility in facility_list :
    fc_dir = os.path.join(facility_dir, facility)
    model1_dir = os.path.join(fc_dir, 'model1')

    os.chdir(model1_dir)
    model1_all = read_excel(f'모델1_{facility}.xlsx')
        
    model1_part = model1_all.iloc[:, 0].tolist()
    model1_part.append(model1_all.columns[0])

    a, b, loc, scale = beta.fit(model1_part)
    a2, b2, loc2, scale2 = beta.fit(model1_all.iloc[:, 0])

    model1_df.loc[:, facility] = [a, b, loc, scale]
    model1_iloc.loc[:, facility] = [a2, b2, loc2, scale2]

    mini = (


print(model1_df)

fig = plt.figure(figsize = [20, 20])
for num, col in enumerate(model1_df.columns) :
    a, b, c, d = model1_df.loc[:, col]
    a1, b1, c1, d1 = model1_iloc.loc[:, col]

    x = np.linspace(min(

    n = 10000
    r = beta.rvs(a, b, loc = c, scale = d, size = n)
    r2 = beta.rvs(a1, b1, loc = c1, scale = d1, size = n)

    ax = fig.add_subplot(2, 1, num + 1)
    ax.plot(


