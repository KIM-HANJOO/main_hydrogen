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

# -------------------------------------
# model 2 merge
# -------------------------------------

facility_list = ['교육시설', '문화시설', '업무시설', '판매및숙박']

save_dir = os.path.join(main_dir, 'temp', 'model2_all')
tdir = os.path.join(facility_dir, '교육시설', 'model2')
os.chdir(tdir)
for excel in os.listdir(tdir) :
    if 'Model2_daily' in excel :
        excel_name = excel

model2 = read_excel(excel_name)

all_model2 = []
for col in model2.columns :
    all_model2 += [x for x in model2.loc[:, col].tolist() if str(x) != 'nan']

c, d, loc, scale = scipy.stats.burr.fit(all_model2)
rv = burr(c, d, loc, scale)

print(all_model2[ : 10])
print(len(all_model2))
print(c, d, loc, scale)

fig = plt.figure(figsize = (7, 7))
x = np.linspace(min(all_model2), max(all_model2), 10000)

plt.plot(x, rv.pdf(x), 'k-', label = 'burr fitted line')
plt.hist(all_model2, density = True, bins = 100)


plt.grid()
plt.legend()
plt.title(f'all facilities, model2\nsample number = {len(all_model2)}\nc, d, loc, scale = {round(c, 3), round(d, 3), round(loc, 3), round(scale, 3)}')

os.chdir(save_dir)
plt.savefig('model2_all.png', dpi = 400)
