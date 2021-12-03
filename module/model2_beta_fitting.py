'''
fitting : 
a, b, s1, s2 = scipy.stats.beta.fit(x)

note : x at here is list of data(variables)

random variates :
r = beta.rvs(a, b, loc = s1, scale = s2, size = n)

note : r is list of random variates with size of n
		if wanting to pull 1 value ot of the beta distribution, n should be 1
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

from scipy.stats import beta, kde
import scipy.stats

import numpy as np
import os
import pandas as pd


import sys
sys.path.append(module_dir)
import model_library as lib

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgunbd.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)



model1_2_dir = main_dir + '\\model_1_2'
model1_dir = model1_2_dir + '\\모델 1'
plot_dir = model1_2_dir + '\\plot'

# pull 'alpha', 'beta' of beta distribution out of model_1 data (beta distribution fitting)

facility = []
for excel in os.listdir(model1_dir) :
	facility.append(str(excel[4 : -18]))
	
m1_ab = pd.DataFrame(columns = facility, index = ['a', 'b', 'loc', 'scale'])

fig = plt.figure(figsize = (16, 12))

for num, excel in enumerate(os.listdir(model1_dir)) :
	os.chdir(model1_dir)	
	facility = excel[4 : -18]
	df = pd.read_excel(excel)
		
	m1 = df.loc[:, df.columns[0]].tolist()
	m1.append(df.columns[0])
	
	drop_list = []
	for i in range(len(m1)) :
		if str(m1[i]) == 'NaN' :
			drop_list.append(i)
			
	if len(drop_list) > 0 :
		m1 = m1.remove(drop_list)
	
	a, b, s1, s2 = scipy.stats.beta.fit(m1)
	

	# plot
	ax = fig.add_subplot(2, 3, 1 + num)
	ax.set_title('{}\nBeta Distribution(a = {}, b = {})'.format(facility, round(a, 3), round(b, 3)))
	ax.set_xlabel('model 1')
	ax.set_ylabel('density')
	
	r = beta.rvs(a, b, loc = s1, scale = s2, size = 10000)
	density_real = kde.gaussian_kde(m1)
	density_sample = kde.gaussian_kde(r)
	x = np.linspace(min(r), max(r), 300)
	
	y_real = density_real(x)
	y_sample = density_sample(x)
	
	ax.grid()
	ax.plot(x, y_sample, 'b--', label = 'random variates')
	ax.plot(x, y_real, 'r', label = 'real value')
	ax.legend()
	
	m1_ab.loc['a', facility] = a
	m1_ab.loc['b', facility] = b
	m1_ab.loc['loc', facility] = s1
	m1_ab.loc['scale', facility] = s2

os.chdir(plot_dir)
fig.subplots_adjust(hspace=1)
# plt.tight_layout()
plt.savefig('model1_fig.png', dpi = 400)
plt.show()
os.chdir(model1_2_dir)

m1_ab.to_excel('model1_beta_fitting.xlsx')


# pull 'alpha', 'beta' of beta distribution out of model_2 data (beta distribution fitting)

os.chdir(model1_2_dir)
m2_all = lib.read_excel('sample(anova).xlsx')

unique_list = m2_all.loc[:, 'group'].unique()
m2_ab = pd.DataFrame(columns = unique_list, index = ['a', 'b'])

fig = plt.figure(figsize = (16, 12))

for num, fc in enumerate(unique_list) :
	facility = fc
	temp = m2_all[m2_all['group'] == fc]
	
	temp.to_excel(f'temp_{facility}.xlsx')
	m2 = temp.loc[:, 'var'].tolist()
	max_var = max(m2)
	
	for i in range(len(m2)) :
		m2[i] = m2[i] / max_var
	
	a, b, s1, s2 = scipy.stats.beta.fit(m2)
	
	
	# plot
	ax = fig.add_subplot(3, 3, 1 + num)
	ax.set_title('{}\nBeta Distribution(a = {}, b = {})'.format(facility, round(a, 3), round(b, 3)))
	ax.set_xlabel('model 1')
	ax.set_ylabel('density')
	
	r = beta.rvs(a, b, loc = s1, scale = s2, size = 10000)
	density_real = kde.gaussian_kde(m2)
	density_sample = kde.gaussian_kde(r)
	x = np.linspace(min(r), max(r), 300)
	
	y_real = density_real(x)
	y_sample = density_sample(x)
	
	ax.grid()
	ax.plot(x, y_sample, 'b--', label = 'random variates')
	ax.plot(x, y_real, 'r', label = 'real value')
	ax.legend()
	
	m2_ab.loc['a', facility] = a
	m2_ab.loc['b', facility] = b
	m2_ab.loc['loc', facility] = s1
	m2_ab.loc['scale', facility] = s2
	

m2_ab.to_excel('model2_beta_fitting.xlsx')

os.chdir(plot_dir)
fig.subplots_adjust(hspace=1)
plt.savefig('model2_fig.png', dpi = 400)
plt.show()

print(m1_ab, '\n', m2_ab)
