'''
fitting : 
a, b, s1, s2 = scipy.stats.beta.fit(x)

note : x at here is list of data(variables)

random variates :
r = beta.rvs(a, b, size = n)

note : r is list of random variates with size of n
		if wanting to pull 1 value ot of the beta distribution, n should be 1
'''


from scipy.stats import beta
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
sys.path.append('C:\\Users\\joo09\\Documents\\GitHub\\main_hydrogen\\module')
import model_library as lib

model1_dir = os.getcwd() + '\\모델 1'

facility = []
for excel in os.listdir(model1_dir) :
	facility.append(str(excel[4 : -18]))
	
m1_ab = pd.DataFrame(columns = facility, index = ['a', 'b'])


for excel in os.listdir(model1_dir) :
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
	
	m1_ab.loc['a', facility] = a
	m1_ab.loc['b', facility] = b
	
print(m1_ab)
