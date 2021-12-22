'''
fitting : 
a, b, s1, s2 = scipy.stats.beta.fit(x)

note : x at here is list of data(variables)

random variates :
r = beta.rvs(a, b, loc = s1, scale = s2, size = n)

note : r is list of random variates with size of n
		if wanting to pull 1 value ot of the beta distribution, n should be 1
'''

from scipy.stats import beta, kde
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

model1_2_dir = os.getcwd()

import sys
sys.path.append(model1_2_dir)
import model_library as lib


def random_variates(df, fc, n) :
	for col in df.columns :
		if fc in col :
			a = df.loc['a', fc]
			b = df.loc['b', fc]
			s1 = df.loc['loc', fc]
			s2 = df.loc['scale', fc]
			
	r = beta.rvs(a, b, loc = s1, scale = s2, size = n)
	
	return r
