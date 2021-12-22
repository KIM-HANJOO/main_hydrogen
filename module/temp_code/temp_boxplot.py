import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(10)
collectn_1 = np.random.normal(100, 10, 200)
collectn_2 = np.random.normal(80, 30, 200)
collectn_3 = np.random.normal(90, 20, 200)
collectn_4 = np.random.normal(70, 25, 200)


data_to_plot = []

for i in range(6) :
	data_to_plot.append(collectn_1)
	data_to_plot.append(collectn_2)
	data_to_plot.append(collectn_3)
	data_to_plot.append(collectn_4)
	
#

red_boxprops = dict(color = 'darkred', facecolor = 'red', linewidth = 1.2)
red_whiskerprops = dict(color = 'red', linestyle = '-', linewidth = 2)
red_flierprops = dict(linestyle = '-', color = 'red', markerfacecolor = 'red', markeredgecolor = 'red', markersize = 10)
red_medprops = dict(color = 'darkred', linewidth = 1.2)
red_capprops = dict(color = 'red', linewidth = 2)


default_boxprops = dict(color = 'black', facecolor = 'dimgrey', linewidth = 2)
default_flierprops = dict(linestyle = 'none', color = 'dimgrey', markerfacecolor = 'dimgrey', linewidth = 2)
default_medprops = dict(color = 'black', linewidth = 2)

#
 
plt.figure(figsize = (14, 6))
 
plt.boxplot(data_to_plot, widths = 0.6, patch_artist = True, \
		flierprops = red_flierprops, boxprops = red_boxprops, medianprops = red_medprops, whiskerprops = red_whiskerprops, \
		capprops = red_capprops)
 
 
plt.show()
