import os
import pathlib
from pathlib import Path
import pandas as pd
import os
import sys

cwdir = os.getcwd()
main_module_dir = str(Path(cwdir).parent.absolute())
sys.path.append(main_module_dir)
import dir_info

def get_info() :
	cwdir = os.getcwd()
	main_module_dir = str(Path(cwdir).parent.absolute())
	dinfo = dir_info.Info(main_module_dir)
	
	return dinfo
	
	
	
'''
### at all subdirs in 'module' dir add 'get_dir.py', at all .py codes add lines
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
'''

'''
### additional imports

sys.path.append(module_dir)
sys.path.append(module_dir + '\\4_directory_module')
import directory_change as dich
import model_library as lib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import os.path
import math
import random
from scipy.stats import beta, kde
import scipy.stats
'''
