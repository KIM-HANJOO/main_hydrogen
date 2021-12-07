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
	dinfo = dir_info.info()
	
	return dinfo
	
	
	
'''
### at all subdirs in 'module' dir add 'get_dir.py', at all .py codes add lines

import get_dir
di = get_dir.get_info()

main_dir = di.main_dir
prep_dir = di.prep_dir
model_dir = di.model_dir
module_dir = di.module_dir
facility_dir = di.facility_dir
plot_dir = di.plot_dir
cluster_dir = di.cluster_dir

###
'''
