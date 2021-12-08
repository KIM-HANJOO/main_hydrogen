'''
note :

'''

###################################################################################
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

sys.path.append(module_dir)
sys.path.append(module_dir + '\\4_directory_moduel')
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
import shutil
from distutils.dir_util import copy_tree

#######################################################


def read_excel(excel) :
    df = pd.read_excel(excel)
    if 'Unnamed: 0' in df.columns :
        df.drop('Unnamed: 0', axis = 1, inplace = True)

    if 'Unnamed: 0.1' in df.columns :
        df.drop('Unnamed: 0.1', axis = 1, inplace = True)

    return df
    
def copyfolderlist(directory1, directory2) :
    folderlist = []
    for folder in os.listdir(directory1) :
        folderlist.append(folder)
    newfolderlist(directory2, folderlist)


def remove(path):
    # """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file

    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
        
    else:
        raise ValueError("file {} is not a file or dir.".format(path))


def copyfile(src_dir, dst_dir, src_file) :
    src = src_dir + '\\' + src_file
    dst = dst_dir
    shutil.copyfile(src, dst)
    print(f'{src_file} copied')
    
    
def copydir(src_dir, dst_dir) :
	shutil.copytree(src_dir, dst_dir)


def copydir_f(src_dir, dst_dir) :
	if os.path.isdir(dst_dir) :
		copy_tree(src_dir, dst_dir)
	
	elif os.path.isfile(dst_dir) :
		shutil.copyfile(src_dir, dst_dir)
		
	else :
		newfolder(dst_dir)
		copy_tree(src_dir, dst_dir)
	
	print(f'{src_dir} copied to {dst_dir}')


def move(src, dst) :
	shutil.move(src, dst)


def newfolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def newfolderlist(directory, folderlist):
    for i, names in enumerate(folderlist):
        directory_temp = directory + '\\' + names
        try:
            if not os.path.exists(directory_temp):
                os.makedirs(directory_temp)
        except OSError:
            print ('Error: Creating directory. ' +  directory_temp)



def deletestr(characters, string) :
    for x in range(len(characters)) :
        string = string.replace(characters[x], "")

    return string



def delun(df) :
    if 'Unnamed: 0' in df.columns :
        df.drop('Unnamed: 0', axis = 1, inplace = True)

    if 'Unnamed: 0.1' in df.columns :
        df.drop('Unnamed: 0.1', axis = 1, inplace = True)

    return df

