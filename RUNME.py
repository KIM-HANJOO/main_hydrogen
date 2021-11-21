import os
import sys


import model_library as lib
import ClusterAnalysis as ca

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
cwdir = os.getcwd()
import sys
sys.path.append(cwdir + '\\module')
import table as tb

import math
import shutil
from sklearn.cluster import KMeans
from sklearn.metrics import *
from distutils.dir_util import copy_tree
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from validclust import dunn
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import k_means
from sklearn import metrics
import matplotlib.cm as cm
import numpy as np
from matplotlib import gridspec
from sklearn.cluster import KMeans
from sklearn.cluster import k_means
from sklearn.metrics import silhouette_samples, silhouette_score

from statsmodels.multivariate.manova import MANOVA

from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgunbd.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
