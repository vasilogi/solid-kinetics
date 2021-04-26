import os
import numpy as np
import pandas as pd
from modules.file_handlers import read_filtrated_datafile, get_data, read_units
from modules.regressors import MSE
import matplotlib.pyplot as plt
from modules.reaction_models import Model
from modules.arrhenius import rateConstant, conv2mass
import random

from scipy.optimize import curve_fit

# Local application imports
from modules.reaction_models import Model

modelNames = ["A2","A3","A4","D1","D2","D3","D4","F0","F1","F2","F3","P2","P3","P4","R2","R3"]

# DIRECTORIES
MAIN_DIR  = os.getcwd()                     # current working directory
DATA      = os.path.join(MAIN_DIR,'data')   # data directory
OUTPUT    = os.path.join(MAIN_DIR,'output') # output directory

# limit conversion fraction
low  = 0.05
high = 0.95

from modules.integral_regression import data2integralFit

data2integralFit(DATA,OUTPUT,modelNames,low,high)

