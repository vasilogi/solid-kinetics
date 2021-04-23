# import matplotlib.pyplot as plt
# import pandas as pd
import os
from modules.graphos import graph_experimental_data

# DIRECTORIES
MAIN_DIR  = os.getcwd()                     # current working directory
DATA      = os.path.join(MAIN_DIR,'data')   # data directory
OUTPUT    = os.path.join(MAIN_DIR,'output') # output directory

graph_experimental_data(DATA,OUTPUT)