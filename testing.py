# import matplotlib.pyplot as plt
# import pandas as pd
import os
from modules.graphos import graph_experimental_data

# basic plot settings
graph_format = 'png'
graph_dpi    = 300
font_size    = 13
lwidth       = 3
palette      = ['#161925','#ba1f33','#1412ad'] # eerie black, cardinal, zaffree

# DIRECTORIES
MAIN_DIR  = os.getcwd()                     # current working directory
DATA      = os.path.join(MAIN_DIR,'data')   # data directory
OUTPUT    = os.path.join(MAIN_DIR,'output') # output directory

graph_experimental_data(DATA,OUTPUT)