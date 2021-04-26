# Standard library imports
import os

# Third party imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Local application imports
from modules.arrhenius import mass2conv
from modules.file_handlers import read_filtrated_datafile, read_units, get_data

# basic plot settings
graph_format = 'png'
graph_dpi    = 300
font_size    = 13
lwidth       = 3

def graph_experimental_data(DATA_DIR,OUTPUT_DIR):

    # concentration range
    low  = 0.0
    high = 1.0
    
    # get the csv data in a list
    Csvs = get_data(DATA_DIR)
    # filnames
    fnames = os.listdir(DATA_DIR)

    # create OUTPUT_DIR directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    fig  = plt.figure()
    # export a graph for the fitting of the integral reaction rate
    Plot = os.path.join(OUTPUT_DIR,'experimental_conversion.'+graph_format)
    
    for indx, Csv in enumerate(Csvs):
        df = pd.read_csv(Csv)
        # read data file
        conversion, time, temperature   = read_filtrated_datafile(df,low,high)
        # read variable units
        timeUnits, massUnits, tempUnits = read_units(df)
        plt.scatter(time,conversion,s=10,label=str(temperature)+tempUnits)
        # export experimental data
        data = {
            'time': time,
            'conversion': conversion
        }
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(OUTPUT_DIR,fnames[indx]),index=False)
    
    plt.xlim(0,)
    plt.ylim(0,1.0)
    plt.legend()
    plt.ylabel('conversion')
    plt.xlabel('time [' + timeUnits +']')
    plt.tight_layout()
    plt.savefig(Plot, format=graph_format, dpi=graph_dpi)
    plt.close() # to avoid memory warnings



# # export a graph for the fitting of the integral reaction rate
# Plot = os.path.join(GRAPH,modelname+'_a_vs_t.png')


# plt.plot(xdata, ydata, lw=lwidth, c=palette[0], label='Experimental rate')
# plt.plot(xdata ,yfit, lw=lwidth, c=palette[1], label='Fit '+r'$R^{2} = '+str(round(r_squared,3))+'$')
# plt.xlabel(r'$ t ('+time_units+') $')
# plt.ylabel(r'$ a $')
# plt.xlim(0.0,)
# plt.ylim(0.0,1.0)
# plt.legend()
# plt.tight_layout()
# plt.savefig(Plot, format=graph_format, dpi=graph_dpi)
# plt.close() # to avoid memory warnings