# PACKAGES
import os
import numpy as np
from modules.reaction_models import Model
from modules.arrhenius import rateConstant, conv2mass
import pandas as pd
import random

# convFlag = True if you are about to generate conversion data
#          = False if not
convFlag = False

# DIRECTORIES
MAIN_DIR  = os.getcwd()                   # current working directory
DATA      = os.path.join(MAIN_DIR,'data') # data directory

# create DATA directory
if not os.path.exists(DATA):
    os.makedirs(DATA)

# models names supported in this software
modelNames = ["A2","A3","A4","D1","D2","D3","D4","F0","F1","F2","F3","P2","P3","P4","R2","R3"]

# pick up a model
model = Model('A2')

# physical properties
enthalpy    = 147.8e+3 # J mol-1
frequency   = 1.75e+22
temperature = np.linspace(320.0,330.0,3)

# random number seed for generating data noise
random.seed(10)

# generate a kinetics (m,t) or (conversion,t) data set for each temperature
for T in temperature:

    # determine an Arrhenius rate constant
    k         = rateConstant(frequency,enthalpy,T)
    
    # calculate the start end end time for the sample data
    alpha     = np.linspace(0.05,0.95,100)
    startTime = model.g(alpha[0])/k
    endTime   = model.g(alpha[-1])/k
    time      = np.linspace(startTime,endTime,100)

    # calculate the conversion based on the chosen model
    conversion = np.array([ model.alpha(t,k)+random.uniform(0.01, 0.04) for t in time])

    # assume initial and final mass
    m0, minf = 10.0, 8.2
    
    # convert converstion fraction to mass
    mass = np.array([conv2mass(m0,minf,alpha) for alpha in conversion])

    # data to dictionary
    if convFlag:
        data = {'mass': mass, 'mass units': 'mg', 'conversion': conversion, 'time': time, 'time units': 'min', 'temperature': T, 'temperature units' : 'Kelvin'}
    else:
        data = {'mass': mass, 'mass units': 'mg', 'time': time, 'time units': 'min', 'temperature': T, 'temperature units' : 'Kelvin'}
    # dictionary to dataframe
    df = pd.DataFrame(data)
    # CSV filename
    Csv = os.path.join(DATA,str(T)+'_Kelvin.csv')
    # dataframe to csv
    df.to_csv(Csv,index=False)