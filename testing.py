# import matplotlib.pyplot as plt
# import pandas as pd
import os
import pandas as pd
from modules.file_handlers import read_filtrated_datafile, get_data
import matplotlib.pyplot as plt

# DIRECTORIES
MAIN_DIR  = os.getcwd()                     # current working directory
DATA      = os.path.join(MAIN_DIR,'data')   # data directory
OUTPUT    = os.path.join(MAIN_DIR,'output') # output directory

import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from modules.reaction_models import Model

def differentialRateRegression(time,conversion,model,k_est):
    # perform Non-Linear Regression
    # fit the experimental differential rate conversion (f)
    # calculate the Arrhenius rate constant (k)

    # k_est: estimation for the Arrhenius constant

    # ODE: da/dt = k f(a)
    def RHS(t, k):
        'Function that returns Ca computed from an ODE for a k'
        def ODE(a, t):
            return k * model.f(a)

        u0          = conversion[0]
        u_numerical = odeint(ODE, u0, t)
        return u_numerical[:,0]

    x          = time
    y          = conversion
    popt, pcov = curve_fit(RHS, x, y, p0=k_est) # p0 : initial guess
    # popt: optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized.
    k          = popt[0]                                        # Arrhenius rate constant
    # yfit       = np.array([simpleLinearFit(t,k) for t in time]) # simulated conversion fraction

    # calculate the determination coefficient
    # r_squared = rSquared(y,yfit)

    return k

# limit conversion fraction
low  = 0.05
high = 0.95

Csvs = get_data(DATA)
Csv = Csvs[0]
df = pd.read_csv(Csv)
conversion, time, temperature = read_filtrated_datafile(df,low,high)

# # ODE: da/dt = k f(a)
# def RHS(t, k):
#     'Function that returns Ca computed from an ODE for a k'
#     def ODE(a, t):
#         return k * model.f(a)

#     u0          = conversion[0]
#     u_numerical = odeint(ODE, u0, t)
#     return u_numerical[:,0]

# model = Model('A2')
# popt, pcov = curve_fit(RHS, time, conversion, p0=0.013) # p0 : initial guess
# k = popt[0]

# 0.013110967942467014

model = Model('A2')
k = differentialRateRegression(time, conversion, model, 0.013 )
tfit = np.linspace(time[0],time[-1])
fit = [model.alpha(t, k) for t in tfit]
# RHS(tfit, k_fit)

plt.plot(time, conversion, 'ro', label='data')
plt.plot(tfit, fit, 'b-', label='fit')
plt.legend(loc='best')
plt.show()


