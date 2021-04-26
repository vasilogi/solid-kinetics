# Third party imports
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d
from scipy.integrate import odeint

# Local application imports
from modules.reaction_models import Model
from modules.gof import rSquared, MSE

def simpleLinearFit(time,k):
    return k*time

def EER(conversion,time,yfit):

    # calculate the mean square erron on the derivative of the experimental conversion
    # EER : Explicit Euler eRror

    # allocale residuals
    residuals = np.zeros(len(conversion))
    
    # in the bulk
    for i in range(len(conversion)-1):
        # experimental reaction rate
        u  = (conversion[i+1] - conversion[i])/( time[i+1] - time[i] )
        # computed reaction rate
        up = yfit[i]
        # calculate residual
        residuals[i] = u - up
    
    # on the boundary
    
    # experimental reaction rate
    u  = (conversion[-1] - conversion[-2])/( time[-1] - time[-2] )
    # computed reaction rate
    up = yfit[-1]
    # calculate residual
    residuals[-1] = u - up

    # calculate sum of square residuals
    ss_res = np.sum(residuals**2.0)

    # return mean square error
    return (1.0/len(conversion))*ss_res

def integralRateRegression(time,conversion,modelName):
    # perform Non-Linear Regression
    # fit the experimental integral rate conversion (g)
    # calculate the Arrhenius rate constant (k)

    # pick up the model
    model = Model(modelName)

    # define data
    x          = time
    y          = np.array([model.g(a) for a in conversion])

    # fit integral rate
    popt, pcov = curve_fit(simpleLinearFit,x,y)
    # popt: optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized.
    k          = popt[0]                 # Arrhenius rate constant
    yfit       = simpleLinearFit(time,k) # modeled integral reaction rate

    return k, yfit

def conversionRegression(time,conversion,modelName):
    # perform Non-Linear Regression
    # fit the experimental conversion (conversion)
    # calculate the Arrhenius rate constant (k)

    # pick up the model
    model = Model(modelName)

    # define data
    x     = time
    y     = conversion

    if modelName not in ['D2','D4']:
        # take estimation from the integral rate regression
        k_est, yfit = integralRateRegression(x,y,modelName)
        # fit conversion
        popt, pcov = curve_fit(model.alpha,x,y,p0=k_est)          # p0 : initial guess
        # popt: optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized.
        k          = popt[0]                                      # Arrhenius rate constant
        yfit       = np.array([model.alpha(t, k) for t in time])  # modeled conversion fraction
    else:
        # measure the mean square error on the linear integral rate
        k, yfit = integralRateRegression(time,conversion,modelName)

    return k, yfit

def differentialRateRegression(time,conversion,modelName):
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

    # pick up the model
    model = Model(modelName)

    # define data
    x          = time
    y          = conversion

    # take estimation from the integral rate regression
    k_est, yfit = integralRateRegression(x,y,modelName)
    # fit ODE
    popt, pcov = curve_fit(RHS, x, y, p0=k_est)                    # p0 : initial guess
    # popt: optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized.
    k          = popt[0]                                           # Arrhenius rate constant from fitting

    if modelName not in ['D2','D4']:
        yfit = np.array([model.alpha(t, k) for t in time])  # modeled conversion fraction
    else:
        # measure the mean square error on the linear integral rate
        y    = np.array( [model.g(a) for a in conversion] ) # experimental integral rate
        yfit = k*time                                       # modeled integral rate

    return k, yfit

def comprehensiveRegressor(time,conversion,models):
    # arguments
    # time       (numpy array)
    # conversion (numpy array)
    # models (list of strings)

    # returns
    # a dataframe containg the fitting information

    rate_constant_alpha    = []
    rate_constant_integral = []
    rate_constant_differen = []
    mse_coef_alpha         = []
    mse_coef_integral      = []
    mse_constant_differen  = []

    # loop over the models
    for modelIndx, modelName in enumerate(models):

        # integral rate regression
        k_integral, mse_integral = integralRateRegression(time, conversion, modelName)

        rate_constant_integral.append(k_integral)
        mse_coef_integral.append(mse_integral)

        # conversion regression
        k_alpha, mse_alpha = conversionRegression(time, conversion, modelName, k_integral)
        
        rate_constant_alpha.append(abs(k_alpha)) # bug: RuntimeWarning: invalid value encountered in sqrt
        mse_coef_alpha.append(mse_alpha)

        # differential rate regression
        k_differen, mse_differen = differentialRateRegression(time, conversion, modelName, k_integral)

        rate_constant_differen.append(abs(k_differen)) # bug: RuntimeWarning: invalid value encountered in sqrt
        mse_constant_differen.append(mse_differen)
    
    # pass the data to a dictionary
    data = {'model': models,
            'rate_constant - alpha': rate_constant_alpha,
            'rate_constant - integral': rate_constant_integral,
            'rate_constant - differential': rate_constant_differen,
            'MSE - alpha': mse_coef_alpha,
            'MSE - integral': mse_coef_integral,
            'MSE - differential': mse_constant_differen}

    # dictionary to dataframe
    df = pd.DataFrame(data)

    return df


def arrheniusEnthalpy(rate_constant,temperature):
    # arguments 
    # rate_constant (numpy array)
    # temperature (numpy array)

    # returns
    # the activation enthalpy (Ea) in a list like [activation enthalpy, mean square error]
    # and the frequency factor (A)
    # and the prediction

    # gas constant 
    R = 8.31446261815324 # J K-1 mol-1

    x = 1.0/temperature
    y = np.log(rate_constant)

    x = x.reshape((-1, 1))

    # linear regression for the logarithmic arrenius equation
    regr   = LinearRegression()
    regr.fit(x, y)
    y_pred = regr.predict(x)

    Ea  = -regr.coef_[0]*R # in J mol-1
    A   = np.exp(regr.intercept_)
    MSE = mean_squared_error(y, y_pred)
    R2   = r2_score(y_pred , y)

    return [Ea,MSE], A

def isocEnthalpy(time,temperature):
    # arguments 
    # time (numpy array)
    # temperature (numpy array)

    # returns
    # the activation enthalpy (Ea) in a list like [activation enthalpy, mean square error]
    # and the ln[g(a)A] factor

    # gas constant 
    R = 8.31446261815324 # J K-1 mol-1

    x = 1.0/temperature
    y = np.log(time)

    x = x.reshape((-1, 1))

    # linear regression for the logarithmic arrenius equation
    regr   = LinearRegression()
    regr.fit(x, y)
    y_pred = regr.predict(x)

    Ea  = regr.coef_[0]*R*1.0e-3 # in kJ mol-1
    gA  = regr.intercept_
    MSE = mean_squared_error(y, y_pred)
    R2   = r2_score(y_pred , y)

    return [Ea,MSE], gA

def interpolateTime(time,conversion,interConversion):
    # arguments
    # time (numpy array)
    # conversion (numpy array)
    # npoints (integer) - the number of the interpolation points
    
    # returns
    # the time for the interpolated points
    y               = time
    x               = conversion
    interpol        = interp1d(x,y,kind='nearest',fill_value="extrapolate")
    return interpol(interConversion)