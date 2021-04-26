# goodness of fit measures

# Third party imports
import numpy as np

def ssRes(y,yfit):
    # calculate the sum of square residuals (ideal = 0)
    residuals = y - yfit
    ss_res    = np.sum(residuals**2.0)
    return ss_res

def rSquared(y,yfit):
    # calculate the determination coefficient for linear fits (ideal = 1)
    residuals = y - yfit
    ss_res    = np.sum(residuals**2.0)
    ss_tot    = np.sum((y-np.mean(y))**2.0)
    return 1.0 - (ss_res / ss_tot)

def MSE(y,yfit):
    # calculate the mean square error (ideal = 0)
    residuals = y - yfit
    ss_res    = np.sum(residuals**2.0)
    return (1.0/len(y))*ss_res

def resEr(y,yfit):
    # calculate the residuals error (ideal = 0)
    residuals = y - yfit
    return np.sum(residuals)

def resAEr(y,yfit):
    # calculate the residuals absolute error (ideal = 0)
    residuals = np.abs(y - yfit)
    return np.sum(residuals)

def resREr(y,yfit):
    # calculate the residuals relative error (ideal = 0)
    residuals = np.abs(y - yfit)/np.abs(y)
    return np.sum(residuals)