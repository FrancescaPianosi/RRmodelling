"""
    Module to calculate model performance metrics  
    Author: Francesca Pianosi, francesca.pianosi@bristol.ac.uk

"""

import numpy as np

from scipy.stats import pearsonr

def CHECK_DIM(y_sim, y_obs):
    
    Nsim = y_sim.shape
    if len(Nsim) > 1: # multiple time series
        N = Nsim[0] # number of rows (no of time series)
        T = Nsim[1] # number of columns (no of time steps)
    elif len(Nsim) == 1: # only one time series
        T = Nsim[0] # number of time steps
        N = 1
 
    Nobs = y_obs.shape
    err_message = ""
    if len(Nobs) > 1:
        if Nobs[0] != 1:
             err_message = '"y_obs" be of shape (T, )' 
        if Nobs[1] != T:
             err_message = 'the number of elements in "y_obs" must be equal to the number of columns in "y_sim"'
    elif len(Nobs) == 1:
        if Nobs[0] != T:
             err_message = 'the number of elements in "y_obs" must be equal to the number of columns in "y_sim"'
        
    return N, T, err_message 

def CORR_ERR(y_sim, y_obs):

    """

     ce = CORR_ERR(Y_sim,y_obs)
     
     calculate the performance metric:
     
     ce = ( r - 1 )^2     
     where r = Pearson correlation between y_sim and y_obs
        
     Usage:
     Y_sim = time series of modelled variable             - numpy.ndarray (T, )
            (N > 1 different time series can be        or - numpy.ndarray (N,T)
            evaluated at once)
     y_obs = time series of observed variable              - numpy.ndarray (T, )

     ce = correlation error                               - scalar 
                                                       or - numpy.ndarray (N, )
    
    """

    N, T, err_message = CHECK_DIM(y_sim, y_obs)
     
    if len(err_message)>0:
        raise ValueError(err_message)
        raise ValueError(err_message)

    if N==1: 
        corr, _ = pearsonr(y_obs,y_sim)
    if N>1:  
        corr=np.zeros((N,))
        for n in range(N):
            corr[n],_= pearsonr(y_obs,y_sim[n,:])
    
    corr_err = (corr-1)**2
    return corr_err


def STD_ERR(y_sim, y_obs):

    """

     se = STD_ERR(Y_sim,y_obs)
     
     calculate the performance metric:
     
     se = ( S_sim/S_obs - 1 )^2     
     where S_sim = standard deviation of y_sim 
           S_obs = standard deviation of y_obs
        
     Usage:
     Y_sim = time series of modelled variable             - numpy.ndarray (T, )
            (N > 1 different time series can be        or - numpy.ndarray (N,T)
            evaluated at once)
     y_obs = time series of observed variable             - numpy.ndarray (T, )

     se = standard deviation error                        - scalar 
                                                       or - numpy.ndarray (N, )
    
    """

    N, T, err_message = CHECK_DIM(y_sim, y_obs)
    
    if len(err_message)>0:
        raise ValueError(err_message)
    
    std_obs = np.std(y_obs)
    if N==1:
        std_sim=np.std(y_sim)
    elif N>=1:
        std_sim = np.std(y_sim,axis=1)
    std_err = (std_sim/std_obs-1)**2
    
    return std_err

def MEAN_ERR(y_sim, y_obs):
 
    """

     me = SMEAN_ERR(Y_sim,y_obs)
     
     calculate the performance metric:
     
     me = ( M_sim/M_obs - 1 )^2     
     where M_sim = mean of y_sim 
           M_obs = mean of y_obs
        
     Usage:
     Y_sim = time series of modelled variable             - numpy.ndarray (T, )
            (N > 1 different time series can be        or - numpy.ndarray (N,T)
            evaluated at once)
     y_obs = time series of observed variable             - numpy.ndarray (T, )

     me = mean error                                      - scalar 
                                                       or - numpy.ndarray (N, )
    
    """
    
    N, T, err_message = CHECK_DIM(y_sim, y_obs)

    if len(err_message)>0:
        raise ValueError(err_message)
    
    mean_obs = np.mean(y_obs)
    if N==1:
        mean_sim=np.mean(y_sim)
    elif N>=1:
        mean_sim = np.mean(y_sim,axis=1)
    mean_err = (mean_sim/mean_obs-1)**2
    
    return mean_err
  

