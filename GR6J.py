"""
    Module to simulate the GR6J model    
    [note: code only compatible with Python 3]

    References for the GRR6J model (and previous versions GR4J, GR5J):
    
    Perrin et al. (2003) Improvement of a parsimonious model for streamflow
         Simulation, Journal of Hydrology, 279(1-4), 275-289
    Pushpalatha et al. (2011) A downward structural sensitivity analysis of 
          hydrological models to improve low-flow simulation, Journal of Hydrology,
          11(1-2), 66-76, doi:10.1016/j.jhydrol.2011.09.034
    https://webgr.inrae.fr/en/models/daily-hydrological-model-gr4j/description-of-the-gr4j-model/
    https://search.r-project.org/CRAN/refmans/airGR/html/RunModel_GR6J.html
    
    This code developed by: francesca.pianosi@bristol.ac.uk
    distributed under GPL 3.0 licence: https://www.gnu.org/licenses/gpl-3.0.html
    


"""
from __future__ import division, absolute_import, print_function

import numpy as np

from numpy.matlib import repmat

from numba import jit # the function jit allows to compile the code and reduce the running time

@jit
def gr6j_sim(param, P, PE, ini):

    """This function simulates the GR6J rainfall-runoff model 

    Usage:
        Q_sim, STATES, FLUXES = gr6j.gr6j_sim(param, P, PE, ini)

    Input:
      param = vector of model parameters                   - numpy.ndarray(6, )
          P = time series of precipitation                 - numpy.ndarray(T, )
         PE = time series of potential evapotranspiration  - numpy.ndarray(T, )
        ini = vector of initial conditions.                - numpy.ndarray(3, )
 
    Output:
      Q_sim = time series of simulated flow (mm/Dt)        - numpy.ndarray(T, )
     STATES = time series of simulated storages (mm)       - numpy.ndarray(T+1,3)
              1: Level of production store (soil moisture)
              2: Level of routing store
              3: Level of exponential store (for GR6J only)
     FLUXES = time series of simulated fluxes (mm/Dt)      - numpy.ndarray(T,11)
              1: Net rainfall
              2: Saturation flow 
              3: Net evapotranspiration capacity
              4: Actual Evapotranspiration
              5: Percolation
              6: Recharge (outflow from production store)
              7: Flow routed through Unit Hydrograph 1 (fast)
              8: Flow routed through Unit Hydrograph 2 (slow)
              9: Catchment water exchange flow
             10: Outflow from routing store
             11: Outflow from exponential store (only used by GR6J)
             
    """      

    #  ---------------------------
    # Recover model parameters
    #  ---------------------------
    x1 = param[0] # production store capacity [mm]
    x2 = param[1] # intercatchment exchange coefficient [mm/Dt]
    x3 = param[2] # routing store capacity [mm]
    x4 = param[3] # unit hydrograph time constant [Dt]
    x5 = param[4] # intercatchment exchange threshold [-]
    x6 = param[5] # exponential store depletion coefficient [mm]

    #  ---------------------------
    # Initialise storage and flux time series
    #  ---------------------------
    
    T   = len(PE)           # number of time steps
    
    Pn  = np.zeros((T, ))   # Net rainfall [mm/Dt]
    En  = np.zeros((T, ))   # Net evapotranspiration capacity [mm/Dt]
    Ps  = np.zeros((T, ))   # Saturation flow [mm/Dt]
    Es  = np.zeros((T, ))   # Actual Evapotranspiration [mm/Dt]
    S   = np.zeros((T+1, )) # Soil Moisture (state of the production storage) [mm]
    Perc= np.zeros((T, ))   # Percolation [mm/Dt]
    Pr  = np.zeros((T, ))   # Recharge (outflow from production store) [mm/Dt]
    
    Q9  = np.zeros((T, ))   # Flow routed through Unit Hydrograph 1 (fast) [mm/Dt]
    Q1  = np.zeros((T, ))   # Flow routed through Unit Hydrograph 2 (slow) [mm/Dt]
    F   = np.zeros((T, ))   # Catchment water exchange flow [mm/Dt]
     
    R1  = np.zeros((T+1, )) # Storage of routing store [mm]
    Qr  = np.zeros((T, ))   # Outflow from routing store [mm/Dt]
    
    R2  = np.zeros((T+1, )) # Storage of exponential store [mm] (only used by GR6J, not in previous model versions)
    Qr2 = np.zeros((T, ))   # Outflow from exponential store [mm/Dt] (only used by GR6J)
    
    Q = np.zeros((T, ))     # Simulated flow
    
    #  ---------------------------
    # Assign initial states
    #  ---------------------------

    S[0]  = ini[0]
    R1[0] = ini[1]
    R2[0] = ini[2]
 
    #  ---------------------------
    # Data for Unit Hydrographs UH1 and UH2
    #  ---------------------------
    # UH ordinates will only depend on the simulation time step
    # and the value of x4, so we can calculate them once and 
    # for all before we start the simulation         
    
    uh_coeff_1 = uh_bell(x4,'half')
    uh_coeff_2 = uh_bell(2*x4,'full')
    uh_m1 = len(uh_coeff_1)
    uh_m2 = len(uh_coeff_2)
    uh_storage_1 = np.zeros((uh_m1,))
    uh_storage_2 = np.zeros((uh_m2,))

    #  ---------------------------
    # Run model simulation
    #  ---------------------------

    for t in range(T):

        #  ---------------------------
        # 1 - Soil moisture accounting
        #  ---------------------------

        # 1.1 - calculate water fluxes at land surface interface:
        Pn[t] = max(P[t]-PE[t],0)              # Net rainfall
        En[t] = max(0,PE[t]-P[t])              # Net evapotranspiration capacity
        Ps[t] =   x1*( 1 - (S[t]/x1)**2 )*np.tanh( Pn[t]/x1 ) / ( 1 + S[t]/x1*np.tanh( Pn[t]/x1 ) )   # Saturation flow 
        Es[t] = S[t]*( 2 - S[t]/x1 )*np.tanh( En[t]/x1 ) / ( 1 + ( 1 - S[t]/x1 )*np.tanh( En[t]/x1 )) # Actual evaporation
        
        # 1.2 - update state (S) of the production store (i.e. soil water content)
        S[t+1] = S[t] - Es[t] + Ps[t]
        
        # 1.3 - calculate percolation and update again production store
        Perc[t] = S[t]*( 1 - ( 1 + (4/9*S[t]/x1)**4 )**(-1/4) )
        S[t+1]  = S[t+1] - Perc[t]       

        # 1.4 - calculate total outflow (Pr) from the production store 
        Pr[t]   = Perc[t] + Pn[t] - Ps[t]
 
        #  ---------------------------
        # 2 - Flow routing
        #  ---------------------------
        
        # 2.1 - propagate outflow from production storage (Pr) through the two unit hydrographs:
        Q9[t] = 0.9*Pr[t]*uh_coeff_1[0] +uh_storage_1[0]
        Q1[t] = 0.1*Pr[t]*uh_coeff_2[0] +uh_storage_2[0]
        # Update unit hydrography routing storages:
        uh_storage_1 = uh_storage_update(uh_storage_1,uh_coeff_1,0.9*Pr[t])
        uh_storage_2 = uh_storage_update(uh_storage_2,uh_coeff_2,0.1*Pr[t])

        # 2.2 - calculate catchment water exchange (F) 
        # This is the flux that is gained/lost through each routing pathway
        # the way this is calculated is the key difference between GR4J
        # and subsequent versions of the model, so I have noted here 
        # as a comment the different versions      
        # For GR4J:
        # F[t] = x2*(R1[t]/x3)**(3.5) # Eq (1) in new paper (Eq 18 in old paper has a typo!)      
        # For GR5J and GR6J:
        F[t] = x2*(R1[t]/x3-x5)       # Eq (2) in new paper
       
        # 2.3 (for GR6J only!) - update state (R2) of an additional exponential store
        SC      = 0.4  # (I think from text before Fig 5 in new paper)
        R2[t+1] = max( 0, R2[t] + SC*Q9[t] + F[t] ) # I suppose
        Qr2[t]  = np.exp( - R2[t]/x6 ) # unclear if this may be the right equation...
        Q9[t]   = (1-SC)*Q9[t]      # overwrite what goes into the routing store!

        # 2.4 - update state (R1) of the routing store
        R1[t+1] = max( 0, R1[t] + Q9[t] + F[t] )
        
        # 2.5 - calculate outflow (Qr) from routing store and update again routing store:
        Qr[t]   = R1[t]*( 1 - ( 1+(R1[t]/x3)**4  )**(-1/4) )
        R1[t+1] = R1[t+1] - Qr[t]

        # 2.6 - apply water exchange to Q1 and calculate total streamflow (Q)
        Q[t] = Qr[t] + max(0,Q1[t]+F[t])
        
        # 2.7 (for GR6J only!) - also add flow from the exponential store
        Q[t] = Q[t] + Qr2[t] 

    # end of simulation loop
    
    #  ---------------------------
    # Return simulation results
    #  ---------------------------
 
    STATES = np.column_stack((S, R1, R2)) 
    FLUXES = np.column_stack((Pn,Ps,En,Es,Perc,Pr,Q9,Q1,F,Qr,Qr2))

    return Q, STATES, FLUXES
    
@jit
def uh_bell(x,type):     # NEEDS CHECKING AS IT MAY BE INCONSISTENT WITH WOUTER's!
    # Calculate the ordinates of a Unit Hydrograph with either a half or a full bell shape.
    # Unit Hydrograph will then be used to spread an input volume over a time period of
    # either x (half) or 2*x (full) days
    #
    # Usage:
    # uh_bell(x,type)
    #
    # Inputs:
    # x  - routing delay [Dt]          - integer
    # half - curve type                - string
    #        can take two values: 'half','full'
    # Output:
    # UH - unit hydrograph ordinates [-]  - numpy.ndarray(m, )
    #      where m = ceil(x) if type = 'half' 
    #         or m = 2*ceil(x) if type = 'full'
    #
    # Example: 
    # x = 3.8 [days]
    # UH = uh_1_half(x,'half')
    # returns:  
    #   UH(1) = 0.04
    #   UH(2) = 0.17
    #   UH(3) = 0.35
    #   UH(4) = 0.45
    # UH = uh_1_half(x,'full')
    # returns:  
    #   UH(1) = 0.02
    #   UH(2) = 0.08
    #   UH(3) = 0.18
    #   UH(4) = 0.29
    #   UH(5) = 0.24
    #   UH(6) = 0.14
    #   UH(7) = 0.05
    #   UH(8) = 0.00

    n = int(np.ceil(x))
    if n == 0:
        n = 1
    
    if type == 'half':
        SH = np.zeros((n+1, ))
        UH = np.zeros((n, ))
    
        for t in range(n):
            SH[t] = (t/n)**(5/2)
        SH[n] = 1

        for t in range(n):
            UH[t]=SH[t+1]-SH[t]

    elif type == 'full':
        
         SH = np.zeros((2*n+1, ))
         UH = np.zeros((2*n, ))
    
         for t in range(2*n):
            if t<=n:
                SH[t] = 0.5*(t/n)**(5/2)
            else:
                SH[t]=1-0.5*(2-t/n)**(5/2)
         SH[2*n] = 1

         for t in range(2*n):
             UH[t]=SH[t+1]-SH[t]

    return UH

@jit
def uh_storage_update(uh_storage,uh_coeff,Qin):
    # Update the states of the Unit Hydrograph routing storages

    uh_storage = uh_storage + Qin*uh_coeff
    m = len(uh_storage)
    for i in range(m-1):
        uh_storage[i] = uh_storage[i+1]
    uh_storage[m-1] = 0

    return uh_storage


