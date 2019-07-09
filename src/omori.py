#!usr/lib/python
"""
estimate Omori-Utsu relation parameters
using MLE and least squares

see e.g. Ogata 1999, Seismicity analysis poitn process

"""
from __future__ import division # include true division 

import numpy as np
import scipy.optimize

#====================================1===========================================
#                   methods to fit aftershock decay
#================================================================================    

def fit_omoriMLE( a_tAS, **kwargs):
    """
    determine omori parameters from constrained optimization of likelihood fct. (from Ogata 1999).
    default: 'SLSQP' ,    -  Sequential Least SQuares Programming
    This function uses the negative log likelihood 
    of observed aftershock time for the parameters k, c and p in the modified
    Omori law
    
    Reference: Ogata, Estimation of the parameters in the modified Omori formula 
    for aftershock sequences by  the maximum likelihood procedure, J. Phys. Earth, 1983
    (equation 6), see also Ogata 1999 equation 10
    
    :input  
              a_tAS    : delta_t aftershocks in days relative to mainshock
              
            kwargs:
              'bounds' : np.array([[cmin,cmax],[Kmin,Kmax],[pmin,pmax]])
                         default:
                         np.array([[1e-2,   2],[   5, 1e3],[  .2,  2]])
               'par0'  : np.array([c0,K0,p0])#starting point for each par.
                         default:
                         np.array([.5,50,1.1]) #p0 should not be 1!
               'method': 'TNC',       - Newton (TNC) algorithm (issue with convergence) 
                         'L-BFGS-B',  - Limited Memory Algorithm for Bound Constrained Optimization, #
                                        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html#scipy.optimize.fmin_l_bfgs_b      
                         'SLSQP' ,    -  Sequential Least SQuares Programming
                         default: SLSQP
               'disp'  : set to 1 for debugging
               
    :return   [fC, fK ,fP, fL]
              fC    -     c-value - short time incompleteness
              fK    -     productivity of aftershock sequence
              fP    -     p value - rate of decay exponent
              fL    -     Maximum likelihood
    adapted from J. Woessner, zmap
    07/14/2018
    """
    disp = 0
    method = 'SLSQP'
    aBounds = np.array([[1e-2,   2],[   2, 1e3],[  .2,  2]])
    aPar0   = np.array([.5, 50, 1.1])
    #------------------------optional arguments------------------------------------------------------ 
    if 'bounds' in kwargs.keys() and kwargs['bounds'] is not None:
        aBounds = kwargs['bounds']
        if len( aBounds) != 3 or len( aBounds.T) != 2:
            error_str = 'bounds input in kwargs has wrong dimension should be 3x2', aBounds
            raise ValueError, error_str
    if 'par0' in kwargs.keys() and kwargs['par0'] is not None:
        aPar0 = kwargs['par0']
        if len( aPar0) != 3:
            error_str = 'par0 input in kwargs has wrong dimension should be 3x1', par0
            raise ValueError, error_str
    if 'disp' in kwargs.keys() and kwargs['disp'] is not None:
        disp = kwargs['disp']
    if 'method' in kwargs.keys() and kwargs['method'] is not None:
        method = kwargs['method']
        print 'method', method
    #---------------------maximize likelihood fct.--------------------------
    objFunc = lambda X: ogata_logL( a_tAS, X )
    # maximize av-log-likelihood
    #if method == 'interior-point':
    #    dPar_fit = scipy.optimize.linprog( objFunc, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=aBounds, method='interior-point', 
    #                                      options={'alpha0': 0.99995, 'beta': 0.1, 'maxiter': 500, 'disp': disp, 'tol': 1e-04})
    #else:
    dPar_fit = scipy.optimize.minimize( objFunc, aPar0, bounds=aBounds, 
                                       tol = 1e-4, method=method,options={'disp': disp, 'maxiter':500})
    if dPar_fit['success'] != True:
        error_str = 'ML solution did not converge, ', dPar_fit
        raise ValueError, error_str
    dOm         = { 'p' : dPar_fit['x'][2], 'K' : dPar_fit['x'][1], 'c' : dPar_fit['x'][0]}
    dOm['logL'] = ogata_logL( a_tAS,  dPar_fit['x'])
    return dOm

def fct_omori(  x, a_par, **kwargs): 
    """ input:  x - vector
                p - vector with parameters
        output: f(x) - results of model values """
    c,K,p = a_par[0], a_par[1], a_par[2]
    return K/( (c + x)**p)  

#---------------------------------------------------------------------------------
#              likelihood function
#---------------------------------------------------------------------------------
def ogata_logL( aT_AS, vParams, **kwargs):
    """ return likelihood function of Omori-Utsu
    see: Ogata, Yosihiko. "Seismicity analysis through point-process modeling: A review." 
    Pure and Applied Geophysics 155.2-4 (1999): 471-507.  Equ. 10
    
    input:    - vParams  - initial guess of parameters  --> in alphabetical order  c,K,p = vParams[0],vParams[1],vParams[2]
              - vOriTime - vector with observed times; important this vector already starts at
                           time of completeness of seismic record
    return:   - average log likelihood of observations
    """
    c,K,p = vParams[0],vParams[1],vParams[2]
    S,T   = aT_AS.min(), aT_AS.max()
    if abs(p - 1) < 1e-8: #for p close to unity
        f_Lcp = np.log( T+c) - np.log( S + c)
    else:#fAcp = ((fTend+c).^(1-p)-(fTstart+c).^(1-p))./(1-p);
        f_Lcp = ( ( T + c)**(1-p) - (S+c)**(1-p)) / (1-p)
    N = aT_AS.shape[0]
    ##negative to use minization algorithm
    fLogL = N*np.log( K) - (p*(np.log( aT_AS+c)).sum()) - K*f_Lcp
    return -fLogL


def K_from_N_and_dt( c, p, dt, N):
    """
    - use omori parameters c and p to calculate productivity K
    - use the cumulative distribution function from Ogata, 1999 and solve for K in a given time window dt
    INPUT: p  - omori p
           c  - omori c
           dt - time window of simulation (tmax - t0)
           N  - total number of simulated events within dt
    OUTPUT:
           K - omori K, aftershock productivity value
    """
    if p != 1:
        K = N*(1-p)/( ( dt+c)**(1-p) - c**(1-p))
    else: #p==1
        K = N/( np.log( dt+c) - np.log( c))
    return K

def integrateOmori( c, K, p, dt):
    """
    - use omori parameters c ,K, p and compute the total number of events
    - use the cumulative distribution function from Ogata, 1999
    INPUT: p  - omori p
           c  - omori c
           K  = omori K
           dt - time window of simulation (tmax - t0)
    OUTPUT:
           N - total number of events in Omori aftershock sequence
    """
    if p != 1:
        N = K/(1-p) * ( ( dt+c)**(1-p) - c**(1-p) )
    else: #p==1
        N = K*( np.log( dt+c) - np.log( c))
    return N

#====================================2===========================================
#                   creates synthetic aftershock times
#================================================================================
def syn_tAS( c, p, tmin,tmax, N):
    """
    Felzer et al. 2002, Triggering of 1999 Mw 7.1 Hector Mine earthquake
    - define create power-law distributed aftershock time vector between tmin and tmax
    - tmin can be = 0, Omori-K parameter is a fct. of all four parameter (tmax-tmin), p, and N
    INPUT:  c, p       - omori parameters describing time shift for complete recording and rate decay exponent
                       - in alphabetical order  
           tmin, tmax  - time window for aftershock catalog
           N           - total number of aftershocks
    """
    vRand = np.random.random_sample( N)
    #===========================================================================
    #          case1:  p != 1
    #===========================================================================    
    #if p != 1.0: #abs(p - 1) < 1e-6:
    p += 1e-5 # this will make it unlikely for p to be exactly 1
    a1 = (tmax + c)**(1-p)
    a2 = (tmin + c)**(1-p)
    a3 = vRand*a1 + (1-vRand)*a2#     
    vt_AS = a3**(1/(1-p))-c
#     else: # p == 1
#         a1 = np.log( tmax + c)
#         a2 = np.log( tmin + c)
#         a3 = vRand*a1 + (1-vRand)*a2
#         vt_AS = np.exp( a3) - c
    vt_AS.sort()
    return vt_AS


