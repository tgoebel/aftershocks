#!/bin/python2.7
"""

--> 1) load ANSS (or comcat) seismicity data using np.genfromtxt
--> 2) select events within area of interest
       create aftershock time vector in days after MS
--> 3) compute aftershock decay rates and power-law fit
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
#------------my modules-----------------------
import src.seis_utils as seis_utils
import src.omori as omori
np.random.seed(123456)
#--------------------------0---------------------------------------------
#                     params, dirs, files
#------------------------------------------------------------------------
plot_file = 'plots/syn_aftershock.png'
dPar     =  {#synthetic data parameters
            'N'  : int(100), 'c_syn' : 0.1, 'p_syn' : 1.0,
            'addNoise' : True, 'sigma' : .5,
            'k'      : 5, # compute aftershock rates using moving sample window, k
             #MLE
             #parameter bounds:            c        K        p
            'a_limit'     : np.array( [[.02, 2],[1, 300],[.2,2.5]]),
            #initial values for inversion c0,K0,   p0
            'a_par0'       : np.array([ .22, 50, .95]),
            # time range for plotting and LS fit
           'tmin'  : .1, 'tmax' : 100,}
#==================================1================================================
#                            create random PL aftershock decay
#===================================================================================
a_tAS = omori.syn_tAS( dPar['c_syn'], dPar['p_syn'], dPar['tmin'], dPar['tmax'], dPar['N'])
if dPar['addNoise'] == True:
    a_tAS += np.random.randn( dPar['N'])*dPar['sigma']
a_tAS.sort()
dPar['K_syn'] = omori.K_from_N_and_dt(dPar['c_syn'], dPar['p_syn'], (a_tAS.max()-a_tAS.min()), dPar['N'])
#==================================3================================================
#                         power-law fitting (AS decay rate)
#===================================================================================
# compute rates
at_bin_AS, aN_bin_AS   = seis_utils.eqRate( a_tAS, dPar['k'])
####A##### power law fit - least squares
sel_t = np.logical_and( at_bin_AS >= dPar['tmin'], at_bin_AS <= dPar['tmax'])
# only events within tmin and tmax, do log-transformation
f_p_LS, f_K_LS, __, __, __ = scipy.stats.linregress( np.log10( at_bin_AS[sel_t]), np.log10( aN_bin_AS[sel_t]))
f_K_LS = 10**f_K_LS
print( 'Omori p-value (least-squares): ', round( f_p_LS,1), 'K', round( f_K_LS,1))
####A##### power law fit - least squares
# for dN/dt = K*t**(-p); remember that we're solving a problem analogous to:
#                         y_hat = alpha * t**beta, with alpha = 10**a and beta = b
####B##### power law fit - maximum likelihood
dOm      = omori.fit_omoriMLE( a_tAS[a_tAS>0],bounds=dPar['a_limit'], par0=dPar['a_par0'], disp = 0)
a_OMrate = omori.fct_omori( at_bin_AS, [dOm['c'], dOm['K'], dOm['p']])
print( 'Omori-fit, MLE', dOm)

#==================================4================================================
#                                 plots
#===================================================================================
plt.figure(1)
ax = plt.subplot()
ax.loglog( at_bin_AS, aN_bin_AS, 'ko', mfc = 'none', mew = 1.5, 
           label = '$N_{tot}$=%i, $c$= %.2f, $K$=%.2f $p$=%.2f'%(  a_tAS.shape[0],dPar['c_syn'],dPar['K_syn'], dPar['p_syn']))
ax.loglog( at_bin_AS, a_OMrate, 'r--', label = 'Omori: $c$= %.2f, $K$=%.2f,  $p$=%.2f'%( dOm['c'], dOm['K'], dOm['p']))
ax.legend( loc = 'upper right')
ax.set_xlabel( 'Time [day]')
ax.set_ylabel( 'events/day')
ax.set_xlim( dPar['tmin'], dPar['tmax'])
#plt.savefig( plot_file)
plt.show()











