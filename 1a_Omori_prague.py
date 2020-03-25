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
#--------------------------0---------------------------------------------
#                     params, dirs, files
#------------------------------------------------------------------------
dir_file = 'data/prague_aftershock_clean.txt'
plot_file = 'plots/prague_aftershock.png'
dPar     =  { 'k'    : 5, # used for seis. rate estimate in moving sample window
                          # larger k result in smoother seismicity rates
             #MLE
             #parameter bounds:            c        K        p
            'a_limit'     : np.array( [[.02, 2],[1, 300],[.2,2.5]]),
            #initial values for inversion c0,K0,   p0
            'a_par0'       : np.array([ .22, 50, .95]),
            # time range for plotting and LS fit
           'tmin'  : 1, 'tmax' : 100,}
#==================================1================================================
#                            load data
#===================================================================================
# import 'clean' file, first 10 columns, 6 time columns, 3 loc, 1 MAG
#                                             date  time la  lo, dep, mag
m_Data   = np.genfromtxt( dir_file, usecols=(0,1,2,3,4,5,6, 7,  8,   9), skip_header = 2).T
print( 'total no. of eqs.', m_Data[0].shape[0])
#==================================2================================================
#                         select events within rmax from MS
#===================================================================================
#A# select events within certain radius from MS
i_MS_ID = np.argmax( m_Data[9])
f_MSmag = m_Data[9].max()
print('MS lon lat: ', m_Data[7][i_MS_ID], m_Data[6][i_MS_ID])
aR    = seis_utils.haversine( m_Data[7][i_MS_ID], m_Data[6][i_MS_ID], m_Data[7], m_Data[6])
# maximum radius of influence from Gardner & Knophoff, 1967
dPar['rmax'] = 10**(0.25*f_MSmag-.22)
selR  = aR <= dPar['rmax']
print( 'maximum radius', dPar['rmax'], 'N  events within r_max: ', selR.sum(), 'N events outside of Rmax', (1-selR).sum())
# only events with rma
m_EvInRmax = m_Data.T[selR].T

#B# update mainshock ID for smaller data-set
i_MS_ID = np.argmax( m_EvInRmax[9])
print('mainshock ID', i_MS_ID)
#C# create time vector --> use obspy UTCDateTime or mx.DateTime
a_t_decYr = np.zeros( m_EvInRmax[0].shape[0])
a_t_decYr2 = np.zeros( m_EvInRmax[0].shape[0])
for i in range( m_EvInRmax[0].shape[0]):
    a_t_decYr[i] = seis_utils.dateTime2decYr( [m_EvInRmax[0,i], m_EvInRmax[1,i], m_EvInRmax[2,i], m_EvInRmax[3,i], m_EvInRmax[4,i], m_EvInRmax[5,i]])

#==================================3================================================
#                         power-law fitting (AS decay rate)
#===================================================================================
# subtract t MS from aftershock times - new vector has only AS with time relative to MS in days
a_tAS_day = (a_t_decYr[i_MS_ID+1::] - a_t_decYr[i_MS_ID])*365.25
# compute rates
at_bin_AS, aN_bin_AS   = seis_utils.eqRate( a_tAS_day, dPar['k'])
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
dOm      = omori.fit_omoriMLE( a_tAS_day[a_tAS_day>0],bounds=dPar['a_limit'], par0=dPar['a_par0'], disp = 0)
a_OMrate = omori.fct_omori( at_bin_AS, [dOm['c'], dOm['K'], dOm['p']])
print( 'Omori-fit, MLE', dOm)

#==================================4================================================
#                                 plots
#===================================================================================
plt.figure(1)
ax = plt.subplot()
ax.loglog( at_bin_AS, aN_bin_AS, 'ko', mfc = 'none', mew = 1.5, label = 'aftershocks, $N_{tot}$=%i'%( a_tAS_day.shape[0]))
ax.loglog( at_bin_AS, a_OMrate, 'r--', label = 'Omori: $c$= %.2f, $K$=%.2f,  $p$=%.2f'%( dOm['c'], dOm['K'], dOm['p']))
ax.legend( loc = 'upper right')
ax.set_xlabel( 'Time [day]')
ax.set_ylabel( 'events/day')
plt.savefig( plot_file)
plt.show()











