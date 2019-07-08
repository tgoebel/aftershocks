# aftershocks
aftershock rate estimates and Omori-type fitting


this module includes python methods for basic aftershock detection, based on space time windowing
and fitting decay rates with least-sqaures and maximum-likelihood approaches.
The latter is commonly preferred.

source code is included in .src
seis_utils.py - basib earthquake catalog processing
omori.py      - omori fitting and creation of random data sets

Two examples are included in:
1a_Omori_prague.py - fit aftershock sequence after the 2011 Prague OKlahoma event
1b_Omori_ranPL.py  - fit random power-law between tmin and tmax, described by Omori parameters: c, K, p
