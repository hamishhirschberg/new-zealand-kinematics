#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hamish Hirschberg
"""
def strainAllMag(model,style,sestyle,sedirec,soln=0,minmag=0,obrange=None):
    """Adjusts magnitudes of all strain rate observation components.
    
    This function adjusts the magnitudes of all strain rate observation
    components based on the magnitude of the observation in the
    previous iteration. It assumes that the three components are:
        0: style*mag: exx+eyy
        1: max shear*mag: e1-e2 = exx'-eyy'
        2: zero shear direction: 2*exy'=0
    Standard error in direction assumed to be in radians.
    """
    import numpy as np
    
    if obrange is None:
        obrange=range(model.neo)
    
    if str(soln)=='0' or soln=='apri' or soln=='apriori':
        mag=np.sqrt(model.e110[:model.ne]**2+model.e220[:model.ne]**2+2*model.e120[:model.ne]**2)
    elif str(soln)=='1' or soln=='apost' or soln=='aposteriori':
        mag=np.sqrt(model.e111[:model.ne]**2+model.e221[:model.ne]**2+2*model.e121[:model.ne]**2)
    else:
        print('Solution not recognised.')
        return
    
    for i in obrange:
        e=model.eeo[i]
        
        # Strain style observation
        model.oec[0,i]=mag[e-1]*style[e-1]
        # Max shear observation
        model.oec[1,i]=mag[e-1]*np.sqrt(2-style[e-1]**2)
        # Zero shear observation = 0
        if mag[e-1]<minmag:
            # use minmag to calculate std error if magnitude is too small
            model.seoec[0,i]=minmag*sestyle[e-1]
#            model.seoec[1,i]=minmag*np.sqrt(2-style[e-1]**2)*0.5
            model.seoec[2,i]=2*minmag*sedirec[e-1]
        else:
            model.seoec[0,i]=mag[e-1]*sestyle[e-1]
#            model.seoec[1,i]=mag[e-1]*np.sqrt(2-style[e-1]**2)*0.5
            model.seoec[2,i]=2*mag[e-1]*sedirec[e-1]
        model.seoec[1,i]=np.abs(style[e-1])/np.sqrt(2-style[e-1]**2)*model.seoec[0,i]+0.5*model.seoec[2,i]