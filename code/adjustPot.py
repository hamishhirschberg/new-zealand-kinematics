#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hamish
"""
def potFitTractionStressZero(model,factor=1):
    """
    This funtion adjusts the force potentials with the aim of fitting 
     observations exactly without consideration of the physical likelihood of
     the resultant force potentials.
    
    The function first adjusts force potentials at points on faults based on
     the difference between the apriori and aposteriori tractions on faults. It
     assumes the tt-component of stress on a fault is zero. It 
     then adjsuts force potentials at points off faults based on the difference
     between the stresses in elements.
    
    This function requires the difference between the apriori and aposteriori
     solutions in both elements and faults to be loaded into the model.
    """
    
    import numpy as np
    
    # arrays of output stress potentials
    phixx=np.zeros((model.ngp))
    phiyy=np.zeros((model.ngp))
    phixy=np.zeros((model.ngp))
    
    # calculate force potentials in elements
    phiel=np.array([[model.s11d,model.s12d],[model.s12d,model.s22d]])
    # assign force potentials to gridpoints
    for e in range(model.ne):
        phixx[model.gp1e[e]-1]+=phiel[0,0,e]
        phixx[model.gp2e[e]-1]+=phiel[0,0,e]
        phixx[model.gp3e[e]-1]+=phiel[0,0,e]
        phiyy[model.gp1e[e]-1]+=phiel[1,1,e]
        phiyy[model.gp2e[e]-1]+=phiel[1,1,e]
        phiyy[model.gp3e[e]-1]+=phiel[1,1,e]
        phixy[model.gp1e[e]-1]+=phiel[0,1,e]
        phixy[model.gp2e[e]-1]+=phiel[0,1,e]
        phixy[model.gp3e[e]-1]+=phiel[0,1,e]
    # account for multiple elements at each point
    phixx=phixx/model.ngpe[:model.ngp]
    phiyy=phiyy/model.ngpe[:model.ngp]
    phixy=phixy/model.ngpe[:model.ngp]
    # reset points on faults
    faults=model.ngpfs[:model.ngp]>0.5
    phixx[faults]=0
    phiyy[faults]=0
    phixy[faults]=0
    
    # force potentials on fault segments
    phitn=model.ttd
    phinn=model.tnd
    phitt=np.zeros_like(phinn)      # assumes tt stress = 0
    rot=np.array([[model.ftx,model.fty],[model.fnx,model.fny]])     # rotation matrices
    for f in range(model.nf):
        for s in range(model.nfs[f]):
            phiseg=np.einsum('ki,lj,kl',rot[:,:,s,f],rot[:,:,s,f],[[phitt[s,f],phitn[s,f]],[phitn[s,f],phinn[s,f]]])
            # assign force potentials to segments
            phixx[model.gpf[s,f]-1]+=phiseg[0,0]
            phixx[model.gpf[s+1,f]-1]+=phiseg[0,0]
            phiyy[model.gpf[s,f]-1]+=phiseg[1,1]
            phiyy[model.gpf[s+1,f]-1]+=phiseg[1,1]
            phixy[model.gpf[s,f]-1]+=phiseg[1,0]
            phixy[model.gpf[s+1,f]-1]+=phiseg[1,0]
    # account for multiple segments at each point
    phixx[faults]=phixx[faults]/model.ngpfs[:model.ngp][faults]
    phiyy[faults]=phiyy[faults]/model.ngpfs[:model.ngp][faults]
    phixy[faults]=phixy[faults]/model.ngpfs[:model.ngp][faults]
    
    # add to existing force potentials
    model.sxxpot[:model.ngp]+=phixx*factor
    model.syypot[:model.ngp]+=phiyy*factor
    model.sxypot[:model.ngp]+=phixy*factor