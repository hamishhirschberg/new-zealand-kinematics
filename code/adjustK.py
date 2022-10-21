#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hamish Hirschberg
"""
import numpy as np

def kSegments(model,factor=1,dtau=0,dumin=1e-12):
    """
    This is the Python version of the Fortran code 'adjust_K_in_setup_input.f'
    that adjusts the slip-rate capacity on a segment-by-segment basis using the
    relative values of apriori and aposteriori slip rates in the fault 
    segments. In comparison to kFaults, this will allow the relative slip-rate
    capacities between different segments on the same fault to vary.
    
    This function takes a permdefmap model as an argument. Optionally, it can
    also take a minimum relative value of slip rate, dumin, which defaults to 1e-12.
    """
    # apriori for all segments
    dumag0=model.tt0*model.dut0+model.tn0*model.dun0        # magnitude of du for each segment
    if np.array(dtau).any():
        dumag0+=model.tt0*(model.kc+model.ks)*dtau[0,:,:]+model.tn0*model.kc*dtau[1,:,:]
    ltot0=np.sum(model.lenfs)                    # total length of all fault segments
    dutot0=np.sum(model.lenfs*dumag0)            # total weighted du
    dumin=dumin*dutot0/ltot0                  # minimum du scaled by weighted average du
    dumag0[np.all([dumag0<dumin,dumag0],axis=0)]=dumin      # implement min du
    
    # aposteriori relative to apriori by segment
    dumag1=model.tt0*model.dut1+model.tn0*model.dun1        # magnitude of du
    dumag1[dumag1<-dumag0]=-dumag0[dumag1<-dumag0]                      # prevent wrong-way weakening
    dumag1=np.abs(dumag1)                               # convert to positive value
    dumag1[np.all([dumag1<dumin,dumag1],axis=0)]=dumin      # implement min du
    duscale=np.divide(dumag1,dumag0,where=dumag0!=0.)**factor        # scale du as ratio of apost and apri
    
    # scale each segment
    model.kc=duscale*model.kc                         # adjust slip-rate cap by scale factor
    model.ks=duscale*model.ks