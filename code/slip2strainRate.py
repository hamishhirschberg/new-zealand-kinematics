#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hamish
"""

import numpy as np

def rectTaper(dist):
    """
    A rectangular 'taper' for use determining the proportion of the strain rate
     being assigned at a given distance relative to the width over which strain
     rates are being assigned (dist=1 for the furthermost point that receives
     some of the strain rate).
    """
    return 1.

def triTaper(dist):
    """
    A triangular taper for use determining the proportion of the strain rate
     being assigned at a given distance relative to the width over which strain
     rates are being assigned (dist=1 for the furthermost point that receives
     some of the strain rate).
    """
    return 2*(1-dist)

def cosTaper(dist):
    """
    A cosine taper for use determining the proportion of the strain rate
     being assigned at a given distance relative to the width over which strain
     rates are being assigned (dist=1 for the furthermost point that receives
     some of the strain rate).
    """
    return 1+np.cos(dist*np.pi*0.5)


def slip2strainRate(model,width,taper='rect',unc=0,clear=False,fill=True):
    """
    This function takes slip rate observations and converts them to strain rate
     observations spread over nearby elements. Contributions are calculated at
     points and the contribution of fault to an element is the mean of the 
     fault's contributions to points on the element.
     
    width is half the distance over which the strain rate is spread. Points
     more than one width from the fault do not get a contribution from that fault.
    
    the options for tapers are:
    rectangular: the strain rate is distributed according to a
     rectangular function with all points with wdith of the fault receiving an
     equal strain rate.
    triangular: the strain rate is distributed according to a
     triangular function with points closer to the fault receiving a greater
     strain rate at a linear rate.
    cosine: the strain rate is distributed according to a cosine function with
     points near the fault receiving similar strain rate, then dropping off
     quickly before smoothly evening out again.
    """
    
    import numpy as np
    
    # choose taper
    if taper=='r' or taper=='rect' or taper=='rectangular':
        taperfun=rectTaper
    elif taper=='t' or taper=='tri' or taper=='triangular':
        taperfun=triTaper
    elif taper=='c' or taper=='cos' or taper=='cosine':
        taperfun=cosTaper
    
    # load points if not already done so
    if model.gp1e[0]==0:
        print('Element points not loaded. Loading now.')
        model.getElementPoints()
    
    if clear:
        model.neo=0     # clear old observations
    
    degw=width/model.re*180/np.pi       # width expressed as degrees
    
    exxgp=np.zeros((model.ngp))
    eyygp=np.zeros((model.ngp))
    exygp=np.zeros((model.ngp))
    varexxgp=np.zeros((model.ngp))
    vareyygp=np.zeros((model.ngp))
    varexygp=np.zeros((model.ngp))
    
    # loop through observations
    for i in range(model.nfo):
        # fault segment geometry
        gp1=model.gp1s[model.sfo[i]-1]
        gp2=model.gp2s[model.sfo[i]-1]
        x1=model.long[gp1-1]
        x2=model.long[gp2-1]
        y1=model.lat[gp1-1]
        y2=model.lat[gp2-1]
        dx=(x2-x1)*np.cos(np.radians((y1+y2)/2))
        dy=y2-y1
        dl=np.hypot(dx,dy)
        tx=dx/dl
        ty=dy/dl
        nx=-dy/dl
        ny=dx/dl
        # convert slip rate observations to strain rates
        A=np.array([[model.codut[0,i],model.codun[0,i]],[model.codut[1,i],model.codun[1,i]]])
        b=np.array([model.oduc[0,i],model.oduc[1,i]])
        Ainvb=np.linalg.solve(A,b)
        odut=Ainvb[0]
        odun=Ainvb[1]
        exxf=0.5*(odun*nx+odut*tx)*nx/width
        eyyf=0.5*(odun*ny+odut*ty)*ny/width
        exyf=0.5*(odun*nx*ny+0.5*odut*(nx*ty+ny*tx))/width
        # get uncertainties
        b=np.array([model.seoduc[0,i],model.seoduc[1,i]])
        Ainvb=np.linalg.solve(A,b)
        seodut=Ainvb[0]
        seodun=Ainvb[1]
        seexxf=0.5*(np.abs(seodun*nx*nx)+np.abs(seodut*tx*nx))/width
        seeyyf=0.5*(np.abs(seodun*ny*ny)+np.abs(seodut*ty*ny))/width
        seexyf=0.5*(np.abs(seodun*nx*ny)+np.abs(0.5*seodut*(nx*ty+ny*tx)))/width
        
        # find points within width of fault
        degwdl=degw+dl
        for p in range(model.ngp):
            # start with quick and dirty calculation to remove far points
            if np.abs(x1-model.long[p])<=degwdl and np.abs(y1-model.lat[p])<=degwdl:
                # now perform proper calculation
                dx1=(model.long[p]-x1)*np.cos(np.radians((model.lat[p]+y1)/2))
                dy1=model.lat[p]-y1
                # check if within width of fault points using dot product with normal to fault
                if np.abs(dx1*nx+dy1*ny)<=degw:
                    distt=dx1*tx+dy1*ty         # transverse distance along fault
                    # check it is within segment length
                    if distt>=0 and distt<=dl:
                        # now we know we want this point
                        distn=np.abs(dx1*nx+dy1*ny)
                        taperfac=taperfun(distn/degw)
                        if distt==0 or distt==dl:
                            taperfac*=0.5   # account for being between segments
                        exxgp[p]+=taperfac*exxf
                        eyygp[p]+=taperfac*eyyf
                        exygp[p]+=taperfac*exyf
                        varexxgp[p]+=(taperfac*seexxf)**2
                        vareyygp[p]+=(taperfac*seeyyf)**2
                        varexygp[p]+=(taperfac*seexyf)**2
    
    # convert strain at points to strain rate in elements
    for e in range(model.ne):
        gpe=np.array([model.gp1e[e],model.gp2e[e],model.gp3e[e]])
        exxe=np.mean(exxgp[gpe-1])
        eyye=np.mean(eyygp[gpe-1])
        exye=np.mean(exygp[gpe-1])
        if exxe != 0. or eyye != 0. or exye != 0.:
            # if strain rate obs nonzero, then store it
            model.oec[0,model.neo]=exxe
            model.oec[1,model.neo]=eyye
            model.oec[2,model.neo]=exye
            model.seoec[0,model.neo]=np.mean(np.sqrt(varexxgp[gpe-1]))
            model.seoec[1,model.neo]=np.mean(np.sqrt(vareyygp[gpe-1]))
            model.seoec[2,model.neo]=np.mean(np.sqrt(varexygp[gpe-1]))
            model.neoc[model.neo]=3
            model.eeo[model.neo]=e+1
            model.coexx[0,model.neo]=1.
            model.coeyy[1,model.neo]=1.
            model.coexy[2,model.neo]=1.
            model.neo+=1
        elif fill:
            # if strain rate obs zero but still setting observation
            model.oec[:,model.neo]=0.
            model.seoec[:,model.neo]=unc
            model.neoc[model.neo]=3
            model.eeo[model.neo]=e+1
            model.coexx[0,model.neo]=1.
            model.coeyy[1,model.neo]=1.
            model.coexy[2,model.neo]=1.
            model.neo+=1
    
def slip2strainTransfer(mold,mnew,width,taper='rect',clear=False,otherels=0.,maxlen=0.):
    """
    This function takes faults from the old model and converts them to strain
     rates which are applied to the new model. It is based on slip2strainRate.
     
    mold: the old model that has the slip rate observations
    mnew: the new model to receive the strain rate observations
    width: max distance from fault that receives a strain rate observation
    clear: remove any pre-existing strain obs from mnew
    otherels: if nonzero, add an observation to all elements with zero observations
        receiving a standard error of otherels.
    maxlen: maximum length of the fault an observation occurs on for it to be
        converted to strain rates. This allows converting only short faults to
        strain rates. maxlen=0 means all faults are processed.
        
    the options for tapers are:
    rectangular: the strain rate is distributed according to a
     rectangular function with all points with wdith of the fault receiving an
     equal strain rate.
    triangular: the strain rate is distributed according to a
     triangular function with points closer to the fault receiving a greater
     strain rate at a linear rate.
    cosine: the strain rate is distributed according to a cosine function with
     points near the fault receiving similar strain rate, then dropping off
     quickly before smoothly evening out again.
    """
    
    import numpy as np
    
    # choose taper
    if taper=='r' or taper=='rect' or taper=='rectangular':
        taperfun=rectTaper
    elif taper=='t' or taper=='tri' or taper=='triangular':
        taperfun=triTaper
    elif taper=='c' or taper=='cos' or taper=='cosine':
        taperfun=cosTaper
    
    # load points if not already done so
    if mnew.gp1e[0]==0:
        print('Element points not loaded. Loading now.')
        mnew.getElementPoints()
    
    if clear:
        mnew.neo=0     # clear old observations
    
    if maxlen:
        if mold.lenfs[0,0]==0.:
            print('Fault segment lengths not loaded. Calculating now.')
            mold.faultSegmentLength()
    else:
        maxlen=np.inf
    
    degw=width/mold.re*180/np.pi       # width expressed as degrees
    
    exxgp=np.zeros((mnew.ngp))
    eyygp=np.zeros((mnew.ngp))
    exygp=np.zeros((mnew.ngp))
    varexxgp=np.zeros((mnew.ngp))
    vareyygp=np.zeros((mnew.ngp))
    varexygp=np.zeros((mnew.ngp))
    
    
    # loop through observations
    for i in range(mold.nfo):
        if np.sum(mold.lenfs[:,mold.ffo[i]-1])>maxlen:
            continue
        # fault segment geometry
        gp1=mold.gp1s[mold.sfo[i]-1]
        gp2=mold.gp2s[mold.sfo[i]-1]
        x1=mold.long[gp1-1]
        x2=mold.long[gp2-1]
        y1=mold.lat[gp1-1]
        y2=mold.lat[gp2-1]
        dx=(x2-x1)*np.cos(np.radians((y1+y2)/2))
        dy=y2-y1
        dl=np.hypot(dx,dy)
        tx=dx/dl
        ty=dy/dl
        nx=-dy/dl
        ny=dx/dl
        # convert slip rate observations to strain rates
        A=np.array([[mold.codut[0,i],mold.codun[0,i]],[mold.codut[1,i],mold.codun[1,i]]])
        b=np.array([mold.oduc[0,i],mold.oduc[1,i]])
        Ainvb=np.linalg.solve(A,b)
        odut=Ainvb[0]
        odun=Ainvb[1]
        exxf=0.5*(odun*nx+odut*tx)*nx/width
        eyyf=0.5*(odun*ny+odut*ty)*ny/width
        exyf=0.5*(odun*nx*ny+0.5*odut*(nx*ty+ny*tx))/width
        # get uncertainties
        b=np.array([mold.seoduc[0,i],mold.seoduc[1,i]])
        Ainvb=np.linalg.solve(A,b)
        seodut=Ainvb[0]
        seodun=Ainvb[1]
        seexxf=0.5*(np.abs(seodun*nx*nx)+np.abs(seodut*tx*nx))/width
        seeyyf=0.5*(np.abs(seodun*ny*ny)+np.abs(seodut*ty*ny))/width
        seexyf=0.5*(np.abs(seodun*nx*ny)+np.abs(0.5*seodut*(nx*ty+ny*tx)))/width
        
        # find points within width of fault
        degwdl=degw+dl
        for p in range(mnew.ngp):
            # start with quick and dirty calculation to remove far points
            if np.abs(x1-mnew.long[p])<=degwdl and np.abs(y1-mnew.lat[p])<=degwdl:
                # now perform proper calculation
                dx1=(mnew.long[p]-x1)*np.cos(np.radians((mnew.lat[p]+y1)/2))
                dy1=mnew.lat[p]-y1
                # check if within width of fault points using dot product with normal to fault
                if np.abs(dx1*nx+dy1*ny)<=degw:
                    distt=dx1*tx+dy1*ty         # transverse distance along fault
                    # check it is within segment length
                    if distt>=0 and distt<=dl:
                        # now we know we want this point
                        distn=np.abs(dx1*nx+dy1*ny)
                        taperfac=taperfun(distn/degw)
                        exxgp[p]+=taperfac*exxf
                        eyygp[p]+=taperfac*eyyf
                        exygp[p]+=taperfac*exyf
                        varexxgp[p]+=(taperfac*seexxf)**2
                        vareyygp[p]+=(taperfac*seeyyf)**2
                        varexygp[p]+=(taperfac*seexyf)**2
    
    # convert strain at points to strain rate in elements
    for e in range(mnew.ne):
        gpe=np.array([mnew.gp1e[e],mnew.gp2e[e],mnew.gp3e[e]])
        exxe=np.mean(exxgp[gpe-1])
        eyye=np.mean(eyygp[gpe-1])
        exye=np.mean(exygp[gpe-1])
        if exxe != 0. or eyye != 0. or exye != 0.:
            # if strain rate obs nonzero, then store it
            mnew.oec[0,mnew.neo]=exxe
            mnew.oec[1,mnew.neo]=eyye
            mnew.oec[2,mnew.neo]=exye
            mnew.seoec[0,mnew.neo]=np.mean(np.sqrt(varexxgp[gpe-1]))
            mnew.seoec[1,mnew.neo]=np.mean(np.sqrt(vareyygp[gpe-1]))
            mnew.seoec[2,mnew.neo]=np.mean(np.sqrt(varexygp[gpe-1]))
            mnew.neoc[mnew.neo]=3
            mnew.eeo[mnew.neo]=e+1
            mnew.coexx[0,mnew.neo]=1.
            mnew.coeyy[1,mnew.neo]=1.
            mnew.coexy[2,mnew.neo]=1.
            mnew.neo+=1
        elif otherels:
            # if strain rate obs zero but still setting observation
            mnew.oec[:,mnew.neo]=0.
            mnew.seoec[:,mnew.neo]=otherels
            mnew.neoc[mnew.neo]=3
            mnew.eeo[mnew.neo]=e+1
            mnew.coexx[0,mnew.neo]=1.
            mnew.coeyy[1,mnew.neo]=1.
            mnew.coexy[2,mnew.neo]=1.
            mnew.neo+=1
            
def slip2strainTransferElements(mold,mnew,clear=False,otherels=0.,mine=0.,prec=1e-12):
    """
    This function transfers slip rates on faults in one model into strain rates
     in elements in a second model. For each fault segment, the function
     determines the length in each element and adds that proportion of the slip
     rate to that element's strain rate.
    """
    
    import numpy as np
    from ..fortran.setup import finde
    
    # load points if not already done so
    if mnew.gp1e[0]==0:
        print('Element points not loaded. Loading now.')
        mnew.getElementPoints()
    
    if clear:
        mnew.neo=0     # clear old observations
    
    if mold.lenfs[0,0]==0.:
        print('Fault segment lengths not loaded. Calculating now.')
        mold.faultSegmentLength()
     
    # initialise strain rates
    exx=np.zeros((mnew.ne))
    eyy=np.zeros((mnew.ne))
    exy=np.zeros((mnew.ne))
    varexx=np.zeros((mnew.ne))
    vareyy=np.zeros((mnew.ne))
    varexy=np.zeros((mnew.ne))
    
    el=1
    isf=np.zeros((mnew.ns),dtype=np.bool)
    # loop through fault observations in mold
    for i in range(mold.nfo):
        # fault segment geometry
        s=mold.fss[mold.sfo[i]-1]
        f=mold.ffo[i]
        gpf=mold.gpf[s-1:s+1,f-1]
        lo0,lo1=mold.long[gpf-1]
        la0,la1=mold.lat[gpf-1]
        dx=(lo1-lo0)*np.cos(np.radians((la0+la1)/2))
        dy=la1-la0
        dl=np.hypot(dx,dy)
        tx=dx/dl
        ty=dy/dl
        nx=-dy/dl
        ny=dx/dl
#        print(lo0,la0,lo1,la1)
        # convert slip rate observations to strain rates
        A=np.array([[mold.codut[0,i],mold.codun[0,i]],[mold.codut[1,i],mold.codun[1,i]]])
        b=np.array([mold.oduc[0,i],mold.oduc[1,i]])
        Ainvb=np.linalg.solve(A,b)
        odut=Ainvb[0]
        odun=Ainvb[1]
        exxf=0.5*(odun*nx+odut*tx)*nx
        eyyf=0.5*(odun*ny+odut*ty)*ny
        exyf=0.5*(odun*nx*ny+0.5*odut*(nx*ty+ny*tx))
        # get uncertainties
        b=np.array([mold.seoduc[0,i],mold.seoduc[1,i]])
        Ainvb=np.linalg.solve(A,b)
        seodut=Ainvb[0]
        seodun=Ainvb[1]
        seexxf=0.5*(np.abs(seodun*nx*nx)+np.abs(seodut*tx*nx))
        seeyyf=0.5*(np.abs(seodun*ny*ny)+np.abs(seodut*ty*ny))
        seexyf=0.5*(np.abs(seodun*nx*ny)+np.abs(0.5*seodut*(nx*ty+ny*tx)))
        
        # find element just after first point (to avoid coincident grid points)
        lo=lo0*(1-1e-3)+lo1*1e-3
        la=la0*(1-1e-3)+la1*1e-3
        out=finde(lo,la,el,mnew.long[:mnew.ngp],mnew.lat[:mnew.ngp],\
                  mnew.e1s[:mnew.ns],mnew.e2s[:mnew.ns],isf,\
                  mnew.gp1e[:mnew.ne],mnew.gp2e[:mnew.ne],mnew.gp3e[:mnew.ne],\
                  mnew.s1e[:mnew.ne],mnew.s2e[:mnew.ne],mnew.s3e[:mnew.ne])
        if out[4]==2:
            print('Unable to find element for start of fault '+str(f))
            return
        el=out[0]
        
        prevdist=0.         # length of segment accounted for so far
        
        # loop through elements on this segment (limited to number of elements)
        for e in range(mnew.ne):
#            print(el,prevdist)
            # find intersect of the line with the sides of element
            lenfrac=[0,0,0]
            gps=np.array([mnew.gp1e[el-1],mnew.gp2e[el-1],mnew.gp3e[el-1],mnew.gp1e[el-1]])
            for side in range(3):
                # find relative intersect of side with segment
                x0,x1=mnew.long[gps[side:side+2]-1]
                y0,y1=mnew.lat[gps[side:side+2]-1]
                det=(lo1-lo0)*(y1-y0)-(la1-la0)*(x1-x0)
                dist=(x1-lo0)*(y1-y0)-(y1-la0)*(x1-x0)
                lenfrac[side]=dist/det
#            print(lenfrac)
            midf=np.median(lenfrac)
            maxf=np.max(lenfrac)
            midi=lenfrac.index(midf)
            maxi=lenfrac.index(maxf)
            if midf<=prevdist+prec:
                # here mid is the segment entering the element
                start=prevdist+0
                end=min(maxf,1)
                nexts=maxi+0
            else:
                # here mid is the segment exiting the element
                start=prevdist+0
                end=min(midf,1)
                nexts=midi+0
            
            length=(end-start)*mold.lenfs[s-1,f-1]
            # calculate the contribution to the element's strain rate
            exx[el-1]+=exxf*length
            eyy[el-1]+=eyyf*length
            exy[el-1]+=exyf*length
            varexx[el-1]+=(seexxf*length)**2
            vareyy[el-1]+=(seeyyf*length)**2
            varexy[el-1]+=(seexyf*length)**2
            
            prevdist=end+0
            if end>=1-prec:
                # end of fault segment
                break
            
            if nexts==0:
                # next side is side 3
                side=mnew.s3e[el-1]
            elif nexts==1:
                # next side is side 1
                side=mnew.s1e[el-1]
            else:
                # next side is side 2
                side=mnew.s2e[el-1]
            # find next element
            if mnew.e1s[side-1]==el:
                el=mnew.e2s[side-1]+0
            else:
                el=mnew.e1s[side-1]+0
            if el==0:
                # boundary reached
                break
        else:
            print('Maximum number of elements reached on observation '+str(i+1))
            return
    
    # store strain rate observations in new model
    for e in range(mnew.ne):
        if np.abs(exx[e])>mine or np.abs(eyy[e])>mine or np.abs(exy[e])>mine:
            # if strain rate obs nonzero, then store it
            mnew.oec[0,mnew.neo]=exx[e]/mnew.area[e]
            mnew.oec[1,mnew.neo]=eyy[e]/mnew.area[e]
            mnew.oec[2,mnew.neo]=exy[e]/mnew.area[e]
            mnew.seoec[0,mnew.neo]=np.mean(np.sqrt(varexx[e]))/mnew.area[e]
            mnew.seoec[1,mnew.neo]=np.mean(np.sqrt(vareyy[e]))/mnew.area[e]
            mnew.seoec[2,mnew.neo]=np.mean(np.sqrt(varexy[e]))/mnew.area[e]
            mnew.neoc[mnew.neo]=3
            mnew.eeo[mnew.neo]=e+1
            mnew.coexx[0,mnew.neo]=1.
            mnew.coeyy[1,mnew.neo]=1.
            mnew.coexy[2,mnew.neo]=1.
            mnew.neo+=1
        elif otherels:
            # if strain rate obs zero but still setting observation
            mnew.oec[:,mnew.neo]=0.
            mnew.seoec[:,mnew.neo]=otherels
            mnew.neoc[mnew.neo]=3
            mnew.eeo[mnew.neo]=e+1
            mnew.coexx[0,mnew.neo]=1.
            mnew.coeyy[1,mnew.neo]=1.
            mnew.coexy[2,mnew.neo]=1.
            mnew.neo+=1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    