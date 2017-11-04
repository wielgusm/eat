import sys, os, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

hrs = [0,24.,48.]




def list_all_triangles(alist):
    all_baselines = set(alist.baseline)
    all_stations = set(''.join( list(all_baselines)))
    foo = list(itertools.combinations(all_stations, 3))
    foo = [list(x) for x in foo if ('R' not in set(x))|('S' not in set(x))]
    foo = [''.join(sorted(x)) for x in foo] 
    return foo

def list_all_quadrangles(alist):
    all_baselines = set(''.join( list(set(alist.baseline))))
    foo = list(itertools.combinations(all_baselines, 4))
    foo = [set(x) for x in foo if ('R' not in set(x))|('S' not in set(x))]
    return foo

def triangles2baselines(tri,alist):
    all_baselines = set(alist.baseline)
    foo_base = []
    signat = []
    for cou in range(len(tri)):
        b0 = tri[cou][0:2]
        b1 = tri[cou][1:3]
        b2 = tri[cou][2]+tri[cou][0]
        #print([b0,b1,b2])
        if b0 in all_baselines:
            base0 = b0
            sign0 = 1
        elif b0[1]+b0[0] in all_baselines:
            base0 = b0[1]+b0[0]
            sign0 = -1
        else:
            base0 = -1
            sign0 = 0
            
        if b1 in all_baselines:
            base1 = b1
            sign1 = 1
        elif b1[1]+b1[0] in all_baselines:
            base1 = b1[1]+b1[0]
            sign1 = -1
        else:
            base1 = -1
            sign1 = 0
            
        if b2 in all_baselines:
            base2 = b2
            sign2 = 1
        elif b2[1]+b2[0] in all_baselines:
            base2 = b2[1]+b2[0]
            sign2 = -1
        else:
            base2 = -1
            sign2 = 0
        baselines = [base0,base1,base2]
        
        baselinesSTR = map(lambda x: type(x)==str,baselines)
        if all(baselinesSTR):
            foo_base.append(baselines)
            signat.append([sign0,sign1,sign2])
    return foo_base, signat

def baselines2triangles(basel):
    tri = [''.join(sorted(list(set(''.join(x))))) for x in basel]
    return tri

def all_bispectra_polar(alist,polar,phaseType='resid_phas'):
    alist = alist[alist['polarization']==polar]
    triL = list_all_triangles(alist)
    tri_baseL, sgnL = triangles2baselines(triL,alist)
    #this is done twice to remove some non-present triangles
    triL = baselines2triangles(tri_baseL)
    tri_baseL, sgnL = triangles2baselines(triL,alist)
    bsp_out = pd.DataFrame({})
    for cou in range(len(triL)):
        Tri = tri_baseL[cou]
        signat = sgnL[cou]
        condB1 = (alist['baseline']==Tri[0])
        condB2 = (alist['baseline']==Tri[1])
        condB3 = (alist['baseline']==Tri[2])
        condB = condB1|condB2|condB3
        alist_Tri = alist.loc[condB,['expt_no','scan_id','source','datetime','baseline',phaseType,'amp','snr','gmst']]
        
        #print(np.shape(alist_Tri))
        #throw away times without full triangle
        tlist = alist_Tri.groupby('datetime').filter(lambda x: len(x) > 2)
        tlist.loc[:,'sigma'] = (tlist.loc[:,'amp']/(tlist.loc[:,'snr']))
        for cou2 in range(3):
            tlist.loc[(tlist.loc[:,'baseline']==Tri[cou2]),phaseType] *= signat[cou2]*np.pi/180.
        tlist.loc[:,'sigma'] = 1./tlist.loc[:,'snr']**2 #put 1/snr**2 in the sigma column to aggregate
        bsp = tlist.groupby(('expt_no','source','scan_id','datetime')).agg({phaseType: lambda x: np.sum(x),'amp': lambda x: np.prod(x), 'sigma': lambda x: np.sqrt(np.sum(x))})
        #sigma above is the CLOSURE PHASE ERROR
        
        bsp.loc[:,'bisp'] = bsp.loc[:,'amp']*np.exp(1j*bsp.loc[:,phaseType])
        bsp.loc[:,'snr'] = 1./bsp.loc[:,'sigma']
        bsp.loc[:,'sigma'] = bsp.loc[:,'amp']*bsp.loc[:,'sigma'] #sigma of bispectrum
        bsp.loc[:,'triangle'] = [triL[cou]]*np.shape(bsp)[0]
        bsp.loc[:,'polarization'] = [polar]*np.shape(bsp)[0]
        #bsp.loc[:,'signature'] = [signat]*np.shape(bsp)[0]
        bsp.loc[:,'cphase'] = np.angle(bsp.loc[:,'bisp'])*180./np.pi
        bsp.loc[:,'amp'] = np.abs(bsp.loc[:,'bisp'])
        bsp.loc[:,'snr'] = bsp.loc[:,'amp']/bsp.loc[:,'sigma']
        bsp.loc[:,'sigmaCP'] = 1./bsp.loc[:,'snr']*180./np.pi #deg
        bsp_out = pd.concat([bsp_out, bsp])
    bsp_out = bsp_out.reset_index()
    bsp_out = bsp_out[['datetime','source','triangle','polarization','cphase','sigmaCP','amp','sigma','snr','scan_id','expt_no']] 
    
    return bsp_out

def only_trivial_triangles(bsp,whichB = 'all'):
    if whichB =='AX':
        condTri = map(lambda x: (('A' in x)&('X' in x)), bsp['triangle'])
    elif whichB =='JS':
        condTri = map(lambda x: (('J' in x)&('S' in x)), bsp['triangle'])
    elif whichB =='JR':
        condTri = map(lambda x: (('J' in x)&('R' in x)), bsp['triangle'])
    else:
        condTri = map(lambda x: (('A' in x)&('X' in x))|(('J' in x)&('S' in x))|(('J' in x)&('R' in x)), bsp['triangle'])
    bsp = bsp[condTri]
    return bsp

def only_non_trivial_triangles(bsp):
    condTri = map(lambda x: (('A' not in x)|('X' not in x))&(('J' not in x)|('S' not in x))&(('J' not in x)|('R' not in x)), bsp['triangle'])
    bsp = bsp[condTri]
    return bsp

def coh_average_bsp(AIPS, tcoh = 5.):
    AIPS.loc[:,'vis'] = AIPS.loc[:,'vis'] = AIPS.loc[:,'amp']*np.exp(1j*AIPS.loc[:,'cphase']*np.pi/180)
    
    if tcoh == 'scan':
        AIPS = AIPS[['datetime','triangle','source','polarization','vis','sigma','scan_id','expt_no']]
        AIPS = AIPS.groupby(('triangle','source','polarization','expt_no','scan_id')).agg({'datetime': 'min', 'vis': np.mean, 'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x) })
    else:
        AIPS.loc[:,'round_time'] = map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/tcoh),AIPS.loc[:,'datetime'])
        AIPS = AIPS[['datetime','triangle','source','polarization','vis','sigma','scan_id','expt_no','round_time']]
        AIPS = AIPS.groupby(('triangle','source','polarization','expt_no','scan_id','round_time')).agg({'datetime': 'min', 'vis': np.mean, 'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x) })
    AIPS = AIPS.reset_index()
    AIPS['amp'] = np.abs(AIPS['vis'])
    AIPS['cphase'] = np.angle(AIPS['vis'])*180/np.pi
    AIPS = AIPS[['datetime','triangle','source','polarization','amp', 'cphase', 'sigma','expt_no','scan_id']]
    return AIPS

def DataBaseline(alist,basename,polar):
    condB = (alist['baseline']==basename)
    condP = (alist['polarization']==polar)
    if 't_coh' in alist.keys():
        alistB = alist.loc[condB&condP,['expt_no','scan_id','source','datetime','baseline','total_phas','amp','snr','gmst','t_coh','t_coh_bias']]
    else:
        alistB = alist.loc[condB&condP,['expt_no','scan_id','source','datetime','baseline','total_phas','amp','snr','gmst']]
    alistB.loc[:,'sigma'] = (alistB.loc[:,'amp']/(alistB.loc[:,'snr']))
    alistB.loc[:,'snrstd'] = (alistB.loc[:,'snr'])
    alistB.loc[:,'ampstd'] = (alistB.loc[:,'amp'])
    alistB.loc[:,'phasestd'] = (alistB.loc[:,'total_phas'])
    if 't_coh' in alist.keys():
        alistB = alistB.groupby(('source','expt_no','scan_id')).agg({'total_phas': lambda x: np.average(x),
        'amp': lambda x: np.average(x), 'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x), 'snr': 'mean','snrstd': 'std',
                        'gmst': 'min', 'ampstd': 'std', 'phasestd': 'std', 't_coh': 'mean','t_coh_bias': 'mean'})
    else:
        alistB = alistB.groupby(('source','expt_no','scan_id')).agg({'total_phas': lambda x: np.average(x),
        'amp': lambda x: np.average(x), 'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x), 'snr': 'mean','snrstd': 'std',
             'gmst': 'min', 'ampstd': 'std', 'phasestd': 'std'})
        
    return alistB

def DataTriangle(alist,Tri,signat,pol):
    condP = (alist['polarization']==pol)
    condB1 = (alist['baseline']==Tri[0])
    condB2 = (alist['baseline']==Tri[1])
    condB3 = (alist['baseline']==Tri[2])
    condB = condB1|condB2|condB3

    alistRR_Tri = alist.loc[condP&condB,['expt_no','scan_id','source','datetime','baseline','total_phas','amp','snr','gmst']]
    #alistRR_Tri = alist.loc[condP&condB,['expt_no','scan_id','source','datetime','baseline','res_phas','amp','snr','gmst']]
    
    #throw away times without full triangle
    tlist = alistRR_Tri.groupby('datetime').filter(lambda x: len(x) > 2)
    tlist.loc[:,'sigma'] = (tlist.loc[:,'amp']/(tlist.loc[:,'snr']))
    for cou in range(3):
        tlist.loc[(tlist.loc[:,'baseline']==Tri[cou]),'total_phas'] *= signat[cou]*np.pi/180.
    #tlist

    tlist.loc[:,'sigma'] = 1./tlist.loc[:,'snr']**2 #put 1/snr**2 in the sigma column to aggregate
    tlist.loc[:,'footime'] = tlist.loc[:,'datetime'] #dummy time to aggregate CP

    #first aggregation to get bispectra
    bsp = tlist.groupby(('expt_no','source','scan_id','footime')).agg({'total_phas': lambda x: np.sum(x),
        'amp': lambda x: np.prod(x), 'sigma': lambda x: np.sqrt(np.sum(x)),'gmst': 'min','datetime': 'min'})
    #sigma above is the CLOSURE PHASE ERROR
    
    bsp.loc[:,'bisp'] = bsp.loc[:,'amp']*np.exp(1j*bsp.loc[:,'total_phas'])
    bsp.loc[:,'snr'] = 1./bsp.loc[:,'sigma']
    bsp.loc[:,'sigma'] = bsp.loc[:,'amp']*bsp.loc[:,'sigma'] #sigma of bispectrum

    #second aggregation to average the bispectra
    bsp = bsp.groupby(('expt_no','source','scan_id')).agg({'bisp': np.average,'gmst' :'min','datetime' :'min','sigma': lambda x: np.sqrt(sum(x**2))/len(x)})
    bsp.loc[:,'cphase'] = np.angle(bsp.loc[:,'bisp'])*180./np.pi #deg
    bsp.loc[:,'amp'] = np.abs(bsp.loc[:,'bisp'])
    bsp.loc[:,'snr'] = bsp.loc[:,'amp']/bsp.loc[:,'sigma']
    bsp.loc[:,'sigmaCP'] = 1./bsp.loc[:,'snr']*180./np.pi #deg
    del bsp['bisp']
    
    return bsp

def DataTriangleNotAv(alist,Tri,signat,pol):
    condP = (alist['polarization']==pol)
    condB1 = (alist['baseline']==Tri[0])
    condB2 = (alist['baseline']==Tri[1])
    condB3 = (alist['baseline']==Tri[2])
    condB = condB1|condB2|condB3

    alistRR_Tri = alist.loc[condP&condB,['expt_no','scan_id','source','datetime','baseline','total_phas','amp','snr','gmst']]
    #alistRR_Tri = alist.loc[condP&condB,['expt_no','scan_id','source','datetime','baseline','res_phas','amp','snr','gmst']]
    
    #throw away times without full triangle
    tlist = alistRR_Tri.groupby('datetime').filter(lambda x: len(x) > 2)
    tlist.loc[:,'sigma'] = (tlist.loc[:,'amp']/(tlist.loc[:,'snr']))
    for cou in range(3):
        tlist.loc[(tlist.loc[:,'baseline']==Tri[cou]),'total_phas'] *= signat[cou]*np.pi/180.
    #tlist

    tlist.loc[:,'sigma'] = 1./tlist.loc[:,'snr']**2 #put 1/snr**2 in the sigma column to aggregate
    tlist.loc[:,'footime'] = tlist.loc[:,'datetime'] #dummy time to aggregate CP

    #first aggregation to get bispectra
    bsp = tlist.groupby(('expt_no','source','scan_id','footime')).agg({'total_phas': lambda x: np.sum(x),
        'amp': lambda x: np.prod(x), 'sigma': lambda x: np.sqrt(np.sum(x)),'gmst': 'min','datetime': 'min'})
    #sigma above is the CLOSURE PHASE ERROR
    bsp0 = bsp
    bsp.loc[:,'bisp'] = bsp.loc[:,'amp']*np.exp(1j*bsp.loc[:,'total_phas'])
    bsp.loc[:,'snr'] = 1./bsp.loc[:,'sigma']
    bsp.loc[:,'sigma'] = bsp.loc[:,'amp']*bsp.loc[:,'sigma'] #sigma of bispectrum

    #second aggregation to average the bispectra
    bsp = bsp.groupby(('expt_no','source','scan_id')).agg({'bisp': np.average,'gmst' :'min','datetime' :'min','sigma': lambda x: np.sqrt(sum(x**2))/len(x)})
    bsp.loc[:,'cphase'] = np.angle(bsp.loc[:,'bisp'])*180./np.pi #deg
    bsp.loc[:,'amp'] = np.abs(bsp.loc[:,'bisp'])
    bsp.loc[:,'snr'] = bsp.loc[:,'amp']/bsp.loc[:,'sigma']
    bsp.loc[:,'sigmaCP'] = 1./bsp.loc[:,'snr']*180./np.pi #deg
    del bsp['bisp']
    
    return bsp0


def DataTriangleRP(alist,Tri,signat,pol):
    condP = (alist['polarization']==pol)
    condB1 = (alist['baseline']==Tri[0])
    condB2 = (alist['baseline']==Tri[1])
    condB3 = (alist['baseline']==Tri[2])
    condB = condB1|condB2|condB3

    alistRR_Tri = alist.loc[condP&condB,['expt_no','scan_id','source','datetime','baseline','resid_phas','amp','snr','gmst']]
    #alistRR_Tri = alist.loc[condP&condB,['expt_no','scan_id','source','datetime','baseline','res_phas','amp','snr','gmst']]
    
    #throw away times without full triangle
    tlist = alistRR_Tri.groupby('datetime').filter(lambda x: len(x) > 2)
    tlist.loc[:,'sigma'] = (tlist.loc[:,'amp']/(tlist.loc[:,'snr']))
    for cou in range(3):
        tlist.loc[(tlist.loc[:,'baseline']==Tri[cou]),'resid_phas'] *= signat[cou]*np.pi/180.
    #tlist

    tlist.loc[:,'sigma'] = 1./tlist.loc[:,'snr']**2 #put 1/snr**2 in the sigma column to aggregate
    tlist.loc[:,'footime'] = tlist.loc[:,'datetime'] #dummy time to aggregate CP

    #first aggregation to get bispectra
    bsp = tlist.groupby(('expt_no','source','scan_id','footime')).agg({'resid_phas': lambda x: np.sum(x),
        'amp': lambda x: np.prod(x), 'sigma': lambda x: np.sqrt(np.sum(x)),'gmst': 'min','datetime': 'min'})
    #sigma above is the CLOSURE PHASE ERROR

    bsp.loc[:,'bisp'] = bsp.loc[:,'amp']*np.exp(1j*bsp.loc[:,'resid_phas'])
    bsp.loc[:,'snr'] = 1./bsp.loc[:,'sigma']
    bsp.loc[:,'sigma'] = bsp.loc[:,'amp']*bsp.loc[:,'sigma'] #sigma of bispectrum

    #second aggregation to average the bispectra
    bsp = bsp.groupby(('expt_no','source','scan_id')).agg({'bisp': np.average,'gmst' :'min','datetime' :'min','sigma': lambda x: np.sqrt(sum(x**2))/len(x)})
    bsp.loc[:,'cphase'] = np.angle(bsp.loc[:,'bisp'])*180./np.pi #deg
    bsp.loc[:,'amp'] = np.abs(bsp.loc[:,'bisp'])
    bsp.loc[:,'snr'] = bsp.loc[:,'amp']/bsp.loc[:,'sigma']
    bsp.loc[:,'sigmaCP'] = 1./bsp.loc[:,'snr']*180./np.pi #deg
    del bsp['bisp']
    
    return bsp


def DataQuadrangle(alist,Quad,pol,method=0, debias=0, signat=[1,1,1,1]):

    condP = (alist['polarization']==pol)
    condB1 = (alist['baseline']==Quad[0])
    condB2 = (alist['baseline']==Quad[1])
    condB3 = (alist['baseline']==Quad[2])
    condB4 = (alist['baseline']==Quad[3])
    condB = condB1|condB2|condB3|condB4

    alist_Quad = alist.loc[condP&condB,['expt_no','scan_id','source','datetime','baseline','total_phas','amp','snr','gmst']]
    tlist = alist_Quad.groupby('datetime').filter(lambda x: len(x) > 3)
    tlist.loc[:,'sigma'] = tlist.loc[:,'amp']/(tlist.loc[:,'snr'])
    tlist.loc[:,'datetime_foo'] = tlist.loc[:,'datetime']
    #if debiasing amplitudes
    if debias != 0:
        tlist.loc[:,'amp'] = tlist.loc[:,'amp']*np.sqrt(1.- debias*2./tlist.loc[:,'snr']**2 )
        
    #####################################################################
    #form quadAmplitudes on 5s and average quadAmplitudes over whole scan
    #####################################################################
    if method == 0:
        for cou in range(2,4):
        #inverse amplitude for the visibilities in the denominator
            tlist.loc[(tlist.loc[:,'baseline']==Quad[cou]),'amp'] = 1./tlist.loc[(tlist.loc[:,'baseline']==Quad[cou]),'amp']

        #aggregating to get quadProducts on 5s segments 
        tlist.loc[:,'sigma'] = 1./tlist.loc[:,'snr']**2#put snr in place of sigma to sum for 4product

        quadAmp = tlist.groupby(('expt_no','source','scan_id','datetime_foo')).agg({'amp': lambda x: np.prod(x), 
                                'gmst': 'min', 'datetime': 'min', 'sigma' : lambda x: np.sqrt(np.sum(x)) })
        quadAmp.loc[:,'sigma'] = quadAmp.loc[:,'sigma']*quadAmp.loc[:,'amp']

        quadAmp = quadAmp.groupby(('expt_no','source','scan_id')).agg({'amp': np.average,'gmst' :'min','datetime' :'min','sigma': lambda x: np.sqrt(np.sum(x**2))/len(x)})
        quadAmp.loc[:,'snr'] = quadAmp.loc[:,'amp']/quadAmp.loc[:,'sigma']

    
    #####################################################################
    #calculate visibilities over whole scan, collapse into quadAmplitude
    #####################################################################
    elif method == 1:
        #aggregation to get visibilities over scan
        quadAmp = tlist.groupby(('expt_no','source','scan_id','baseline')).agg({'amp': lambda x: np.average(x), 
                                'gmst': 'min','sigma': lambda x: np.sqrt(np.sum(x**2))/len(x)})
        quadAmp.reset_index(level=3, inplace=True)

        for cou in range(2,4):
            #inverse complex number for the visibilities in the denominator
            #tlist.loc[(tlist['baseline']==Quad[cou]),'total_phas'] *= -1.
            quadAmp.loc[(quadAmp.loc[:,'baseline']==Quad[cou]),'amp'] = 1./quadAmp.loc[(quadAmp.loc[:,'baseline']==Quad[cou]),'amp']

        tlist.loc[:,'sigma'] = tlist.loc[:,'sigma']/tlist.loc[:,'amp']#put sigma by amplitude for 4product
        quadAmp = quadAmp.groupby(('expt_no','source','scan_id')).agg({'amp': np.prod,'gmst' :'min', 'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x)})
        quadAmp.loc[:,'sigma'] = quadAmp.loc[:,'sigma']*quadAmp.loc[:,'amp']
        quadAmp.loc[:,'snr'] = quadAmp.loc[:,'amp']/quadAmp.loc[:,'sigma']

    
    #####################################################################
    #coherent averaging with phase information
    #####################################################################
    #conjugation because we have data for YX not XY
    elif method == 2:
        for cou in range(4):
            tlist.loc[(tlist.loc[:,'baseline']==Quad[cou]),'total_phas'] *= signat[cou]*np.pi/180.   

        for cou in range(2,4):
            tlist.loc[(tlist['baseline']==Quad[cou]),'total_phas'] *= -1.
            tlist.loc[(tlist.loc[:,'baseline']==Quad[cou]),'amp'] = 1./tlist.loc[(tlist.loc[:,'baseline']==Quad[cou]),'amp']

        tlist.loc[:,'sigma'] = 1./tlist.loc[:,'snr']**2#put snr in place of sigma to sum for 4product
        quadAmp = tlist.groupby(('expt_no','source','scan_id','datetime')).agg({'amp': lambda x: np.prod(x), 
                                    'gmst': 'min', 'total_phas': np.sum, 'sigma' : lambda x: np.sqrt(np.sum(x))})
        quadAmp.loc[:,'sigma'] = quadAmp.loc[:,'sigma']*quadAmp.loc[:,'amp']


        quadAmp.loc[:,'quadProd'] = quadAmp.loc[:,'amp']*np.exp(1j*quadAmp.loc[:,'total_phas'])

        quadAmp = quadAmp.groupby(('expt_no','source','scan_id')).agg({'quadProd': np.average,'gmst' :'min','sigma': lambda x: np.sqrt(np.sum(x**2))/len(x)})

        quadAmp.loc[:,'amp'] = np.abs(quadAmp.loc[:,'quadProd'])
        quadAmp.loc[:,'snr'] = quadAmp.loc[:,'amp']/quadAmp.loc[:,'sigma']


    #####################################################################
    #use log amplitudes !!WORK IN PROGRESS!!
    #####################################################################
    elif method==3:
        quadAmp = tlist
        quadAmp.loc[:,'logamp'] = np.log(quadAmp.loc[:,'amp'])
        #approximated formula for std of log(X)
        quadAmp.loc[:,'sigma'] = quadAmp.loc[:,'sigma']/(quadAmp.loc[:,'amp']) 

        for cou in range(2,4):
        #negative sign for denominator amplitudes
            tlist.loc[(tlist.loc[:,'baseline']==Quad[cou]),'logamp'] *= -1.

        #aggregating to get quadProducts on 5s segments 
        quadAmp = quadAmp.groupby(('expt_no','source','scan_id','datetime')).agg({'logamp': lambda x: np.sum(x), 
                                'gmst': 'min', 'sigma' : lambda x: np.sqrt(np.sum(x**2)) })

        quadAmp = quadAmp.groupby(('expt_no','source','scan_id')).agg({'logamp': np.average,'gmst' :'min','sigma': lambda x: np.sqrt(np.sum(x**2))/len(x)})
        quadAmp.loc[:,'snr'] = quadAmp.loc[:,'logamp']/quadAmp.loc[:,'sigma']

    return quadAmp

def Baseline(alist,basename,source):
    condB = (alist['baseline']==basename)
    alistB = alist.loc[condB,['expt_no','scan_id','source','datetime','baseline','total_phas','polarization','amp','snr','gmst']]
    alistB.loc[:,'sigma'] = (alistB.loc[:,'amp']/(alistB.loc[:,'snr']))
    
    snr_listB = tlist.groupby(('expt_no','source','scan_id','datetime','polarization')).agg({'total_phas': lambda x: np.average(x),
        'amp': lambda x: np.average(x), 'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x),'gmst': 'min'})
   
    #aggregate for source on baseline



def PreparePlot(bsp,source,days,hrs):
    t = []; cp =[]; yerr = []
    for x in range(len(days)):
        t.append(hrs[x] + np.asarray(bsp.loc[(bsp.index.get_level_values('source') == source)&(bsp.index.get_level_values('expt_no') == days[x]),['gmst']]))
        cp.append(np.asarray(bsp.loc[(bsp.index.get_level_values('source') == source)&(bsp.index.get_level_values('expt_no') == days[x]),['cphase']]))
        yerr.append(np.asarray(bsp.loc[(bsp.index.get_level_values('source') == source)&(bsp.index.get_level_values('expt_no') == days[x]),['sigmaCP']]))
    tf = np.concatenate(t).ravel()
    cpf = np.concatenate(cp).ravel()
    yerrf = np.concatenate(yerr).ravel()
    return tf, cpf, yerrf



def plotTrianglePolar(alist,Triangle,Signature,pol,MaxAngle,printSummary=1):

    bsp = DataTriangle(alist,Triangle,Signature,pol)
    days = [3597,3600,3601]; source = '3C279'
    #hrs = [0,24.,48.] #to add to time
    #'OJ287' '3C279' 'J1924-2914' 'CENA' '1055+018'
    t3C,cp3C,yerr3C = PreparePlot(bsp,source,days,hrs)
    source = 'OJ287'
    tOJ,cpOJ,yerrOJ = PreparePlot(bsp,source,days,hrs)
    source = 'J1924-2914'
    tJ1,cpJ1,yerrJ1 = PreparePlot(bsp,source,days,hrs)
    source = '1055+018'
    t10,cp10,yerr10 = PreparePlot(bsp,source,days,hrs)
    source = '1749+096'
    t17,cp17,yerr17 = PreparePlot(bsp,source,days,hrs)
    source = '1921-293'
    t19,cp19,yerr19 = PreparePlot(bsp,source,days,hrs)
    source = 'CYGX-3'
    tCY,cpCY,yerrCY = PreparePlot(bsp,source,days,hrs)
    source = 'J1733-1304'
    tJ17,cpJ17,yerrJ17 = PreparePlot(bsp,source,days,hrs)
    source = 'CENA'
    tCE,cpCE,yerrCE = PreparePlot(bsp,source,days,hrs)

    #plot
    #-----------------------------------------------
    plt.figure(figsize=(15,6))
    if len(t3C)> 0:
        plt.errorbar(t3C,cp3C,xerr=0, yerr = 1.*yerr3C, fmt='bo', label = '3C279')
    if len(tOJ)> 0:
        plt.errorbar(tOJ,cpOJ,xerr=0, yerr = 1.*yerrOJ, fmt='ro', label = 'OJ287')
    if len(tJ1)> 0:
        plt.errorbar(tJ1,cpJ1,xerr=0, yerr = 1.*yerrJ1, fmt='go', label = 'J1924-2914')
    if len(t10)> 0:
        plt.errorbar(t10,cp10,xerr=0, yerr = 1.*yerr10, fmt='mo', label = '1055+018')
    if len(t17)> 0:
        plt.errorbar(t17,cp17,xerr=0, yerr = 1.*yerr17, fmt='ko', label = '1921-293')
    if len(t19)> 0:
        plt.errorbar(t19,cp19,xerr=0, yerr = 1.*yerr19, fmt='yo', label = 'CYGX-3')
    if len(tJ17)> 0:
        plt.errorbar(tJ17,cpJ17,xerr=0, yerr = 1.*yerrJ17, fmt='sb', label = 'J1733-1304')
    if len(tCE)> 0:
        plt.errorbar(tCE,cpCE,xerr=0, yerr = 1.*yerrCE, fmt='co', label = 'CENA')

    plt.xlabel('time',fontsize=15)
    plt.ylabel('closure phase [deg]',fontsize=15)
    plt.axhline(y=0,linewidth=2, color='k')
    plt.axvline(x=24.,linewidth=1,color='k',linestyle='--')
    plt.axvline(x=48.,linewidth=1,color='k',linestyle='--')
    plt.axis([np.amin(np.concatenate((t3C,tOJ,tJ1,t10,t17,t19,tJ17,tCE)))-0.5,np.amax(np.concatenate((t3C,tOJ,tJ1,t10,t17,t19,tJ17,tCE)))+0.5,-MaxAngle,MaxAngle])
    plt.title(str(Triangle)+', '+pol, fontsize=15)
    plt.legend()
    plt.show()
    if printSummary != 0:
        PrintSummaryTri(bsp)



def plotQuadranglePolar(alist,Quadrangle,pol,MaxErr=2.,method=0,Signature = [1,1,1,1]):

    DataLabel='amp'
    ErrorLabel = 'sigma'
    alist = DataQuadrangle(alist,Quadrangle,pol,method,Signature)
    days = [3597,3600,3601]; source = '3C279'
    #hrs = [0,24.,48.] #to add to time
    t3C,cp3C,yerr3C = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = 'OJ287'
    tOJ,cpOJ,yerrOJ = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = 'J1924-2914'
    tJ1,cpJ1,yerrJ1 = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = '1055+018'
    t10,cp10,yerr10 = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = '1749+096'
    t17,cp17,yerr17 = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = '1921-293'
    t19,cp19,yerr19 = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = 'CYGX-3'
    tCY,cpCY,yerrCY = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = 'J1733-1304'
    tJ17,cpJ17,yerrJ17 = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = 'CENA'
    tCE,cpCE,yerrCE = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)

    #plot
    #-----------------------------------------------
    plt.figure(figsize=(15,6))
    if len(t3C)> 0:
        plt.errorbar(t3C,cp3C,xerr=0, yerr = 1.*yerr3C, fmt='bo', label = '3C279')
    if len(tOJ)> 0:
        plt.errorbar(tOJ,cpOJ,xerr=0, yerr = 1.*yerrOJ, fmt='ro', label = 'OJ287')
    if len(tJ1)> 0:
        plt.errorbar(tJ1,cpJ1,xerr=0, yerr = 1.*yerrJ1, fmt='go', label = 'J1924-2914')
    if len(t10)> 0:
        plt.errorbar(t10,cp10,xerr=0, yerr = 1.*yerr10, fmt='mo', label = '1055+018')
    if len(t17)> 0:
        plt.errorbar(t17,cp17,xerr=0, yerr = 1.*yerr17, fmt='ko', label = '1921-293')
    if len(t19)> 0:
        plt.errorbar(t19,cp19,xerr=0, yerr = 1.*yerr19, fmt='yo', label = 'CYGX-3')
    if len(tJ17)> 0:
        plt.errorbar(tJ17,cpJ17,xerr=0, yerr = 1.*yerrJ17, fmt='sb', label = 'J1733-1304')
    if len(tCE)> 0:
        plt.errorbar(tCE,cpCE,xerr=0, yerr = 1.*yerrCE, fmt='co', label = 'CENA')

    plt.xlabel('time',fontsize=15)
    #plt.xticks(x, my_xticks)
    plt.ylabel('closure amplitudes',fontsize=15)
    plt.axhline(y=1.,linewidth=2, color='k')
    plt.axvline(x=24.,linewidth=1,color='k',linestyle='--')
    plt.axvline(x=48.,linewidth=1,color='k',linestyle='--')
    plt.axis([np.amin(np.concatenate((t3C,tOJ,tJ1,t10,t17,t19,tJ17,tCE)))-0.5,np.amax(np.concatenate((t3C,tOJ,tJ1,t10,t17,t19,tJ17,tCE)))+0.5,1./(1.+MaxErr),1.+MaxErr])
    plt.title(str(Quadrangle)+', '+pol, fontsize=15)
    plt.legend()
    plt.show()
    PrintSummaryQuad(alist)
    

def plotBaseline(alist,basename,pol,DataLabel='snr',ErrorLabel='snrstd',logscaley=False):

    #DataLabel='amp'
    #ErrorLabel = 'sigma'
    alist = DataBaseline(alist,basename,pol)
    days = [3597,3600,3601]; source = '3C279'
    #hrs = [0,24.,48.] #to add to time
    t3C,cp3C,yerr3C = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = 'OJ287'
    tOJ,cpOJ,yerrOJ = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = 'J1924-2914'
    tJ1,cpJ1,yerrJ1 = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = '1055+018'
    t10,cp10,yerr10 = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = '1749+096'
    t17,cp17,yerr17 = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = '1921-293'
    t19,cp19,yerr19 = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = 'CYGX-3'
    tCY,cpCY,yerrCY = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = 'J1733-1304'
    tJ17,cpJ17,yerrJ17 = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = 'CENA'
    tCE,cpCE,yerrCE = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)

    #plot
    #-----------------------------------------------
    plt.figure(figsize=(15,6))
    if len(t3C)> 0:
        plt.errorbar(t3C,cp3C,xerr=0, yerr = 1.*yerr3C, fmt='bo', label = '3C279')
    if len(tOJ)> 0:
        plt.errorbar(tOJ,cpOJ,xerr=0, yerr = 1.*yerrOJ, fmt='ro', label = 'OJ287')
    if len(tJ1)> 0:
        plt.errorbar(tJ1,cpJ1,xerr=0, yerr = 1.*yerrJ1, fmt='go', label = 'J1924-2914')
    if len(t10)> 0:
        plt.errorbar(t10,cp10,xerr=0, yerr = 1.*yerr10, fmt='mo', label = '1055+018')
    if len(t17)> 0:
        plt.errorbar(t17,cp17,xerr=0, yerr = 1.*yerr17, fmt='ko', label = '1921-293')
    if len(t19)> 0:
        plt.errorbar(t19,cp19,xerr=0, yerr = 1.*yerr19, fmt='yo', label = 'CYGX-3')
    if len(tJ17)> 0:
        plt.errorbar(tJ17,cpJ17,xerr=0, yerr = 1.*yerrJ17, fmt='sb', label = 'J1733-1304')
    if len(tCE)> 0:
        plt.errorbar(tCE,cpCE,xerr=0, yerr = 1.*yerrCE, fmt='co', label = 'CENA')

    plt.xlabel('time',fontsize=15)
    #plt.xticks(x, my_xticks)
    if logscaley==True:
        plt.yscale('log')
    plt.ylabel(DataLabel+' in 5s',fontsize=15)
    if (DataLabel=='t_coh_bias')|(DataLabel=='t_coh'):
        plt.ylabel(DataLabel,fontsize=15)
    #plt.axhline(y=1.,linewidth=2, color='k')
    plt.axvline(x=24.,linewidth=1,color='k',linestyle='--')
    plt.axvline(x=48.,linewidth=1,color='k',linestyle='--')
    plt.xlim((np.amin(np.concatenate((t3C,tOJ,tJ1,t10,t17,t19,tJ17,tCE)))-0.5,np.amax(np.concatenate((t3C,tOJ,tJ1,t10,t17,t19,tJ17,tCE)))+0.5,))
    plt.title(str(basename)+', '+pol, fontsize=15)
    plt.legend()
    plt.show()

    



def PreparePlot(bsp,source,days,hrs):
    t = []; cp =[]; yerr = []
    for x in range(len(days)):
        t.append(hrs[x] + np.asarray(bsp.loc[(bsp.index.get_level_values('source') == source)&(bsp.index.get_level_values('expt_no') == days[x]),['gmst']]))
        cp.append(np.asarray(bsp.loc[(bsp.index.get_level_values('source') == source)&(bsp.index.get_level_values('expt_no') == days[x]),['cphase']]))
        yerr.append(np.asarray(bsp.loc[(bsp.index.get_level_values('source') == source)&(bsp.index.get_level_values('expt_no') == days[x]),['sigmaCP']]))
    tf = np.concatenate(t).ravel()
    cpf = np.concatenate(cp).ravel()
    yerrf = np.concatenate(yerr).ravel()
    return tf, cpf, yerrf

def GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs):
    t = []; dat =[]; yerr = []
    for x in range(len(days)):
        tfoo = np.asarray(alist.loc[(alist.index.get_level_values('source') == source)&(alist.index.get_level_values('expt_no') == days[x]),['gmst']])
        tfoo =  tfoo + hrs[x]
        t.append(tfoo)
        dat.append(np.asarray(alist.loc[(alist.index.get_level_values('source') == source)&(alist.index.get_level_values('expt_no') == days[x]),[DataLabel]]))
        yerr.append(np.asarray(alist.loc[(alist.index.get_level_values('source') == source)&(alist.index.get_level_values('expt_no') == days[x]),[ErrorLabel]]))
    tf = np.concatenate(t).ravel()
    datf = np.concatenate(dat).ravel()
    yerrf = np.concatenate(yerr).ravel()
    return tf, datf, yerrf


def PrintSummaryQuad(alist):
    #alist with just this quadrangle
    qaf = alist
    sigmaLim = 0.25
    n1 = len(qaf['amp'])
    n2 = len(qaf.loc[qaf['sigma']<sigmaLim,'sigma'])
    n3 = len(qaf.loc[(qaf['sigma']<sigmaLim)&(np.abs(qaf['amp'] - 1. )<3.*qaf['sigma']),'sigma'])
    n4 = len(qaf.loc[(np.abs(qaf['amp'] - 1. )<0.1)&(np.abs(qaf['amp'] - 1. )>=3.*qaf['sigma']),'sigma'])
    n5 = len(qaf.loc[(qaf['sigma']>=sigmaLim)&(np.abs(qaf['amp'] - 1. )<3.*qaf['sigma']),'sigma'])
    print 'Total scans: ', n1 
    print 'Scans with sigma <', sigmaLim,': ', n2
    print 'Scans with sigma <', sigmaLim,', consistent with 4AMP==1 within 3 sigma: ', n3
    print 'Scans inconsistent with 4AMP==1 within 3 sigma, but error smaller than 0.1: ', n4
    print 'Scans consistent with 4AMP==1 within 3 sigma, but sigma > ', sigmaLim,': ', n5


def PrintSummaryTri(alist):
    #alist with just this quadrangle
    qaf = alist
    sigmaLim = 2.5 #deg
    n1 = len(qaf['cphase'])
    n2 = len(qaf.loc[qaf['sigmaCP']<sigmaLim,'sigmaCP'])
    n3 = len(qaf.loc[(qaf['sigmaCP']<sigmaLim)&(np.abs(qaf['cphase'] - 0. )<3.*qaf['sigmaCP']),'sigmaCP'])
    n4 = len(qaf.loc[(np.abs(qaf['cphase'] - 0.)< 1.5)&(np.abs(qaf['cphase'] - 0. )>=3.*qaf['sigmaCP']),'sigmaCP'])
    n5 = len(qaf.loc[(qaf['sigmaCP']>=sigmaLim)&(np.abs(qaf['cphase'] - 0. )<3.*qaf['sigmaCP']),'sigmaCP'])
    print 'Total scans: ', n1 
    print 'Scans with sigma <', sigmaLim,'deg: ', n2
    print 'Scans with sigma <', sigmaLim,'deg, consistent with CP==0 within 3 sigma: ', n3
    print 'Scans inconsistent with CP==0 within 3 sigma, but error smaller than 1.5 deg: ', n4
    print 'Scans consistent with CP==0 within 3 sigma, but sigma > ', sigmaLim,'deg: ', n5


def DataTriangle3(alist,Tri,signat,pol):
    condP = (alist['polarization']==pol)
    condB1 = (alist['baseline']==Tri[0])
    condB2 = (alist['baseline']==Tri[1])
    condB3 = (alist['baseline']==Tri[2])
    condB = condB1|condB2|condB3

    alistRR_Tri = alist.loc[condP&condB,['expt_no','scan_id','source','datetime','baseline','total_phas','amp','snr','gmst']]
    #throw away times without full triangle
    tlist = alistRR_Tri.groupby('datetime').filter(lambda x: len(x) > 2)
    tlist.loc[:,'sigma'] = (tlist.loc[:,'amp']/(tlist.loc[:,'snr']))
    for cou in range(3):
        tlist.loc[(tlist.loc[:,'baseline']==Tri[cou]),'total_phas'] *= signat[cou]*np.pi/180.
    #tlist
    #print('scan_idttt', sorted(list(set(tlist.loc[tlist['scan_id']==1,'gmst']))))
    #print(set(alistRR_Tri['gmst']))
    tlist.loc[:,'sigma'] = 1./tlist.loc[:,'snr']**2 #put 1/snr**2 in the sigma column to aggregate
    #first aggregation to get bispectra
    bsp = tlist.groupby(('expt_no','source','scan_id','datetime')).agg({'total_phas': lambda x: np.sum(x),
        'amp': lambda x: np.prod(x), 'sigma': lambda x: np.sqrt(np.sum(x)),'gmst': 'min'})
    #sigma above is the CLOSURE PHASE ERROR
    #print('scan_idttt', list(set(tlist.loc[bsp['scan_id']==1,'gmst'])).sorted)
    bsp.loc[:,'bisp'] = bsp.loc[:,'amp']*np.exp(1j*bsp.loc[:,'total_phas'])
    bsp.loc[:,'snr'] = 1./bsp.loc[:,'sigma']
    bsp.loc[:,'sigma'] = bsp.loc[:,'amp']*bsp.loc[:,'sigma'] #sigma of bispectrum

    #second aggregation to average the bispectra
    #print('ddd', set(bsp['gmst']))
    #print('scan_id', set(bsp['scan_id']))
    #bsp = bsp.groupby(('expt_no','source','scan_id')).agg({'bisp': np.average,'gmst' :'min','sigma': lambda x: np.sqrt(sum(x**2))/len(x)})
    #print('eee', set(bsp['gmst']))
    bsp.loc[:,'cphase'] = np.angle(bsp.loc[:,'bisp'])*180./np.pi #deg
    bsp.loc[:,'amp'] = np.abs(bsp.loc[:,'bisp'])
    bsp.loc[:,'snr'] = bsp.loc[:,'amp']/bsp.loc[:,'sigma']
    bsp.loc[:,'sigmaCP'] = 1./bsp.loc[:,'snr']*180./np.pi #deg
    #del bsp['bisp']
    bsp = bsp.reset_index()
    return bsp




def DataQuadrangle3(alist,Quad,pol, debias=1):

    condP = (alist['polarization']==pol)
    condB1 = (alist['baseline']==Quad[0])
    condB2 = (alist['baseline']==Quad[1])
    condB3 = (alist['baseline']==Quad[2])
    condB4 = (alist['baseline']==Quad[3])
    condB = condB1|condB2|condB3|condB4

    alist_Quad = alist.loc[condP&condB,['expt_no','scan_id','source','datetime','baseline','total_phas','amp','snr','gmst']]
    tlist = alist_Quad.groupby('datetime').filter(lambda x: len(x) > 3)
    tlist.loc[:,'sigma'] = tlist.loc[:,'amp']/(tlist.loc[:,'snr'])
    tlist.loc[:,'datetime_foo'] = tlist.loc[:,'datetime']
    #if debiasing amplitudes
    if debias != 0:
        tlist.loc[:,'amp'] = tlist.loc[:,'amp']*np.sqrt(1.- debias*2./tlist.loc[:,'snr']**2 )
        
    #####################################################################
    #form quadAmplitudes on 5s and average quadAmplitudes over whole scan
    #####################################################################
    
    for cou in range(2,4):
    #inverse amplitude for the visibilities in the denominator
        tlist.loc[(tlist.loc[:,'baseline']==Quad[cou]),'amp'] = 1./tlist.loc[(tlist.loc[:,'baseline']==Quad[cou]),'amp']

    #aggregating to get quadProducts on 5s segments 
    tlist.loc[:,'sigma'] = 1./tlist.loc[:,'snr']**2#put snr in place of sigma to sum for 4product

    quadAmp = tlist.groupby(('expt_no','source','scan_id','datetime_foo')).agg({'amp': lambda x: np.prod(x), 
                            'gmst': 'min', 'datetime': 'min', 'sigma' : lambda x: np.sqrt(np.sum(x)) })
    quadAmp.loc[:,'sigma'] = quadAmp.loc[:,'sigma']*quadAmp.loc[:,'amp']
    return quadAmp
