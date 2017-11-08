import numpy as np
import pandas as pd
from closures import *

def polar_CP_error_hi_lo(bisp_rr_hi, bisp_ll_hi, bisp_rr_lo, bisp_ll_lo):

    bisp_rr_hi, bisp_ll_hi = match_2_dataframes(bisp_rr_hi, bisp_ll_hi, 'triangle')
    bisp_rr_lo, bisp_ll_lo = match_2_dataframes(bisp_rr_lo, bisp_ll_lo, 'triangle')
    
    sigmaDif = np.sqrt(np.asarray(bisp_rr_hi['sigmaCP'])**2+np.asarray(bisp_ll_hi['sigmaCP'])**2)
    bisp_rr_hi['TotErr'] = np.abs(((np.asarray(bisp_rr_hi['cphase'])-np.asarray(bisp_ll_hi['cphase']))))
    bisp_rr_hi['TotErr'] = np.minimum(np.asarray(bisp_rr_hi['TotErr']),np.abs(np.asarray(bisp_rr_hi['TotErr']) -360))
    bisp_rr_hi['RelErr'] = np.asarray(bisp_rr_hi['TotErr'])/sigmaDif
    sigmaDif = np.sqrt(np.asarray(bisp_rr_lo['sigmaCP'])**2+np.asarray(bisp_ll_lo['sigmaCP'])**2)
    bisp_rr_lo['TotErr'] = np.abs(((np.asarray(bisp_rr_lo['cphase'])-np.asarray(bisp_ll_lo['cphase']))))
    bisp_rr_lo['TotErr'] = np.minimum(np.asarray(bisp_rr_lo['TotErr']),np.abs(np.asarray(bisp_rr_lo['TotErr']) -360))
    bisp_rr_lo['RelErr'] = np.asarray(bisp_rr_lo['TotErr'])/sigmaDif

    TriStat = pd.DataFrame({})
    AllTri = sorted(list(set(bisp_rr_hi.triangle)))
    TriStat['triangle'] = AllTri
    scans_RR_LL_lo = [np.shape(bisp_rr_lo[bisp_rr_lo.triangle == Tri])[0] for Tri in AllTri]
    scans_RR_LL_hi = [np.shape(bisp_rr_hi[bisp_rr_hi.triangle == Tri])[0] for Tri in AllTri]
    scans_RR_LL_lo_3sig = [np.shape(bisp_rr_lo[(bisp_rr_lo.triangle == Tri)&(bisp_rr_lo.RelErr < 3.)])[0] for Tri in AllTri]
    scans_RR_LL_hi_3sig = [np.shape(bisp_rr_hi[(bisp_rr_hi.triangle == Tri)&(bisp_rr_hi.RelErr < 3.)])[0] for Tri in AllTri]
    TriStat['sc_lo'] = scans_RR_LL_lo
    TriStat['sc__hi'] = scans_RR_LL_hi
    TriStat['sc_total'] = np.asarray(scans_RR_LL_hi)+np.asarray(scans_RR_LL_lo)
    TriStat['sc_3sig_lo'] = scans_RR_LL_lo_3sig
    TriStat['sc_3sig_hi'] = scans_RR_LL_hi_3sig
    TriStat['sc_3sig'] = np.asarray(scans_RR_LL_hi_3sig)+np.asarray(scans_RR_LL_lo_3sig)
    TriStat['sc_3sig_proc'] = np.asarray(map(float,TriStat['sc_3sig']))/np.asarray(map(float,TriStat['sc_total']))
    #TriStat['MedianAbs_lo'] = [np.median(np.asarray(bisp_rr_lo.loc[bisp_rr_lo['triangle'] == Tri,['TotErr']])) for Tri in AllTri]
    #TriStat['MedianAbs_hi'] = [np.median(np.asarray(bisp_rr_hi.loc[bisp_rr_hi['triangle'] == Tri,['TotErr'] ])) for Tri in AllTri]
    TriStat['MedianAbs'] = [np.median(np.asarray(list(bisp_rr_lo.loc[bisp_rr_lo['triangle'] == Tri,'TotErr'])+list(bisp_rr_hi.loc[bisp_rr_hi['triangle'] == Tri,'TotErr' ]))) for Tri in AllTri]
    TriStat = TriStat.sort_values('sc_3sig_proc')

    return TriStat


def polar_CP_error(bisp_rr, bisp_ll):

    bisp_rr, bisp_ll = match_2_dataframes(bisp_rr, bisp_ll, 'triangle')
    
    sigmaDif = np.sqrt(np.asarray(bisp_rr['sigmaCP'])**2+np.asarray(bisp_ll['sigmaCP'])**2)
    bisp_rr['TotErr'] = np.abs(((np.asarray(bisp_rr['cphase'])-np.asarray(bisp_ll['cphase']))))
    bisp_rr['TotErr'] = np.minimum(np.asarray(bisp_rr['TotErr']),np.abs(np.asarray(bisp_rr['TotErr']) -360))
    bisp_rr['RelErr'] = np.asarray(bisp_rr['TotErr'])/sigmaDif
    
    TriStat = pd.DataFrame({})
    AllTri = sorted(list(set(bisp_rr.triangle)))
    TriStat['triangle'] = AllTri
    scans_RR_LL = [np.shape(bisp_rr[bisp_rr.triangle == Tri])[0] for Tri in AllTri]
    scans_RR_LL_3sig = [np.shape(bisp_rr[(bisp_rr.triangle == Tri)&(bisp_rr.RelErr < 3.)])[0] for Tri in AllTri]
    TriStat['sc_total'] = np.asarray(scans_RR_LL)
    TriStat['sc_3sig'] = scans_RR_LL_3sig
    TriStat['sc_3sig_proc'] = np.asarray(map(float,TriStat['sc_3sig']))/np.asarray(map(float,TriStat['sc_total']))
    #TriStat['MedianAbs_lo'] = [np.median(np.asarray(bisp_rr_lo.loc[bisp_rr_lo['triangle'] == Tri,['TotErr']])) for Tri in AllTri]
    #TriStat['MedianAbs_hi'] = [np.median(np.asarray(bisp_rr_hi.loc[bisp_rr_hi['triangle'] == Tri,['TotErr'] ])) for Tri in AllTri]
    TriStat['MedianAbs'] = [np.median(np.asarray(list(bisp_rr.loc[bisp_rr['triangle'] == Tri,'TotErr']))) for Tri in AllTri]
    TriStat = TriStat.sort_values('sc_3sig_proc')

    return TriStat

def polar_CP_error_source(bisp_rr, bisp_ll):

    AllSo = sorted(list( set(bisp_rr.source) ))
    bisp_rr, bisp_ll = match_2_dataframes(bisp_rr, bisp_ll, 'triangle')
    
    sigmaDif = np.sqrt(np.asarray(bisp_rr['sigmaCP'])**2+np.asarray(bisp_ll['sigmaCP'])**2)
    bisp_rr['TotErr'] = np.abs(((np.asarray(bisp_rr['cphase'])-np.asarray(bisp_ll['cphase']))))
    bisp_rr['TotErr'] = np.minimum(np.asarray(bisp_rr['TotErr']),np.abs(np.asarray(bisp_rr['TotErr']) -360))
    bisp_rr['RelErr'] = np.asarray(bisp_rr['TotErr'])/sigmaDif
    
    TriStat = pd.DataFrame({})
    #AllTri = sorted(list(set(bisp_rr.triangle)))
    TriStat['source'] = AllSo
    scans_RR_LL = [np.shape(bisp_rr[bisp_rr.source== So])[0] for So in AllSo]
    scans_RR_LL_3sig = [np.shape(bisp_rr[(bisp_rr.source == So)&(bisp_rr.RelErr < 3.)])[0] for So in AllSo]
    TriStat['sc_total'] = np.asarray(scans_RR_LL)
    TriStat['sc_3sig'] = scans_RR_LL_3sig
    TriStat['sc_3sig_proc'] = np.asarray(map(float,TriStat['sc_3sig']))/np.asarray(map(float,TriStat['sc_total']))
    TriStat['MedianAbs'] = [np.median(np.asarray(list(bisp_rr.loc[bisp_rr['source'] == So,'TotErr']))) for So in AllSo]
    TriStat = TriStat.sort_values('sc_3sig_proc')

    return TriStat


def polar_CP_error_station(bisp_rr, bisp_ll):

    AllSt = list(set(''.join(list(set(bisp_rr.triangle)))))
    bisp_rr, bisp_ll = match_2_dataframes(bisp_rr, bisp_ll, 'triangle')
    
    sigmaDif = np.sqrt(np.asarray(bisp_rr['sigmaCP'])**2+np.asarray(bisp_ll['sigmaCP'])**2)
    bisp_rr['TotErr'] = np.abs(((np.asarray(bisp_rr['cphase'])-np.asarray(bisp_ll['cphase']))))
    bisp_rr['TotErr'] = np.minimum(np.asarray(bisp_rr['TotErr']),np.abs(np.asarray(bisp_rr['TotErr']) -360))
    bisp_rr['RelErr'] = np.asarray(bisp_rr['TotErr'])/sigmaDif
    
    TriStat = pd.DataFrame({})
    #AllTri = sorted(list(set(bisp_rr.triangle)))
    TriStat['station'] = AllSt
    scans_RR_LL = [np.shape(bisp_rr[map(lambda x: St in x, bisp_rr.triangle)])[0] for St in AllSt]
    scans_RR_LL_3sig = [np.shape(bisp_rr[(map(lambda x: St in x, bisp_rr.triangle))&(bisp_rr.RelErr < 3.)])[0] for St in AllSt]
    TriStat['sc_total'] = np.asarray(scans_RR_LL)
    TriStat['sc_3sig'] = scans_RR_LL_3sig
    TriStat['sc_3sig_proc'] = np.asarray(map(float,TriStat['sc_3sig']))/np.asarray(map(float,TriStat['sc_total']))
    TriStat['MedianAbs'] = [np.median(np.asarray(list(bisp_rr.loc[map(lambda x: St in x, bisp_rr['triangle']),'TotErr']))) for St in AllSt]
    TriStat = TriStat.sort_values('sc_3sig_proc')

    return TriStat

def triv_CP_error(bisp):

    AllTri = sorted(list( set(bisp.triangle) ))
    
    bisp['TotErr'] = np.mod(np.asarray(bisp['cphase']),360.)
    bisp['TotErr'] = np.minimum( np.asarray(bisp['TotErr']), np.abs(np.asarray(bisp['TotErr']) -360.))
    bisp['RelErr'] = np.asarray(bisp['TotErr'])/np.asarray(bisp['sigmaCP'])

    #print(bisp)
    TriStat = pd.DataFrame({})
    
    TriStat['triangle'] = AllTri
    scans_tot = [np.shape(bisp[bisp.triangle == Tri])[0] for Tri in AllTri]
    scans_3sig = [np.shape(bisp[(bisp.triangle == Tri)&(bisp.RelErr < 3.)])[0] for Tri in AllTri]
    TriStat['sc_total'] = scans_tot
    TriStat['sc_3sig'] = scans_3sig
    TriStat['sc_3sig_proc'] = np.asarray(map(float,TriStat['sc_3sig']))/np.asarray(map(float,TriStat['sc_total']))
    TriStat['MedianAbs'] = [np.median(np.asarray(list(bisp.loc[bisp['triangle'] == Tri,'TotErr']))) for Tri in AllTri]
    TriStat['MedianSigma'] = [np.median(np.asarray(list(bisp.loc[bisp['triangle'] == Tri,'sigmaCP']))) for Tri in AllTri]
    
    TriStat = TriStat.sort_values('sc_3sig_proc')

    return TriStat


def triv_CP_error_station(bisp):

    AllTri = sorted(list( set(bisp.triangle) ))
    AllSt = list(set(''.join(list(set(bisp.triangle)))))

    bisp['TotErr'] = np.mod(np.asarray(bisp['cphase']),360.)
    bisp['TotErr'] = np.minimum( np.asarray(bisp['TotErr']), np.abs(np.asarray(bisp['TotErr']) -360.))
    bisp['RelErr'] = np.asarray(bisp['TotErr'])/np.asarray(bisp['sigmaCP'])

    TriStat = pd.DataFrame({})
    
    TriStat['station'] = AllSt

    scans_tot = [np.shape(bisp[map(lambda x: St in x, bisp.triangle)])[0] for St in AllSt]
    scans_3sig = [np.shape(bisp[map(lambda x: St in x, bisp.triangle)&(bisp.RelErr < 3.)])[0] for St in AllSt]
    TriStat['sc_total'] = scans_tot
    TriStat['sc_3sig'] = scans_3sig
    TriStat['sc_3sig_proc'] = np.asarray(map(float,TriStat['sc_3sig']))/np.asarray(map(float,TriStat['sc_total']))
    TriStat['MedianAbs'] = [np.median(np.asarray(list(bisp.loc[map(lambda x: St in x, bisp.triangle),'TotErr']))) for St in AllSt]
    TriStat['MedianSigma'] = [np.median(np.asarray(list(bisp.loc[map(lambda x: St in x, bisp.triangle),'sigmaCP']))) for St in AllSt]
    
    TriStat = TriStat.sort_values('sc_3sig_proc')

    return TriStat


def triv_CP_error_baseline(bisp):
    #IN PROGRESS
    #AllTri = sorted(list( set(bisp.triangle) ))
    #AllBa = list(set(''.join(list(set(bisp.triangle)))))

    #bisp['TotErr'] = np.mod(np.asarray(bisp['cphase']),360.)
    #bisp['TotErr'] = np.minimum( np.asarray(bisp['TotErr']), np.abs(np.asarray(bisp['TotErr']) -360.))
    #bisp['RelErr'] = np.asarray(bisp['TotErr'])/np.asarray(bisp['sigmaCP'])

    #TriStat = pd.DataFrame({})
    
    #TriStat['station'] = AllSt

    #scans_tot = [np.shape(bisp[map(lambda x: St in x, bisp.triangle)])[0] for St in AllSt]
    #scans_3sig = [np.shape(bisp[map(lambda x: St in x, bisp.triangle)&(bisp.RelErr < 3.)])[0] for St in AllSt]
    #TriStat['sc_total'] = scans_tot
    #TriStat['sc_3sig'] = scans_3sig
    #TriStat['sc_3sig_proc'] = np.asarray(map(float,TriStat['sc_3sig']))/np.asarray(map(float,TriStat['sc_total']))
    #TriStat['MedianAbs'] = [np.median(np.asarray(list(bisp.loc[map(lambda x: St in x, bisp.triangle),'TotErr']))) for St in AllSt]
    #TriStat['MedianSigma'] = [np.median(np.asarray(list(bisp.loc[map(lambda x: St in x, bisp.triangle),'sigmaCP']))) for St in AllSt]
    
    #TriStat = TriStat.sort_values('sc_3sig_proc')

    return bisp

def triv_CP_error_source(bisp):

    AllSo = sorted(list( set(bisp.source) ))
    #AllSt = list(set(''.join(list(set(bisp.triangle)))))

    bisp['TotErr'] = np.mod(np.asarray(bisp['cphase']),360.)
    bisp['TotErr'] = np.minimum( np.asarray(bisp['TotErr']), np.abs(np.asarray(bisp['TotErr']) -360.))
    bisp['RelErr'] = np.asarray(bisp['TotErr'])/np.asarray(bisp['sigmaCP'])

    TriStat = pd.DataFrame({})
    
    TriStat['source'] = AllSo

    scans_tot = [np.shape(bisp[bisp.source==So])[0] for So in AllSo]
    scans_3sig = [np.shape(bisp[(bisp.source==So)&(bisp.RelErr < 3.)])[0] for So in AllSo]
    TriStat['sc_total'] = scans_tot
    TriStat['sc_3sig'] = scans_3sig
    TriStat['sc_3sig_proc'] = np.asarray(map(float,TriStat['sc_3sig']))/np.asarray(map(float,TriStat['sc_total']))
    TriStat['MedianAbs'] = [np.median(np.asarray(list(bisp.loc[bisp['source']==So,'TotErr']))) for So in AllSo]
    TriStat['MedianSigma'] = [np.median(np.asarray(list(bisp.loc[bisp['source']==So,'sigmaCP']))) for So in AllSo]
    
    TriStat = TriStat.sort_values('sc_3sig_proc')

    return TriStat



def add_band_error(bisp,band):

    bisp['TotErr'] = np.mod(np.asarray(bisp['cphase']),360.)
    bisp['TotErr'] = np.minimum( np.asarray(bisp['TotErr']), np.abs(np.asarray(bisp['TotErr']) -360.))
    bisp['RelErr'] = np.asarray(bisp['TotErr'])/np.asarray(bisp['sigmaCP'])
    bisp['band'] = [band]*np.shape(bisp)[0]
    return bisp

