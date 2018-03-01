from __future__ import print_function
import sys, os, datetime, itertools
import scipy.special as ss
import scipy.optimize as so
import scipy.stats as st 
import pandas as pd
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from eat.io import hops, util
from eat.hops import util as hu
from eat.aips import aips2alist as a2a
from eat.inspect import closures as closures
import statsmodels.stats.stattools as sss
import statsmodels.robust.scale as srs
from sklearn.cluster import KMeans
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, ICRS, Angle
from astropy.time import Time, TimeDelta

#BUNCH OF FUNCTIONS FOR GENERAL DESCRIPTION OF SCAN

def circular_mean(theta):
    theta = np.asarray(theta, dtype=np.float32)*np.pi/180.
    theta = theta[theta==theta]
    if len(theta)==0:
        return None
    else:
        C = np.mean(np.cos(theta))
        S = np.mean(np.sin(theta))
        mt = np.arctan2(S,C)*180./np.pi
        return np.mod(mt,360)

def unb_amp_no_trend(t_A): 
    #t_A = list(zip(fooLoc.mjd,fooLoc.amp))
    time = np.asarray([x[0] for x in t_A])
    time = time - np.mean(time)
    amp = np.asarray([x[1] for x in t_A])
    #removing linear trend, but not the mean value
    amp = amp-np.polyfit(time,amp,1)[0]*time
    amp2 = np.asarray(amp, dtype=np.float32)**2
    m = np.mean(amp2)
    s = np.std(amp2)
    delta = m**2 - s**2
    if delta >=0:
        a0 = delta**(0.25)
        #s0 = np.sqrt((m -np.sqrt(delta))/2.)
    else:
        a0 = np.nan
        #s0 = np.nan
    return a0

def unb_std_no_trend(t_A): 
    #t_A = list(zip(fooLoc.mjd,fooLoc.amp))
    time = np.asarray([x[0] for x in t_A])
    time = time - np.mean(time)
    amp = np.asarray([x[1] for x in t_A])
    #removing linear trend, but not the mean value
    amp = amp-np.polyfit(time,amp,1)[0]*time
    amp2 = np.asarray(amp, dtype=np.float32)**2
    m = np.mean(amp2)
    s = np.std(amp2)
    delta = m**2 - s**2
    if delta >=0:
        #a0 = delta**(0.25)
        s0 = np.sqrt((m -np.sqrt(delta))/2.)
    else:
        #a0 = np.nan
        s0 = np.nan
    return s0

def cut_outliers_circ(vector,no_sigmas):
    #cuts outliers that are in distance from mean value larger than no_sigmas
    vector = vector[vector==vector]
    sigma = circular_std(vector)
    m = circular_mean(vector)
    dist = np.minimum(np.abs(vector - m), np.abs(360. - np.abs(vector - m)))
    vector = vector[dist < no_sigmas*sigma]
    return vector

def circ_cut_and_mean(vector,no_sigmas):
    vector = vector[vector==vector]
    vector = cut_outliers_circ(vector,no_sigmas)
    return circular_mean(vector)

def circ_cut_and_std(vector,no_sigmas):
    vector = vector[vector==vector]
    vector = cut_outliers_circ(vector,no_sigmas)
    return circular_std(vector)

def circular_std(theta):
    theta = theta[theta==theta]
    theta = np.asarray(theta, dtype=np.float32)*np.pi/180.
    C = np.mean(np.cos(theta))
    S = np.mean(np.sin(theta))
    st = np.sqrt(-2.*np.log(np.sqrt(C**2+S**2)))*180./np.pi
    return st

def circular_std_of_mean(theta):
    return circular_std(theta)/np.sqrt(len(theta))

def mean(theta):
    theta = np.asarray(theta, dtype=np.float32)
    return np.mean(theta)

def median(theta):
    theta = np.asarray(theta, dtype=np.float32)
    return np.median(theta)

def minv(theta):
    theta = np.asarray(theta, dtype=np.float32)
    return np.min(theta)

def maxv(theta):
    theta = np.asarray(theta, dtype=np.float32)
    return np.max(theta)

def std(theta):
    theta = np.asarray(theta, dtype=np.float32)
    return np.std(theta)

def unbiased_amp(amp):
    amp2 = np.asarray(amp, dtype=np.float32)**2
    m = np.mean(amp2)
    s = np.std(amp2)
    delta = m**2 - s**2
    if delta >=0:
        a0 = delta**(0.25)
    else:
        a0 = 0.*(m**2)**(0.25)
    return a0

def unbiased_amp_boot(amp):
    amp2 = np.asarray(amp, dtype=np.float32)**2
    #m = np.mean(amp2)
    #s = np.std(amp2)
    m = bootstrap(amp2,N,np.mean)[0]
    s = bootstrap(amp2,N,np.std)[0]
    delta = m**2 - s**2
    if delta >=0:
        a0 = delta**(0.25)
    else:
        a0 = 0.*(m**2)**(0.25)
    return a0

def unbiased_std(amp):
    amp2 = np.asarray(amp, dtype=np.float32)**2
    m = np.mean(amp2)
    s = np.std(amp2)
    delta = m**2 - s**2
    if delta >= 0:
        s0 = np.sqrt((m -np.sqrt(delta))/2.)
    else:
        s0 = 0.*np.sqrt(m/2.)
    return s0

def unbiased_snr(amp):
    return unbiased_amp(amp)/unbiased_std(amp)

def skew(theta):
    theta = np.asarray(theta, dtype=np.float32)
    return st.skew(theta)

def kurt(theta):
    theta = np.asarray(theta, dtype=np.float32)
    return st.kurtosis(theta)

def mad(theta):
    theta = np.asarray(theta, dtype=np.float32)
    madev = float(srs.mad(theta))
    return madev

def circular_mad(theta):
    theta = np.asarray(theta, dtype=np.float32)*np.pi/180.
    C = np.median(np.cos(theta))
    S = np.median(np.sin(theta))
    st = np.sqrt(-2.*np.log(np.sqrt(C**2+S**2)))*180./np.pi
    return st

def medcouple(theta):
    theta = np.asarray(theta, dtype=np.float32)
    mc = float(sss.medcouple(theta))
    return mc

def circular_median(theta):
    theta = np.asarray(theta, dtype=np.float32)*np.pi/180.
    theta = theta[theta==theta]  
    if len(theta)==0:
        return None
    else:
        C = np.median(np.cos(theta))
        S = np.median(np.sin(theta))
        mt = np.arctan2(S,C)*180./np.pi
        return mt

def do_quart(theta):
    theta = np.asarray(theta, dtype=np.float32)
    q1 = np.percentile(theta,25)
    return q1

def up_quart(theta):
    theta = np.asarray(theta, dtype=np.float32)
    q3 = np.percentile(theta,75)
    return q3

def iqr(theta):
    theta = np.asarray(theta, dtype=np.float32)
    return st.iqr(theta)

def range_adjust_box(vec,scaler=1.5):
    vec = np.asarray(vec, dtype=np.float32)
    #print(len(vec))
    quart3 = np.percentile(vec,75)
    quart1 = np.percentile(vec,25)
    iqr = quart3-quart1
    mc = float(sss.medcouple(vec))
    if mc > 0:
        whisk_plus = scaler*iqr*np.exp(3*mc)
        whisk_min = scaler*iqr*np.exp(-4*mc)
    else:
        whisk_plus = scaler*iqr*np.exp(4*mc)
        whisk_min = scaler*iqr*np.exp(-3*mc)       
    range_plus = quart3 + whisk_plus
    range_min = quart1 - whisk_min   
    return [range_min, range_plus]
def adj_box_outlier(vec,scaler=2.):
    vec = np.asarray(vec, dtype=np.float32)
    vec_no_nan=vec[vec==vec]
    if len(vec_no_nan)>0:
        range_box = range_adjust_box(vec_no_nan,scaler)
    else: range_box=[0,0]
    is_in = (vec<range_box[1])&(vec>range_box[0])
    return 1-is_in

def adj_box_outlier_minus(vec):
    range_box = range_adjust_box(vec)
    is_out = (vec<range_box[0])
    return is_out
def adj_box_outlier_plus(vec):
    range_box = range_adjust_box(vec)
    is_out = (vec>range_box[1])
    return is_out

def number_out(vec):
    return len(adj_box_outlier(vec))

def correlate_tuple(x):
    time = np.asarray([y[0] for y in x])
    sth = np.asarray([y[1] for y in x])
    r = st.pearsonr(time,sth)[0]
    return r

def detect_dropouts_kmeans(x):
    foo_data = list(zip(np.zeros(len(x)),x))
    dataKM = [list(y) for y in foo_data]
    dataKM = np.asarray(dataKM)
    test1 = KMeans(n_clusters=1, random_state=0).fit(dataKM)
    test2 = KMeans(n_clusters=2, random_state=0).fit(dataKM)
    return test1.inertia_/test2.inertia_

def get_dropout_indicator(x,inertia_treshold=3.9):
    foo_data = list(zip(np.zeros(len(x)),x))
    dataKM = [list(y) for y in foo_data]
    dataKM = np.asarray(dataKM)
    test1 = KMeans(n_clusters=1, random_state=0).fit(dataKM)
    test2 = KMeans(n_clusters=2, random_state=0).fit(dataKM)
    inertia_ratio = test1.inertia_/test2.inertia_
    indic = np.zeros(len(x))
    #print(test2.cluster_centers_)
    #if inertia_ratio > inertia_treshold:
    #    indic = test2.labels_
    #    if test2.cluster_centers_[1][1] > test2.cluster_centers_[0][1]:
    #        indic = 1-indic
    return indic

def get_outlier_indicator(x,scaler=2.):
    out_ind = adj_box_outlier(x,scaler)
    return out_ind

def bootstrap(data, num_samples, statistic, alpha=0.05):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""
    data = np.asarray(data)
    data = np.asarray(data)
    n = len(data)
    idx = npr.randint(0, n, (num_samples, n))
    samples = data[idx]
    stat = np.sort(statistic(samples, 1))
    return np.median(stat),stat[int((alpha/2.0)*num_samples)], stat[int((1-alpha/2.0)*num_samples)]
    #return stat[int(num_samples/2.)]

def bootstrap2(data, num_samples, statistic, alpha=0.05):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""

    stat = np.zeros(num_samples)
    data = np.asarray(data)
    n = len(data)
    idx = npr.randint(0, n, (num_samples, n))
    samples = data[idx]
    for cou in range(num_samples):
        stat[cou] = statistic(samples[cou,:])
    stat = np.sort(stat)
    return np.median(stat),stat[int((alpha/2.0)*num_samples)], stat[int((1-alpha/2.0)*num_samples)]
    #return stat[int(num_samples/2.)]

def scans_statistics(alist,statL='all'):
    if 'band' not in alist.columns:
        alist = add_band(alist,None)
    if 'amp_no_ap' not in alist.columns:
        alist['amp_no_ap']=alist['amp']
    if 'scan_id' not in alist.columns:
        alist['scan_id']=alist['scan_no_tot']
    if 'resid_phas' not in alist.columns:
        alist['resid_phas']=alist['phase']
    foo = alist[['scan_id','polarization','expt_no','band','datetime','resid_phas','amp','baseline','source','amp_no_ap','mjd']]
    time_0 = list(foo['datetime'])[0]
    # means
    foo['mean_amp'] = foo['amp']
    foo['mean_phase'] = foo['resid_phas']
    foo['median_amp'] = foo['amp']
    foo['median_phase'] = foo['resid_phas']
    foo['unbiased_amp'] = foo['amp']
    foo['unb_dtr_amp'] = list(zip(foo.mjd,foo.amp))
    foo['rice_amp'] = foo['amp']
    foo['inv_amp'] = foo['amp']
    
    # variation
    foo['std_amp'] = foo['amp']
    foo['std_phase'] = foo['resid_phas']
    foo['unbiased_std'] = foo['amp']
    foo['unbiased_std_no_ap'] = foo['amp_no_ap']
    foo['unbiased_snr'] = foo['amp']
    foo['unb_dtr_std'] = list(zip(foo.mjd,foo.amp))
    foo['rice_std'] = foo['amp']
    foo['inv_std'] = foo['amp']
    foo['mad_amp'] = foo['amp']
    foo['mad_phase'] = foo['resid_phas']
    foo['iqr_amp'] = foo['amp']
    foo['iqr_phase'] = foo['resid_phas']
    foo['q1_amp'] = foo['amp']
    foo['q3_amp'] = foo['amp']
    foo['q1_phase'] = foo['resid_phas']
    foo['q3_phase'] = foo['resid_phas']
    foo['max_amp'] = foo['amp']
    foo['min_amp'] = foo['amp']
    foo['max_phas'] = foo['resid_phas']
    foo['min_phas'] = foo['resid_phas']

    # skewness
    '''
    foo['skew_amp'] = foo['amp']
    foo['skew_phas'] = foo['resid_phas']
    foo['medc_amp'] = foo['amp']
    foo['medc_phas'] = foo['resid_phas']
    '''
    #time correlation
    time_sec = list(map(lambda x: float((x - time_0).seconds), foo['datetime']))
    foo['corr_phase'] = list(zip(time_sec,foo['resid_phas']))
    foo['corr_amp'] = list(zip(time_sec,foo['amp']))

    # other
    
    foo['length'] = foo['amp']
    '''
    foo['number_out'] = foo['amp']

    foo['kurt_amp'] = foo['amp']
    foo['kurt_phase'] = foo['resid_phas']
    
    # dropout detection
    foo['dropout'] = foo['amp']
    '''

    scans_stats = foo.groupby(('scan_id','polarization','expt_no','band','baseline','source')).agg(
    { 'datetime': min,
    'mean_amp': mean,
    'mean_phase': circular_mean,
    'median_amp': median,
    'median_phase': circular_median,
    'unbiased_amp': unbiased_amp,
    #'unb_dtr_amp': unb_amp_no_trend,
    'rice_amp': find_rice_amp_low_snr,
    'inv_amp': Koay_inv_amp,
    'std_amp': std,
    'std_phase': circular_std,
    'unbiased_std': unbiased_std,
    #'unb_dtr_std': unb_std_no_trend,
    'rice_std': find_rice_sig_low_snr,
    'inv_std': Koay_inv_std,
    'unbiased_std_no_ap': unbiased_std,
    'unbiased_snr': unbiased_snr,
    'mad_amp': mad,
    'mad_phase': circular_mad,
    'iqr_amp': iqr,
    'iqr_phase': iqr,
    'q1_amp': do_quart,
    'q3_amp': up_quart,
    'q1_phase': do_quart,
    'q3_phase': up_quart,
    'max_amp': maxv,
    'min_amp': minv,
    'max_phas': maxv,
    'min_phas': minv,
    'skew_amp': skew,
    'skew_phas': skew,
    'medc_amp': medcouple,
    'medc_phas': medcouple,
    'corr_phase': correlate_tuple,
    'corr_amp': correlate_tuple,
    'length': len,
    'number_out': number_out,
    'kurt_amp': kurt,
    'kurt_phase': kurt,
    'dropout': detect_dropouts_kmeans
    })

    return scans_stats.reset_index()

def add_drop_out_indic(alist):
    alist.sort_values('datetime', inplace=True)
    #foo = alist[['scan_id','polarization','expt_no','band','baseline','source','amp']]
  
    #print(alist)
    foo = alist[['scan_id','polarization','expt_no','band','baseline','source','amp']]
    #foo['drop_ind'] = list(zip(np.arange(len(foo['amp'])),foo['amp']))
    foofoo = foo.groupby(('scan_id','polarization','expt_no','band','baseline','source')).transform(lambda x: get_dropout_indicator(x))
    #foo = foo.transform()
    alist['drop_ind'] = foofoo
    return alist

def add_outlier_indic(alist,scaler=2.0):
    if 'outl_ind' in alist.columns:
        alist.drop('outl_ind',axis=1,inplace=True)
    #alist.sort_values('datetime', inplace=True)
    foo = alist[['scan_id','polarization','expt_no','band','baseline','source','amp']]
    foofoo = foo.groupby(('scan_id','polarization','expt_no','band','baseline','source')).transform(lambda x: get_outlier_indicator(x,scaler=scaler))
    alist['outl_ind'] = foofoo
    return alist


def coh_avg_vis_empiric(alist, tcoh='scan'):

    if 'sigma' not in alist.columns:
        alist.loc[:,'sigma'] = alist.loc[:,'amp']/alist.loc[:,'snr']
    if 'band' not in alist.columns:
        alist.loc[:,'band'] = [None]*np.shape(alist)[0]
    if 'phase' not in alist.columns:
        alist.loc[:,'phase'] = alist.loc[:,'resid_phas']
    if 'u' not in alist.columns:
        alist.loc[:,'u'] = [np.nan]*np.shape(alist)[0]
    if 'v' not in alist.columns:
        alist.loc[:,'v'] = [np.nan]*np.shape(alist)[0]
    if 'snr' not in alist.columns:
        alist.loc[:,'snr'] = [np.nan]*np.shape(alist)[0]
    
    alist_loc = alist[['expt_no','source','band','scan_id','polarization','baseline','datetime','u','v','amp','phase']]
    #these quantities are estimated from statistics of amp so we substitute amp in here
    alist_loc['sigma'] = alist_loc['amp']
    alist_loc['snr'] = alist_loc['amp']

    if tcoh=='scan':
        alist_loc = alist_loc.groupby(('expt_no','source','band','scan_id','polarization','baseline')).agg({'datetime': 'min','u': 'mean', 'v': 'mean', 'amp': unbiased_amp, 'phase': circular_mean, 'sigma': lambda x: unbiased_std(x)/np.sqrt(len(x)), 'snr': lambda x : unbiased_snr(x)*np.sqrt(len(x)) })
    else:
        alist_loc['round_time'] = list(map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/tcoh),alist_loc['datetime']))
        alist_loc = alist_loc.groupby(('expt_no','source','band','scan_id','polarization','baseline','round_time')).agg({'datetime': 'min','u': 'mean', 'v': 'mean', 'amp': unbiased_amp,
           'phase': circular_mean, 'sigma': lambda x: unbiased_std(x)/np.sqrt(len(x)), 'snr': lambda x : unbiased_snr(x)*np.sqrt(len(x)) })

    return alist_loc.reset_index()


def coh_avg_vis_thermal(alist, tcoh='scan'):
    if 'sigma' not in alist.columns:
        alist.loc[:,'sigma'] = alist.loc[:,'amp']/alist.loc[:,'snr']
    if 'band' not in alist.columns:
        alist.loc[:,'band'] = [None]*np.shape(alist)[0]
    if 'phase' not in alist.columns:
        alist.loc[:,'phase'] = alist.loc[:,'resid_phas']
    if 'u' not in alist.columns:
        alist.loc[:,'u'] = [np.nan]*np.shape(alist)[0]
    if 'v' not in alist.columns:
        alist.loc[:,'v'] = [np.nan]*np.shape(alist)[0]
    if 'snr' not in alist.columns:
        alist.loc[:,'snr'] = [np.nan]*np.shape(alist)[0]
    
    #alist.loc[:,'median'] = alist.loc[:,'amp']
    #list.loc[:,'mad'] = alist.loc[:,'amp']
    alist_loc = alist[['expt_no','source','band','scan_id','polarization','baseline','datetime','u','v','snr','sigma']]
    alist_loc['vis'] = alist['amp']*np.exp(1j*alist['phase']*np.pi/180)
    
    if tcoh=='scan':
        alist_loc = alist_loc.groupby(('expt_no','source','band','scan_id','polarization','baseline')).agg({'datetime': 'min','u': 'mean', 'v': 'mean', 'vis': np.mean, 
            'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x), 'snr': lambda x : np.sqrt(np.sum(x**2)) })
    else:
        alist_loc['round_time'] = list(map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/tcoh),alist_loc['datetime']))
        alist_loc = alist_loc.groupby(('expt_no','source','band','scan_id','polarization','baseline','round_time')).agg({'datetime': 'min','u': 'mean', 'v': 'mean', 'vis': np.mean,
            'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x), 'snr': lambda x : np.sqrt(np.sum(x**2))  })
        #alist_loc = alist_loc.groupby(('expt_no','source','band','scan_id','polarization','baseline','round_time')).agg({ 'vis': np.mean })


    alist_loc['phase'] = np.angle(np.asarray(alist_loc['vis']))*180/np.pi
    alist_loc['amp'] = np.abs(np.asarray(alist_loc['vis']))
    
    return alist_loc.reset_index()

def add_baselength(alist):
    alist['baselength'] = np.sqrt(np.asarray(alist.u)**2+np.asarray(alist.v)**2)
    return alist

def drop_outliers(alist):
    alist[np.isfinite(alist['amp'])&(alist['outl_ind']==0)]
    return alist

def add_mjd(alist):
    alist['mjd'] = Time(list(alist.datetime)).mjd
    return alist

def add_fmjd(alist):
    alist['fmjd'] = list(map(lambda x: x%1 ,Time(list(alist.datetime)).mjd))
    return alist

def add_band(alist,band):
    alist['band'] = [band]*np.shape(alist)[0]
    return alist

def add_sigma(alist):
    alist['sigma'] = alist['amp']/alist['snr']
    return alist

def add_col(alist,what_add,what_val):
    alist[what_add] = [what_val]*np.shape(alist)[0]
    return alist


def fit_circ(x,y):
    
    x_m = np.mean(x)
    y_m = np.mean(y)
    u = x - x_m
    v = y - y_m
    Suv  = sum(u*v)
    Suu  = sum(u**2)
    Svv  = sum(v**2)
    Suuv = sum(u**2 * v)
    Suvv = sum(u * v**2)
    Suuu = sum(u**3)
    Svvv = sum(v**3) 
    A = np.array([ [ Suu, Suv ], [Suv, Svv]])
    B = np.array([ Suuu + Suvv, Svvv + Suuv ])/2.0
    uc, vc = np.linalg.solve(A, B)
    xc_1 = x_m + uc
    yc_1 = y_m + vc
    Ri_1 = np.sqrt((x-xc_1)**2 + (y-yc_1)**2)
    R_1  = np.mean(Ri_1)
    residu_1 = sum((Ri_1-R_1)**2)
    return R_1, xc_1, yc_1



def MC_CP_dist(amp,sig,bsp_avg=1, N=int(1e5)):

    V1x = amp[0] + sig[0]*npr.randn(N)
    V1y = sig[0]*npr.randn(N)
    V2x = amp[1] + sig[1]*npr.randn(N)
    V2y = sig[1]*npr.randn(N)
    V3x = amp[2] + sig[2]*npr.randn(N)
    V3y = sig[2]*npr.randn(N)
    V1 = V1x + 1j*V1y
    V2 = V2x + 1j*V2y
    V3 = V3x + 1j*V3y
    bsp = V1*V2*V3
    bsp = bsp.reshape((int(N/bsp_avg),bsp_avg))
    bsp = np.mean(bsp,1)

    cphase = np.angle(bsp)*180/np.pi
    kde = st.gaussian_kde(cphase)
    MCsig = circular_std(cphase)
    return kde, MCsig

def fake_CP_data(amp,sig,bsp_avg=1, N=int(1e5)):

    V1x = amp[0] + sig[0]*npr.randn(N)
    V1y = sig[0]*npr.randn(N)
    V2x = amp[1] + sig[1]*npr.randn(N)
    V2y = sig[1]*npr.randn(N)
    V3x = amp[2] + sig[2]*npr.randn(N)
    V3y = sig[2]*npr.randn(N)
    V1 = V1x + 1j*V1y
    V2 = V2x + 1j*V2y
    V3 = V3x + 1j*V3y
    bsp = V1*V2*V3
    bsp = bsp.reshape((int(N/bsp_avg),bsp_avg))
    bsp = np.mean(bsp,1)
    cphase = np.angle(bsp)*180/np.pi
    return cphase

def find_rice_sig_low_snr(A):
    A = np.asarray(A)
    m = np.mean(A)
    s = np.mean(A**2)
    L12 = lambda x: np.exp(x/2)*((1-x)*ss.iv(0,-x/2)-x*ss.iv(1,-x/2))
    eq = lambda x: x*L12(1-s/x**2/2) - m*np.sqrt(2/np.pi)
    Esig = so.fsolve(eq,0.5)[0]
    #EA0 = np.sqrt(s - 2*Esig**2)
    return Esig

def find_rice_amp_low_snr(A):
    A = np.asarray(A)
    m = np.mean(A)
    s = np.mean(A**2)
    L12 = lambda x: np.exp(x/2)*((1-x)*ss.iv(0,-x/2)-x*ss.iv(1,-x/2))
    eq = lambda x: x*L12(1-s/x**2/2) - m*np.sqrt(2/np.pi)
    sig0 = np.std(A)
    Esig = so.fsolve(eq,sig0)[0]
    EA0 = np.sqrt(s - 2*Esig**2)
    return EA0


def Koay_inv_amp(A):
    A = np.asarray(A)
    m = np.mean(A)
    s = np.std(A)
    r = m/s
    zeta = lambda x: 2 + x**2 - (np.pi/8)*np.exp(-x**2/2)*((2+x**2)*ss.iv(0,x**2/4) + (x**2)*ss.iv(1,x**2/4))**2
    g = lambda x: (zeta(x)*(1+r**2)-2) #no sqrt

    rel = lambda x: g(x) - x**2
    x0 = r
    sol = so.fsolve(rel,r)[0]

    unb_std = s/np.sqrt(zeta(sol))
    unb_amp = np.sqrt(m**2 +(zeta(sol) - 2)*unb_std**2)
    return unb_amp

def Koay_inv_std(A):
    A = np.asarray(A)
    m = np.mean(A)
    s = np.std(A)
    r = m/s
    zeta = lambda x: 2 + x**2 - (np.pi/8)*np.exp(-x**2/2)*((2+x**2)*ss.iv(0,x**2/4) + (x**2)*ss.iv(1,x**2/4))**2
    g = lambda x: (zeta(x)*(1+r**2)-2) #no sqrt

    rel = lambda x: g(x) - x**2
    x0 = r
    sol = so.fsolve(rel,r)[0]

    unb_std = s/np.sqrt(zeta(sol))
    unb_amp = np.sqrt(m**2 +(zeta(sol) - 2)*unb_std**2)
    return unb_std
#def chi2