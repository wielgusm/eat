import numpy as np
import pandas as pd
#import sys
#sys.path.append('/Users/mwielgus/Works/MyEAT/eat/')
from eat.inspect import closures as cl

lat_dict = {'A': -23.02922,
            'X': -23.00578,
            'Z': 32.70161,
            'L': 18.98577,
            'P': 37.06614,
            'J': 19.82284,
            'S': 19.82423,
            'R': 19.82423,
            'T':  -90.00000}

lon_dict = {'A': 67.75474,
            'X': 67.75914,
            'Z': 109.89124,
            'L': 97.31478,
            'P': 3.39260,
            'J': 155.47703,
            'S': 155.47755,
            'R': 155.47755,
            'T':  -45.00000}

ant_locat ={
    'A': [2225061.16360, -5440057.36994, -2481681.15054],
    'X': [2225039.52970, -5441197.62920, -2479303.35970],
    'P': [5088967.74544, -301681.18586, 3825012.20561],
    'T': [0.01000, 0.01000, -6359609.70000],
    'L': [-768715.63200, -5988507.07200, 2063354.85200],
    'Z': [-1828796.20000, -5054406.80000, 3427865.20000],
    'J': [-5464584.67600, -2493001.17000, 2150653.98200],
    'S': [-5464555.49300, -2492927.98900, 2150797.17600],
    'R': [-5464555.49300, -2492927.98900, 2150797.17600]
}

ras_dict = {'1055+018': 10.974891,
            '1749+096': 17.859116,
            '1921-293': 19.414182999999998,
            '3C273': 12.485194,
            '3C279': 12.936435000000001,
            '3C454.3': 22.899373999999998,
            '3C84': 3.3300449999999997,
            'BLLAC': 22.045359000000001,
            'CENA': 13.424337,
            'CTA102': 22.543447,
            'CYGX-3': 20.540490999999999,
            'J0006-0623': 0.10385899999999999,
            'J0132-1654': 1.5454129999999999,
            'J1733-1304': 17.550751000000002,
            'J1924-2914': 19.414182999999998,
            'NGC1052': 2.684666,
            'OJ287': 8.9135759999999991} #in hours

dec_dict = {'1055+018': 1.5663400000000001,
            '1749+096': 9.6502029999999994,
            '1921-293': -29.241701000000003,
            '3C273': 2.0523880000000001,
            '3C279': -5.7893120000000007,
            '3C454.3': 16.148211,
            '3C84': 41.511696000000001,
            'BLLAC': 42.277771000000001,
            'CENA': -43.019112,
            'CTA102': 11.730805999999999,
            'CYGX-3': 40.957751999999999,
            'J0006-0623': -6.3931490000000002,
            'J0132-1654': -16.913479000000002,
            'J1733-1304': -13.08043,
            'J1924-2914': -29.241701000000003,
            'NGC1052': -8.2557639999999992,
            'OJ287': 20.108511} #in deg



def compute_elev(ra_source, dec_source, xyz_antenna, time):
    #this one is by Michael Janssen
   """
   given right ascension and declination of a sky source [ICRS: ra->(deg,arcmin,arcsec) and dec->(hour,min,sec)]
   and given the position of the telescope from the vex file [Geocentric coordinates (m)]
   and the time of the observation (e.g. '2012-7-13 23:00:00') [UTC:yr-m-d],
   returns the elevation of the telescope.
   Note that every parameter can be an array (e.g. the time)
   """
   from astropy import units as u
   from astropy.coordinates import EarthLocation, AltAz, ICRS, Angle
   #angle conversions:
   ra_src_deg       = Angle(ra_source, unit=u.hour)
   ra_src_deg       = ra_src_deg.degree * u.deg
   dec_src_deg      = Angle(dec_source, unit=u.deg)

   source_position  = ICRS(ra=ra_src_deg, dec=dec_src_deg)
   antenna_position = EarthLocation(x=xyz_antenna[0]*u.m, y=xyz_antenna[1]*u.m, z=xyz_antenna[2]*u.m)
   altaz_system     = AltAz(location=antenna_position, obstime=time)
   trans_to_altaz   = source_position.transform_to(altaz_system)
   elevation        = trans_to_altaz.alt
   return elevation.degree


def paralactic_angle(alist):
    from astropy.time import Time
    get_GST_rad = lambda x: Time(x).sidereal_time('mean','greenwich').hour*2.*np.pi/24.
    GST = np.asarray(list(map(get_GST_rad,alist.datetime))) #in radians
    station1 = [x[0] for x in alist.baseline]
    station2 = [x[1] for x in alist.baseline]
    #print(set(station1))
    #print(set(station2))
    lon1 = np.asarray(list(map(lambda x: lon_dict[x],station1)))*np.pi*2./360. #converted from deg to radians
    lat1 = np.asarray(list(map(lambda x: lat_dict[x],station1)))*np.pi*2./360. #converted from deg to radians
    lon2 = np.asarray(list(map(lambda x: lon_dict[x],station2)))*np.pi*2./360. #converted from deg to radians
    lat2 = np.asarray(list(map(lambda x: lat_dict[x],station2)))*np.pi*2./360. #converted from deg to radians
    #ras = np.asarray(alist.ra_hrs)*np.pi*2./24. #converted from hours to radians
    #dec = np.asarray(alist.dec_deg)*np.pi*2./360. #converted from deg to radians    
    ras = np.asarray(list(map(lambda x: ras_dict[x], alist.source)))*np.pi*2./24. #converted from hours to radians
    dec = np.asarray(list(map(lambda x: dec_dict[x], alist.source)))*np.pi*2./360. #converted from deg to radians
    HA1 = GST - lon1 - ras #in rad
    HA2 = GST - lon2 - ras #in rad
    par1I = np.sin(HA1)
    par1R = np.cos(dec)*np.tan(lat1) - np.sin(dec)*np.cos(HA1)
    par1 = np.angle(par1R + 1j*par1I )
    par2I = np.sin(HA2)
    par2R = np.cos(dec)*np.tan(lat2) - np.sin(dec)*np.cos(HA2)
    par2 = np.angle(par2R + 1j*par2I )
    alist.loc[:,'par1'] = par1
    alist.loc[:,'par2'] = par2

    return alist

def field_rotation(fra_data):
    par = fra_data[0]
    elev = fra_data[1]
    station = fra_data[2]
    if station in {'A','J'}:
        fra = par
    elif station in {'X','Z'}:
        fra = par+elev
    elif station in {'L','P'}:
        fra = par-elev
    elif station=='S':
        fra = 45. - elev + par
    else:
        fra = None
    return fra

    
def add_computed_elev(alist):
    station1 = [x[0] for x in alist.baseline]
    station2 = [x[1] for x in alist.baseline]
    elev_data1 = zip(alist.source,station1,alist.datetime)
    elev_data2 = zip(alist.source,station2,alist.datetime)
    prep_elev = lambda x: compute_elev(ras_dict[x[0]],dec_dict[x[0]],ant_locat[x[1]],str(x[2]))
    ref_elev = list(map(prep_elev,elev_data1))
    rem_elev = list(map(prep_elev,elev_data2))
    alist_out = alist
    alist_out.loc[:,'ref_elev'] = ref_elev
    alist_out.loc[:,'rem_elev'] = rem_elev
    return alist_out
   

def add_field_rotation(alist):
    
    station1 = [x[0] for x in alist.baseline]
    station2 = [x[1] for x in alist.baseline]
    fra_data1 = zip(alist.par1*180./np.pi,alist.ref_elev,station1)
    fra_data2 = zip(alist.par2*180./np.pi,alist.rem_elev,station2)
    
    fra1 = list(map(field_rotation,fra_data1))
    fra2 = list(map(field_rotation,fra_data2))

    alist.loc[:,'fra1'] = fra1
    alist.loc[:,'fra2'] = fra2
    return alist

def total_field_rotation(alist):
    alist.loc[:,'tot_fra'] = -2.*(alist.fra1 - alist.fra2)
    return alist

def add_total_field_rotation(alist, recompute_elev = False):
    alist_out = alist
    alist = paralactic_angle(alist)
    if recompute_elev:
        alist = add_computed_elev(alist)
    alist = add_field_rotation(alist)
    alist = total_field_rotation(alist)
    alist_out.loc[:,'tot_fra'] = alist['tot_fra']
    return alist_out


def solve_amp_ratio(alist,no_sigmas=4,weightsA=True, weightsP=True):
    fooR, fooL = cl.match_2_bsp_frames(alist,alist,match_what='polarization',dt = 0.5,what_is_same='baseline')
    fooR = fooR.sort_values(['datetime','baseline'])
    fooL = fooL.sort_values(['datetime','baseline'])
    #print(np.shape(fooL))
    #print(np.shape(fooR))
    #solving for amplitudes of RR to LL ratios
    #prepare mean values of ratios amplitudes
    amp_ratio = []
    amp_weights = []
    list_baselines = sorted(list(set(fooR.baseline)))
    for base in list_baselines:
        fooRb = fooR[fooR.baseline==base]
        fooLb = fooL[fooL.baseline==base]
        vec = np.asarray(fooRb.amp)/np.asarray(fooLb.amp)
        vec = cut_outliers(vec,no_sigmas)
        amp_ratio.append(np.mean(vec))
        amp_weights.append(np.mean(fooRb.snr))
    #print(amp_weights)
    phase_weights = amp_weights
    if weightsA==False:
        amp_weights = np.ones(len(list_baselines))
    else:
        amp_weights = np.asarray(amp_weights)
    list_stations = sorted(list(set(''.join(list_baselines) )))
    amp_matrix = np.zeros((len(list_baselines),len(list_stations)))
    for couS in range(len(list_stations)):
        st = list_stations[couS]
        for couB in range(len(list_baselines)):
            bs = list_baselines[couB]
            if st in bs:
                amp_matrix[couB,couS] = 1.*amp_weights[couB]
    #print(amp_matrix)
    amp_ratio_st =  (np.linalg.lstsq(amp_matrix,np.transpose(amp_weights*np.log(np.asarray(amp_ratio))))[0])
    approx_res = np.exp(np.multiply(amp_matrix,amp_ratio_st))
    amp_ratio_st = np.exp(amp_ratio_st)

    #solving for phases
    phas_diff = []
    if weightsP==False:
        phase_weights = np.ones(len(list_baselines))
    else:
        phase_weights = np.asarray(phase_weights)
    for base in list_baselines:
        fooRb = fooR[fooR.baseline==base]
        fooLb = fooL[fooL.baseline==base]
        vec = np.asarray(fooRb.resid_phas) - np.asarray(fooLb.resid_phas) - (-fooRb.tot_fra)
        #print(cl.circular_mean(vec))
        old_mean = cl.circular_mean(vec)
        #print(old_mean)
        vec = cut_outliers_circ(vec,no_sigmas)
        new_mean = cl.circular_mean(vec)
        if new_mean!=new_mean:
            new_mean = old_mean
        #print(new_mean)
        #print(vec)
    
        phas_diff.append(new_mean)
    
    #print('Now phas_diff at the begin:')
    #print(phas_diff)

    phas_matrix = np.zeros((len(list_baselines),len(list_stations)))
    for couS in range(len(list_stations)):
        st = list_stations[couS]
        for couB in range(len(list_baselines)):
            bs = list_baselines[couB]
            if bs[0]==st:
                phas_matrix[couB,couS] = 1.
            elif bs[1]==st:
                phas_matrix[couB,couS] = -1.

    func = lambda x: np.sum(phase_weights**2*np.abs(np.exp(1j*(2.*np.pi/360.)*(phas_matrix.dot(x) - phas_diff))-1.)**2)
    from scipy.optimize import minimize
    x0 =  180.*np.ones(len(list_stations))
    #x0[0] = 150.
    #meth = 'Nelder-Mead'
    #meth = 'Powell'
    #meth = 'CG'
    #meth = 'dogleg'
    #meth='trust-krylov'
    #meth='COBYLA'
    meth='L-BFGS-B'
    bounds = [(0.,360.)]*len(list_stations)
    #foo = minimize(func, x0,method=meth,bounds=bounds)
    foo = minimize(func, x0,method=meth)

    phas_ratio_st = foo.x
    #print(list_baselines)
    #print('Now phas_diff at the end:')
    #print(phas_diff)
    return amp_ratio_st, np.mod(phas_ratio_st,360), list_stations


def solve_ratios_scans(alist, recompute_elev=False,remove_phas_dof=False, zero_station=''):

    src_sc_list = list(set(zip(alist.source,alist.scan_id)))
    sc_list = [x for x in src_sc_list]
    scan_id_list = [x[1] for x in sc_list]
    list_stations_tot = list(set(''.join(list(set(alist.baseline)))))
    foo_columns = sorted(list(map(lambda x: x+'_amp',list_stations_tot))+list(map(lambda x: x+'_phas',list_stations_tot)))
    columns = ['datetime','source','scan_id']+foo_columns
    ratios = pd.DataFrame(columns=columns)
    for cou in range(len(sc_list)):
        local_scan_id = sc_list[cou][1] 
        local_source = sc_list[cou][0] 
        print(local_scan_id,' ',local_source)
        fooHloc = alist[alist.scan_id==local_scan_id]
        fooHloc = add_total_field_rotation(fooHloc, recompute_elev=recompute_elev)
        amp, pha, list_stations_loc = solve_amp_ratio(fooHloc,weightsA=True,weightsP=True)
        
        print(list_stations_loc)
        print(amp)
        print(pha)
        print('\n')
        Ratios_loc = pd.DataFrame(columns=columns)
        Ratios_loc['scan_id'] = [local_scan_id]
        Ratios_loc['source'] = [local_source]
        Ratios_loc['datetime'] = [min(fooHloc.datetime)]
        for cou_st in range(len(list_stations_loc)):
            stat = list_stations_loc[cou_st]
            Ratios_loc[stat+'_amp'] = amp[cou_st]
            Ratios_loc[stat+'_phas'] = pha[cou_st]
        ratios = pd.concat([ratios,Ratios_loc],ignore_index=True)

    if remove_phase_dof==True:
        ratios = remove_phase_dof(ratios,zero_station)

    return ratios



def remove_phase_dof(ratios,zero_phase_station=''):
    '''
    Phase calculation with solve_amp_ratio keeps a degree of freedom i.e.
    if ALL phases are shifted by any phi_0, results remain the same.
    This function just assumes that phase at one chosen station is zero, subtracting
    estimated phase at this station from all stations present in this scan
    '''

    ratios_fixed = ratios.copy()
    phas_list = ratios.columns[list(map(lambda x:'phas' in x, ratios.columns))]
    
    #if zero phase station unspecified, choose the one with best coverage
    #that is not ALMA
    if zero_phase_station=='':
        phas_list_noA = phas_list.drop('A_phas')
        zero_phase_station = ratios_fixed[phas_list_noA].isnull().sum().argmin()[0]


    #when zero station is active, subtract its phase from all ratios
    phase_on_zero_station = np.asarray(ratios_fixed[zero_phase_station+'_phas'].copy())
    phase_on_zero_station[phase_on_zero_station!=phase_on_zero_station]=0

    for station in phas_list:
        ratios_fixed.loc[:,station] = np.mod(np.asarray(ratios_fixed.loc[:,station]) - phase_on_zero_station,360)
    
    #when zero phase station isn't present, solve for value to subtract
    phase_on_zero_station = np.asarray(ratios_fixed[zero_phase_station+'_phas'].copy())
    #print(ratios_fixed.loc[phase_on_zero_station!=phase_on_zero_station,phas_list])
    #print(ratios_fixed.loc[phase_on_zero_station==phase_on_zero_station,phas_list].mean())
    
    zps_present = ratios_fixed.loc[phase_on_zero_station==phase_on_zero_station,phas_list]
    zps_not_present = ratios_fixed.loc[phase_on_zero_station!=phase_on_zero_station,phas_list]
    #print(zps_present)
    phas_list_noS = phas_list.drop('S_phas')
    zps_present_mean_phases = zps_present.apply(cl.circular_mean)
    zps_differences = zps_not_present-zps_present_mean_phases
    #print(zps_differences)

    error_m360 = lambda vect,x: np.sum(np.minimum(vect - x, 360 - (vect-x))**2)
    #from scipy.optimize import minimize
    for indexR, row in ratios_fixed.iterrows():
        if indexR in zps_differences.index:
            #print(indexR)
            row = zps_differences[phas_list_noS][zps_differences.index==indexR]
            rowLoc = np.asarray(row,dtype=np.float32)
            rowLoc = rowLoc[rowLoc==rowLoc]
            #print(rowLoc)

            delta_phase_scan = cl.circular_mean(rowLoc)
            #print(ratios_fixed.iloc[index])
            #print(index)
            #print()
            foo = np.asarray(ratios_fixed.loc[(ratios_fixed.index==indexR),phas_list]) - delta_phase_scan 
            #print(foo)
            ratios_fixed.loc[(ratios_fixed.index==indexR),phas_list] = foo
            #print(ratios_fixed.loc[(ratios_fixed.index==indexR),phas_list])
            #print(ratios_fixed[(ratios_fixed.index==indexR)][phas_list])
            #print(foo)
            #ratios_fixed[phas_list].replace([(ratios_fixed.index==indexR)], foo, inplace=True)
            #print(ratios_fixed[(ratios_fixed.index==indexR)][phas_list])
    return ratios_fixed

#def minimize_with_one_subtraction(vect):



def cut_outliers(vector,no_sigmas):
    #cuts outliers that are in distance from mean value larger than no_sigmas
    sigma = np.std(vector)
    m = np.mean(vector)
    vector = vector[np.abs(vector - m) < no_sigmas*sigma]
    return vector


def cut_outliers_circ(vector,no_sigmas):
    #cuts outliers that are in distance from mean value larger than no_sigmas
    sigma = cl.circular_std(vector)
    m = cl.circular_mean(vector)
    dist = np.minimum(np.abs(vector - m), np.abs(360. - np.abs(vector - m)))
    vector = vector[dist < no_sigmas*sigma]
    return vector

'''
reload(polcal)
hopsDd = hopsD[hopsD.expt_no==3597]
hopsDd = hopsDd[map(lambda x: x[0]!=x[1], hopsDd.baseline)]
hopsDd = hopsDd[map(lambda x: x[0]==x[1], hopsDd.polarization)]
src_sc_list = list(set(zip(hopsDd.source,hopsDd.scan_id)))
sc_list = [x for x in src_sc_list if x[0]=='OJ287']
scan_id_list = [x[1] for x in sc_list]
list_stations_tot = list(set(''.join(list(set(hopsDd.baseline)))))
foo_columns = sorted(map(lambda x: x+'_amp',list_stations_tot)+map(lambda x: x+'_phas',list_stations_tot))
columns = ['datetime','scan_id']+foo_columns
Ratios = pd.DataFrame(columns=columns)
#Ratios['scan_id']= scan_id_list
for cou in range(len(sc_list)):
    local_scan_id = sc_list[cou][1] 
    print(local_scan_id)
    fooHloc = hopsDd[hopsDd.scan_id==local_scan_id]
    fooHloc = polcal.add_total_field_rotation(fooHloc)
    amp, pha, list_stations_loc = polcal.solve_amp_ratio(fooHloc)
    print(list_stations_loc)
    print(amp)
    print(pha)
    Ratios_loc = pd.DataFrame(columns=columns)
    Ratios_loc['scan_id'] = [local_scan_id]
    Ratios_loc['datetime'] = [min(fooHloc.datetime)]
    for cou_st in range(len(list_stations_loc)):
        stat = list_stations_loc[cou_st]
        Ratios_loc[stat+'_amp'] = amp[cou_st]
        Ratios_loc[stat+'_phas'] = pha[cou_st]
    Ratios = pd.concat([Ratios,Ratios_loc],ignore_index=True)

'''