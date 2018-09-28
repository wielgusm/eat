'''
Imports uvfits using ehtim library
and remodels them into pandas dataframes

'''

#import pandas as pd
#import ehtim as eh



def get_info(observation='EHT2017',path_vex='VEX/'):
    '''
    gets info about stations, scans, expt for a given observation,
    by default it's EHT2017 campaign
    '''
    if observation=='EHT2017':

        stations_2lett_1lett = {'AZ': 'Z', 'PV': 'P', 'SM':'S', 'SR':'R','JC':'J', 'AA':'A','AP':'X', 'LM':'L','SP':'Y'}
        jd_expt = jd2expt2017
        scans = make_scan_list_EHT2017(fpath)
    
    return stations_2lett_1lett, jd_expt, scans

'''
def jd2expt2017(jd):
    '''
    Function translating from jd to expt for April 2017 EHT
    '''
    if (jd > 2457853.470)&(jd < 2457854.132 ):
        return 3600
    elif (jd > 2457849.531)&(jd < 2457850.177):
        return 3598
    elif (jd > 2457850.667)&(jd < 2457851.363):
        return 3599
    elif (jd > 2457848.438)&(jd < 2457849.214):
        return 3597
    elif (jd > 2457854.427)&(jd < 2457855.141):
        return 3601
    else:
        return None

def make_scan_list_EHT2017(fpath):
    '''
    generates data frame with information about scans for EHT2017
    '''
    import ehtim.vex as vex
    import os
    import pandas as pd
    from astropy.time import Time, TimeDelta
    import datetime as datetime
    nam2lett = {'ALMA':'A','APEX':'X','LMT':'L','PICOVEL':'P','SMTO':'Z','SPT':'T','JCMT':'J','SMAP':'S'}
    track2expt ={'D':3597,'B':3598, 'C':3599,'A':3600,'E':3601}
    list_files = os.listdir(fpath)
    scans = pd.DataFrame({'source' : []})

    for fi in list_files:#loop over vex files in folder
        track_loc = fi[3].upper()
        vpath = fpath+fi
        aa = vex.Vex(vpath)
        dec = []
        for cou in range(len(aa.source)):
            dec_h = float(aa.source[cou]['dec'].split('d')[0])
            dec_m = float((aa.source[cou]['dec'].split('d')[1])[0:2])
            dec_s = float((aa.source[cou]['dec'].split('d')[1])[3:-1])
            dec.append(tuple((dec_h,dec_m,dec_s)))    
        ra = []
        for cou in range(len(aa.source)):
            ra_d = float(aa.source[cou]['ra'].split('h')[0])
            ra_m = float(aa.source[cou]['ra'].split('h')[1].split('m')[0])
            ra_s = float(aa.source[cou]['ra'].split('h')[1].split('m')[1][:-1])
            ra.append(tuple((ra_d,ra_m,ra_s)))      
        sour_name = [aa.source[x]['source'] for x in range(len(aa.source))]
        dict_ra = dict(zip(sour_name,ra))
        dict_dec = dict(zip(sour_name,dec))
        t_min = [aa.sched[x]['start_hr'] for x in range(len(aa.sched))]
        sour = [aa.sched[x]['source'] for x in range(len(aa.sched))]
        datet = []
        elev = []
        antenas = []
        duration=[]
        for x in range(len(aa.sched)):#loop over scans in given file
            t = Time(aa.sched[x]['mjd_floor'], format='mjd', scale='utc')
            tiso = Time(t, format='iso', scale='utc')
            tiso = tiso + TimeDelta(t_min[x]*3600., format='sec')
            datet.append(tiso)
            ant_foo = set([nam2lett[aa.sched[x]['scan'][y]['site']] for y in range(len(aa.sched[x]['scan']))])
            antenas.append(ant_foo)
            duration_foo =max([aa.sched[x]['scan'][y]['scan_sec'] for y in range(len(aa.sched[x]['scan']))])
            duration.append(duration_foo)
        #time_min = [pd.tslib.Timestamp(datet[x].datetime) for x in range(len(datet))]
        time_min = [pd.Timestamp(datet[x].datetime) for x in range(len(datet))]
        time_max = [time_min[x] + datetime.timedelta(seconds=duration[x]) for x in range(len(aa.sched))]
        foo = pd.DataFrame(aa.sched)
        foo = foo[['source','mjd_floor','start_hr']]
        foo['time_min']=time_min
        foo['time_max']=time_max
        foo['scan_no'] = foo.index
        foo['scan_no'] = list(map(int,foo['scan_no']))
        foo['track'] = [track_loc]*foo.shape[0]
        foo['expt'] = [int(track2expt[track_loc])]*foo.shape[0]
        foo['antenas'] = antenas
        foo['duration'] = duration
        scans = pd.concat([scans,foo], ignore_index=True)
    scans = scans.reindex_axis(['mjd_floor','expt','track','scan_no','source','time_min','time_max','duration','antenas'],axis=1)
    scans = scans.sort_values('time_max')
    scans = scans.reset_index(drop=True)
    scans['scan_no_tot'] = scans.index
    return scans

def match_scans(scans,data):
    '''
    matches data with scans
    '''
    import pandas as pd
    from astropy.time import Time, TimeDelta
    import datetime as datetime
    bins_labels = [None]*(2*scans.shape[0]-1)
    bins_labels[1::2] = map(lambda x: -x-1,list(scans['scan_no_tot'])[:-1])
    bins_labels[::2] = list(scans['scan_no_tot'])
    dtmin = datetime.timedelta(seconds = 2.) 
    dtmax = datetime.timedelta(seconds = 2.) 
    binsT = [None]*(2*scans.shape[0])
    binsT[::2] = list(map(lambda x: x - dtmin,list(scans.time_min)))
    binsT[1::2] = list(map(lambda x: x + dtmax,list(scans.time_max))) 
    ordered_labels = pd.cut(data.datetime, binsT,labels = bins_labels)
    data['scan_no_tot'] = ordered_labels
    data = data[list(map(lambda x: x >= 0, data['scan_no_tot']))]
    data['scan_id'] = data['scan_no_tot']
    data.drop('scan_no_tot',axis=1,inplace=True)
    return data


    
def get_df_all_pol(pathf,pathVex=''):
    import pandas as pd

    obsRR = eh.io.load.load_obs_uvfits(pathf,  force_singlepol='R')
    obsLL = eh.io.load.load_obs_uvfits(pathf,  force_singlepol='L')
    obsRL = eh.io.load.load_obs_uvfits(pathf,  force_singlepol='RL')
    obsLR = eh.io.load.load_obs_uvfits(pathf,  force_singlepol='LR')

    obsRR.add_vis_df()
    obsLL.add_vis_df()
    obsRL.add_vis_df()
    obsLR.add_vis_df()
    
    dfRR = obsRR.vis_df
    dfLL = obsLL.vis_df
    dfRL = obsRL.vis_df
    dfLR = obsLR.vis_df
    
    dfRR['polarization'] = 'RR'
    dfLL['polarization'] = 'LL'
    dfRL['polarization'] = 'RL'
    dfLR['polarization'] = 'LR'
    
    df = pd.concat([dfRR,dfLL,dfLR,dfRL],ignore_index=True)
    df['expt_no'] = list(map(jd2expt2017,df['jd']))
    df['baseline'] = list(map(lambda x: AZ2Z[x[0].decode('unicode_escape')]+AZ2Z[x[1].decode('unicode_escape')],zip(df['t1'],df['t2'])))
    is_alphabetic = list(map(lambda x: float(x== ''.join(sorted([x[0],x[1]]))),df['baseline']))
    df['amp'] = list(map(np.abs,df['vis']))
    df['phase'] = list(map(lambda x: (2.*x[1]-1.)*(180./np.pi)*np.angle(x[0]),zip(df['vis'],is_alphabetic)))
    df['baseline'] = list(map(lambda x: ''.join(sorted([x[0],x[1]])),df['baseline']))
    if pathVex!='':
        scans = make_scan_list_EHT2017(pathVex)
    df = match_scans(scans,df)
            
    return df


def get_df_all_pol_netcal(pathf,pathVex=''):
    import pandas as pd

    if 'RR' in pathf:   
        obs = eh.io.load.load_obs_uvfits(pathf,  force_singlepol='R')
    elif 'LL' in pathf:
        obs = eh.io.load.load_obs_uvfits(pathf,  force_singlepol='L')
        #obsRL = eh.io.load.load_obs_uvfits(pathf,  force_singlepol='RL')
        #obsLR = eh.io.load.load_obs_uvfits(pathf,  force_singlepol='LR')

    #obsRR.add_vis_df()
    #obsLL.add_vis_df()
    #obsRL.add_vis_df()
    #obsLR.add_vis_df()
    obs.add_vis_df()
    
    #dfRR = obsRR.vis_df
    #dfLL = obsLL.vis_df
    #dfRL = obsRL.vis_df
    #dfLR = obsLR.vis_df
    df=obs.vis_df
    
    #dfRR['polarization'] = 'RR'
    #dfLL['polarization'] = 'LL'
    #dfRL['polarization'] = 'RL'
    #dfLR['polarization'] = 'LR'
    if 'RR' in pathf:   
        df['polarization'] = 'RR'
    elif 'LL' in pathf:
        df['polarization'] = 'LL' 
    #df = pd.concat([dfRR,dfLL,dfLR,dfRL],ignore_index=True)
    df['expt_no'] = list(map(jd2expt2017,df['jd']))
    df['baseline'] = list(map(lambda x: AZ2Z[x[0].decode('unicode_escape')]+AZ2Z[x[1].decode('unicode_escape')],zip(df['t1'],df['t2'])))
    is_alphabetic = list(map(lambda x: float(x== ''.join(sorted([x[0],x[1]]))),df['baseline']))
    df['amp'] = list(map(np.abs,df['vis']))
    df['phase'] = list(map(lambda x: (2.*x[1]-1.)*(180./np.pi)*np.angle(x[0]),zip(df['vis'],is_alphabetic)))
    df['baseline'] = list(map(lambda x: ''.join(sorted([x[0],x[1]])),df['baseline']))
    if pathVex!='':
        scans = make_scan_list_EHT2017(pathVex)
        df = match_scans(scans,df)
            
    return df
'''