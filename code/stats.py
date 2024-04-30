#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 14:11:39 2022

@author: sopsla
"""
import numpy as np
import scipy.stats as stats
from pyeeg.utils import lag_span
import mne
from meg import CH_NAMES, SUBJECTS

lags = lag_span(-0.2, 0.8, 120)

feat_id2name = {0:'Envelope', 1:'Word onset', 2:'Entropy', 3:'Surprisal', 4:'Word frequency'}
model2feats = {'envelope':[0],
               'onset':[0,1],
               'entropy':[0,1,2],
               'surprisal':[0,1,3],
               'entropy_surprisal':[0,1,2,3],
               'frequency':[0,1,4],
               'frequency_entropy':[0,1,2,4],
               'frequency_surprisal':[0,1,3,4],
               'frequency_entropy_surprisal':[0,1,2,3,4]}

def cluster_2samp(trfs, model, info, tmin=None, tmax=None, statfun=None, alpha_statfun=0.05,
            cluster_alpha=0.01, multiplication=3):
        """
        Computes cluster-based permutation test.
        Contrasts sentence and list data for each
        feature separately in a given model.
        
        INPUT
        -----
        trfs : list of meg.TRF
        model : str 
        threshold : float | dict
            vertices w/ data values more extreme than threshold will be used to form clusters
            if dict (default), threshold-free cluster enhancement is used
        cluster-alpha: float
            Cut-off value to determine significant clusters. Default = 0.01
        tmin : float | None
            Min value for subset of times, default None --> tmin is first sample
        tmax : float | None
            Max value for subset of times, default None --> tmax is final sample
        lags : list
            Original lag matrix
        plot : boolean
            if True: plots clusters evoked and scalp maps 
            
        OUTPUT
        ------
        out : dict of dict
            dict of feature: output of cluster-based permutation test
        
        """
        ttest = lambda x, y: mne.stats.cluster_level.ttest_1samp_no_p(x-y)
        
        idx = mne.channel_indices_by_type(info, picks=info.ch_names)
        adjacency, _ = mne.channels.read_ch_adjacency('ctf275', idx['mag'])
        
        nchans = len(info.ch_names)
        nsubj = len(SUBJECTS)
        times = lags/120
        
        if statfun == None:
            tail = 1
            threshold = stats.f.ppf(1-alpha_statfun, 2-1, (nsubj-1)*2) * multiplication
        else:
            statfun = ttest
            tail = 0
            threshold = stats.t.ppf(1-alpha_statfun/2, df=nsubj-1) * multiplication
        
        ### Prepare the data ###
        # allocate memory
        if tmin is None:
            tmin = times[0]
            lagmin = 0
        else:
            lagmin = np.argmin(abs(tmin-times))
            tmin = times[lagmin]
            
        if tmax is None:
            tmax = times[-1]
            lagmax = len(times)-1
        else:
            lagmax = np.argmin(abs(tmax-times))
            tmax = times[lagmax]
        samp_idx = np.arange(lagmin, lagmax+1)
        nsamp = len(samp_idx)
        lst = np.zeros((nsubj, nsamp, nchans))
        snt = np.zeros_like(lst)
        out = {}
    
        # accumulate data according to cond
        for k_feat, name_id in enumerate(model2feats[model]):
            print("------- Stats for feat: %s ---------"%(feat_id2name[name_id]))
            for condition in ['list', 'sentence']:
                for i, participant in enumerate(trfs[model][condition]):
                    dat = participant.coef_[samp_idx, k_feat, :]
                    if condition == 'list':
                        lst[i,:,:] = np.squeeze(dat)
                    elif condition == 'sentence':
                        snt[i,:,:] = np.squeeze(dat)
                        
            ### Permutation test ###
            print("Using a threshold of %.3f"%threshold)    
            cluster_stats = mne.stats.spatio_temporal_cluster_test([lst, snt], 
                                                n_permutations = 1024,
                                                threshold = threshold,
                                                tail = tail,
                                                stat_fun = statfun,
                                                max_step = 1,
                                                n_jobs = 1,
                                                buffer_size = None,
                                                adjacency = adjacency)
        
            out[feat_id2name[name_id]] = cluster_stats
            print("There are %d significant clusters\n"%(cluster_stats[2]<cluster_alpha).sum())
            
        return out                   

def cluster_1samp(trfs, info, model, feature='word frequency', statfun='ttest', alpha_statfun=0.05, 
                  cluster_alpha=0.01, multiplication=3):
    ### Adjacency matrix ###
    print(f'running permutation for feat {feature} in model {model}')
    idx = mne.channel_indices_by_type(info, picks=info.ch_names)
    adjacency, _ = mne.channels.read_ch_adjacency('ctf275', idx['mag'])

    nsubj = len(trfs[model]['sentence'])
    
    if statfun == 'ttest':
        tail = 0
        statfun = lambda x: mne.stats.cluster_level.ttest_1samp_no_p(x)
        threshold = stats.t.ppf(1-alpha_statfun/2, df=nsubj-1) * multiplication
    else:
        statfun = None #f-test
        tail = 1
        threshold = stats.f.ppf(1-alpha_statfun, 2-1, (nsubj-1)*2) * multiplication
    
    # get data in shape (participants, lags, channels)
    X_list = np.zeros((nsubj, 120, len(info.ch_names)))
    X_sent = np.zeros((nsubj, 120, len(info.ch_names)))
    for sub in list(range(nsubj)):
        X_list[sub,:,:] = trfs[model]['list'][sub][feature].data.T
        X_sent[sub,:,:] = trfs[model]['sentence'][sub][feature].data.T
    
    results = []
    for dat in [X_list, X_sent]:
        cluster_stats = mne.stats.spatio_temporal_cluster_1samp_test(dat,
                                                                    threshold = threshold,
                                                                    n_permutations = 1024,
                                                                    tail = tail,
                                                                    stat_fun = statfun,
                                                                    adjacency = adjacency)
        print("There are %d significant clusters\n"%(cluster_stats[2]<cluster_alpha).sum())
        results.append(cluster_stats)
        
    return results

# here: source permutation 
def source_cluster_2samp(list_dat, sent_dat, threshold, adjacency, cluster_alpha=0.05, exclude=None):

    """
    Computes cluster-based permutation test in source space. T-test.
    Contrasts sentence and list data.
    
    INPUT
    -----
    list_dat : np.array
    sent_dat : np.array
    threshold : float | dict
        vertices w/ data values more extreme than threshold will be used to form clusters
        if dict (default), threshold-free cluster enhancement is used
    cluster-alpha: float
        Cut-off value to determine significant clusters. Default = 0.01
    adjacency : output of mne.spatial_src_adjacency
    exclude : tuple of list | indices of sources to exclude
    
    OUTPUT
    ------
    cluster_stats : dict of dict
        dict of feature: output of cluster-based permutation test
    
    """
    X = [sent_dat, list_dat]   
    ttest = lambda x, y: mne.stats.cluster_level.ttest_1samp_no_p(x-y)
    
    cluster_stats = mne.stats.spatio_temporal_cluster_test(X,
                                                       n_permutations=1024,
                                                       tail=0,
                                                       stat_fun=ttest,
                                                       adjacency=adjacency,
                                                       threshold=threshold,
                                                       spatial_exclude=exclude)
    print("There are %d significant clusters\n"%(cluster_stats[2]<cluster_alpha).sum())
    return cluster_stats

def source_cluster_1samp(dat, threshold, adjacency, cluster_alpha=0.05, exclude=None):

    """
    Computes cluster-based permutation test in source space. T-test.
    Contrasts sentence and list data.
    
    INPUT
    -----
    dat : np.array | data
    threshold : float | dict
        vertices w/ data values more extreme than threshold will be used to form clusters
        if dict (default), threshold-free cluster enhancement is used
    cluster-alpha: float
        Cut-off value to determine significant clusters. Default = 0.01
    adjacency : output of mne.spatial_src_adjacency
    exclude : tuple of list | indices of sources to exclude
    
    OUTPUT
    ------
    cluster_stats : dict of dict
        dict of feature: output of cluster-based permutation test
    
    """
    cluster_stats = mne.stats.spatio_temporal_cluster_1samp_test(dat,
                                                       n_permutations=1024,
                                                       tail=0,
                                                       stat_fun=None,
                                                       adjacency=adjacency,
                                                       threshold=threshold,
                                                       spatial_exclude=exclude)
    print("There are %d significant clusters\n"%(cluster_stats[2]<cluster_alpha).sum())
    return cluster_stats