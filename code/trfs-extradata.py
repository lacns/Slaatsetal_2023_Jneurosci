#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 21:55:07 2021

@author: sopsla
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
import mne

from text import loadmat, get_word_frequency
from meg import TRF, crossval
from utils import temporal_smoother

from pyeeg.utils import signal_envelope, lag_matrix, lag_span
from scipy.io.wavfile import read as wavread
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

FEAT_DICT = {'envelope':['envelope'],
             'onset':['envelope', 'word onset'],
             'frequency':['envelope', 'word onset', 'word frequency']}

def epoch_variable(raw, log, prestim=0, poststim=0):
    print('\nReading events...')
    events = mne.find_events(raw, stim_channel='UPPT001')
    
    onsets = events[::2]
    offsets = events[1::2]
        
    epochs = []
    for onset, offset in zip(onsets, offsets):
        t_idx = raw.time_as_index([onset[0]/1200-prestim, offset[0]/1200+poststim])
        epoch, times = raw[:, t_idx[0]:t_idx[1]]
        epochs.append(mne.EpochsArray(np.expand_dims(epoch, 0), raw.info, tmin=-prestim, verbose=False))
        
    print('Epoching done: {0} trials found.'.format(len(epochs)))
    
    return epochs, onsets[:np.shape(offsets)[0],2], offsets[:,2]
    

conditions = {'sentence': [11, 12, 13, 14],
            'word list': [31, 32, 33, 34]}

tasks = {'passive': [11, 31],
       'phrase': [12, 32],
       'word': [13, 33],
       'syllable': [14, 34]}

### PATHS ###
SUBJECTS = [s for s in os.listdir('/project/3027006.01/raw') if len(s) == 7]

# subject 11 does not have word lists
SUBJECTS.remove('sub-011')
SUBJECTS.remove('sub-015')
subject = SUBJECTS[int(sys.argv[1])] # int(sys.argv[1])

if subject == 'sub-013':
    megpath = '/project/3027006.01/raw/{0}/ses-meg02/meg/'.format(subject)
else:
    megpath = '/project/3027006.01/raw/{0}/ses-meg01/meg/'.format(subject)
meg = os.listdir(megpath)[0]
savedir = '/project/3027005.01/extra_data/theta_reg'


# logfile
print("\nLoading the logfile...")
logpath = '/home/lacnsg/sopsla/tmp_logf'
fname = os.path.join(logpath, 'log_{0}.mat'.format(subject))
log = loadmat(fname)['log']['data']

# Annotations
sent_annot = pd.read_csv('/project/3027006.01/01_Task/Materials/Annotations/sentence_triggers.txt')
list_annot = pd.read_csv('/project/3027006.01/01_Task/Materials/Annotations/wordlist_triggers.txt')

if not os.path.isdir(os.path.join(savedir, subject)):
    os.mkdir(os.path.join(savedir, subject))

### VARIABLES ###
hp = 4.
lp = 10.
sfreq = 120
tmin = -0.2
tmax = 0.8
fit_intercept = True
#alpha = 5000
alpha = np.logspace(-2, 2, 20) * 60000  # alphas: logspace from mean Eigenvalue of XtX 60470.94

### START ANALYSIS ###
if not os.path.isfile(os.path.join(savedir, subject, 'epochs.pkl')):
    # load the data & filter
    raw = mne.io.read_raw_ctf(os.path.join(megpath, meg))
    
    # filter
    print('Filtering ...')
    raw.load_data()
    raw = raw.filter(hp, lp, fir_design='firwin')
    
    # epoch
    epochs, condition, trialidx = epoch_variable(raw, log, prestim=0, poststim=0)
    del raw 
    
    for epoch in epochs:
        epoch.apply_baseline(baseline=(None, None), verbose=False)
        epoch.resample(sfreq)
        epoch = epoch.pick_types(meg='mag', ref_meg=False)
        
    ep_dict = {}
    ep_dict['epochs'] = epochs
    ep_dict['condition'] = condition
    ep_dict['trialidx'] = trialidx
    
    with open(os.path.join(os.path.join(savedir, subject, 'epochs.pkl')), 'wb') as f:
        pickle.dump(ep_dict, f)
        
    del ep_dict
    
else:
    with open(os.path.join(os.path.join(savedir, subject, 'epochs.pkl')), 'rb') as f:
        ep_dict = pickle.load(f)
        
    epochs, condition, trialidx = ep_dict.values()
    del ep_dict
    
for epoch in epochs:
    epoch = epoch.pick_types(meg='mag', ref_meg=False)

#if subject == 'sub-018':
 #   epochs = [epoch for (epoch,idx) in zip(epochs,trialidx) if idx !=15]
  #  trialidx = [idx for idx in trialidx if idx !=15]
    
for model, features in FEAT_DICT.items():
    ### START TRF-STUFF ###
    info = epochs[0].info
    n_pred = len(features)
    nchans = len(epochs[0].info['ch_names'])
    lags = lag_span(tmin, tmax, sfreq)
    
    xylist = []
    
    if model == 'frequency':
    # get all frequencies from logfile
    # and calculate mean/std over it
        all_freq = []
        for trial in log[7]:
            for word in trial:
                all_freq.append(get_word_frequency(word))
                
        mean = np.mean(all_freq)
        std = np.std(all_freq)    

    # Allocate memory for covariance matrix
    if fit_intercept:
        e = 1
    else:
        e = 0
    
    XtX_sent = np.zeros((len(lags)*n_pred + e, len(lags)*n_pred + e))
    Xty_sent = np.zeros((len(lags)*n_pred + e, nchans))
    XtX_list = np.zeros((len(lags)*n_pred + e, len(lags)*n_pred + e))
    Xty_list = np.zeros((len(lags)*n_pred + e, nchans))
    
    
    # Divide data into train and test splits
    traini, testi = train_test_split(list(range(60)), test_size=0.2, random_state=42)
    traini = traini + [i + 60 for i in traini] + [i + 120 for i in traini] + [i + 180 for i in traini] + [i + 240 for i in traini]
    testi = testi + [i + 60 for i in testi] + [i + 120 for i in testi] + [i + 180 for i in testi] + [i + 240 for i in testi]
    xytest = {'sentence':[], 'list':[]}
    
    # calculate mean (mu) and std (sigma) across training trials
    mu = np.hstack([np.squeeze(tr._data) for tr in epochs]).mean(1)
    sigma = np.hstack([np.squeeze(tr._data) for tr in epochs]).std(1)
    
    # here we accumulate the annotations etc
    print("\nStarting the accumulation of covmat...")
    for i, (ep, cond, trial) in enumerate(zip(epochs, condition, trialidx)):
        
        # normalize the trial data
        y = (ep._data[0].T - mu)/sigma
        
        try:
           # this makes sure the auditory localizer is not included in the TRF
            if i < np.shape(log)[-1]:
                # get envelope and annotations
                N = np.shape(ep._data)[-1]
        
                if cond in conditions['sentence']:
                    trigger = '101'
                    annotations = sent_annot
                else:
                    trigger = '105' # word list
                    annotations = list_annot
        
                audio = '/project/3027006.01/01_Task/Materials/Stimuli/{0}_{1}.wav'.format(trigger, trial)
                fs, signal = wavread(audio)
                env = signal_envelope(signal, fs, resample=sfreq, verbose=False)
                
                if N is not None and len(env)!=N:
                    if len(env) < N:
                        env = np.pad(env, (N-len(env), 0))
                    else:
                        env = env[:N]
                        
                # create matrix
                x = zscore(env)
            
                if model in ['onset', 'frequency']:
        
                    # get word onsets in seconds, words, and wf
                    onsets = annotations.loc[(annotations['ItemTrigger'] == trial) & 
                                            ((annotations['Segment'] == 'word') | 
                                            (annotations['Segment'] == 'de') |
                                            (annotations['Segment'] == 'het') |
                                            (annotations['Segment'] == 'een') |
                                            (annotations['Segment'] == 'en'))
                                            , 'SegmentStart']
        
                    words = log[7][i]
                    x_w = np.zeros((N, n_pred-1))
                    samples = np.round(np.asarray(onsets)*sfreq).astype(int)
                    x_w[samples,0] = 1 # word onset feature
                
                    if model == 'frequency':
                        frequency = [get_word_frequency(word) for word in words]
                        x_w[samples,1] = [value - mean / std for value in frequency] # wf feature
                   
                    x_w = temporal_smoother(x_w, fs=sfreq, std_time=0.015)
                    x = np.hstack([x[:, None], x_w])
                    
                # Create lag matrix and remove the NaN-rows
                X = lag_matrix(x, lags) #, filling=0. drop_missing=False
                nan_rows = np.isnan(X.mean(1))
                X = X[~nan_rows]
                y = y[~nan_rows] # also for the meg
        
                # Add the trial to the training or testing list
                if i in traini:
                    # compute xylist for crossval
                    xylist.append((trial, X, y))
                    if fit_intercept:
                        X = np.hstack([np.ones((X.shape[0],1)), X])
                        
                    # + Accumulate covmat for fitting the model afterwards
                    if cond in conditions['sentence']:
                        XtX_sent += X.T @ X
                        Xty_sent += X.T @ y
                    else:
                        XtX_list += X.T @ X
                        Xty_list += X.T @ y
                        
                elif i in testi:
                    if cond in conditions['sentence']:
                        xytest['sentence'].append((trial, X, y))
                    else:
                        xytest['list'].append((trial, X, y))
                            
        # handling exceptions just by passing them for now...           
        except (ValueError, IndexError):
            print("\nSomething is wrong with trial {0} for subject {1}. Continuing to the next trial...".format(trial, subject))
            pass    
        
    
    # compute TRF and save
    if isinstance(alpha, np.ndarray):
        print('\nCovariance matrices computed. Crossvalidating...')
        best_alpha, scores, _, _ = crossval(xylist, lags, sfreq, alpha=alpha, features=features, \
                                 info=info, picks=info['ch_names'], n_splits=8, fit_intercept=True,
                                 plot=True)
        
        print("\nDone! Best alpha: {0}".format(str(best_alpha)))
    
    else:
        best_alpha = alpha
        
    print("\nFitting TRF and saving...")
    beta_list = np.linalg.inv(XtX_list + np.eye(len(lags)*n_pred+e)*best_alpha) @ Xty_list
    beta_sent = np.linalg.inv(XtX_sent + np.eye(len(lags)*n_pred+e)*best_alpha) @ Xty_sent
        
    if fit_intercept:
        beta_list = beta_list[1:,:]
        beta_sent = beta_sent[1:,:]
        
    beta_list = beta_list.reshape((len(lags), n_pred, -1))
    beta_sent = beta_sent.reshape((len(lags), n_pred, -1))
        
    args = lags, epochs[0].info, features
        
    TRF_list = TRF(beta_list, *args)
    TRF_sent = TRF(beta_sent, *args)
    
    with open(os.path.join(savedir, str(subject), 'TRF_{0}.pkl'.format(model)), 'wb') as f:
        trf = {}
        trf['alpha'] = best_alpha
        trf['list'] = TRF_list
        trf['sentence'] = TRF_sent
        pickle.dump(trf, f)
        
    print("\nDone! Predicting...")
    
    R = dict.fromkeys(['sentence', 'list'])
    
    for c, trf_tmp in zip(['sentence', 'list'], [TRF_sent, TRF_list]):
        xylist = xytest[c]
        
        # betas  np.reshape(self.coef_[::-1, :, :], (len(self.lags) * self.n_feats_, self.n_chans_))
        betas = np.reshape(trf_tmp.coef_, (len(lags)*n_pred, len(info['ch_names'])))
        
        # test the model 
        r_ = np.zeros((len(xylist), len(info['ch_names'])))  # accumulate r scores per trial
        
        for i, (s_id, X, y) in enumerate(xylist):
            yhat = X @ betas  
            r_[i, :] = np.diag(np.corrcoef(yhat[:, :], y, rowvar=False), k=len(info['ch_names']))  # (1, channels)
    
        R[c] = r_.mean(0)
    
    print("Done! Saving R-values...")
    with open(os.path.join(savedir, str(subject), 'R_{0}.pkl'.format(model)), 'wb') as fr:
        pickle.dump(R, fr)
    