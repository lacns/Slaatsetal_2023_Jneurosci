#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 15:26:58 2021

@author: sopsla
"""
# general modules
import numpy as np
import os
import sys
import pickle
#import argparse

# pyeeg
from pyeeg.utils import lag_span

# local modules
from meg import CH_NAMES, SUBJECTS
from meg import fieldtrip2mne, crossval, fit, predict
from utils import train_test, Xy_flexibly

bp_dict = {'delta': [0.5, 4], 'theta': [4, 10]}
FEATURES = {'envelope':['envelope'], 
            'onset':['envelope', 'word onset'],
            'frequency':['envelope', 'word onset', 'word frequency'],
            'entropy':['envelope', 'word onset', 'entropy'],
            'surprisal':['envelope', 'word onset', 'surprisal'],
            'entropy_surprisal':['envelope', 'word onset', 'entropy', 'surprisal'],
            'frequency_entropy':['envelope', 'word onset', 'entropy', 'word frequency'],
            'frequency_surprisal':['envelope', 'word onset', 'surprisal', 'word frequency'],
            'frequency_entropy_surprisal':['envelope', 'word onset', 'entropy', 'surprisal', 'word frequency']
            }

# bandpass
BANDPASS = 'delta'
PATH_TO_MEG = f'/project/3027005.01/TRF_meg/preprocessed/bandpass_{bp_dict[BANDPASS][0]}-{bp_dict[BANDPASS][1]}'
MEG_FILES = os.listdir(PATH_TO_MEG)

# Set arguments
tmin = -0.2
tmax = 0.8
fs = 120
fit_intercept=True
picks=CH_NAMES
alpha = np.logspace(-3, 3, 20) * 60470.93937147979 # logspace from mean Eigenvalue of XtX 60470.94
save_dir = f'/project/3027005.01/results/publication/sensor/{BANDPASS}' # directory for saving

# prepare subjects
MEG_FILES.sort()
SUBJECTS.sort()
BAD_SUBJECTS = ['A2001','A2012','A2018','A2022','A2023','A2026','A2043','A2044','A2045','A2048','A2054','A2060',
                'A2074','A2081','A2082','A2087','A2093', 'A2100','A2107','A2112','A2115','A2118','A2123','AP02']

try:
    os.makedirs(save_dir)
except FileExistsError:
    # directory already exists
    pass

subject = SUBJECTS[int(sys.argv[1])] # for bash scripting
dirname = os.path.join(save_dir, str(subject))
if str(subject) in BAD_SUBJECTS:
    dirname = os.path.join(save_dir, ''.join(['BAD_', str(subject)]))
    
try:
    os.makedirs(os.path.join(save_dir, str(subject)))
except FileExistsError:
    # directory already exists
    pass
 
# read data
trials, sent_id = fieldtrip2mne(fname=os.path.join(PATH_TO_MEG, MEG_FILES[int(sys.argv[1])]))
trials = [tr.pick(picks) for tr in trials]

# take the info object for the first trial for further use
info = trials[0].info

# split data into train and test set
train, test = train_test(sent_id, ratio=0.2)

# calculate mean (mu) and std (sigma) across training trials
mu = np.hstack([np.squeeze(tr._data) for tr in [trials[i] for i in train]]).mean(1)
sigma = np.hstack([np.squeeze(tr._data) for tr in [trials[i] for i in train]]).std(1)

# prepare X, y, and lags
lags = lag_span(tmin, tmax, fs)

for model, features in FEATURES.items():
    TRF_fname = os.path.join(save_dir, str(subject), 'TRF_{0}.pkl'.format(model))
    R_fname = os.path.join(save_dir, str(subject), 'R_{0}.pkl'.format(model))
    
    xylist = list(Xy_flexibly(trials, sent_id, fs, lags, mu, sigma, picks, features, tim=0.015))
    
    # split into train and test sets - now for X and y
    xytrain = [xylist[i] for i in train]
    xytest = [xylist[j] for j in test]

    # crossvalidate on train data
    best_alpha, scores, betas, peaks = crossval(xytrain, lags=lags, fs=fs, alpha=alpha, 
                                            features=features, info=info,
                                            picks=CH_NAMES, n_splits=8, 
                                            fit_intercept=fit_intercept, plot=False)

    # fit model on train
    TRF_list, TRF_sent = fit(xytrain, lags, best_alpha, features=features, info=info, 
                         fit_intercept=fit_intercept)
    
    with open(os.path.join(save_dir, str(subject), 'TRF_{0}.pkl'.format(model)), 'wb') as f:
        trf = {}
        trf['alpha'] = best_alpha
        trf['list'] = TRF_list
        trf['sentence'] = TRF_sent
        pickle.dump(trf, f)

    # predict using test data
    r_list = predict(TRF=TRF_list, xylist=xytest, condition='word list', features=features, lags=lags)
    r_sent = predict(TRF=TRF_sent, xylist=xytest, condition='sentence', features=features, lags=lags)

    with open(os.path.join(save_dir, str(subject), 'R_{0}.pkl'.format(model)), 'wb') as fr:
        r_values = {}
        r_values['list'] = r_list
        r_values['sentence'] = r_sent
        pickle.dump(r_values, fr)
        