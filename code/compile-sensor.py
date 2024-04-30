#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 13:36:18 2022

@author: sopsla
"""
from collect import compile_trfs, grandaverage, compile_rvals
from meg import CH_NAMES
import mne
import os

BANDPASS = 'delta'
resultsdir = f'/project/3027005.01/results/srilm/sensor/{BANDPASS}'

#### PREPARE MEG DATA ####
picks = CH_NAMES
info = mne.io.read_info(os.path.join('/project/3027005.01/', 'fif_files', 'sub-A2003', '120Hz_raw.fif'))
info.pick_channels(CH_NAMES)

conditions = ['list','sentence']
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
    
## ACTUAL COMMANDS
trfs = compile_trfs(resultsdir, conditions=conditions, models=FEATURES.keys(), save=True, source=False)
GA = grandaverage(trfs, info=info, models=FEATURES.keys(), conditions=conditions, savedir=resultsdir, save=True)
r_data = compile_rvals(resultsdir, info=info, models=FEATURES.keys(), source=False, save=True)
