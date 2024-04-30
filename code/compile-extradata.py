#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 13:36:18 2022

@author: sopsla
"""
from collect import compile_trfs, grandaverage, compile_rvals
#from meg import CH_NAMES
#import mne

BANDPASS = 'delta'
resultsdir = f'/project/3027005.01/extra_data/{BANDPASS}_reg'

#### PREPARE MEG DATA ####
#picks = CH_NAMES
#info = mne.create_info(picks, sfreq=120, ch_types='mag')

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
trfs = compile_trfs(resultsdir, conditions=conditions, models=['envelope', 'onset', 'frequency'], save=True, source=False)
GA = grandaverage(trfs, info=trfs['frequency']['sentence'][0].info, models=['envelope', 'onset', 'frequency'], conditions=conditions, savedir=resultsdir, save=True)
r_data = compile_rvals(resultsdir, info=trfs['frequency']['sentence'][0].info, models=['envelope', 'onset', 'frequency'], source=False, save=True)
