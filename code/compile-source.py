#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 14:48:24 2021

@author: sopsla
"""
from collect import compile_trfs, grandaverage, compile_rvals
import mne

BANDPASS = 'delta'
resultsdir = f'/project/3027005.01/results/srilm/source/delta/'

#### PREPARE MEG DATA ####
n_vert = 8196
picks = [str(no) for no in list(range(n_vert))]
info = mne.create_info(picks, sfreq=120, ch_types='mag')

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
trfs = compile_trfs(resultsdir, conditions=conditions, models=['frequency_entropy_surprisal'], save=True, source=True)
GA = grandaverage(trfs, info=info, models=['frequency_entropy_surprisal'], conditions=conditions, savedir=resultsdir, save=True)
r_data = compile_rvals(resultsdir, info=info, models=['entropy_surprisal', 'frequency_entropy_surprisal'], source=True, save=True)
