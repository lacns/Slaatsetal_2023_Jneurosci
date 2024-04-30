#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 10:45:09 2021

@author: sopsla
"""
# compute covariance matrices per participant
import sys
import os
import mne
from orig import load_raw
from meg import SUBJECTS, CH_NAMES

BANDPASS = 'delta' # this for both frequency bands

if BANDPASS == 'delta':
    hp = 0.5
    lp = 4
elif BANDPASS == 'theta':
    hp = 4
    lp = 10
    
# directories
SUBJECTS_DIR = '/project/3027005.01/SUBJECTS'
DATA_PATH = '/project/3011020.13/bids'

# get a subject
SUBJECTS.sort()
subject = SUBJECTS[int(sys.argv[1])]

####### first: resting state data #######
rest = load_raw(subject=subject, rest=True)

# Filter the resting state data
print('Filtering ...')
rest.load_data()
rest = rest.filter(float(hp), float(lp), fir_design='firwin')
rest = rest.resample(sfreq=120, npad='auto', window='boxcar')
rest.pick_channels(CH_NAMES)

# computing the covariance matrix & plotting it
noise_cov = mne.compute_raw_covariance(rest)
rest_info = rest.info

# save memory
del rest

noise_cov.save(fname=os.path.join(SUBJECTS_DIR, subject, '{0}_noise_cov.fif'.format(BANDPASS)))