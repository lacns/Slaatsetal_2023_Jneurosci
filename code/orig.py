#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 17:56:49 2020

@author: sopsla
"""

import os
import pandas as pd
import numpy as np
import mne
import re

from text import loadmat
from meg import SUBJECTS, CH_NAMES


def read_head_position(subject, return_dig=False, unit='m', DATA_PATH='/project/3011020.13/bids/'):
    """
    Read head shape digitised position from Polhemus file.
    @ hugwei
    """
    data = pd.read_csv(os.path.join(DATA_PATH, 'sub-{0}/meg/sub-{0}_headshape.pos'.format(subject)),
                       delim_whitespace=True, skiprows=1, names=['pnts', 'x', 'y', 'z'])
    pos_col = ['x', 'y', 'z']
    if unit == 'cm':
        scale = 1e2
    elif unit == 'mm':
        scale = 1e3
    else:
        scale = 1
    nasion = data.loc[data.pnts == 'nasion', pos_col].to_numpy().ravel()/1e2
    lpa = data.loc[data.pnts == 'left', pos_col].to_numpy().ravel()/1e2
    rpa = data.loc[data.pnts == 'right', pos_col].to_numpy().ravel()/1e2
    hsp = data.loc[data.pnts.str.match(r'\d'), pos_col].to_numpy()/1e2
    dig =  {'nasion': nasion, 'rpa': rpa, 'lpa': lpa, 'hsp': hsp}
    
    if return_dig:
        return mne.channels.make_dig_montage(**dig)
    return {k: v*scale for k,v in dig.items()}

def load_raw(subject, DATA_PATH='/project/3011020.13/bids/', rest=False):
    """
    Loads raw MEG data for MOUS project. Some participants have two runs.
    They are appended to form one mne.Raw object. 
    Adds Polhemus to Raw. info.
    
    Input
    -----
    subject : str
    DATA_PATH : str
        path to raw meg
        
    Output
    ------
    raw : mne.Raw data structure
    stimuli : pd.DataFrame
        stimulus information for subject
    """
    if rest:
        runtype = 'rest'
    else:
        runtype = 'auditory'
        
    print('\nGetting data from task type: {0}'.format(runtype))
    
    MEG_PATH = os.path.join(DATA_PATH, 'sub-{0}/meg/'.format(subject))

    # load data, incl. check for multiple task runs
    print('\nReading dataset of subject {0} ...\n'.format(subject))
    datasets = [ctf for ctf in os.listdir(MEG_PATH) if ctf.endswith('.ds') and re.search(runtype, ctf)]
    
    for dataset in datasets:
        tmp_raw = mne.io.read_raw_ctf(os.path.join(MEG_PATH, dataset)) 
        #tmp_raw.pick_channels(CH_NAMES) # all channels need to be the same?
    
        if len(datasets) == 1 or datasets.index(dataset) == 0:
            raw = tmp_raw
        else:
            raw.append(tmp_raw)
            
    # add polhemus
    print("Reading Polhemus file for subject {0}...".format(subject))
    dig = read_head_position(subject, return_dig=True)
    _ = raw.get_montage() # this is to get the original DigMontage inside "raw"

    # Transform to head coordinate frame
    print("Adding Polhemus data to info...")
    dig_head = mne.channels.montage.transform_to_head(dig)
    raw.info['dig'] += dig_head.dig[3:] # not adding fiducials, only HSP
    #raw = raw.pick_types(meg=True, stim=True, eeg=False, ref_meg=False)
    
    if rest:
        return raw
    
    else:
        # load stimulus info, incl. check for multiple task runs
        print('\nReading stimulus file...')
        stimfiles = [tsv for tsv in os.listdir(MEG_PATH) if tsv.endswith('events.tsv') and re.search(runtype, tsv)]


        for stimfile in stimfiles:
            tmp_stimuli = pd.read_table(os.path.join(MEG_PATH, stimfile))

            if len(stimfiles) == 1 or stimfiles.index(stimfile) == 0:
                stimuli = tmp_stimuli
            else:
                stimuli.append(tmp_stimuli)

            stimuli.reset_index(inplace=True)

        return raw, stimuli

def epoch_to_offset(raw, stimuli, prestim = 0.2, poststim = 0):
    """
    Epochs the MOUS-data around the duration of one stimulus.
    
    Input
    -----
    raw : mne.Raw
    stimuli : pd.DataFrame
        events.tsv file from same path as raw
        output by load_raw
    prestim : float
        time (s) to add pre stimulus
    poststim : float
        time (s) to add post stimulus
        
    Output
    -----
    epochs : list of mne.Epochs
    stim_ids : list of int
        ids of the stimuli corresponding to the epochs
    """
    print('Epoching ...')
    # taking the events from the stimulus file bc of trigger issues

    onset_samples = stimuli.loc[stimuli['value'].str.contains(r'[1357] Audio onset', regex=True, na=False), 'sample']
    offset_samples = stimuli.loc[stimuli['value'] == '15 End of file', 'sample']
    stim_ids = [int(item[-7:-4]) for item in stimuli.loc[stimuli['value'].str.contains(r'14 Start File .*.wav', regex=True, na=False), 'value']]
    
    # todo: incorporate stim_ids
    epochs = []
    for onset, offset, stim_id in zip(onset_samples, offset_samples, stim_ids):        
        t_idx = raw.time_as_index([onset/1200-prestim, offset/1200+poststim])  
        epoch, times = raw[:, t_idx[0]:t_idx[1]]  
        
        epochs.append(mne.EpochsArray(np.expand_dims(epoch, 0), raw.info, tmin=-prestim, verbose=False))
        
    print('Epoching done: {0} trials found.'.format(len(epochs)))
    return epochs, stim_ids
