#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 14:40:23 2021

@author: sopsla
"""
import mne
from mne.beamformer import apply_lcmv_epochs

def lcmv_apply(epochs, filters):
    for epoch in epochs:
        yield apply_lcmv_epochs(epoch, filters, max_ori_out='signed')[0]
        
def morph_apply(source_epochs, subject, src_to, subjects_dir):
    for epoch in source_epochs:
        morph = mne.compute_source_morph(epoch, subject_from=subject,
                                 subject_to='fsaverage',
                                 src_to=src_to,
                                 subjects_dir=subjects_dir)
        yield morph.apply(epoch)
        
def parcel_labels(hemisphere, labels, all_labels, brain=None, plot=False):
    """
    """
    if hemisphere == 'left':
        labnames = [f'L_{parcel}_ROI-lh' for parcel in labels]
    elif hemisphere == 'right':
        labnames = [f'R_{parcel}_ROI-rh' for parcel in labels]
    all_labnames = [lab.name for lab in all_labels]

    parcels = []
        
    for parcel in labnames:
        idx = all_labnames.index(parcel)
        if plot:
            brain.add_label(all_labels[idx])
  
        parcels.append(all_labels[idx])

    return parcels