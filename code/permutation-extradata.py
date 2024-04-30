#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 14:53:45 2022

@author: sopsla
"""
import os
import pickle
from stats import cluster_1samp, cluster_2samp

resultsdir = '/project/3027005.01/extra_data/theta_reg'

with open(os.path.join(resultsdir, 'trfs.pkl'), 'rb') as f:
    trfs = pickle.load(f)
    
with open(os.path.join(resultsdir, 'GA.pkl'), 'rb') as fd:
    GA = pickle.load(fd)
    
info = GA['frequency']['sentence'].info

# small permutation (includes only envelope, word onset, word frequency)
small_permutation = cluster_2samp(trfs, info=info, model='frequency', statfun='ttest', multiplication=1)
with open(os.path.join(resultsdir, 'small-permutation.pkl'), 'wb') as f:
    pickle.dump(small_permutation, f)
    
# onset permutation (only envelope & word onset)
onset_permutation = cluster_2samp(trfs,info=info, model='onset', statfun='ttest', multiplication=1)
with open(os.path.join(resultsdir, 'onset-permutation.pkl'), 'wb') as f:
    pickle.dump(onset_permutation, f)

# one-sample tests
one_sample = cluster_1samp(trfs, info=info, model='frequency', multiplication=1)

for clusters,name in zip(one_sample, ['list', 'sentence']):
    with open(os.path.join(resultsdir, f'{name}-1samp-permutation.pkl'), 'wb') as f:
        pickle.dump(clusters,f) # these files include the results for the word frequency trf only
        