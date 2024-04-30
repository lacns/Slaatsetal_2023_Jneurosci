#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 13:12:06 2022

@author: sopsla
"""
import os
import pickle
from stats import cluster_1samp, cluster_2samp

resultsdir = '/project/3027005.01/results/timelag_800ms/zscore-adventure/delta_nan'

with open(os.path.join(resultsdir, 'trfs.pkl'), 'rb') as f:
    trfs = pickle.load(f)
    
with open(os.path.join(resultsdir, 'GA.pkl'), 'rb') as fd:
    GA = pickle.load(fd)
    
info = GA['frequency_entropy_surprisal']['sentence'].info

# small permutation (includes only envelope, word onset, word frequency)
small_permutation = cluster_2samp(trfs, info=info, model='frequency', statfun='ttest')
with open(os.path.join(resultsdir, 'small-permutation.pkl'), 'wb') as f:
    pickle.dump(small_permutation, f)
    
# large permutation (includes all models)
large_permutation = cluster_2samp(trfs,info=info, model='frequency_entropy_surprisal', statfun='ttest')
with open(os.path.join(resultsdir, 'large-permutation.pkl'), 'wb') as f:
    pickle.dump(large_permutation, f) # these files include the results for all features in the model!
    
# onset permutation (only envelope & word onset)
onset_permutation = cluster_2samp(trfs,info=info, model='onset', statfun='ttest')
with open(os.path.join(resultsdir, 'onset-permutation.pkl'), 'wb') as f:
    pickle.dump(onset_permutation, f)

# one-sample tests
one_sample = cluster_1samp(trfs, info=info, model='frequency_entropy_surprisal')

for clusters,name in zip(one_sample, ['list', 'sentence']):
    with open(os.path.join(resultsdir, f'{name}-1samp-permutation.pkl'), 'wb') as f:
        pickle.dump(clusters,f) # these files include the results for the word frequency trf only
        