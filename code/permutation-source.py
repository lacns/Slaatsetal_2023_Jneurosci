#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 11:43:49 2022

@author: sopsla
"""
import os
import numpy as np
import pandas as pd
import pickle
import scipy.stats as stats

# MEG stuff
import mne
from mne import spatial_src_adjacency
from pyeeg.utils import lag_span

# local modules
from meg import SUBJECTS
from stats import source_cluster_2samp, source_cluster_1samp
from mne.stats import f_mway_rm, f_threshold_mway_rm
ttest = lambda x, y: mne.stats.cluster_level.ttest_1samp_no_p(x-y)

flatten = lambda t: [item for sublist in t for item in sublist]

SUBJECTS.sort()
SUBJECTS_DIR = '/project/3027005.01/SUBJECTS'

resultsdir = '/project/3027005.01/results/srilm/source/'

with open(os.path.join(resultsdir, 'trfs.pkl'), 'rb') as f:
    trfs = pickle.load(f)
    
rvals = pd.read_csv(os.path.join(resultsdir, 'r_data.csv'))

with open(os.path.join(resultsdir, 'GA.pkl'), 'rb') as f:
    GA = pickle.load(f)
    
# adjacency matrix
src = mne.read_source_spaces(fname=os.path.join(SUBJECTS_DIR, 'fsaverage/bem/fsaverage-oct-6-src.fif'))
adjacency = spatial_src_adjacency(src)
vertices=[src[0]['vertno'], src[1]['vertno']]

## get the indices of the vertices of the medial wall to exclude
tmp_src = mne.SourceEstimate(data=trfs['frequency_entropy_surprisal']['sentence'][0]['word frequency'].data,
                                 vertices=vertices,tmin=-0.2, tstep=1/120, subject='fsaverage')
sample_path = mne.datasets.sample.data_path() + '/subjects'
mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=sample_path)
labels = mne.read_labels_from_annot('fsaverage', 'HCPMMP1_combined', 'both', surf_name='pial',subjects_dir=sample_path)

# get indices of vertices to exclude
midbrain_label = labels[0] + labels[1]
bad_src = tmp_src.in_label(midbrain_label)

idx = []
for hemi in [0,1]:
    i = [list(tmp_src.vertices[hemi]).index(v) for v in bad_src.vertices[hemi]]
    if hemi == 1:
        i = [v + 4098 for v in i]
    idx.append(np.asarray(i))
    
idx = np.concatenate((idx[0],idx[1]))

# adjust data structure for permutation
info = trfs['frequency_entropy_surprisal']['sentence'][0].info
nsubj = len(trfs['frequency_entropy_surprisal']['sentence'])
nvert = len(info.ch_names)

X_list = np.zeros((nsubj, 120, nvert))
X_sent = np.zeros((nsubj, 120, nvert))

for sub in list(range(nsubj)):
    X_list[sub,:,:] = trfs['frequency_entropy_surprisal']['list'][sub]['word frequency'].data.T
    X_sent[sub,:,:] = trfs['frequency_entropy_surprisal']['sentence'][sub]['word frequency'].data.T
    
# get the time windows
lags = lag_span(-0.2,0.8,120)
trf_times = list(lags/120)
early_idx = [trf_times.index(0.2),trf_times.index(0.4)]
late_idx = [trf_times.index(0.5),trf_times.index(0.7)]

early_sent = X_sent[:,early_idx[0]:early_idx[1],:]
early_list = X_list[:,early_idx[0]:early_idx[1],:]
late_sent = X_sent[:,late_idx[0]:late_idx[1],:]
late_list = X_list[:,late_idx[0]:late_idx[1],:]

# two-sample test
# threshold
multiplication= 1
alpha_statfun = 0.025
threshold = stats.t.ppf(1-alpha_statfun/2, df=nsubj-1) * multiplication

# early time-window
cluster_stats = source_cluster_2samp(early_list, early_sent, threshold, adjacency,exclude=idx)

with open(os.path.join(resultsdir, f'trf-source-mask-early-thresh{multiplication}.pkl'), 'wb') as f:
    pickle.dump(cluster_stats, f)

cluster_stats2 = source_cluster_2samp(late_list, late_sent, threshold, adjacency,exclude=idx)

with open(os.path.join(resultsdir, f'trf-source-mask-late-thresh{multiplication}.pkl'), 'wb') as f:
    pickle.dump(cluster_stats2, f)
    
# one-sample tests
# sentence
sent_early = source_cluster_1samp(early_sent, threshold, adjacency, cluster_alpha=0.01, exclude=idx)
sent_late = source_cluster_1samp(late_sent, threshold, adjacency, cluster_alpha=0.01, exclude=idx)
list_early = source_cluster_1samp(early_list, threshold, adjacency, cluster_alpha=0.01, exclude=idx)
list_late = source_cluster_1samp(late_list, threshold, adjacency, cluster_alpha=0.01, exclude=idx)

with open(os.path.join(resultsdir, f'trf-source-mask-1samp-early-sentence-thresh{multiplication}.pkl'), 'wb') as f:
    pickle.dump(sent_early, f)

with open(os.path.join(resultsdir, f'trf-source-mask-1samp-late-sentence-thresh{multiplication}.pkl'), 'wb') as f:
    pickle.dump(sent_late, f)
    
with open(os.path.join(resultsdir, f'trf-source-mask-1samp-early-list-thresh{multiplication}.pkl'), 'wb') as f:
    pickle.dump(list_early, f)

with open(os.path.join(resultsdir, f'trf-source-mask-1samp-late-list-thresh{multiplication}.pkl'), 'wb') as f:
    pickle.dump(list_late, f)
    
### the reconstruction accuracies ###
model1 = 'entropy_surprisal'
model2 = 'frequency_entropy_surprisal'

data = rvals
nsubj = len(set(rvals['subject']))

# prepare the data
list_onset = np.asarray(data.loc[(data['model'] == model1) & (data['condition'] == 'list'), 'r_values'])
sent_onset = np.asarray(data.loc[(data['model'] == model1) & (data['condition'] == 'sentence'), 'r_values'])
list_freq = np.asarray(data.loc[(data['model'] == model2) & (data['condition'] == 'list'), 'r_values'])
sent_freq = np.asarray(data.loc[(data['model'] == model2) & (data['condition'] == 'sentence'), 'r_values'])

list_onset = list_onset.reshape((nsubj,8196))
sent_onset = sent_onset.reshape((nsubj,8196))
list_freq = list_freq.reshape((nsubj,8196))
sent_freq = sent_freq.reshape((nsubj,8196))

X = [list_onset, list_freq, sent_onset, sent_freq]

# stats settings
factor_levels = [2,2]
effects = 'A:B'
return_pvals = False

n_times = 1
n_conditions = 4
n_subjects = nsubj

def stat_fun(*args):
    # get f-values only
    return f_mway_rm(np.swapaxes(args,1,0), factor_levels = factor_levels, effects=effects,
                    return_pvals=return_pvals)[0]

pthresh = 0.05
n_permutations = 1024
f_thresh = f_threshold_mway_rm(n_subjects, factor_levels, effects, pthresh)

# create mask to exclude the correct sources
mask = np.zeros((1,8196),dtype=bool)
idx_array = np.asarray(idx)
mask[:,idx_array] = True
mask = mask.ravel()

# permutation
cluster_stats_r = mne.stats.permutation_cluster_test(X, adjacency=adjacency, n_jobs=1,
                                                    threshold=f_thresh, stat_fun=stat_fun,
                                                    n_permutations=n_permutations,
                                                    buffer_size=None, exclude=mask)

print("There are %d significant clusters\n"%(cluster_stats_r[2]<0.05).sum())

with open(os.path.join(resultsdir, 'r-source-permutation.pkl'), 'wb') as f:
    pickle.dump(cluster_stats_r, f)

# effect of model per condition (main effects)
X_list = [X[1], X[0]]
X_sent = [X[3], X[2]]

threshold = stats.t.ppf(1-0.05/2, df=nsubj-1)

cluster_r_sent = mne.stats.permutation_cluster_test(X_sent, adjacency=adjacency, n_jobs=1,
                                                    threshold=threshold, stat_fun=ttest,
                                                    n_permutations=n_permutations,
                                                    buffer_size=None, exclude=mask)

print("There are %d significant clusters\n"%(cluster_r_sent[2]<0.05).sum())

with open(os.path.join(resultsdir, 'r-source-permutation-sent.pkl'), 'wb') as f:
    pickle.dump(cluster_r_sent, f)
    
cluster_r_list = mne.stats.permutation_cluster_test(X_list, adjacency=adjacency, n_jobs=1,
                                                    threshold=threshold, stat_fun=ttest,
                                                    n_permutations=n_permutations,
                                                    buffer_size=None, exclude=mask)

print("There are %d significant clusters\n"%(cluster_r_list[2]<0.05).sum())

with open(os.path.join(resultsdir, 'r-source-permutation-list.pkl'), 'wb') as f:
    pickle.dump(cluster_r_list, f)
