#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:53:15 2022

@author: sopsla
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 16:50:57 2021

@author: sopsla
"""
# general modules
import os
import re
import sys
import numpy as np
import pickle

# MEG stuff
import mne
from mne.beamformer import make_lcmv
from pyeeg.utils import lag_span

# local modules
from meg import CH_NAMES, SUBJECTS
from meg import fieldtrip2mne, crossval, fit, predict
from utils import Xy_flexibly, split_X
from morphing import lcmv_apply, morph_apply

from sklearn.model_selection import train_test_split
from statistics import mode

flatten = lambda t: [item for sublist in t for item in sublist]

# subjects with svd problem
#SUBJECTS = ['A2062', 'A2063', 'A2009', 'A2049', 'A2069', 'A2076', 'A2084']
#SUBJECTS = ['A2010', 'A2009', 'A2017','A2049', 'A2062', 'A2063','A2069', \
 #           'A2076', 'A2084', 'A2108', 'A2119']
# likely problematic: A2010, A2017, A2108
    
SUBJECTS.sort()
bp_dict = {'delta': [0.5, 4.0], 'theta': [4.0, 10.0]}


### VARIABLES ###
subject = SUBJECTS[int(sys.argv[1])]
BANDPASS = 'delta' 
tmin=-0.2
tmax=0.8
fs=120
n_splits=5
alpha = np.logspace(-2, 2, 10) * 60470.93937147979# mean Eigenvalue of XtX 60470.94
fit_intercept=True
picks=CH_NAMES

### PATHS ###
if BANDPASS == 'delta':
    DATA_PATH = '/project/3027005.01/TRF_meg/preprocessed/bandpass_0.5-4'
elif BANDPASS == 'theta':
    DATA_PATH = '/project/3027005.01/TRF_meg/preprocessed/bandpass_4-10'

MEG_FILES = os.listdir(DATA_PATH)
MEG_FILES.sort()

print(f'Computing beamformer & TRF for subject {subject} in the {BANDPASS} band...')

SUBJECTS_DIR = '/project/3027005.01/SUBJECTS'
save_dir = f'/project/3027005.01/results/source-analysis/whole-brain-mask/{BANDPASS}/{subject}'
try:
    os.makedirs(save_dir)
except FileExistsError:
     # directory already exists
    pass

#### MODELS ####
# we fit only the largest two models
FEATURES = {'frequency_entropy_surprisal':['envelope', 'word onset', 'entropy', 'surprisal', 'word frequency'],
            'entropy_surprisal':['envelope', 'word onset', 'entropy', 'surprisal']}

megfile = os.path.join(DATA_PATH, [file for file in MEG_FILES if re.search(subject, file)][0])

### ANALYSIS ###
# 1. Epochs
# read data
epochs, stim_ids = fieldtrip2mne(fname=megfile) 
epochs = [tr.pick(picks) for tr in epochs]
stim_ids = list(stim_ids)

# take the info object for the first trial for further use
info = epochs[0].info

# 2. Covariance matrices
noise_cov = mne.read_cov(fname=os.path.join(SUBJECTS_DIR, subject, '{0}_noise_cov.fif'.format(BANDPASS)))

# previous: rank not specified, method = 'empirical'
# changed to see if it would solve the convergence issues, but it did not
data_cov = mne.compute_covariance(mne.EpochsArray(np.concatenate(([ep._data for ep in epochs]), axis=2), info) \
    , tmin=0, tmax=None, rank=None, method='empirical')
    
# 3. Forward solution
try:
    fwd = mne.read_forward_solution(fname=os.path.join(SUBJECTS_DIR, subject, 'new-8196src-fwd.fif'))
except FileNotFoundError:
    raise FileNotFoundError(f'No forward solution for subject {subject}. Aborting')
        
src_fwd=fwd['src']

# 4. Create the filters
filters = make_lcmv(epochs[0].info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                   pick_ori='max-power', rank=None) # scalar beamformer: one source estimate per source

# 5. Apply the filters to the epochs in a generator
source_epochs = lcmv_apply(epochs, filters)

# save memory
del epochs, filters, noise_cov, data_cov, fwd

# 6. Morph the epochs to fsaverage in a generator
src_to = mne.read_source_spaces(fname=os.path.join(SUBJECTS_DIR, 'fsaverage/bem/fsaverage-oct-6-src.fif'))
fsaverage_epochs = list(morph_apply(source_epochs, subject, src_to, SUBJECTS_DIR))

# save memory
del source_epochs, src_to, src_fwd

# pick the sources that are not in the center of the head
#print('Discarding the sources in the center of the head')
#sample_path = mne.datasets.sample.data_path() + '/subjects'
#mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=sample_path)
#labels = mne.read_labels_from_annot('fsaverage', 'HCPMMP1_combined', 'both', surf_name='pial',subjects_dir=sample_path)
#good_vertices = labels[2]
#for label in labels[3:-1]:
 #   good_vertices += label
#fsaverage_epochs = [ep.in_label(good_vertices) for ep in fsaverage_epochs]    

print('Epochs have been source-localized and morphed to fsaverage. Onto TRF computation!')

picks = flatten(fsaverage_epochs[0].vertices)
lags = lag_span(tmin, tmax, fs)
info = mne.create_info([str(p) for p in picks], sfreq=120, ch_types='mag')

# split data into train and test set
train_id, test_id = train_test_split(stim_ids, test_size=0.2, random_state=42)
train = [stim_ids.index(i) for i in train_id]
test = [stim_ids.index(i) for i in test_id]

# calculate mean (mu) and std (sigma) across training epochs
mu = np.hstack([np.squeeze(tr._data) for tr in [fsaverage_epochs[i] for i in train]]).mean(1)
sigma = np.hstack([np.squeeze(tr._data) for tr in [fsaverage_epochs[i] for i in train]]).std(1)

# prepare X, y once, then use splitX
xylist = list(Xy_flexibly(fsaverage_epochs, stim_ids, fs, lags, mu, sigma, picks, 
                          features=['envelope', 'word onset', 'entropy', 'surprisal', 'word frequency'], 
                          tim=0.015))

# ensure float32 instead of float64
xylist = [(stim_id, np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)) for (stim_id, X, y) in xylist]

del fsaverage_epochs, train, test

# prepare X, y
for model, features in FEATURES.items():
    TRF_fname = os.path.join(save_dir, 'sTRF_{0}.pkl'.format(model))
    R_fname = os.path.join(save_dir, 'sR_{0}.pkl'.format(model))
    
    # select features
    print(f"Getting features for model {model}...")
    sub_xylist = [(stim_id, split_X(X, lags, features), y) for (stim_id, X, y) in xylist]
    scores = np.zeros((n_splits, len(alpha), len(info.ch_names)))
    
    chunk_size = int(len(info.ch_names)/12)
          
    for i in range(0, len(info.ch_names), chunk_size):
        sub_sub_xylist = [(stim_id, X, y[:,i:i+chunk_size]) for (stim_id, X, y) in sub_xylist]
        info_tmp = info.copy()
        info_tmp.pick_channels(info.ch_names[i:i+chunk_size])
        
        # split into train and test sets - now for X and y
        xytrain = [xy for xy in sub_sub_xylist if xy[0] in train_id]
        xytest = [xy for xy in sub_sub_xylist if xy[0] in test_id]
        
        del sub_sub_xylist
        
        # crossvalidate on train
        print("Crossvalidating...")
        _, scores[:,:,i:i+chunk_size], _, _ = crossval(xytrain, lags=lags, fs=fs, alpha=alpha, 
                                            features=features, info=info_tmp,
                                            picks=info_tmp.ch_names, n_splits=n_splits, 
                                            fit_intercept=fit_intercept, plot=True)
        
    # Get the best alpha 
    # Take the mean over sensors & maximum value over alphas
    peaks = scores.mean(-1).argmax(1)
    
    # catch function: if reconstruction accuracy never peaks, take value
    # BEFORE reconstruction accuracy takes a steep dive.
    catch_r = scores.mean(-1)

    for kf in list(range(n_splits)):
        if all([catch_r[kf,i+1] < catch_r[kf,i] for i in list(range(len(alpha)-1))]):
            deriv = [catch_r[kf,i+1] - catch_r[kf,i] for i in list(range(len(alpha)-1))]
            peaks[kf] = deriv.index(min(deriv))
                
    best_alpha = alpha[mode(peaks)]     
    print(f"Best alpha: {best_alpha}")
    
    del info_tmp, xytrain, xytest
    
    # split into train and test sets - now for X and y
    xytrain = [xy for xy in sub_xylist if xy[0] in train_id]
    xytest = [xy for xy in sub_xylist if xy[0] in test_id]
    
    del sub_xylist
        
    # # fit model on train
    print(f'Computing TRF for model {model}...')
    TRF_list, TRF_sent = fit(xytrain, lags, best_alpha, picks=info.ch_names, features=features, info=info, 
                          fit_intercept=fit_intercept)
    
    # we only save the TRFs for the largest model to save memory
    if model == 'frequency_entropy_surprisal':
        with open(TRF_fname, 'wb') as f:
            trf = {}
            trf['alpha'] = best_alpha
            trf['list'] = TRF_list
            trf['sentence'] = TRF_sent
            pickle.dump(trf, f)

    # predict
    r_list = predict(TRF=TRF_list, xylist=xytest, condition='word list', features=features, picks=picks, lags=lags)
    r_sent = predict(TRF=TRF_sent, xylist=xytest, condition='sentence', features=features, picks=picks, lags=lags)

    with open(R_fname, 'wb') as fr:
        r_values = {}
        r_values['list'] = r_list
        r_values['sentence'] = r_sent
        pickle.dump(r_values, fr)
    
