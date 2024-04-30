#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 14:56:30 2022

@author: sopsla
"""
import os
import numpy as np
import h5py
import mne
import matplotlib.pyplot as plt
from statistics import mode
from sklearn.model_selection import KFold

PATH_TO_MEG = '/project/3027005.01/TRF_meg/preprocessed/bandpass_0.5-4/'
MEG_FILES = [fname for fname in os.listdir(PATH_TO_MEG) if '120Hz' in fname]
SUBJECTS = [s.rstrip('.mat').split('_')[-1] for s in MEG_FILES]

with open(os.path.join('/project/3027005.01/', '272_chnames.csv')) as f:
    CH_NAMES = f.read().split()

class TRF(object):
    "A quick wrapper class to use TRF as Evoked mne data"
    def __init__(self, beta, lags, info, features=["Envelope"], intercept=None):
        self.coef_ = beta
        self.intercept_ = intercept
        self.info = info
        self.lags = lags
        self.times = lags/info['sfreq']
        self.trf = {}
        self.feat_names = features
        for k, f in enumerate(features):
            self.trf[f] = mne.EvokedArray(beta[:, k, :].T, info, tmin=lags[0]/info['sfreq'])
        self.nfeats = len(features)
        
    def __getitem__(self, f):
        return self.trf[f]
    
    def plot_single(self, f, *args, **kwargs):
        self.trf[f].plot(*args, **kwargs)
        
    def plot_all(self, spatial_colors=True, sharey=True, **kwargs):
        f, ax = plt.subplots(1, self.nfeats, figsize=(6*self.nfeats, 4), squeeze=False, sharey=sharey)
        for k, aax in enumerate(ax[0, :]):
            #aax.plot(self.times, self.coef_[:, k, :], 'k', alpha=0.3)
            self.trf[self.feat_names[k]].plot(axes=aax, spatial_colors=spatial_colors, **kwargs)
            aax.set_title(self.feat_names[k])
        if spatial_colors:
            while len(f.get_axes()) > self.nfeats + 1: # remove all but one color legend
                f.get_axes()[self.nfeats].remove()
        return f
            
    def plot_joint_single(self, f, *args, **kwargs):
        self.trf[f].plot_joint(*args, **kwargs)

def fieldtrip2mne(fname, verbose=True):
    """
    Clumsy attempt at transferring Fieldtrip data to MNE instance...

    Parameters
    ----------
    fname : str
        File name of .mat file in which data structure is stored.

    Returns
    -------
    mne.Epochs or list of such
    """
    trials = []
    try:
        f = h5py.File(fname, 'r')
        if 'MEG' in f.keys():
            loc = 'MEG'
        elif 'data' in f.keys():
            loc = 'data'
            
        ntrials, _ = f[loc]['trial'].shape
        if verbose: print("Importing trials (%d trials in total)"%ntrials)
        for k in range(ntrials):
            ref = f[loc]['trial'][k][0]
            trials.append(np.asarray(f[ref]))
        sfreq = f[loc]['fsample'][0][0]
        events = np.asarray(f[loc]['trialinfo'][:, :], dtype=np.int)
        sentence_id = events[-1, :]
        _, nchans = f[loc]['label'].shape
        ch_names = []
        if verbose: print("Getting channel names (%d channels found)"%nchans)
        for k in range(nchans):
            ch_names.append(''.join(chr(c if np.isscalar(c) else c[0]) for c in f[f[loc]['label'][0][k]][()])+'-4304')
            
        info = mne.create_info(ch_names, sfreq, 'mag')
        #standard_info = mne.io.read_info(os.path.join('/project/3027005.01/', 'fif_files', 'sub-A2003', '120Hz_raw.fif'))
        epochs = []
        for tr in trials:
            epochs.append(mne.EpochsArray(np.expand_dims(tr.T, 0), info, verbose=False))
    finally:
        f.close()
    
    return epochs, sentence_id

def crossval(xylist, lags, fs, alpha, features=[],info=None, picks=CH_NAMES, 
             n_splits=5, fit_intercept=False, plot=False):
    """
    Use cross-validation to find the best lambda values (regularization) among
    the list given for one subject. Mean is computed over sensors. Best lambda is
    determined as the most frequent one over all folds.
    
    Input
    -----
    xylist : list of tuples
        List of output of Xy()
    lags : np.array
        Output of lag_span()
    fs : int
        Sampling frequency
    alpha : list of int
        Regularization parameters to test
    features : list of str
        Features to include in the model.
        Options: 'envelope', 'word onset', 'word frequency'
    info : mne info struct
        If None, will load one from project folder
    picks : list of str
        List of channels to pick
    n_splits : int
        Number of folds. Default = 5
    fit_intercept : bool
        Whether to add a column of 1s to the design matrix
        to fit the intercept.
    plot : bool
        Plot scores of crossvalidation
    
    Returns
    -------
    alpha : int
        best regularization parameter for subject
    scores : np.array
        R-values
    betas_new :
        betas
    TRF_list : class TRF
        Instance of class TRF for the best alpha for both conditions
    """
    # make sure we have several alphas
    if np.ndim(alpha) < 1 or len(alpha) <= 1:
        raise ValueError("Supply more than one alpha to use this cross-validation method.")   
    
    if info is None:
        info = mne.io.read_info(os.path.join('/project/3027005.01/', 'fif_files', 'sub-A2003', '120Hz_raw.fif'))
    info["sfreq"] = int(fs)

    # create KFold object for splitting...
    kf = KFold(n_splits=n_splits, shuffle=True)
    
    all_betas = []
    scores = np.zeros((n_splits, len(alpha), len(picks)))
    
    # allocate memory for covmat
    if fit_intercept:
        e = 1
    else:
        e = 0
    
    array_size = len(features)
    
    if 'pos' in features:
        array_size += 9
    
    print(array_size)
        
    XtX = np.zeros((len(lags)*array_size + e, len(lags)*array_size + e))
    Xty = np.zeros((len(lags)*array_size + e, len(picks)))   

    # start cross-validation
    for kfold, (train, test) in enumerate(kf.split(xylist)):
        print("Training/Evaluating fold {0}/{1}".format(kfold+1, n_splits))
        
        # reset covmat!
        XtX = np.zeros((len(lags)*array_size + e, len(lags)*array_size + e))
        Xty = np.zeros((len(lags)*array_size + e, len(picks)))   

        # accumulate covariance matrices and compute betas
        for s_id, X, y in (xylist[i] for i in train):
            if fit_intercept:
                X = np.hstack([np.ones((X.shape[0],1)), X])
           
            XtX += X.T @ X
            Xty += X.T @ y
            
        u, s, v = np.linalg.svd(XtX) 
        
        if np.ndim(alpha) > 0:
            betas = [(u @ np.diag(1/(s+a))) @ v @ Xty for a in alpha] # EDIT 2 HERE
        else:
            betas = np.linalg.inv(XtX + alpha * np.eye(len(lags))) @ Xty
            
        all_betas.append(betas)
        
        # test the model 
        r_ = []  # list accumulates r scores per trial
        for s_id, X, y in (xylist[k] for k in test):
            if fit_intercept:
                X = np.hstack([np.ones((X.shape[0],1)), X])
                
            yhat = X @ betas  # shape: len of (alphas, samples, channels)     
            r_.append(np.asarray([np.diag(np.corrcoef(yhat[a, :, :], y, rowvar=False), k=len(picks))
                             for a in range(len(alpha))]))  # shape: alphas, channels
        
        r_means = (np.asarray(r_)).mean(0)  # first dimension of array from list = trials. Shape: (alpha, channel)
        scores[kfold, :, :] = r_means
        print(r_means.mean(-1).mean(0))

   # plt.plot(scores)
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
    
    # plotting
    if plot:
        scores_plot = scores.mean(-1).T  # .mean(-1)
        _ = plt.semilogx(alpha, scores_plot)
        plt.semilogx(alpha[peaks], 
                   [scores.mean(-1)[i,peak] for i,peak in enumerate(peaks)], '*k')
                
    # reshaping and getting coefficients
    if fit_intercept:
        betas_new = np.asarray(all_betas)[:, :, 1:, :]
        
    else:
        betas_new = np.asarray(all_betas)
    
    print("Returning the best alpha, r-values, all betas, and highest per fold.")
    print("Caution: betas have the shape {0}. Need reshaping for TRF.".format(np.shape(betas_new)))
    
    return best_alpha, scores, betas_new, peaks

def fit(xylist, lags, alpha, picks=CH_NAMES, features=[], info=None, fit_intercept=False):
    """
    Computes TRF for given features between given lags.
    N.B. use crossvalidation first to get the best alpha value.
    
    Input
    -----
    xylist : list 
        lagged features for fitting (X) and normalized response (y)
        training set
    lags : lag matrix
    alpha : int
        regularization parameter
    features : list of str
        features to be regressed. 'Envelope', 'Word onset', 'Word frequency'
    info : mne raw structure
    plot : boolean
    
    Output
    ------
    TRF_sent : class TRF
    TRF_list : class TRF
    """
    n_pred = len(features)
    if 'pos' in features:
        n_pred += 9
        
    nchans = len(picks)
    
    if info is None:
        info = mne.io.read_info(os.path.join('/project/3027005.01/', 'fif_files', 'sub-A2003', '120Hz_raw.fif'))
        
     # accumulate covmatrices
    if fit_intercept:
        e = 1
    else:
        e = 0
        
    XtX_sent = np.zeros((len(lags)*n_pred+e, len(lags)*n_pred+e))
    Xty_sent = np.zeros((len(lags)*n_pred+e, nchans))
    XtX_list = np.zeros((len(lags)*n_pred+e, len(lags)*n_pred+e))
    Xty_list = np.zeros((len(lags)*n_pred+e, nchans))
    
    for s_id, X, y in xylist:
        if fit_intercept:
            X = np.hstack([np.ones((X.shape[0],1)), X])
            
        if s_id < 500:
            XtX_sent += X.T @ X
            Xty_sent += X.T @ y
        else:
            XtX_list += X.T @ X
            Xty_list += X.T @ y
        
    # compute TRF and return
    beta_list = np.linalg.inv(XtX_list + np.eye(len(lags)*n_pred+e)*alpha) @ Xty_list
    beta_sent = np.linalg.inv(XtX_sent + np.eye(len(lags)*n_pred+e)*alpha) @ Xty_sent
    
    if fit_intercept:
        beta_list = beta_list[1:,:]
        beta_sent = beta_sent[1:,:]
    
    beta_list = beta_list.reshape((len(lags), n_pred, -1))
    beta_sent = beta_sent.reshape((len(lags), n_pred, -1))
    
    args = lags, info, features
    
    TRF_list = TRF(beta_list, *args)
    TRF_sent = TRF(beta_sent, *args)
    
    return TRF_list, TRF_sent

def predict(TRF, xylist, condition, lags,features=[], picks=CH_NAMES, baseline=False):
    """
    Computes reconstruction accuracy of X on basis of TRF.
    
    Input
    -----
    TRF : class TRF
    trials : list of mne.Epoch
    sent_id : list of int
    picks : list of str
        channel names
    
    Output
    ------
    scores : numpy array
        R values
    """
    n_feat = len(features)
    if 'pos' in features:
        n_feat += 9
    
    # to do: stop looping lol
    if condition == 'sentence':
        xys = []
        for s_id, X, y in xylist:
            if s_id < 500:
                xys.append((s_id, X, y))
                
    elif condition == 'word list':
        xys = []
        for s_id, X, y in xylist:
            if s_id > 500:
                xys.append((s_id, X, y))
                
    elif condition == 'all':
        xys = xylist
    else:
        print("Condition must be 'sentence', 'word list', or 'all'. Computing r for sentence and word list combined")
        xys = xylist
    
    # betas  
    betas = np.reshape(TRF.coef_, (len(lags)*n_feat, len(picks)))
    
    # test the model 
    r_ = np.zeros((len(xylist), len(picks)))  # accumulate r scores per trial
    
    for i, (s_id, X, y) in enumerate(xylist):
        yhat = X @ betas  
        r_[i, :] = np.diag(np.corrcoef(yhat[:, :], y, rowvar=False), k=len(picks))  # (1, channels)
        
    return r_.mean(0)
