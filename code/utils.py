#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 14:42:54 2022

@author: sopsla
"""
import os
import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from itertools import compress
from scipy.signal import correlate, gaussian
from scipy.io.wavfile import read as wavread
from pyeeg.utils import signal_envelope, lag_matrix
from text import get_wordlevel_feats_flexibly, zscore_predictor
from matplotlib.style import available as avail_styles
import srilmtext as st

def index_transform(indices, index_map):
    """
    Maps indices between source spaces.
    
    input
    -----
    indices : list of int
    index_map : pd.DataFrame
        output of get_index_map
        
    output
    ------
    transformed : list of int
    
    """
    transformed = [float(index_map.loc[index_map['old']==index, 'fwd']) for index in indices]
    nan_vals = (~np.isnan(transformed)).T.tolist()
    transformed = list(compress(transformed, nan_vals))
    transformed = [int(idx) for idx in transformed]
    return transformed    

def get_index_map(src_orig, src_fwd):
    """
    Creates a mapping between sources used in the
    forward solution and those present in the old
    source space. Necessary for restrict_to_atlas.
    
    input
    -----
    src_orig : mne.SourceSpaces
    src_fwd : mne.SourceSpaces
    
    output
    ------
    index_map : pd.DataFrame
        mapping between vertices present in both

    """
    index_map = []

    for hemi in ['lh', 'rh']:
        idx = 0 if hemi == 'lh' else 1
        index_map_tmp = np.empty((np.shape(src_orig[hemi]['inuse'])[0],4))
        index_map_tmp[:] = np.NaN
        index_map_tmp[:,0] = idx
        index_map_tmp[src_orig[idx]['inuse'] == True,0] = list(range(idx*sum(src_orig[idx]['inuse']),(idx+1)*sum(src_orig[idx]['inuse'])))
        index_map_tmp[src_fwd[idx]['inuse'] == True,1] = list(range(idx*sum(src_fwd[idx]['inuse']),(idx+1)*sum(src_fwd[idx]['inuse'])))
        index_map_tmp[src_orig[idx]['inuse'] == True,2] = src_orig[idx]['vertno']
        index_map_tmp = pd.DataFrame(index_map_tmp, columns=['hemi', 'old','fwd', 'vertno'])
        index_map.append(index_map_tmp)
    
    del index_map_tmp

    return pd.concat(index_map,ignore_index=True)

def coherence_across_trials(trials, sent_ids):
    """
    nperseg=1024
    f, Pxy = csd(meg.T, env, nperseg=nperseg, axis=0, window='hann', fs=120, scaling='density')
    _, Pxx = welch(meg.T, fs=120, nperseg=nperseg, axis=0, scaling='density', window='hann')
    _, Pyy = welch(env, fs=120, nperseg=nperseg, axis=0, scaling='density', window='hann')
    plt.plot(f, abs(Pxy).mean(1), label='csd');
    plt.plot(f, np.mean(abs(Pxy)**2/(Pxx*Pyy[:, None]), 1), label='my coherence');
    plt.plot(f, coherence(meg.T, env[:, None], fs=120, axis=0, nperseg=nperseg)[1].mean(1), label='coherence')
    plt.legend()


    Parameters
    ----------
    trials : TYPE
        DESCRIPTION.
    sent_ids : TYPE
        DESCRIPTION.

    Raises
    ------
    NotImplementedError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    raise NotImplementedError

def find_matching_lag(signal, pattern):
    """
    Find the lag at which patter occurs in signal.
    
    signal: array-like
        Must be 1-dimensional.
    pattern: array-like
        If pattern is a list of arrays (each possibly different lengths),
        will repeat the function for each element of the extra axis.
    """
    signal = np.asarray(signal)
    #pattern = npasarray(pattern, dtype=object)

    assert signal.ndim == 1, "First argument must be one dimensional."

    if isinstance(pattern, (list, tuple)) or (isinstance(pattern, np.ndarray) and pattern.ndim == 2):
        lags = []
        for p in pattern:
            lags.append(find_matching_lag(signal, p))
        return lags
    else:
        out = correlate(signal, pattern, mode='valid')
        lag = np.argmax(out)
        return lag
    
def get_audio_envelope(sent_id, sfreq=120, N=None, audio_path='/project/3027005.01/audio_files/'):
    """
    Directly compute the envelope, resample and truncate/pad to the right length.
    
    Parameters
    ----------
    sent_id: int
        Sentence number id.
    sfreq : int or float
        Target sampling frequency
    N : int
        Final length wanted
    audio_path : str
        Location of all .wav files (each file should start with)
        'EQ_Ramp_Int2_Int1LPF'
        
    Returns
    -------
    env : ndarray
        Broad band envelope for this sentence.
    """
    fname = 'EQ_Ramp_Int2_Int1LPF%.03d.wav'%sent_id
    #print("Loading envelope for " + fname)
    fs, y = wavread(os.path.join(audio_path, fname))
    env = signal_envelope(y[:, 0], fs, resample=sfreq, verbose=False)
    if N is not None and len(env)!=N:
        if len(env) < N:
            env = np.pad(env, (N-len(env), 0))
        else:
            env = env[:N]
    return env

def temporal_smoother(x, win=None, fs=120, std_time=0.015, axis=0):
    """
    Smooth (by convolution) a time series (possibly n-dimensional).
    
    Default smoothing window is gaussian (a.k.a gaussian filtering), which has the nice property of being an ideal
    temporal filter, it is equivalent to low-pass filtering with a gaussian window in fourier domain. The window
    is symmetrical, hence this is non causal. To control for the cut-off edge, or standard deviation in the frequency
    domain, use the following formula to go from stndard deviation in time domain to fourier domain:
    
    .. math::
           \sigma_f = \frac{1}{2\pi\sigma_t}
           
    Hence for a standard deviation of 15ms (0.015) in time domain we will get 42Hz standard deviation in the frequency
    domain. Hence roughly cutting off frequencies above 20Hz (as the gaussian spread both negative and positive 
    frequencies).
    
    Parameters
    ----------
    x: ndarray (n_time, any)
        Time series, axis of time is assumed to be the first one (axis=0).
        Change `axis` argument if not.
    win : ndarray (n_win,)
        Smoothing kernel, if None (default) will apply a gaussian window
        with std given by the `std_time` parameter.
    fs : int
        Sampling frequency
    std_tim : float
        temporal standard deviation of gaussian window (default=0.015 seconds)
    axis : int (default 0)
    
    Returns
    -------
    x : ndarray
        Same dimensionality as input x.
    """
    assert np.asarray(x).ndim <= 2, "Toom may dimensions (maximum 2)"
    if win is None:
        win = gaussian(fs, std=std_time * fs)
    x = np.apply_along_axis(np.convolve, axis, x, win, 'same')
    return x

def select_plot_style(style_name):
    """
    Select a plotting style, can use some shortcuts, hardcoded here:
    - "poster" -> "seaborn-poster"
    - "paper" -> "seaborn-paper"
    If style not found, throw an error.

    Parameters
    ----------
    style_name : str
        A style among plt.style.available

    Returns
    -------
    style : str
        Style mapped to matplotlib ones.
    """
    style_name = style_name.lower()
    if style_name == 'poster':
        style = "seaborn-poster"
    elif style_name == 'paper':
        style = 'seaborn-paper'
    else:
        style = style_name
    if style not in avail_styles and style != "default":
        raise NameError("Style supplied does not exist.")
    return style

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Test
    y = np.random.randn(1000,)
    original_lags = [200, 350, 600]
    extracts = []
    for l in original_lags:
        extracts.append(y[l:l+np.random.randint(100, 200)])

    lags = find_matching_lag(y, extracts)
    
    plt.plot(y)
    for k, l in enumerate(lags):
        plt.plot(np.arange(l, l+len(extracts[k])), extracts[k], label="extract %d"%(k+1))
    plt.vlines(original_lags, *plt.gca().get_ylim(), color='k', linestyle='--', label="original lags")
    plt.legend()
    plt.show()
        
def Xy_flexibly(meg_trials, stim_ids, fs, lags, mu, sigma, picks, 
       features, tim=0.015, reject_nan=True):
    """
    Computes X (lag matrix) and normalized y for all trials.
    
    Input
    -----
    meg_trials : list of mne.Epoch
    stim_ids : list of int
    lags : np.array
    fs : int
        Sampling frequency
    mu : int
    sigma : int
    picks : list of str
        Channel names to be included
    tim : float
        temporal standard deviation of gaussian window (default=0.015 seconds)
        
    Output
    ------
    Generator object with (s_id, X, y) tuples
        s_id : stimulus id
        X : time-lagged features
        y : normalized trial data
    """
    
    print("Computing lag matrix and normalizing MEG trials...")
    
    for tr, s_id in tqdm(zip(meg_trials, stim_ids), total=len(meg_trials), leave=False):
        try:
            N = tr._data.shape[-1]
        
            x = zscore_predictor(get_audio_envelope(s_id, N=N), 'envelope')  # get envelope
            if reject_nan:
                if len(features) == 1 and 'envelope' in features:
                    X = lag_matrix(x, lags)
                
                else:
                    x_w = get_wordlevel_feats_flexibly(sentence_id=s_id, N=N, fs=fs,
                                                       features=features, do_zscore=True)
            
                    x_w = temporal_smoother(x_w,fs=fs, std_time=tim)     
                    X = lag_matrix(np.hstack([x[:, None], x_w]), lags)  
                   
                y = (np.squeeze(tr._data).T-mu)/sigma  # Zscore trial data
                
                # Remove the NaN-rows
                nan_rows = np.isnan(X.mean(1))
                X = X[~nan_rows]
                y = y[~nan_rows] # also for the meg
                
            else:
                if len(features) == 1 and 'envelope' in features:
                    X = lag_matrix(x, lags, filling=0., drop_missing=False)
                
                else:
                    x_w = get_wordlevel_feats_flexibly(sentence_id=s_id, N=N, fs=fs,
                                                       features=features, do_zscore=True)
            
                    x_w = temporal_smoother(x_w,fs=fs, std_time=tim)     
                    X = lag_matrix(np.hstack([x[:, None], x_w]), lags, filling=0., drop_missing=False)  
                   
                y = (np.squeeze(tr._data).T-mu)/sigma  # Zscore trial data
            yield s_id, X, y
            
        # handling exceptions just by passing them for now...
        except (ValueError, IndexError):
            print("Something is wrong with trial {0}. Continuing to the next trial...".format(s_id))
            pass

def Xy_srilm(meg_trials, stim_ids, fs, lags, mu, sigma, picks, 
       features, tim=0.015, reject_nan=True):
    """
    Computes X (lag matrix) and normalized y for all trials.
    
    Input
    -----
    meg_trials : list of mne.Epoch
    stim_ids : list of int
    lags : np.array
    fs : int
        Sampling frequency
    mu : int
    sigma : int
    picks : list of str
        Channel names to be included
    tim : float
        temporal standard deviation of gaussian window (default=0.015 seconds)
        
    Output
    ------
    Generator object with (s_id, X, y) tuples
        s_id : stimulus id
        X : time-lagged features
        y : normalized trial data
    """
    
    print("Computing lag matrix and normalizing MEG trials...")
    
    for tr, s_id in tqdm(zip(meg_trials, stim_ids), total=len(meg_trials), leave=False):
        try:
            N = tr._data.shape[-1]
        
            x = zscore_predictor(get_audio_envelope(s_id, N=N), 'envelope')  # get envelope
            if reject_nan:
                if len(features) == 1 and 'envelope' in features:
                    X = lag_matrix(x, lags)
                
                else:
                    x_w = st.get_wordlevel_feats_srilm(sentence_id=s_id, N=N, fs=fs,
                                                       features=features, do_zscore=True)
            
                    x_w = temporal_smoother(x_w,fs=fs, std_time=tim)     
                    X = lag_matrix(np.hstack([x[:, None], x_w]), lags)  
                   
                y = (np.squeeze(tr._data).T-mu)/sigma  # Zscore trial data
                
                # Remove the NaN-rows
                nan_rows = np.isnan(X.mean(1))
                X = X[~nan_rows]
                y = y[~nan_rows] # also for the meg
                
            else:
                if len(features) == 1 and 'envelope' in features:
                    X = lag_matrix(x, lags, filling=0., drop_missing=False)
                
                else:
                    x_w = st.get_wordlevel_feats_srilm(sentence_id=s_id, N=N, fs=fs,
                                                       features=features, do_zscore=True)
            
                    x_w = temporal_smoother(x_w,fs=fs, std_time=tim)     
                    X = lag_matrix(np.hstack([x[:, None], x_w]), lags, filling=0., drop_missing=False)  
                   
                y = (np.squeeze(tr._data).T-mu)/sigma  # Zscore trial data
            yield s_id, X, y
            
        # handling exceptions just by passing them for now...
        except (ValueError, IndexError):
            print("Something is wrong with trial {0}. Continuing to the next trial...".format(s_id))
            pass                
        
def split_X(X, lags, features=[]):
    """
    Works for envelope, onset, frequency, entropy, surprisal
    
    Input
    ------
    X : np.array
        lag matrix for all features
    features : list of str
        feature to be included in the model
        options: 'envelope', word onset', 'entropy', 'surprisal', 'word frequency'
    
    Output
    ------
    X : np.array
        lag matrix of specified features
    """   
    ft = {
        'envelope': 0,
        'word onset': 1,
        'onset':1,
        'entropy':2,
        'surprisal':3,
        'word frequency':4,
        'frequency':4
    }
    
    if len(features) == 5: # return all
        outX = X
    
    else:
        resX = X.reshape(np.shape(X)[0],len(lags),5)
    
        if len(features) > 1:
            features = [feat.lower() for feat in features]
            idx = np.asarray([ft[feat] for feat in features])
            outX = resX[:,:,idx]

        else:
            outX = resX[:,:,0] # for one feature
            
        outX = outX.reshape(np.shape(X)[0],len(features)*len(lags))  # melt it back into 2d-shape

    return outX
  
    
def train_test(sent_id, ratio=0.2):
    """
    Creates a random split of indices for
    training and testing. Ensures an equal
    split for word list and sentence items.
    
    If ratio leads to float for split, the
    number is rounded.
    
    Input
    -----
    sent_id : list
    ratio : float
        ratio to split by
        
    Returns
    -------
    train : list of int
    test : list of int
        indices to select for training and testing
    """
    test = []
    train = []
    sentences = [index for index, value in enumerate(sent_id) if value < 500]
    wordlists = [index for index, value in enumerate(sent_id) if value > 500]
    
    p_s = np.random.permutation(list(range(len(sentences))))
    p_w = np.random.permutation(list(range(len(wordlists))))
    
    test.extend(p_s[0:round(ratio*len(p_s))])
    test.extend(p_w[0:round(ratio*len(p_w))])
    
    train.extend(p_s[round(ratio*len(p_s)):-1])
    train.extend(p_w[round(ratio*len(p_w)):-1])
    
    return train, test

def get_MEGfiles_path(fmin=1, fmax=12, bandname='broadband'):
    """
    Returns the root path to directory containg preprocessed data for the chosen frequency band.
    If data do not exist, raises an error.
    """
    if bandname == 'delta':
        fmin = 1
        fmax= 4.5
    elif bandname == 'theta':
        fmin = 4
        fmax = 8
    if (fmin == 1) and (fmax == 12) and (bandname=='broadband'):
        from meg import PATH_TO_MEG
        return PATH_TO_MEG
    if fmin!=np.round(fmin):
        if fmax == np.round(fmax):
            folder_name = f"bandpass_{fmin:1.1f}-{int(fmax):1d}"
        else:
            folder_name = f"bandpass_{fmin:1.1f}-{fmax:1.1f}"
    else:
        if fmax == np.round(fmax):
            folder_name = f"bandpass_{int(fmin):1d}-{int(fmax):1d}"
        else:
            folder_name = f"bandpass_{int(fmin):1d}-{fmax:1.1f}"
    if not os.path.exists(os.path.join('/project/3027005.01/TRF_meg/preprocessed', folder_name)):
        raise FileNotFoundError("This processed data folder (%s) seems to not exist..."%folder_name)
    return os.path.join('/project/3027005.01/TRF_meg/preprocessed', folder_name)