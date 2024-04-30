#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 14:49:15 2022

@author: sopsla
"""
import os
import re
import pandas as pd
import numpy as np
import textgrid as tg
from scipy.stats import zscore
from loadmat import loadmat
import pickle

PROJECT_PATH = '/project/3027005.01' # only used for audio files or preprocessed MEG
ANNOTATIONS_PATH = f'{os.getcwd()}/annotations'
SENTENCES = {}

with open(f'{PROJECT_PATH}/predictor_musigma.pkl', 'rb') as f:
    mu_sigma = pickle.load(f)

with open(os.path.join(ANNOTATIONS_PATH, '20200609-MOUS-stimuli.txt'), 'r') as f:
    raw = f.readlines()
    for line in raw:
        idx, sents = line.split()[0], ' '.join(line.split()[1:])
        SENTENCES[int(idx)] = sents

STIMULI = loadmat(os.path.join(PROJECT_PATH, 'mous_stimuli.mat'))
stim_map = pd.read_csv(os.path.join(PROJECT_PATH, 'stim_map.csv'))

def extract_word_onsets(sent_id,):
    """
    Returns word and their corresponding onset for a given sentence.

    Parameters
    ----------
    sent_id : int
        Sentence id, an integer as listed in /project/3027005.01/stim

    Returns
    -------
    words : list
        List of words in the sentence
    tonset : list
        Corresponding onset times
    """
    # Extract onset from TextGrid data
    
    try:
        tgfile = tg.TextGrid.fromFile(os.path.join(ANNOTATIONS_PATH, 'TextGrids/stim%d.TextGrid'%sent_id))
        onsets = [interval.minTime for interval in tgfile.getFirst('words')[:] if interval.mark != '']
        offsets = [interval.maxTime for interval in tgfile.getFirst('words')[:] if interval.mark != '']
    except: # wrong format for 400+ files
        with open(f'{os.getcwd()}/annotations/TextGrids/stim%d.TextGrid'%sent_id) as f:
            raw = f.read()
            lines = raw.splitlines()
            idx = np.where(np.asarray(lines) == '"word"')[0]
            #words = [lines[k] for k in idx]
            onsets = [float(lines[k-2]) for k in idx]
            offsets = [float(lines[k-1]) for k in idx]
    # Extract word list from stim data
    sentence = SENTENCES[sent_id]
    words = sentence.split()
    # Check they have similar length
    if len(words) != len(onsets):
        # Check that last word is not dummy empty space
        if abs(len(words)-len(onsets))==1 and offsets[-1]-onsets[-1] < 1e-3:
            onsets.pop()
        else:
            #print(tgfile, f)
            print(sent_id)
            raise ValueError("Mismatch in length between annotations and transcript data!")
    
    return words, onsets

def get_word_frequency(word, wf_file='SUBTLEX-NL.cd-above2.txt', fallback=0.301):
    """
    Get word frequencies as given by the SUBTLEX-NL corpus.

    Parameters
    ----------
    word : str
        Word (lowered characters)
    fallback : float
        Value to fall back to if word is not in corpus. Default to the minimum 
        value encountered in the corpus.

    Returns
    -------
    float
        -log(Word frequency) of a word

    """
    if "df_wf" not in globals():
        global df_wf
        df_wf = pd.read_csv(os.path.join(PROJECT_PATH, 'wordfreq', wf_file), delim_whitespace=True)
    if word.lower() not in df_wf.Word.to_list():
        return fallback
    else:
        return df_wf.loc[df_wf.Word==word.lower(), 'Lg10WF'].to_numpy()[0]
    
if __name__ == '__main__':
    sent_id = 18
    words, onset = extract_word_onsets(sent_id)
    wf = [get_word_frequency(w.lower()) for w in words]
    
    print(pd.DataFrame({'Word':words, 'onset':onset, 'WordFreq':wf}))
    
    
def get_surprisal(sentence_id):
    """
    Get trigram perplexity for all words in a stimulus
    as in MOUS-stimulus annotations.
    
    Parameters
    ----------
    sentence_id : int
        stimulus id
    
    Returns
    -------
    list of int
        perplexity values for all words in a stimulus
    """
    
    idx = int(stim_map.loc[stim_map['sent_id']==sentence_id, 'index'])
    return [np.log10(STIMULI['stimuli'][idx].words[i].perplexity) 
                for i in list(range(len(STIMULI['stimuli'][idx].words)))]
    
    
def get_perplexity(sentence_id):
    """
    Get trigram perplexity for all words in a stimulus
    as in MOUS-stimulus annotations.
    
    Parameters
    ----------
    sentence_id : int
        stimulus id
    
    Returns
    -------
    list of int
        perplexity values for all words in a stimulus
    """
    # get the right stimulus from the mat file 
    # by taking the idx from stim_map
    idx = int(stim_map.loc[stim_map['sent_id']==sentence_id, 'index'])
    return [STIMULI['stimuli'][idx].words[i].perplexity 
                for i in list(range(len(STIMULI['stimuli'][idx].words)))]


def get_entropy(sentence_id):
    """
    Get trigram entropy for all words in a stimulus
    as in MOUS-stimulus annotations.
    
    Parameters
    ----------
    sentence_id : int
        stimulus id
    
    Returns
    -------
    list of int
        perplexity values for all words in a stimulus
    """
    # get the right stimulus from the mat file 
    # by taking the idx from stim_map
    idx = int(stim_map.loc[stim_map['sent_id']==sentence_id, 'index'])
    entropy = [STIMULI['stimuli'][idx].words[i].entropy 
                for i in list(range(len(STIMULI['stimuli'][idx].words)))]
    
    # replace non-existing with overall mean
    entropy = [5.777279852384063 if np.isnan(value) else value for value in entropy]
    return entropy


pattern = re.compile('.*(?=\()')

def woordsoort(wrd):
    """
    MOUS-stimulus annotations are long.
    Pattern cuts off the first part of
    annotation.
    
    Parameters
    ----------
    wrd : str
    
    Returns
    -------
    str 
        Part of speech. One from:
        'ADJ', 'BW', 'LID', 'N', 'SPEC', 
        'TW', 'VG', 'VNW', 'VZ', 'WW'
    
    """
    match = pattern.search(wrd)
    return match.group()


def get_pos(sent_id, N, samples=[], fs=120):  
    """
    Create a predictor for each part of speech
    
    Parameters
    ----------
    sent_id : int
        ID of stimulus
    samples : list of int
        output of get_word_onsets
    fs : int
        sampling frequency
        Default 120
    
    Returns
    -------
    x : ndarray (N, len(pos))
        Array of part-of-speech predictors
        len(pos) = 10
    
    """
    if len(samples) == 0:
        _, onset = extract_word_onsets(sent_id)
        samples = np.round(np.asarray(onset)*fs).astype(int)
        
    parts_of_speech = {'ADJ', 'BW', 'LID', 'N', 'SPEC', 'TW', 'VG', 'VNW', 'VZ', 'WW'}
    
    # initiate dataframe
    x = np.zeros((N, len(parts_of_speech)))
    
    # get the right stimulus from the mat file 
    # by taking the idx from stim_map
    idx = int(stim_map.loc[stim_map['sent_id']==sent_id, 'index'])
    sent_pos = [woordsoort(STIMULI['stimuli'][idx].words[i].POS) 
                for i in list(range(len(STIMULI['stimuli'][idx].words)))]
    
    # loop over parts of speech & change 0 to 1 if onset
    # corresponds to given pos
    for sample, pos in zip(samples, sent_pos):
        for i,part in enumerate(parts_of_speech):
            if part == pos:
                x[sample , i] = 1    
    
    return x

def zscore_predictor(value, predictor):
    mean = mu_sigma[predictor][0]
    std = mu_sigma[predictor][1]
    
    return (value - mean) / std

def get_wordlevel_feats_flexibly(sentence_id, N, fs=120, 
                                 features=['word onset', 'entropy', 'surprisal', 'word frequency'],
                                 do_zscore=True):
    """
    Extract directly the array structure needed for TRF computation.
    N.B. CANNOT calculate both perplexity and entropy (or/or)
    
    Parameters
    ----------
    sentence_id : int
        ID of stimulus
    N : int
        Number of samples for this trial.
    fs : int
        Sampling frequency (needed to map word onset
        to actual sample).
    features : list of str
        Features to fit. Options: onset, entropy, surprisal, frequency,
        perplexity, part-of-speech
    do_zscore : bool
        Whether to normalize word frequencies (default: True).
        
    Returns
    -------
    x : ndarray (N, n_feat)
        Array of features. 
    """
    word, onset = extract_word_onsets(sentence_id)
    feats = features.copy()
    
    if 'envelope' in feats:
        feats.remove('envelope')
    
    if 'pos' in features:
        x = np.zeros((N, len(feats)+9))
    else:
        x = np.zeros((N, len(feats)))
    samples = np.round(np.asarray(onset)*fs).astype(int)
    
    for idx, feature in enumerate(feats):
        if feature == 'word onset':
            x[samples, idx] = 1
        elif feature == 'word frequency':
            wf = [get_word_frequency(w) for w in word]
            x[samples, idx]  = [zscore_predictor(f, feature) for f in wf] if do_zscore else wf
        elif feature =='entropy':
            en = get_entropy(sentence_id)
            x[samples, idx] = [zscore_predictor(f, feature) for f in en] if do_zscore else en
        elif feature == 'surprisal':
            ss = get_surprisal(sentence_id) 
            x[samples, idx] = [zscore_predictor(f, feature) for f in ss] if do_zscore else ss
        elif feature == 'perplexity':
            pp = get_perplexity(sentence_id)
            x[samples, idx] = do_zscore(pp) if do_zscore else pp
        
    if 'pos' in feats:
        x[:, -10:] = get_pos(sentence_id, N, samples)
        features.remove('pos')
        features += ['ADJ', 'BW', 'LID', 'N', 'SPEC', 'TW', 'VG', 'VNW', 'VZ', 'WW']

    return x