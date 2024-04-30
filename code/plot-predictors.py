#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:26:32 2022

@author: sopsla
"""
from text import extract_word_onsets, get_word_frequency, get_surprisal, get_entropy
from utils import get_audio_envelope, temporal_smoother, zscore_predictor
from pyeeg.utils import lag_span
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# %% 
sid = 1
fs = 120

env = get_audio_envelope(1, fs)

x = np.zeros((len(env),5))
words, onsets = extract_word_onsets(sid)
samples = np.round(np.asarray(onsets)*fs).astype(int)
wf = [get_word_frequency(w) for w in words]
surp = get_surprisal(sid)
ent = get_entropy(sid)

x[samples,1] = 1
x[samples,2] = [zscore_predictor(f, 'word frequency') for f in wf]
x[samples,3] = [zscore_predictor(f, 'surprisal') for f in surp]
x[samples,4] = [zscore_predictor(f, 'entropy') for f in ent]
x = temporal_smoother(x)
x[:,0] = zscore_predictor(env, 'envelope')

# %%
lags = lag_span(-0.2, 0.8, fs)

style = 'seaborn-paper'
plt.style.use(style)

fig,ax = plt.subplots(nrows=5, ncols=1, figsize=(6,4), sharey=True, sharex=True)

for idx in range(0,5):
    ax[idx].plot(x[:,idx])

sns.despine()
plt.tight_layout()

