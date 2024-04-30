#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 14:56:57 2022

@author: sopsla
"""
import os
import pickle
import numpy as np
from pyeeg.utils import lag_span
import mne

# local modules
from meg import CH_NAMES
from viz import _rgb

# correlation
from scipy.signal import correlate, correlation_lags, find_peaks
from itertools import chain

# plotting
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mne.viz.evoked import _handle_spatial_colors
import matplotlib.pyplot as plt
import seaborn as sns

lags = lag_span(-0.2, 0.8, 120)

bandpass = 'delta'
resultsdir = '/project/3027005.01/results/timelag_800ms/zscore-adventure/delta_nan' #f'/project/3027005.01/results/publication/sensor/{bandpass}'

# %% load the TRFs and the permutations
with open(os.path.join(resultsdir, 'GA.pkl'), 'rb') as f:
    GA = pickle.load(f)
    
info = GA['frequency_entropy_surprisal']['sentence'].info

# load the cluster-based permutation test
with open(os.path.join(resultsdir, 'list-1samp-permutation.pkl'), 'rb') as f:
    listperm = pickle.load(f)

with open(os.path.join(resultsdir, 'sentence-1samp-permutation.pkl'), 'rb') as f:
    sentperm = pickle.load(f)

_, clusters_list, pvals_list, _ = listperm
_, clusters_sent, pvals_sent, _ = sentperm

# %% get the shared sensors
ch_idx_list = [clu[1] for clu in clusters_list]
ch_idx_sent = [clu[1] for clu in clusters_sent]

common_channels = list(set(chain(*ch_idx_list)).intersection(set(chain(*ch_idx_sent))))
#print(np.asarray(CH_NAMES)[np.asarray(common_channels)])

# copy the GA and take away the edge artifact
sentence_trf = GA['frequency_entropy_surprisal']['sentence']['word frequency'].copy()
sentence_trf = sentence_trf.crop(tmin=-0.17, tmax=0.77)

list_trf = GA['frequency_entropy_surprisal']['list']['word frequency'].copy()
list_trf = list_trf.crop(tmin=-0.17, tmax=0.77)

sent_channels = sentence_trf.data[common_channels]
list_channels = list_trf.data[common_channels]

# %% perform cross correlation
lags_xcorr = correlation_lags(len(sent_channels[0, :]), len(list_channels[0, :]), 'same')

xcorr = []
for ch in range(len(common_channels)):
    c = correlate(list_channels[ch, :], sent_channels[ch, :], mode='same')
    c /= np.max(c) # normalize the correlation
    xcorr.append(c)

xcorr = np.asarray(xcorr)

# %% Get the peaks & roll each signal by its correlation peak
#peaks = [find_peaks(xcorr[i,:])[0] for i in list(range(len(common_channels)))]
#peaks = [i[0] if len(i) != 0 else lags_xcorr[-1] for i in peaks]
peaks = [lags_xcorr[find_peaks(xcorr[i,:])[0][0]] for i in list(range(len(common_channels)))]

rolled_sent = np.zeros(np.shape(sent_channels))

for ch in list(range(np.shape(sent_channels)[0])):
    rolled_sent[ch,:] = np.roll(sent_channels[ch,:], peaks[ch], axis=0)

pearson = np.diag(np.corrcoef(rolled_sent, list_channels), k=len(common_channels))

# %% pick random channels + random lags for comparison (=> many times = distribution for comp.)
iterations = 10000
rdm_results = []

for i in range(iterations):
    rdm_idx = np.random.choice(range(272), replace=False, size=(len(common_channels),))
    rdm_channels = sentence_trf.data[rdm_idx]
    contrast_channels = list_trf.data[rdm_idx]
    
    rdm_peaks_mean = np.random.randint(120)
    rdm_peaks = np.random.normal(loc=rdm_peaks_mean, scale=np.std(peaks), size=len(common_channels))
    #rdm_peaks = peaks.copy()
    rdm_rolled = np.zeros(np.shape(sent_channels))
    for ch in range(len(common_channels)):
        rdm_rolled[ch,:] = np.roll(rdm_channels[ch,:], np.int(np.round(rdm_peaks[ch])), axis=0)

    rdm_pearson = np.diag(np.corrcoef(rdm_rolled, contrast_channels), k=len(common_channels))
    rdm_results.append(rdm_pearson)

# %% ## PLOT THE RESULTS ###
# channel colors
from matplotlib.lines import Line2D
plt.rcParams['axes.xmargin'] = 0

tmp_info = sentence_trf.info

style = 'seaborn-paper'
plt.style.use(style)

idx = mne.channel_indices_by_type(tmp_info, picks=CH_NAMES)['mag']
trf_times = lag_span(-0.2, 0.8, 120) / 120
chs = [tmp_info['chs'][i] for i in idx]

#chs = tmp_info['chs']
locs3d = np.array([ch['loc'][:3] for ch in chs])
x, y, z = locs3d.T
colors = _rgb(x, y, z)

fig,axes = plt.subplots(nrows=2, ncols=2, figsize=(6,4))

# plot the original sensors
for i,ch in enumerate(common_channels):
    axes[0,0].plot(sentence_trf.times, sentence_trf.data[ch,:].T,
                  color=colors[ch], linewidth=1, alpha=1, linestyle='solid')
    axes[0,0].plot(list_trf.times, list_trf.data[ch,:].T,
              color=colors[ch], linewidth=1, alpha=0.8, linestyle='dashed')

# set legend 
custom_lines = [Line2D([0], [0], lw=1, color='black', linestyle='solid'), Line2D([0], [0], lw=1, color='black', linestyle='dashed')]
axes[0,0].legend(custom_lines, ['sentence', 'word list'],loc='upper left', frameon=False)
axes[0,0].xaxis.set_visible(False)

for i,ch in enumerate(common_channels):
 #   print(idx)
    axes[1,0].plot(sentence_trf.times, np.roll(sent_channels[i,:].T, 40, axis=0), color=colors[ch], linewidth=1)
    axes[1,0].plot(sentence_trf.times, list_channels[i,:].T, color=colors[ch], linewidth=1, alpha=0.8, linestyle='dashed')
    
_handle_spatial_colors(colors=colors[np.asarray(common_channels)], info=tmp_info, 
                       idx=np.asarray(common_channels), ch_type='mag', psd=False, ax=axes[1,0], sphere=None)

axes[1,0].set_xlabel('Time (s)')

# Plot the correlations
sub_colors = sns.color_palette(palette=[colors[ch] for ch in common_channels])
for i,ch in enumerate(common_channels):
    axes[0,1].plot(lags_xcorr/info['sfreq'], xcorr.T[:,i], color=sub_colors[i])

axes[0,1].set_xlabel('Lag of sent relative to list (s)')
axes[0,1].axvline(0, c='black',linestyle='dashed')
#axes[0,1].set_title('Cross-correlation')

# Plot the Pearson's values against the random ones
# make two axes out of the last one
ax_divider = make_axes_locatable(axes[1,1])
cax = ax_divider.append_axes("right", size='100%', pad='20%', sharey=axes[1,1])
std = [np.std(r) for r in rdm_results]
mu = [np.mean(r) for r in rdm_results]

for name,value,actual_result,ax,title in zip(['mean', 'standard deviation'],[mu, std], [np.mean(pearson), np.std(pearson)], 
                                             [axes[1,1], cax],['mean', 'std of Pearsons R']):
    sns.kdeplot(x=value, ax=ax, fill=True)
    ax.axvline(x=actual_result, color='red')
    ax.set_xlabel(name)
    ax.set_ylabel('')
    #ax.set_title(title)
    if name == 'standard deviation':
      ax.yaxis.set_visible(False)  
    elif name == 'mean':
        ax.legend(['Random', 'Observed'], frameon=False)
    
    ax.margins(x=0.1)

sns.despine()
plt.tight_layout()
#fig.savefig(os.path.join(resultsdir, 'correlation-analysis-minimalist.svg'))

#%% poster plot

tmp_info = sentence_trf.info

style = 'seaborn-poster'
plt.style.use(style)

idx = mne.channel_indices_by_type(tmp_info, picks=CH_NAMES)['mag']
trf_times = lag_span(-0.2, 0.8, 120) / 120
chs = [tmp_info['chs'][i] for i in idx]

#chs = tmp_info['chs']
locs3d = np.array([ch['loc'][:3] for ch in chs])
x, y, z = locs3d.T
colors = _rgb(x, y, z)

#plt.style.use('seaborn-paper')
fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(8,3))

# plot the original sensors
for i,ch in enumerate(common_channels):
    axes[0].plot(sentence_trf.times, sentence_trf.data[ch,:].T,
                  color=colors[ch], linewidth=1, alpha=1, linestyle='solid')
    axes[0].plot(list_trf.times, list_trf.data[ch,:].T,
              color=colors[ch], linewidth=1, alpha=0.8, linestyle='dashed')

_handle_spatial_colors(colors=colors[np.asarray(common_channels)], info=tmp_info, 
                       idx=np.asarray(common_channels), ch_type='mag', psd=False, ax=axes[0], sphere=None)
#axes[0].legend(['sentence', 'word list'],loc='upper right')
axes[0].set_title('Shared sensors')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Coeff (a.u.)')

# Plot the correlations
sub_colors = sns.color_palette(palette=[colors[ch] for ch in common_channels])
for i,ch in enumerate(common_channels):
    axes[1].plot(lags_xcorr/info['sfreq'], xcorr.T[:,i], color=sub_colors[i])

#axes[0,1].plot(lags_xcorr/info['sfreq'], xcorr.T, cmap=my_cmap)
axes[1].set_title("Cross-correlation")
axes[1].set_xlabel('Lag (s)')
axes[1].axvline(0, c='black',linestyle='dashed')
sns.despine()
plt.tight_layout()

fig.savefig(os.path.join(resultsdir, 'correlation-analysis-POSTER.svg'))