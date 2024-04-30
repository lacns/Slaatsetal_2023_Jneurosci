#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 13:58:58 2022

@author: sopsla
"""
import os
import re
import pickle
import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# local modules
from viz import plot_1samp, plot_2samp

listcolors = sns.color_palette('flare')[2:5]
sentcolors = sns.color_palette('crest')[2:5]

resultsdir = '/project/3027005.01/results/timelag_800ms/zscore-adventure/delta_nan' #'/project/3027005.01/results/srilm/sensor/delta' #

# %% load the grandaverage & stats
with open(os.path.join(resultsdir, 'GA.pkl'), 'rb') as fd:
    GA = pickle.load(fd)
    
info = GA['frequency_entropy_surprisal']['sentence'].info

# load the stats
with open(os.path.join(resultsdir, 'large-permutation.pkl'), 'rb') as f:
    twosamp = pickle.load(f)

with open(os.path.join(resultsdir, 'list-1samp-permutation.pkl'), 'rb') as f:
    onesamp_list = pickle.load(f)
    
with open(os.path.join(resultsdir, 'sentence-1samp-permutation.pkl'), 'rb') as f:
    onesamp_sent = pickle.load(f)
    
with open(os.path.join(resultsdir, 'onset-permutation.pkl'), 'rb') as f:
    onset = pickle.load(f)

# %% load the reconstruction accuracies and prepare the data
rvals = pd.read_csv(os.path.join(resultsdir,'r_data.csv'))

# average over sensors
avg = rvals.groupby(by=['subject','condition', 'model']).aggregate(r_values=pd.NamedAgg('r_values', np.mean))
avg.reset_index(inplace=True)

for mod in ['surprisal', 'frequency', 'entropy']:
    avg[mod] = [1 if re.match(r'.*{0}.*'.format(mod), i) else 0 for i in avg['model']]  
    
# remove the envelope
env = avg.loc[avg['model'] == 'envelope']
env_concat = pd.DataFrame()

for subject in set(avg['subject']):
    for condition in ['list', 'sentence']:
        for model in range(len(set(avg['model']))-1):
            env_concat = env_concat.append(env.loc[(env['subject']==subject) & (env['condition']==condition)])

env_concat.reset_index(drop=True, inplace=True)

avg = avg.loc[avg['model'] != 'envelope']
avg.reset_index(drop=True, inplace=True)
avg['r_values'] = avg['r_values'] - env_concat['r_values']
   
#avg['r_values'] = avg['difference']

# %% set up the plot
lcolors = sns.color_palette('flare', n_colors=272)
scolors = sns.color_palette('crest', n_colors=272)

plt.rcParams['axes.xmargin'] = 0

style = 'seaborn-paper'
plt.style.use(style)
fig,ax = plt.subplots(nrows=4, ncols=2, figsize=(6,9), gridspec_kw={'height_ratios':[1,1,1,1], 'width_ratios':[3,1]})

# plot the grand average
ax_divider = make_axes_locatable(ax[0,1])
cax = ax_divider.append_axes("right", size='10%', pad=0.1)
axes = np.array([ax[0,0], ax[0,1], cax])
plot_2samp(GA['frequency_entropy_surprisal'], analysis='large', out=twosamp, info=info, feature='word frequency', use_style=style,
           plot_style='clusters', topomap=True, lag_of_interest=[0.6], ax=axes, resultsdir='', legend=True)
ax[0,0].get_xaxis().set_visible(False)

ax_divider = make_axes_locatable(ax[1,1])
cax = ax_divider.append_axes("right", size='10%', pad=0.1)
axes = np.array([ax[1,0], ax[1,1], cax])
plot_1samp(GA['frequency_entropy_surprisal']['sentence']['word frequency'], onesamp_sent, plot_type='butterfly',colors=scolors,
           condition='sentence', use_style=style, topomap=True, topolags=[0.32],
           info=info, ax=axes, save=False, resultsdir='')
ax[2,0].get_xaxis().set_visible(False)

ax_divider = make_axes_locatable(ax[2,1])
cax = ax_divider.append_axes("right", size='10%', pad=0.1)
axes = np.array([ax[2,0], ax[2,1], cax])
plot_1samp(GA['frequency_entropy_surprisal']['list']['word frequency'], onesamp_list, plot_type='butterfly',colors=lcolors,
           condition='word list', use_style=style, topomap=True, topolags=[0.6],
           info=info, ax=axes, save=False, resultsdir='')
ax[1,0].get_xaxis().set_visible(False)

# word onset TRF
ax_divider = make_axes_locatable(ax[3,1])
cax = ax_divider.append_axes("right", size='10%', pad=0.1)
axes = np.array([ax[3,0], ax[3,1], cax])
plot_2samp(GA['frequency_entropy_surprisal'], analysis='onset', out=twosamp, info=info, feature='word onset', use_style=style,
           plot_style='clusters', topomap=True, ax=axes, lag_of_interest=[0.08], resultsdir='', legend=False)

fig.tight_layout()
fig.savefig(f'{resultsdir}/trfs.svg')


# %% reconstruction accuracies (demeaned)
#TODO! change error bars in the line plot
from viz import plot_all_bar
plt.rcdefaults()

style = 'seaborn-paper'
plt.style.use(style)

fig,ax = plt.subplots(nrows=1, ncols=3, figsize=(6,3), sharey=True)

# first two are the barplots, the third one is the line plot
plot_all_bar(avg, axes=ax[:2])
sns.lineplot(x='frequency', y='r_values', data=avg, hue='condition',
             ax=ax[2], palette=[listcolors[-1], sentcolors[-1]], zorder=2, ci=95,
             legend=False, err_style='bars', err_kws={'capsize':2, 'fmt':'', 'capthick' :1}, markers=True)

plt.legend(['word list', 'sentence'], bbox_to_anchor=(0.8,-0.3), frameon=False)
ax[2].set_xticks([0, 1])
ax[2].set_xticklabels(['No frequency', 'Frequency'])
ax[2].set_xlabel("Frequency")
plt.setp(ax[2].xaxis.get_majorticklabels(), rotation=90)
ax[2].xaxis.label.set_visible(False)
ax[2].get_yaxis().set_visible(False)
ax[2].set_title('Word frequency effect')
sns.despine()
fig.tight_layout()
fig.savefig(f'{resultsdir}/RAs.svg')

#%% POSTER TRFS
lcolors = sns.color_palette('flare', n_colors=272)
scolors = sns.color_palette('crest', n_colors=272)

plt.rcParams['axes.xmargin'] = 0

style = 'seaborn-poster'
plt.style.use(style)
fig,ax = plt.subplots(nrows=6, ncols=2, figsize=(8,20), gridspec_kw={'height_ratios':[1,1,1,1,1,1], 'width_ratios':[3,1]})

# plot the grand average
ax_divider = make_axes_locatable(ax[0,1])
cax = ax_divider.append_axes("right", size='10%', pad=0.1)
axes = np.array([ax[0,0], ax[0,1], cax])
plot_2samp(GA['frequency_entropy_surprisal'], analysis='large', out=twosamp, info=info, feature='word frequency', use_style=style,
           plot_style='clusters', topomap=True, lag_of_interest=[0.6], ax=axes, resultsdir='', legend=True)
ax[0,0].get_xaxis().set_visible(False)
ax[0,0].title.set_text("Word frequency")
ax[0,0].set_ylabel('Coeff. (a.u.)')

# word onset TRF
ax_divider = make_axes_locatable(ax[1,1])
cax = ax_divider.append_axes("right", size='10%', pad=0.1)
axes = np.array([ax[1,0], ax[1,1], cax])
plot_2samp(GA['frequency_entropy_surprisal'], analysis='onset', out=twosamp, info=info, feature='word onset', use_style=style,
           plot_style='clusters', topomap=True, ax=axes, lag_of_interest=[0.08], resultsdir='', legend=False)
ax[1,0].title.set_text("Word onset")
ax[1,0].set_ylabel('Coeff. (a.u.)')
ax[1,0].get_xaxis().set_visible(False)

ax_divider = make_axes_locatable(ax[2,1])
cax = ax_divider.append_axes("right", size='10%', pad=0.1)
axes = np.array([ax[2,0], ax[2,1], cax])
plot_1samp(GA['frequency_entropy_surprisal']['sentence']['word frequency'], onesamp_sent, plot_type='butterfly',colors=scolors,
           condition='sentence', use_style=style, topomap=True, topolags=[0.32],
           info=info, ax=axes, save=False, resultsdir='')
ax[2,0].title.set_text("Sentence")
ax[2,0].set_ylabel('Coeff. (a.u.)')
ax[2,0].get_xaxis().set_visible(False)

ax_divider = make_axes_locatable(ax[3,1])
cax = ax_divider.append_axes("right", size='10%', pad=0.1)
axes = np.array([ax[3,0], ax[3,1], cax])
plot_1samp(GA['frequency_entropy_surprisal']['list']['word frequency'], onesamp_list, plot_type='butterfly',colors=lcolors,
           condition='word list', use_style=style, topomap=True, topolags=[0.6],
           info=info, ax=axes, save=False, resultsdir='')
ax[3,0].title.set_text("Word list")
ax[3,0].set_ylabel('Coeff. (a.u.)')

# add the cross-correlation
tmp_info = info

idx = mne.channel_indices_by_type(tmp_info, picks=CH_NAMES)['mag']
trf_times = lag_span(-0.2, 0.8, 120) / 120
chs = [tmp_info['chs'][i] for i in idx]
locs3d = np.array([ch['loc'][:3] for ch in chs])
x, y, z = locs3d.T
colors = _rgb(x, y, z)

axes = np.array([ax[4,0], ax[4,1]])
# this only works after running crosscorrelation-sensor.py
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

axes[1].get_xaxis().set_visible(False)
axes[1].get_yaxis().set_visible(False)

axes = np.array([ax[5,0], ax[5,1]])

axes[1].get_xaxis().set_visible(False)
axes[1].get_yaxis().set_visible(False)

# Plot the correlations
sub_colors = sns.color_palette(palette=[colors[ch] for ch in common_channels])
for i,ch in enumerate(common_channels):
    axes[0].plot(lags_xcorr/info['sfreq'], xcorr.T[:,i], color=sub_colors[i])

#axes[0,1].plot(lags_xcorr/info['sfreq'], xcorr.T, cmap=my_cmap)
axes[0].set_title("Cross-correlation")
axes[0].set_xlabel('Lag (s)')
axes[0].axvline(0, c='black',linestyle='dashed')

sns.despine()
fig.tight_layout()
fig.savefig(f'{resultsdir}/trfs-poster.svg')

# %%
lcolors = sns.color_palette('flare', n_colors=272)
scolors = sns.color_palette('crest', n_colors=272)

style = 'seaborn-poster'
plt.style.use(style)

fig,ax = plt.subplots(nrows=2, ncols=2, figsize=(6,5), gridspec_kw={'height_ratios':[1,1], 'width_ratios':[3,1]})

# plot the grand average
plot_1samp(GA['frequency_entropy_surprisal']['sentence']['word frequency'], onesamp_sent, plot_type='butterfly',
           condition='sentence', use_style=style, topomap=True, topolags=[0.3],colors=scolors,title=False,
           info=info, ax=ax[0,:], save=False, resultsdir='', colorbar=False)
ax[0,0].get_xaxis().set_visible(False)
ax[0,0].title.set_text("Sentence")
ax[0,0].set_ylabel('Coeff. (a.u.)')

plot_1samp(GA['frequency_entropy_surprisal']['list']['word frequency'], onesamp_list, plot_type='butterfly',
           condition='word list', use_style=style, topomap=True, topolags=[0.6],colors=lcolors,title=False,
           info=info, ax=ax[1,:], save=False, resultsdir='', colorbar=False)
ax[1,0].title.set_text("Word list")
ax[1,0].set_ylabel('Coeff. (a.u.)')
#ax[1,0].get_xaxis().set_visible(False)

fig.tight_layout()
#fig.savefig(f'{resultsdir}/TRFs-1samp-poster.svg')

#%% POSTER TRFS 2

lcolors = sns.color_palette('flare', n_colors=272)
scolors = sns.color_palette('crest', n_colors=272)

style = 'seaborn-poster'
plt.style.use(style)

fig,ax = plt.subplots(nrows=2, ncols=1, figsize=(5,5)) #, gridspec_kw={'height_ratios':[1,1], 'width_ratios':[3,1]})

# plot the grand average
plot_2samp(GA['frequency_entropy_surprisal'], analysis='large', out=twosamp, info=info, feature='word frequency', use_style=style,
           plot_style='clusters', topomap=False, lag_of_interest=None, ax=ax[0], resultsdir='', legend=False)
ax[0].title.set_text("Word frequency")
ax[0].set_ylabel('Coeff. (a.u.)')

plot_2samp(GA['frequency_entropy_surprisal'], analysis='onset', out=twosamp, info=info, feature='word onset', use_style=style,
           plot_style='clusters', topomap=False, ax=ax[1], lag_of_interest=None, resultsdir='', legend=False)

ax[1].title.set_text("Word onset")
ax[1].set_ylabel('Coeff. (a.u.)')
#ax[1,0].get_xaxis().set_visible(False)

fig.tight_layout()
#fig.savefig(f'{resultsdir}/TRFs-2samp-poster.svg')

#%%

#%% POSTER RAs
style = 'seaborn-paper'
plt.style.use(style)

"""
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
"""
fig,ax = plt.subplots(nrows=1, ncols=3, figsize=(6,3), sharey=True)

# first two are the barplots, the third one is the line plot
plot_all_bar(avg, axes=ax[:2])
sns.lineplot(x='frequency', y='r_values', data=avg, hue='condition',
             ax=ax[2], palette=[listcolors[-1], sentcolors[-1]], zorder=2, ci=95,
             legend=False, err_style='bars', err_kws={'capsize':2, 'fmt':'', 'capthick' :1}, markers=True)

plt.legend(['word list', 'sentence'], bbox_to_anchor=(0.8,-0.3))
ax[2].set_xticks([0, 1])
ax[2].set_xticklabels(['No frequency', 'Frequency'])
ax[2].set_xlabel("Frequency")
plt.setp(ax[2].xaxis.get_majorticklabels(), rotation=90)
ax[2].xaxis.label.set_visible(False)
ax[2].get_yaxis().set_visible(False)
ax[2].set_title('Word frequency effect')
sns.despine()
fig.tight_layout()
