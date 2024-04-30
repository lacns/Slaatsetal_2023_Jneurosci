#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 14:52:15 2022

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

resultsdir = '/project/3027005.01/extra_data/delta_reg'

# %% load the grandaverage & stats
with open(os.path.join(resultsdir, 'GA.pkl'), 'rb') as fd:
    GA = pickle.load(fd)
    
info = GA['frequency']['sentence'].info

# load the stats
with open(os.path.join(resultsdir, 'small-permutation.pkl'), 'rb') as f:
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

# %% set up the plot
style = 'seaborn-paper'
plt.style.use(style)
fig,ax = plt.subplots(nrows=4, ncols=2, figsize=(6,9), gridspec_kw={'height_ratios':[1,1,1,1], 'width_ratios':[3,1]})

# plot the grand average
ax_divider = make_axes_locatable(ax[0,1])
cax = ax_divider.append_axes("right", size='10%', pad=0.1)
axes = np.array([ax[0,0], ax[0,1], cax])
plot_2samp(GA['frequency'], analysis='large', out=twosamp, info=info, feature='word frequency', use_style=style,
           plot_style='clusters', topomap=True, lag_of_interest=[0.17], ax=axes, resultsdir='', flip=False)
ax[0,0].get_xaxis().set_visible(False)

ax_divider = make_axes_locatable(ax[1,1])
cax = ax_divider.append_axes("right", size='10%', pad=0.1)
axes = np.array([ax[1,0], ax[1,1], cax])
plot_1samp(GA['frequency']['sentence']['word frequency'], onesamp_sent, plot_type='butterfly',
           condition='sentence', use_style=style, topomap=True, topolags=[0.17],
           info=info, ax=axes, save=False, resultsdir='')
ax[2,0].get_xaxis().set_visible(False)

ax_divider = make_axes_locatable(ax[2,1])
cax = ax_divider.append_axes("right", size='10%', pad=0.1)
axes = np.array([ax[2,0], ax[2,1], cax])
plot_1samp(GA['frequency']['list']['word frequency'], onesamp_list, plot_type='butterfly',
           condition='word list', use_style=style, topomap=True, topolags=[0.37],
           info=info, ax=axes, save=False, resultsdir='')
ax[1,0].get_xaxis().set_visible(False)

# word onset TRF
ax_divider = make_axes_locatable(ax[3,1])
cax = ax_divider.append_axes("right", size='10%', pad=0.1)
axes = np.array([ax[3,0], ax[3,1], cax])
plot_2samp(GA['frequency'], analysis='onset', out=twosamp, info=info, feature='word onset', use_style=style,
           plot_style='clusters', topomap=True, ax=axes, lag_of_interest=[0.1], resultsdir='', legend=False, flip=False)

fig.tight_layout()

fig.savefig(f'{resultsdir}/trfs.svg')

# %% reconstruction accuracies (demeaned)
#TODO! change error bars in the line plot
from viz import plot_all_bar

fig,ax = plt.subplots(nrows=1, ncols=3, figsize=(6,3), sharey=True)

# first two are the barplots, the third one is the line plot
plot_all_bar(avg, axes=ax[:2], models=['onset', 'frequency'],
             abbrev=['Onset', 'Freq.'])
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
fig.savefig(f'{resultsdir}/RAs.svg')
