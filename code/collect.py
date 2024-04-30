#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 15:18:51 2021

@author: sopsla
"""
import os
import pickle
import numpy as np
import pandas as pd
from pyeeg.utils import lag_span
from meg import TRF, CH_NAMES

conditions = ['list','sentence']
FEATURES = {'envelope':['envelope'], 
            'onset':['envelope', 'word onset'],
            'entropy':['envelope', 'word onset', 'entropy'],
            'surprisal':['envelope', 'word onset', 'surprisal'],
            'frequency':['envelope', 'word onset', 'word frequency'],
            'entropy_surprisal':['envelope', 'word onset', 'entropy', 'surprisal'],
            'frequency_entropy':['envelope', 'word onset', 'entropy', 'word frequency'],
            'frequency_surprisal':['envelope', 'word onset', 'surprisal', 'word frequency'],
            'frequency_entropy_surprisal':['envelope', 'word onset', 'entropy', 'surprisal', 'word frequency']
            }    
    

def compile_trfs(resultsdir, conditions, models, save=True, roi=False, 
                 roi_name=None, hemi=None, source=False):  
    if (roi_name == None and hemi != None) or (roi_name != None and hemi == None):
        raise ValueError('supply both the name of the ROI and the hemisphere')
        
    folders = [f for f in os.listdir(resultsdir) if os.path.isdir(os.path.join(resultsdir, f))]
     
    # initiate dictionary
    trfs = {}
    trfs = dict.fromkeys(models)
    for key in trfs.keys():
        trfs[key]={}
        trfs[key]=dict.fromkeys(conditions)
        for ks in trfs[key].keys():
            trfs[key][ks]=[]
    
    for folder in folders:
        for model in models:
            if roi == False:
                fname = os.path.join(resultsdir, folder, ''.join(['TRF_', model, '.pkl']))
            else:
                if roi_name == None and hemi == None:
                    fname = os.path.join(resultsdir, folder, ''.join(['rTRF_', model, '.pkl']))
                else:
                    fname = os.path.join(resultsdir, folder, f'rTRF_{model}_{roi_name}-{hemi}.pkl')
                    
            if source:
                fname = os.path.join(resultsdir, folder, ''.join(['sTRF_', model, '.pkl']))

            try:
                with open(fname, 'rb') as f:
                    trf_dict = pickle.load(f)
                
                for condition, trf_values in trf_dict.items():
                    if condition in trfs[model].keys(): # why this check?
                        trfs[model][condition].append(trf_values)
                        
            except FileNotFoundError:
                print('{0} is not a file. Check the code for subject {1}.'.format(fname, folder))
                
    if save:
        if roi_name == None or hemi == None:
            with open(os.path.join(resultsdir, 'trfs.pkl'), 'wb') as fr:
                pickle.dump(trfs, fr)
            
        else:
             with open(os.path.join(resultsdir, f'trfs_{roi_name}-{hemi}.pkl'), 'wb') as fw:
                pickle.dump(trfs, fw)
            
    return trfs

def grandaverage(trf, info, models, conditions=None, savedir=None, save=False, 
                 roi_name=None, hemi=None):
    """
    Calculates Grand Average for TRFs.
    
    INPUT
    -----
    conditions : list of str
        conditions to be averaged, default all
    models : list of str
        models to be included
        
    OUTPUT
    ------
    GA : dict of class meg.TRF
    """
    lags = lag_span(-0.2, 0.8, 120)
    if conditions == None:
        conditions = ['list', 'sentence']
            
    # initiate dict
    GA = dict.fromkeys(models)
    for key in GA.keys():
        GA[key]=dict.fromkeys(conditions)
        
    for model in models:
        features = FEATURES[model]
            
        for condition in conditions:
            avg_coef = np.mean([t.coef_ for t in trf[model][condition]], 0)
            GA[model][condition] = TRF(beta=avg_coef, lags=lags, info=info, features=features)
        
    if save and savedir is not None:
        if roi_name != None and hemi != None:
            with open(os.path.join(savedir, f'GA_{roi_name}-{hemi}.pkl'), 'wb') as fw:
                pickle.dump(GA, fw)
        else:
            with open(os.path.join(savedir, 'GA.pkl'), 'wb') as fw:
                pickle.dump(GA, fw)
            
    return GA


def compile_rvals(resultsdir, models, save=True, roi=False, info=None, roi_name=None, hemi=None, source=False):
    """
    Loops over folders in directory to create
    a pandas dataframe of all reconstruction
    accuracies (r-values). 
    
    N.B. number & type of models = hard-coded! @TODO
    
    INPUT
    -----
    resultsdir : str
        directory with results from reconstruct.py:
        .pkl files in separate folder for a subject
    models : list of str
    save : Boolean (default True)
        if True, saves dataframe to disk ('r_data.csv')
    roi : Bool
    info : mne.info
    roi_name : str
    hemi : str
    
    OUTPUT
    ------
    r_data : pd.DataFrame
    """
    if roi==True and info==None:
        raise ValueError("Supply info with ROI names.")
    
    folders = [f for f in os.listdir(resultsdir) if os.path.isdir(os.path.join(resultsdir, f))]
    ch_names_short = [CH_NAME[0:5] for CH_NAME in info.ch_names]
    
    dfs = []
    for folder in folders:
        print('Compiling arrrrrr-values for participant {0}, matey...'.format(folder))
        for model in models:
            if source:
                if roi:
                    if roi_name != None:
                        fname = os.path.join(resultsdir, folder, f'rR_{model}_{roi_name}-{hemi}.pkl')
                    else:    
                        fname = os.path.join(resultsdir, folder, "".join(['rR_', model, '.pkl']))
                else:   
                    fname = os.path.join(resultsdir, folder, f'sR_{model}.pkl')
                
            else:
                fname = os.path.join(resultsdir, folder, "".join(['R_', model, '.pkl']))
            try:
                with open(fname, 'rb') as f:
                    r_dict = pickle.load(f)
                
            except FileNotFoundError:
                print('{0} is not a file. Check the code for subject {1}.'.format(fname, folder))
                continue
            
            df = pd.DataFrame.from_dict(r_dict, orient='index').T.melt()
            df.columns = ['condition', 'r_values']
            df['subject'] = folder
            df['model'] = model
            
            if roi or source:
                df['sensor'] = info.ch_names * 2
            else:
                df['sensor'] = ch_names_short * 2
            dfs.append(df)
            
    r_data = pd.concat(dfs, ignore_index=True)
    r_data.reset_index(inplace=True)
    del dfs
    
    print('Arrrrrr-values have been compiled. Ahoy!')
    if save:
        if roi_name != None and hemi != None:
            r_data.to_csv(os.path.join(resultsdir, f'r_data_{roi_name}_{hemi}.pkl'))
        else:
            r_data.to_csv(os.path.join(resultsdir, 'r_data.csv'))
        
    return r_data

