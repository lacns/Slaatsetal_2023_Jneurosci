#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 13:58:41 2022

@author: sopsla
"""

from text import get_wordlevel_feats_flexibly
from utils import get_audio_envelope
import matplotlib.pyplot as plt

sid = 205

env = get_audio_envelope(sid)
N=len(env)
x = get_wordlevel_feats_flexibly(sid, N=N)

fig,ax=plt.subplots()
plt.plot(env)
plt.plot(x)