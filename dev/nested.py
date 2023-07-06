#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 09:42:56 2023

@author: julieneg
"""

import cfr
import matplotlib.pyplot as plt
cfr.__version__
pdb = cfr.ProxyDatabase().fetch('PAGES2kv2')


fig, axs = plt.subplots(nrows=5, ncols=2, figsize=[10,2])
axs = axs.flatten()
start = 1000
for k in range(9):
    # find a nest
    century = [start+k*100+1, start+(k+1)*100]
    pid = pdb.nest_indices(time_period=century)  
    # filter
    comp = pdb.filter(by='pid', keys=pid).make_composite(bin_width=5,n_bootstraps=10)
    comp.plot_composite(ax=axs[k])
    axs[k].set_title(f'{century}') 
