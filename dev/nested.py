#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 09:42:56 2023

@author: julieneg
"""

import cfr
cfr.__version__
pdb = cfr.ProxyDatabase().fetch('PAGES2kv2')

# standardize
ref_period = [1951,1980]
pdb_s = pdb.standardize(ref_period)

# find a nest
ind = pdb_s.nest_indices([1300, 1400])  
