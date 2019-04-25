# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:57:17 2018

@author: 264401k
"""

import pandas as pd

rawIN = pd.read_csv(r"F:\Buckland Hill Gun Reserve\GPR_v2\NewCoords\RAW.dat", header = None)
rawIN_grp = rawIN.groupby(0)

for name, group in rawIN_grp:
    group.to_csv('Profile_00{}_A1.cor'.format(name), header = None, index = None, sep = '\t', columns = range(1,10))
