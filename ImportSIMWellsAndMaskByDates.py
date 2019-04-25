# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 15:16:11 2018

@author: 264401k

Import Excel sheet with water levels.

-- Import Excel
-- Filter to specific wells
-- Graph wells
-- Get meaningful statistics
"""

import pandas as pd
import matplotlib.pyplot as plt

raw_WL = pd.read_excel(r"G:/PROJECTS/PAPER -- Quinns Rocks Urban Expansion & Saline Water Interface/Data/QuinnsRocks_WirBores/WaterLevelsDiscreteForSiteCrossTab.xlsx")

wells = [61611701,61611702,61611703,61611704,61611706]
raw_WL_grp = pd.groupby(raw_WL.loc[raw_WL['Site'].isin(wells)], 'Site')


dateMask_pre1990 = raw_WL['Date'] < '01/01/1998'
dateMask_post1990 = raw_WL['Date'] > '01/01/1998'
dateMask_post2005 = raw_WL['Date'] > '01/01/2005'

raw_WL_grp_dateMask_pre1990 = raw_WL.loc[raw_WL['Site'].isin(wells) & dateMask_pre1990].groupby('Site')
raw_WL_grp_dateMask_post1990 = raw_WL.loc[raw_WL['Site'].isin(wells) & dateMask_post1990].groupby('Site')
raw_WL_grp_dateMask_post2005 = raw_WL.loc[raw_WL['Site'].isin(wells) & dateMask_post2005].groupby('Site').median()


fig = plt.figure()
ax = plt.axes()
for label, df in raw_WL_grp:
    df.plot(x = 'Date', y = 'WaterLevel_AHD', ax = ax, label = label)


x_pre1990 = [raw_WL_grp_dateMask_pre1990['Date'].min(), raw_WL_grp_dateMask_pre1990['Date'].max()]
y_pre1990 = [raw_WL_grp_dateMask_pre1990['WaterLevel_AHD'].median(), raw_WL_grp_dateMask_pre1990['WaterLevel_AHD'].median()]

x_post1990 = [raw_WL_grp_dateMask_post1990['Date'].min(), raw_WL_grp_dateMask_post1990['Date'].max()]
y_post1990 = [raw_WL_grp_dateMask_post1990['WaterLevel_AHD'].median(), raw_WL_grp_dateMask_post1990['WaterLevel_AHD'].median()]

plt.plot(x_pre1990,y_pre1990, x_post1990, y_post1990)
