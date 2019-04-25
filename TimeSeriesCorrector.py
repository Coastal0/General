# -*- coding: utf-8 -*-
"""
Created on Mon May  7 13:18:33 2018

@author: 264401k

Load and correct for seasonal trends in tidal data.
"""
import pandas as pd
import os
import glob
import re
from statsmodels.tsa.seasonal import seasonal_decompose

data = []
workDir = r"G:\PROJECTS\PAPER -- Quinns Rocks Urban Expansion & Saline Water Interface\Data\Hillarys_Tides&Sim\Tide Data (Hillarys)" #Set data directory.
os.chdir(workDir) #Change to data directory.
yearsList = glob.glob("*.csv") #Make a list of the csv's in the directory.
years = [re.split(r'[_|.]',y)[1] for y in yearsList] #Split the year from the filename.

data = pd.read_csv(yearsList[0], index_col = [' Date & UTC Time'], parse_dates = True)
data = data.replace(-9999, "")
data.interpolate(inplace=True)
data.plot()
result = seasonal_decompose(data['Sea Level'], model = 'additive', freq = 24*7)
result.plot()

expweighted_avg = pd.ewma(data['Sea Level'], halflife = 24)
ts_diff = data['Sea Level'] - expweighted_avg
data['Sea Level'].plot()
expweighted_avg.plot()
ts_diff.plot()

#ts_diff.to_csv('ts_diff.csv')


#%% FFT
import numpy as np

test = np.fft.fft(a = data['Sea Level'])
