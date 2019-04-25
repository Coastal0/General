# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:48:25 2017

@author: 264401k
"""

import os
import numpy as np
import pygimli as pg
import pygimli.frameworks
import matplotlib.pyplot as plt
os.chdir(r"U:\EGP_Data\Projects\Research\WATER Projects\DoW_Hydrogeophysics_2015_2018\WP3 - MT and ERI\WP3-RA2-ERI_Radar\Data\Bold Park Camel Lake")
tt, hh ,xx ,yy = np.loadtxt('gps.txt', unpack = True)

resampVector = np.arange(72) * 5.
x = pygimli.frameworks.harmfit(xx,tt, nc = 10, resample = resampVector)[0]
y = pygimli.frameworks.harmfit(yy,tt, nc = 10, resample = resampVector)[0]
z = pygimli.frameworks.harmfit(hh,tt, nc = 10, resample = resampVector)[0]
t= np.hstack((0.,np.cumsum(np.sqrt(np.diff(x)**2+np.diff(y)**2))))

x_i = x.array()
y_i = y.array()
z_i = z.array()

fOut = np.array([t,x_i,y_i,z_i]).T
np.savetxt('NEW.dat',fOut, fmt='%.4f',header = 't x y z',delimiter=',')

fig, ax = plt.plot(tt,hh, 'bx-', tt, z, 'r-')
fig, ax = plt.plot(xx,yy, 'bx-',x,y,'r-')
