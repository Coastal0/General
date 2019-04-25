# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 08:22:44 2018

@author: 264401k
"""
import pygimli as pg
import pygimli.frameworks
import numpy as np
import matplotlib.pyplot as plt

def interp(tt,xx,yy,zz,nc):

    return()

tt,xx,yy,zz = np.loadtxt("U:/EGP_Data/Projects/Research/WATER Projects/Environmental_2018/ERI/Raw/2D_coords.txt", unpack=True)

tt_x = np.cumsum(abs(xx[0]-xx))

tt_y =  np.cumsum(abs(yy[0] - yy))

robust = True
nc = 10
error = 0.0001
lam = 1
resample = None
resample = np.linspace(0,70,71)

for nc in range(2,4):
    print(nc)
    nc = 4
    x = np.asarray(pygimli.frameworks.harmfit(xx,tt_x, nc = nc, resample = resample,error = error, lam = lam, robust = robust)[0])
    y = np.asarray(pygimli.frameworks.harmfit(yy,tt_y, nc = nc, resample = resample,error = error, lam = lam, robust = robust)[0])
    z = np.asarray(pygimli.frameworks.harmfit(zz,tt, nc = nc, resample = resample,error = error, lam = lam, robust = robust)[0])
    fig, ax = plt.plot(xx,yy, 'bx-',x,y,'ro-')
