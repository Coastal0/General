# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 17:07:23 2018

@author: 264401k

Makes a conditional cumulative sum for each year of a given textfile (year,value)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn import linear_model
from scipy.optimize import curve_fit
from tqdm import tqdm
from datetime import datetime as dt

# %%Load and define data
fIn = "G:/PROJECTS/PAPER -- Quinns Rocks Urban Expansion & Saline Water Interface/Data/PerthAirport_Rainfall_1944+/IDCJAC0001_009021/IDCJAC0001_009021_Data1.csv"
rData = pd.read_csv(fIn)

cumulativeSum = rData.iloc[:,4].expanding(1).sum()
cumulativeSum.rename('CumulativeSum', inplace = True)
rData = rData.join(cumulativeSum)
del cumulativeSum
rData.drop(labels = rData.columns[[0,1,5]], axis = 1, inplace = True)

date = pd.to_datetime({'year': rData['Year'],
                       'month': rData['Month'],
                       'day': rData['Year']*0+1})
rData = rData.join(date.rename('Date'))

# Plot raw data
#rData.plot(x= 'Date', y = 'CumulativeSum')
#rData.plot(x= 'Date', y = rData.columns[2])

# %% Setup linear regression
def linest(x,y):
#    xvals = x.values.reshape([x.size,1]).astype('datetime64[M]').astype(float)
    if type(y) != np.ndarray:
        print('reshaping Y array...')
        y = y.values.reshape([y.size,1])
    if np.ndim(x) == 1:
        print('Reshaping X array...')
        x.values.reshape([x.size,1]).astype('datetime64[M]')
#    x_date = x.astype('datetime64[M]')
#    xvals = x.astype(int)
    regr = linear_model.LinearRegression()
    regr.fit(x,y)
    regr.coef_
#    plt.scatter(x,y, color='black', s = 1)
#    plt.plot(x, regr.predict(xvals), linewidth=3)
    if np.ndim(regr.coef_) > 1:
        return regr.coef_[0][0], regr
    else:
        return regr.coef_[0], regr

blueShade = np.array([0, 112, 192])/255
redShade = np.array([255, 0, 0])/255
greenShade = np.array([51, 153, 51])/255
# %% SETUP SUBPLOTS
fig, ax = plt.subplots(1,2)
ax1 = ax[0]
ax2 = ax[1]
#ax1.minorticks_on()
#ax2.minorticks_on()
#ax1.grid(alpha = 0.3)
#ax2.grid(alpha = 0.3)
fig.set_size_inches(10,5)

# %%
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

#axins = zoomed_inset_axes(ax1, 4, loc=2)
axins2 = zoomed_inset_axes(ax1, 12, loc=4, borderpad=2)
plt.sca(ax1)
# % Large-range estimates
plt.xlabel('Year')
plt.ylabel('Cumulative Rainfall [mm]')
# Whole-Period
x_all = rData['Date']
x_all = rData['Date'].values.astype('datetime64[M]').reshape([x_all.size,1])
y_all = rData['CumulativeSum']
ax1.set_xlim(min(rData['Date']), max(rData['Date']))
#ax1.set_ylim(0, max(rData['CumulativeSum']))
ax1.set_ylim(0, 60000)
ax1.tick_params(direction = 'in')

coef, regr = linest(x_all,y_all)
ax1.scatter(x_all,y_all, color='black',  s = 1)
#ax1.plot(x_all, regr.predict(x_all.astype(int)), linewidth=1, linestyle = '-', color = 'k')

# First 20-years
dMask_f20 = rData['Date'] < min(rData['Date']) + np.timedelta64(20, 'Y')
x_f20 = rData['Date'][dMask_f20]
x_f20 = x_f20.values.reshape([x_f20.size,1]).astype('datetime64[M]')
y_f20 = rData['CumulativeSum'][dMask_f20]
coef_f20, regr_f20 = linest(x_f20,y_f20)
x_f20_incEnd = np.append(x_f20, x_all[-1]).reshape([x_f20.size+1,1])
#plt.plot(x_f20, regr_f20.predict(x_f20.astype(int)), linewidth=1, linestyle = '-', color = 'g')
ax1.plot(x_f20_incEnd, regr_f20.predict(x_f20_incEnd.astype(int)), linewidth=1, linestyle = '-', color = 'k')
#xy_t1 = (dt(1986,1,1).toordinal(),getNearestDate(rData, dt(1986,1,1))+20000)
xy_t1 = (725007, 51800)
t1 = ax1.annotate("Trend - 1944 to 1964", xy = xy_t1, rotation = 46, color = 'k')

# Last 20-years
dMask_l20 = rData['Date'] > max(rData['Date']) - np.timedelta64(20, 'Y')
x_l20 = rData['Date'][dMask_l20]
x_l20 = x_l20.values.reshape([x_l20.size,1]).astype('datetime64[M]')
y_l20 = rData['CumulativeSum'][dMask_l20]

coef_l20, regr_l20 = linest(x_l20,y_l20)
x_l20_incStart = np.insert(x_l20,[0], x_all[0]).reshape([x_l20.size+1,1])
# ax1.plot(x_l20, regr_l20.predict(x_l20.astype(int)), linewidth=1, linestyle = '-', color = 'g')
ax1.plot(x_l20_incStart, regr_l20.predict(x_l20_incStart.astype(int)), linewidth=1, linestyle = '-', color = greenShade)
#steam_l20 = ax1.stem(x_l20[0], y_l20.values[0], alpha = 0.2)

#xy_t2 = (dt(1951,5,1).toordinal(),getNearestDate(rData, maxDate))
xy_t2  = (712343,26500)
t2 = ax1.annotate("Trend - 1998 to 2018", xy = xy_t2, rotation = 42, color = greenShade)
#t2 = ax1.annotate("Trend - 1998 to 2018, m = {} mm/yr".format(int(round(12*coef_l20,0))), xy = xy_t2, rotation = 42, color = greenShade)

# Control Zoomed Axis
def getNearestDate(rData, date):
    inDate = rData['Date'].dt.to_pydatetime() == date
    inDate_IDX = [i for i, x in enumerate(inDate) if x]
    value = rData['CumulativeSum'][inDate_IDX].values[0]
    return value

# axins (top-left subplot)
#minDate = dt(1944,5,1)
#maxDate = dt(1951,5,1)
#x1 = minDate.toordinal()
#x2 = maxDate.toordinal()
#y1 = getNearestDate(rData, minDate)
#y2 = getNearestDate(rData, maxDate)
#
#axins.set_xlim(x1, x2)  # apply the x-limits
#axins.set_ylim(y1, y2)  # apply the y-limits
#axins.scatter(x_all,y_all, color='black',  s = 1)
#axins.plot(x_all, regr.predict(x_all.astype(int)), linewidth=1, linestyle = '--', color = 'k')
##axins.plot(x_f20, regr_f20.predict(x_f20.astype(int)), linewidth=1, linestyle = '-', color = 'g')
#axins.plot(x_f20_incEnd, regr_f20.predict(x_f20_incEnd.astype(int)), linewidth=1, linestyle = '--', color = 'darkorange')
#
#axins.xaxis.set_visible(True)
#axins.xaxis.set_label_position("bottom")
#axins.xaxis.set_ticks_position('bottom')
#
#axins.yaxis.set_visible(True)
#axins.yaxis.set_label_position("right")
#axins.yaxis.set_ticks_position("right")
#axins.set_xticks(axins.get_xticks()[1:-1:3])
#axins.set_yticks(axins.get_yticks()[1:-1:2])
#mark_inset(ax1, axins, loc1=3, loc2=4, fc="none", ec="0.5")

# axins2 (bottom-right subplot)
minDate2 = dt(2016,1,1)
maxDate2 = dt(2018,1,1)
x1 = minDate2.toordinal()
x2 = maxDate2.toordinal()
y1 = getNearestDate(rData, minDate2)
y2 = getNearestDate(rData, maxDate2)

axins2.set_xlim(x1, x2) # apply the x-limits
axins2.set_ylim(y1, y2) # apply the y-limits
axins2.scatter(x_all,y_all, color='black',  s = 1)
#axins2.plot(x_all, regr.predict(x_all.astype(int)), linewidth=1, linestyle = '-', color = 'k')
#axins2.plot(x_l20, regr_l20.predict(x_l20.astype(int)), linewidth=1, linestyle = '-', color = 'g')
axins2.plot(x_l20_incStart, regr_l20.predict(x_l20_incStart.astype(int)), linewidth=1, linestyle = '-', color = greenShade)

axins2.xaxis.set_visible(True)
axins2.xaxis.set_label_position("bottom")
axins2.xaxis.set_ticks_position('bottom')

axins2.yaxis.set_visible(True)
axins2.yaxis.set_label_position("left")
axins2.yaxis.set_ticks_position("left")
axins2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axins2.set_xticks([axins2.get_xticks()[0],axins2.get_xticks()[-1]])
axins2.set_yticks([axins2.get_yticks()[0],axins2.get_yticks()[-1]])

#axins2.grid(which = 'both')

mark_inset(ax1, axins2, loc1=1, loc2=2, fc="none", ec="0.5")


# %%Sort into months and fit line for each month
summerMonths = [10,11,12,1,2,3]
winterMonths = [4,5,6,7,8,9]

monthDict = {}
for s in summerMonths:
    monthDict.update({s : 'S'})
for w in winterMonths:
    monthDict.update({w : 'W'})


# %%Filter and prepare dataframe
rData['Season'] = rData['Month'].map(monthDict)

#### WARNING: THIS IS SLOW!!! ###
#rData['Season'] = np.nan
#rDataSeason = rData['Season']
#for n,months in tqdm(enumerate(rData['Month'])):
#    if months in summerMonths:
#        rDataSeason[n] = 'S'
##        rData['Season'][n] = 'S'
#    if months in winterMonths:
#        rDataSeason[n] = 'W'
##        rData['Season'][n] = 'W'

rData2 = rData.drop(rData.index[0:5])
rData2 = rData2[:-4]
rData2.dropna(inplace=True)

rData2['s_s'] = rData2['Season'].shift(1)
rData2.fillna(value = 'S', inplace = True)
rData2.reset_index(drop = True, inplace = True)

# %% Setup arrays
x = np.zeros([len(summerMonths),1], dtype = 'datetime64[M]')
y = np.zeros([len(summerMonths),1])
rCoef_W = np.zeros([int(np.ceil(rData2.shape[0]/(2*len(summerMonths)))),1])
rCoef_S = np.zeros([int(np.ceil(rData2.shape[0]/(2*len(summerMonths)))),1])

i = a = b = 0
for index, row in tqdm(rData2.iterrows()):
    # Collect x's (dates) and y's (cumulative rainfall) while season matches.
    if row[5] == row[6]:
        x[i] = rData2.iloc[index]['Date'].to_pydatetime()
        y[i] = rData2.iloc[index]['CumulativeSum']
        i = i + 1
    else:
        i = len(summerMonths)-1
        x[i] = rData2.iloc[index]['Date'].to_pydatetime()
        y[i] = rData2.iloc[index]['CumulativeSum']
        if row[6] == 'S':
            rCoef_S[a], regr = linest(x,y)
            plt.scatter(x,y, color='black', s = 1)
            plt.plot(x, regr.predict(x.astype(int)), linewidth=1, color = redShade)
            axins.plot(x, regr.predict(x.astype(int)), linewidth=1, color = redShade)
            axins2.plot(x, regr.predict(x.astype(int)), linewidth=1, color = redShade)
            a = a + 1
        elif row[6] == 'W':
            rCoef_W[b], regr = linest(x,y)
            plt.scatter(x,y, color='black', s = 1)
            plt.plot(x, regr.predict(x.astype(int)), linewidth=1, color = blueShade)
            axins.plot(x, regr.predict(x.astype(int)), linewidth=1, color = blueShade)
            axins2.plot(x, regr.predict(x.astype(int)), linewidth=1, color = blueShade)
            b = b + 1
        i = 0
ax = plt.gca()
plt.xlabel('Year')
plt.ylabel('Cumulative Rainfall (mm)')
TitleString = 'Cumulative Rainfall Since {}'.format(min(x_all)[0].astype('datetime64[Y]'))
#plt.title(TitleString)
from matplotlib.lines import Line2D
custom_legend = [Line2D([0], [0], color=redShade, lw=1),
                Line2D([0], [0], color=blueShade, lw=1),
                Line2D([0], [0], color=greenShade, lw=1),
                Line2D([0], [0], color='k', lw=1)]
ax1.legend(custom_legend, ['Summer', 'Winter', 'm = {} mm/yr'.format(int(round(12*coef_l20,0)), '%0d'), 'm = {} mm/yr'.format(int(round(12*coef_f20,0)), '%:d')], frameon=False)
fig = plt.gcf()
fig.get_size_inches()
fig.set_tight_layout('tight')

# %% Get Gradients and plots
plt.sca(ax2)
ax2.set_xlim(min(years), max(years))
ax2.set_ylim(0,200)
ax2.tick_params(direction = 'in')
rCoef_S[rCoef_S == 0] = rCoef_W[rCoef_W == 0] = np.nan
mask = ~np.any(np.isnan(rCoef_S), axis = 1)
rCoef_S = rCoef_S[mask]
rCoef_W = rCoef_W[mask]

x_rCoef = np.arange(0,rCoef_S.size).reshape([rCoef_S.size,1])
plt.gcf()
rCoef_rCoefS, regrS = linest(x_rCoef, rCoef_S)
rCoef_rCoefW, regrW = linest(x_rCoef, rCoef_W)

ax = plt.gca()
#plt.minorticks_on()
plt.xlabel("Year")
plt.ylabel("Seasonal Gradient (mm/month)")
#plt.title('Comparison of Linear Fit Gradients')
years = pd.unique(rData2['Year'])[1:-1]
#plt.plot(years, rCoef_S, color = 'firebrick', linestyle = '--', label = "Summer Linear Coefficients")
summer_scatter = ax2.scatter(years, rCoef_S, marker = 'o', edgecolor = 'None', alpha = 0.5, color= redShade, s= 20, label = "Summer Gradients")
#plt.plot(years, rCoef_W, color = 'steelblue', linestyle = '--' ,label = "Winter Linear Coefficients")
winter_scatter = ax2.scatter(years, rCoef_W, marker = 'o', edgecolor = 'None', alpha = 0.5, color=blueShade, s = 20, label = "Winter Gradients")
summer_gradient = ax2.plot(years, regrS.predict(x_rCoef), label = 'm = '+ str(round(12*rCoef_rCoefS,2)) + ' mm/yr$^2$', color=redShade)
winter_gradient = ax2.plot(years, regrW.predict(x_rCoef), label = 'm = '+ str(round(12*rCoef_rCoefW,2)) + ' mm/yr$^2$', color=blueShade)
plt.legend(frameon=False)

