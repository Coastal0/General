# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:04:12 2015

@author: Alex
"""

import numpy as np


linenumber=0
topo=[];

topodata=[];
topodata = open("D:\Topography.txt");        
for line in topodata:
    linenumber = linenumber + 1;
    pointarray = [];
    point = line.split('\t')
    for i in range (0, len(point)) :
        pointarray.append(float(point[i]));
    topo.append(pointarray);   
        
topodata.close();

#d = {}
#with open("D:\Topography.txt") as t:
#    for line in t:
#        splitline = line.split()
#        d[int(splitline[0])] = ",".join(splitline[1:])


data = [];
depthnumber = 0;

file = open("D:\TestFile1990.txt");
f = file.readlines();
nth = 0;
for i in range (0,len(f)):
    l = list(f[i]);
    s = [];
    for i in range (0,len(l)):
        if ((l[i] >= '0') and (l[i] <= '9')) : s.append(float(l[i]));
        elif l[i] == 'a' : s.append(10.0);
        elif l[i] == 'b' : s.append(11.0);
        elif l[i] == 'c' : s.append(12.0);
        elif l[i] == 'd' : s.append(13.0);
        elif l[i] == 'e' : s.append(14.0);
        elif l[i] == 'f' : s.append(15.0);
  
    data.append(s)
    while (depthnumber < len(f)):    
        depthnumber=depthnumber+1
        
datnp = np.array(data);
datnp = datnp.transpose();
tpad = int(round(max(np.array(topo)[:,1])));
datpad = [];
for i in range(0, len(datnp)) :
    padf = int(round(topo[i/2][1]));  
    pad3 = tpad-padf;
    profile = datnp[i];
    datpadl = [];
    for j in range(0,padf) :
        datpadl.append(15);
    for j in range(0,len(profile)) :
        datpadl.append(profile[j]);
    for j in range(0,pad3) :
        datpadl.append(3);
    datpad.append(datpadl);
datpadnp = np.array(datpad).transpose();
#    l = list(f[i])
#    s=[];
#    pad = int(round(topo[nth][1])); 
#    nth = nth+1;
#    for i in range(0, pad):
#        s.append(15);
file.close();

fileout = open("D:\Out.txt", 'w')
np.savetxt(fileout,datpadnp,fmt='%d')