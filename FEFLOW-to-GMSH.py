# -*- coding: utf-8 -*-

"""
Convert FEFLOW mesh output to GMSH-style input and then call pyGimli convertor

"""
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import spatial

#%% Import Data
os.chdir('F:\Scripts\PYthon')
fName = 'qr_v13_lowpoly_test.pnt'

print('Reading raw data from', fName)
#point_type = [('ID',float),('x',float),('y',float),('conc',float)]
point_type = float
data = np.loadtxt(fName,dtype=point_type,comments='END')
data_raw = data
print('Imported ', data.shape[0], 'nodes')

formation_factor = 14
sea_temp = 19.5 # Sea Surface Temp. (deg C)
sea_conc = 35.8 # Sea Surface Salinity (psu)
conc_factor = 0.6 # To-Do; Find proper formula describing this. (will range from 0.46 to 0.8 probably)

data_rho = formation_factor/(((data[:,3])/1000)*conc_factor)
repRho = 100
data_rho[data_rho == float('Inf')] = repRho # Solve any inf's remaining in model.
data = np.c_[data, data_rho]

box_extent = 1000
box_val = 999.99
# bounding box; left-top, left-bottom, right-bottom, right-top
boundingbox = [(data[-1,0]+1,data[0,1]-box_extent,data[0,2],data[0,3],box_val),
                (data[-1,0]+2,data[1,1]-box_extent,data[1,2]-box_extent,data[0,3],box_val),
                (data[-1,0]+3,data[2,1]+box_extent,data[2,2]-box_extent,data[0,3],box_val),
                (data[-1,0]+4,data[3,1]+box_extent,data[3,2],data[0,3],box_val)]

data = np.r_[data, boundingbox]

# %% Catch topography
#data_sort = np.sort(data_raw[data_raw[:,2]>=0],0)
data_sort = data_raw[data_raw[:,1].argsort()]

topo = [];


iSkip = 10
max_x = data_sort[-1,1]
curr_x = 0

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

j_track = [];angle = [];dist = [];
i=0; j=i+1;timesteps=1;
max_timestep = 1000;
while j < int(data_sort[-1,0]):
    if timesteps < max_timestep:
        angle = [];dist = []
        for i in range(0,data_sort.shape[0]-1,iSkip):
            p1 = data_sort[i,1],data_sort[i,2]
            p2 = data_sort[j,1],data_sort[j,2]
            angN = angle_between(p2,p1)
            angle.append([data_sort[i,0],angN])
        
            distN = np.hypot(p2[0]-p1[0],p2[1]-p1[1])
            dist.append([data_sort[i,0],distN])
            
            angle_array = np.array(angle)
            dist_array = np.array(dist)
            
            maxDist = 100
            maxAng = 90
            maxAngInDist = angle_array[dist_array[:,1] <= maxDist]
            maxAngInDistRestricted = maxAngInDist[maxAngInDist[:,1] < maxAng]
    
            if maxAngInDistRestricted.shape[0]>0:
                maxAng = np.amax(maxAngInDistRestricted[:,1])
                maxAngLoc = np.argmax(angle_array[:,1],axis=0)
                maxAngLocID = int(angle_array[maxAngLoc][0])
                iddx = [dist_array == maxAngLocID]
                iddxx = np.where(iddx)[1][0]
                p2_max_angle_dist = float(dist_array[iddxx][1])
            
                jdx = np.where([data_sort[:,0] == maxAngLocID])[1][0]
                j = int(jdx)
                curr_x = p2_max_angle_dist
                
                j_track.append(j)
                topo.append([maxAngLocID])
            print('Min. Angle is ' + str(maxAng) +' degrees and ' + str(p2_max_angle_dist) +' metres')   
        else:
            break
    print(timesteps)
    timesteps += 1
    plt.figure(1)
    plt.scatter(timesteps,j)
    
topo_array = np.array(topo)
topo_x = data[topo_array][:,0,:][:,1]
topo_y = data[topo_array][:,0,:][:,2]
plt.figure(2)
plt.scatter(topo_x,topo_y)

# %% Format to *.geo
fNameOut = 'out.geo'
with open(fNameOut,'w') as f:
    for x in data:
        index = int(x[0]);
        px = x[1];
        py = x[2];
        val = x[3];
        outstring = 'Point(' + str(index) + ') = {' + str(px) + ', ' + str(py) + ', 0.0, ' + str(val) + '};\n';
        f.write(outstring);
    for i in range(4,1,-1):
        outstring = 'Line(newl) = {' + str(data[-i,0]) + ', ' + str(data[-i+1,0]) +'};\n'
#        print(outstring)
        f.write(outstring);
    f.write('Line(newl) = {' + str(data[-4,0]) + ', ' + str(data[0,0]) +'};\n')
    f.write('Line(newl) = {' + str(data[-1,0]) + ', ' + str(data[3,0]) +'};\n')
    f.write('Line(newl) = {' + str(data[0,0]) + ', ' + str(data[1,0]) +'};\n')
    f.write('Line(newl) = {' + str(data[1,0]) + ', ' + str(data[2,0]) +'};\n')
    f.write('Line(newl) = {' + str(data[2,0]) + ', ' + str(data[3,0]) +'};\n')
    f.write('Line(newl) = {' + str(data[3,0]) + ', ' + str(data[0,0]) +'};\n')
f.closed