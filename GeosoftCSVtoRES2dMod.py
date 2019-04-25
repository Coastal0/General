# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 08:12:35 2015

@author: Alex
"""
import numpy as np
import copy


"""
Data import and splitting
"""
numberofelectrodes = 290 #input('#Electrodes?');
electrodespacing = 2.0 #input('Spacing?');
lineNumber = 0;
data = [];

file = open("D:\SWI1990_XYZ.xyz")
for line in file:
    lineNumber = lineNumber+1;
    if (lineNumber != 1):
        stringSplit = line.split('\t')
        singleLineArray = [];
        for i in range (0,len(stringSplit)):
            singleLineArray.append(float(stringSplit[i]));
        data.append(singleLineArray);
        #print(singleLineArray)

        
"""
Define extents of grid
"""

x = np.array([data]); #Finds largest list value in data
xmax = np.amax(x[:,:,0]);
xmin = np.amin(x[:,:,0]); #Finds smallest list value in data
ymax = np.amax(x[:,:,1]); #Finds largest value for list depths
ymin = np.amin(x[:,:,1]); #Finds smallest value for list depths

#print(xmax,xmin,ymax,ymin,totalnumblocks)



"""
Assign ranges of resistivity values to indexed numbers
"""

resvalues = ([]);
for i in range(0, 16):
    z=0
    z = z+i
    i=i+1
    resvalues.append(z)
totalZ = len(resvalues);

#print(z, resvalues)

ResData=data

intervals=[]

intervals_raw = open("D:\intervals.csv")
intervals=intervals_raw.readlines()
intervals=[int(i) for i in intervals]
#if ((intervals[len(intervals)-1] < np.amax(x[:,:,2]))) :
#    intervals[len(intervals)-1] = (int(np.amax(x[:,:,2]))+1)
if (intervals[0] == 0) :
    intervals[0] = 0.1
print(intervals)

formfactor = 14

#intervals = [600, 3500, 6500, 9500, 12500, 15500, 18300, 21300, 24300,27300,30200,33200,36100,39100,42000,45001];
intervalCode = ['0', '1', '2', '3', '4', '5', '6', '7', '8','9','a','b','c','d','e','f'];

for i in (range(0,x.shape[1])):
    d = data[i];
    d[2] = (10000/d[2])*formfactor
    res = d[2];
    
    for n in range(0,len(intervals)) :
            intervalNumber = intervals[n];
            if(res <  intervalNumber) :
                data[i].append(copy.deepcopy(n));
                break;  

x_app = np.array([data]);
totalnumblocks = len(np.unique(x_app[:,:,0]));
#ints_str=str(intervals);
#ints_str=ints_str.strip('[')
#ints_str=ints_str.strip(']')
#ints_str=ints_str.replace('.',',');
#ints_str=ints_str.replace('\n','');
#ints_str=ints_str.replace(" ","");
#ints_str=ints_str.strip()    
                
"""
Define depth levels of grid
"""

datadepths = np.array([data]);
y=np.unique(abs(x[:,:,1]))
y_str=str(y);
y_str=y_str.strip('[').strip(']')
#y_str=y_str.strip(']')
y_str=y_str.replace('.',',').replace('\n','').replace(" ","");
#y_str=y_str.replace('\n','');
#y_str=y_str.replace(" ","");
y_str=y_str.strip()


datalevel = len(y);
psuedolevel = len(y);

file.close()

"""
Write output file for RES2DMOD software
"""

fileOut = open("D:\SWI1990_XYZ_out.xyz.txt", 'w')

fileOut.write("SWI" + "\n"); #File title
fileOut.write(str(numberofelectrodes) + "\n"); #Number of electrode, max 500
fileOut.write(str(psuedolevel) + "\n"); #Number of pseudosection levels
fileOut.write("0" + "\n"); #Underwater flag
fileOut.write(str(electrodespacing) + "\n"); #Unit electrode spacing
fileOut.write("2" + "\n"); #grid model flag, 2 for userdefined
fileOut.write("0" + "\n"); #First block offset (0)
fileOut.write(str(totalnumblocks) + "\n"); #Number of blocks in array
fileOut.write(str(totalZ) + "\n") #Total number of res values
fileOut.write("2" + "\n") #Nodes per spacing
fileOut.write(str(intervals) + "\n") #Values of resistivity
fileOut.write(str(datalevel) + "\n") #Total number  of depth levels
fileOut.write((str(y_str[:-1]) + "\n")) #values of depth levels i.e rows of blocks

#lastZ = data[0][1];
##print(lastZ)
#for i in (range(0,len(data))):
#    point = data[i];
#    z = (point[1]);
#    resIndex = point[3]
#    resChar = intervalCode[resIndex];
#        
#    if(z == lastZ) :
#        fileOut.write(str(resChar));
#    else :
#        fileOut.write("\n");
#        fileOut.write(str(resChar));
#        lastZ = z;
#        print(i,point,z,lastZ)

lastZ = x[0,1,1];
#print(lastZ)
for i in (range(0,x.shape[1])):
    point = x_app[:,i];
    z = point[0,1];
    resIndex = int(point[0,3]);
    resChar = intervalCode[resIndex];
        
    if(z == lastZ) :
        fileOut.write(str(resChar));
    else :
        fileOut.write("\n");
        fileOut.write(str(resChar));
        lastZ = z;
        print(i,point,z,lastZ)

fileOut.write(("\n" + str(1) + "\n")) #array type


i=0
for i in range(3):
    if i == 3:
        break
    else:
        fileOut.write((str(0) + "\n")) #"end with a few zeros"
        i=i+1


print("Done")
file.close();