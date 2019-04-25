# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 08:46:12 2018

@author: 264401k
"""
#
#import matplotlib.pyplot as plt
##
#import pygimli as pg
##pg.__version__
#from pygimli.meshtools import readGmsh
import numpy as np
import pygimli as pg

def readGmsh(fname, verbose=True):
    inNodes, inElements, inEntities, inHeader, ncount, entcount, elecount = 0, 0, 0, 0, 0, 0, 0
    nreads, readahead = 0,0
    fid = open(fname)
    if verbose:
        print('Reading %s... \n' % fname)
    
    for line in fid:
    #        print(line)
        if line[0] == '$':
            if line.find('Nodes') > 0:
                inNodes = 1
            if line.find('EndNodes') > 0:
                inNodes = 0
            if line.find('Elements') > 0:
                inElements = 1
                firstLine = 1
            if line.find('EndElements') > 0:
                inElements = 0
            if line.find('Entities') > 0:
                inEntities = 1
            if line.find('EndEntities') > 0:
                inEntities = 0
            if line.find('MeshFormat') > 0:
                inHeader = 1
            if line.find('EndMeshFormat') > 0:
                inHeader = 0
        else:
            if inHeader:
                meshVersion = float(line.split()[0])
                print('Mesh version: %s' % meshVersion)
                
            if inEntities: # v4 meshes only.
                tLine = [float(e_) for e_ in line.split()]
                if len(tLine) == 4: # Header line
                    nPoints = int(tLine[0])
                    # Initialize arrays with maximum number of entries
                    # Docs suggest this should be 8, but adding phys node to it makes it 9 units long...
                    pointsList = np.zeros((nPoints,9))
                    nCurves = int(tLine[1])
                    curvesList = np.zeros((nCurves,12))
                    nSurfaces = int(tLine[2])
                    surfacesList = np.zeros((nSurfaces,18))
                    nVolumes = int(tLine[3])
                    volumesList = np.zeros((nVolumes,10))
                    
                    pCount, cCount, sCount, vCount = 0, 0, 0, 0

                    nEntities = (nPoints + nCurves + nSurfaces + nVolumes)
#                    entityList = np.zeros((nEntities,12))
                    print('Expecting %s total entitites...' % nEntities, "(",nPoints, nCurves, nSurfaces, nVolumes,")")
               
                else: # Load entities (This could be improved)
                    if entcount in range(nPoints):
                        pointsList[pCount,:] = np.pad(tLine,(0,abs(len(tLine) - pointsList.shape[1])),'constant')
                        pCount += 1
                    elif entcount in range(nPoints, nCurves + nPoints):
                        curvesList[cCount,:] = np.pad(tLine,(0,abs(len(tLine) - curvesList.shape[1])),'constant')
                        cCount += 1
                    elif entcount in range(nCurves + nPoints, nSurfaces + (nCurves + nPoints)):
                        surfacesList[sCount,:] = np.pad(tLine,(0,abs(len(tLine) - surfacesList.shape[1])),'constant')
                        sCount += 1
                    elif entcount in range(nEntities - nVolumes, nEntities + nVolumes):
                        volumesList[vCount,:] = np.pad(tLine,(0,abs(len(tLine) - volumesList.shape[1])),'constant')
                        vCount += 1
                    entcount += 1
  
            elif inNodes == 1:
                if meshVersion < 4: # Old importer
                    if len(line.split()) == 1:
                        nodes = np.zeros((int(line), 3))
                        if verbose:
                            print('  Nodes: %s' % int(line))
                    else:
                        nodes[ncount, :] = np.array(line.split(), 'float')[1:]
                        ncount += 1        
                else: # New 
                    tLine = [float(e_) for e_ in line.split()]
                    tLine[0] = int(tLine[0])
                    if len(line.split()) == 2: # 2 Entries is the header for Nodes section
#                        entities = np.zeros((int(line.split()[0]), 3))
                        nodes = np.zeros((int(line.split()[1]), 3))
                        if verbose:
                            print('   Nodes: %s' % tLine[0])
                            print('   Entities: %s' % tLine[1])
                    elif len(line.split()) == 4: # 4 Entries is for each nodes or entity 
                        headerline = 0
                        if tLine[1] in range(nSurfaces+1) and tLine[2] == 0:
                            readahead = tLine[-1]
                            headerline = 1
                            nreads = 0
                        elif nreads < readahead and headerline != 1:
                            nodes[tLine[0]-1, :] = tLine[1:]
                            nreads += 1
                        
            elif inElements == 1:
                if meshVersion < 4: # Standard import sequence
                    if len(line.split()) == 1:
                        if verbose:
                            print('  Entries: %s' % int(line))
                        points, lines, triangles, tets = [], [], [], []
    
                    else:
                        entry = [int(e_) for e_ in line.split()][1:]
    
                        if entry[0] == 15:
                            points.append((entry[-2], entry[-3]))
                        elif entry[0] == 1:
                            lines.append((entry[-2], entry[-1], entry[2]))
                        elif entry[0] == 2:
                            triangles.append((entry[-3], entry[-2], entry[-1],
                                              entry[2]))
                        elif entry[0] == 4:
                            tets.append((entry[-4], entry[-3], entry[-2],
                                         entry[-1], entry[2]))
                        elif entry[0] in [3, 6]:
                            pg.error("Quadrangles and prisms are not supported yet.")                 
                else: # New import sequence
                    if firstLine == 1:
                        totalEles = int(line.split()[1])
                        print('#Elements = %s' %totalEles)
                        firstLine = 0 # Deflag section header
                        numEle = 0
                        eCount, addPoints, addTriangles, addLines = 0, 0, 0, 0 # Initialize variables
                        points, lines, triangles, tets = [], [], [], []
                    else:
                        tLine = [int(e_) for e_ in line.split()]
                        if eCount < numEle:
                            if addPoints == 1:
                                points.append(tLine[-1])
                                eCount += 1
                            elif addLines == 1:
                                lines.append((tLine[1],tLine[2]))
                                eCount += 1
                            elif tLine[1] == tLine[2] == 2:
                                addTriangles = 0
                            elif addTriangles == 1:
                                triangles.append((tLine[1],tLine[2],tLine[3], tagEnt))
                                eCount += 1
                                
                        elif eCount >= numEle:
                            addPoints = 0
                            addLines = 0
                            addTriangles = 0
                            eCount = 0
 
                        if len(line.split()) == 4 and addTriangles == 0:
                            tagEnt = tLine[0]
                            dimEnt = tLine[1]
                            typeEle = tLine[2]
                            numEle = tLine[3]
                            
                            if typeEle == 15: # Flag points on
                                addPoints = 1
                            elif typeEle == 1: # Flag lines on
                                addLines = 0 # Disabled until I work out BOUNDS
                            elif typeEle == 2: # Flag triangles on
                                addTriangles = 1
                            else:
                                print('typeEle not initialized;')
    fid.close()
    lines = np.asarray(lines)
    triangles = np.asarray(triangles)
    tets = np.asarray(tets)

    if verbose:
        print('    Points: %s' % len(points))
        print('    Lines: %s' % len(lines))
        print('    Triangles: %s' % len(triangles))
        print('    Tetrahedra: %s \n' % len(tets))
        
        print('Creating mesh object... \n')

    # check dimension
    if len(tets) == 0:
        dim, bounds, cells = 2, lines, triangles
        zero_dim = np.abs(nodes.sum(0)).argmin()  # identify zero dimension
    else:
        dim, bounds, cells = 3, triangles, tets
    if verbose:
        print('  Dimension: %s-D' % dim)

    # creating instance of GIMLI::Mesh class
    mesh = pg.Mesh(dim)

    # replacing boundary markers (gmsh does not allow negative phys. regions)
    bound_marker = (pg.MARKER_BOUND_HOMOGEN_NEUMANN, pg.MARKER_BOUND_MIXED,
                    pg.MARKER_BOUND_HOMOGEN_DIRICHLET,
                    pg.MARKER_BOUND_DIRICHLET)

    if bounds.any():
        for i in range(4):
            bounds[:, dim][bounds[:, dim] == i + 1] = bound_marker[i]

        # account for CEM markers
        bounds[:, dim][bounds[:, dim] >= 10000] *= -1

        if verbose:
            bound_types = np.unique(bounds[:, dim])
            print('  Boundary types: %s ' % len(bound_types) + str(
                tuple(bound_types)))
    else:
        print("WARNING: No boundary conditions found.",
              "Setting Neumann on the outer edges by default.")

    if verbose:
        regions = np.unique(cells[:, dim + 1])
        print('  Regions: %s ' % len(regions) + str(tuple(regions)))

    for node in nodes:
        if dim == 2:
            mesh.createNode(node[0], node[3 - zero_dim], 0)
        else:
            mesh.createNode(node)

    for cell in cells:
        if dim == 2:
            mesh.createTriangle(
                mesh.node(int(cell[0] - 1)), mesh.node(int(cell[1] - 1)),
                mesh.node(int(cell[2] - 1)), marker=int(cell[3]))
        else:
            mesh.createTetrahedron(
                mesh.node(int(cell[0] - 1)), mesh.node(int(cell[1] - 1)),
                mesh.node(int(cell[2] - 1)), mesh.node(int(cell[3] - 1)),
                marker=int(cell[4]))

    mesh.createNeighbourInfos()

    # Set Neumann on outer edges by default (can be overriden by Gmsh info)
    for b in mesh.boundaries():
        if not b.leftCell() or not b.rightCell():
            b.setMarker(pg.MARKER_BOUND_HOMOGEN_NEUMANN)

    for bound in bounds:
        if dim == 2:
            mesh.createEdge(
                mesh.node(int(bound[0] - 1)), mesh.node(int(bound[1] - 1)),
                marker=int(bound[2]))
        else:
            mesh.createTriangleFace(
                mesh.node(int(bound[0] - 1)), mesh.node(int(bound[1] - 1)),
                mesh.node(int(bound[2] - 1)), marker=int(bound[3]))

    # assign marker to corresponding nodes (sensors, reference nodes, etc.)
    if points:
        try:
            points = np.c_[points, pointsList[pointsList[:,7] > 0][:nPoints,-1]]
            points = [[int(p_) for p_ in p] for p in points]
            for point in points:
                mesh.node(point[0] - 1).setMarker(-point[1])
        except ValueError:
            print("Something went wrong with setting point markers.")
    if verbose:
        if points:
            points = np.asarray(points)
            node_types = np.unique(points[:, 1])
            print('  Marked nodes: %s ' % len(points) + str(tuple(node_types)))
        print('\nDone. \n')
        print('  ' + str(mesh))
    return mesh

#fname = r"C:\Users\264401k\Downloads\gmsh-4.0.2-Windows64\gmsh-4.0.2-Windows64\PyGIMLI_CO2_Test.msh4"
fname = r"C:\Users\264401k\Downloads\inv1 (1).msh"
mesh = pg.meshtools.readGmsh(fname)

pg.show(mesh, markers = True, showMesh = True)
