###############################################################################
#
# ATTENTION:
# ----------
#
# The station numbers MUST currently be exactly the same 
# as the values for the X-coordinate in the grid file.
#
#
# DESCRIPTION:
# ------------
# 
# Converts a regular XYZD ascii grid (e.g. export from Surfer) to a flat-file
# format, which can be read by OpendTect for subsequent conversion to SEGY.
# 
# The output file format is as follows:
#
#     Easting1, Northing1, Value1, Value2, ..., ValueN
#     Easting2, Northing2, Value1, Value2, ..., ValueN
#     ...
#     EastingM, NorthingM, Value1, Value2, ..., ValueN
#
# for M coordinates and data values at N depth levels.
# 
# The output for OpendTect has one header line with the grid info:
# surface level, depth level spacing, number of depth levels.
#
# Optional, additional column headers and data columns can be written:
# station number in the first column and topography in the 4th column.
#
# 
# USER INPUT:
# -----------
# 
# 1) The path to the working directory has to be set. `
# 2) The ascii grid file has to be specified.
# 3) The data file with GPS coordinates has to be specified
# 4) The output file name has to be specified.
# 5) Specify a value flagging invalid numbers (e.g. 1.0e30)
# 6) Optional: data file with full headers can be written
#    (not compatible with OpendTect)
#
# NOTES:
# ------
#
# The GPS data file has to be without headers in XYZ format, e.g. as follows:
#
#       0    657531.93150    5733793.31810    48.70
#       5    657536.46880    5733793.45090    48.60
#      10    657541.28720    5733794.57740    48.50
#      ...
#
# where the 1st column contains the station number followed by Easting,  
# Northing and Elevation (station number must exactly match the grid x-values)
#
#
# AUTHOR:
# --------
# Ralf Schaa, Curtin University 2016
#
###############################################################################

import os
import sys
import numpy
import collections


# -----------------------------------------------------------------------------
# REQUIRED USER INPUT:
# -----------------------------------------------------------------------------

# 
# <NOTE> The GPS coordinates must currently coincide with the first grid column 
#

# Set the working folder where the data files are:
work_dir = r'z:\_Curtin\_Projects\_CO2CRC\2016-04_Otway_ERI\_Data\ERI_Data\21-04-2016\DIPDIP\Surfer\SEGY_Prep'

# Regular ascii grid (XYZD) file as ouptut from Sufer:
grid_file = r'Otway_Filt_21042016_L2_DPDP_IP_Joint_modip_ascii_grid.dat' 

# GPS file
gps_file = r'z:\_Curtin\_Projects\_CO2CRC\2016-04_Otway_ERI\_Data\ERI_Data\21-04-2016\DIPDIP\Surfer\SEGY_Prep\Otway_GPS_LINE2.dat'

# Output filename:
out_file = r'segy_prep.dat'

# Invalid data values (usually > 1.0e30 in Res2DInv:
INVALID = 1.0e30

# Optional (default=False): this writes header lines and additional data columns (not OpendTect compatible):
WRITE_HEAD = False
   

# -----------------------------------------------------------------------------
# READ FILES INTO NUMPY ARRAYS
# -----------------------------------------------------------------------------

# Change to the working directory:
os.chdir( work_dir )

# Open the ASCII GRID file, assuming uniform grid:
with open( grid_file ) as f:
    
    # Prepare work arrays for storing the data while parsing:
    xgrd = []
    zgrd = []
    vgrd = []
    
    # Cycle lines and read into the work arrays:
    for line in f:
        # Read the XYZ data of the current line:         
        xn,zn,vn = numpy.fromstring(line, dtype=float, sep=" ")
        # Append as lists:
        xgrd.append(xn), zgrd.append(zn), vgrd.append(vn) 
# Reading done!
f.close()   



# Open the GPS file -- Format: Station (Electrode), Easting, Northing, Elevation:
# (Stations must match those in the ASCII Grid file)
with open( gps_file ) as f:
    
    # Save as ordered dictionary with stations as keys:
    gps = collections.OrderedDict()
    
    # Cycle lines and read into the work arrays:
    for line in f:
               
        # Read the XYZ data of the current line:         
        try:
            sn,xn,zn,vn = numpy.fromstring(line, dtype=float, sep=" ")
        except Exception:
            continue
        # Save as lists with station as key:
        if WRITE_HEAD:
            gps[ sn ] = (xn,zn,vn)
        else:
            gps[ sn ] = (xn,zn)    
        
# Reading done!
f.close()   



# -----------------------------------------------------------------------------
# INTERPOLATE GPS COORDINATES  
# -----------------------------------------------------------------------------

# not yet implemented
#---------------------

# Find EXACT GPS stations at grid nodes (station number must be equal):
stn_mask = [i for i, item in enumerate(gps.keys()) if item not in set(xgrd)]
if len(stn_mask)  <= 1:
    print("Error: At least two GPS station number (1st column) must be equal to the data station number (1st column)")
    sys.exit()

    



# -----------------------------------------------------------------------------
# SORT ACCORDING TO DEPTHS 
# -----------------------------------------------------------------------------

# (like an Excel sheet where values are sorted according to two columns)

# Convert to numpy arrays first:
xgrd = numpy.array(xgrd)
zgrd = numpy.array(zgrd)
vgrd = numpy.array(vgrd)

# Get the sort-indices -- sort first w.r.t Z, start at the surface, and then w.r.t. X:
sorter = numpy.lexsort((-zgrd,xgrd))

# The values sorted here with respect to depth and station:
sorted_values = vgrd[ sorter ]

# Number of depth levels, starting at the surface:
depth_levels = numpy.unique( zgrd  )[::-1]




# -----------------------------------------------------------------------------
# WRITE OUT 
# -----------------------------------------------------------------------------

# Open the file for writing ..
with file(out_file,'w') as outf:

    # Optional extra info:
    if WRITE_HEAD:
        # Write a header line with the depth levels in one row:    
        outf.write( '/' +  ('{:>13s}'.format("\t") )*3 + '{:>13s}'.format("DEPTHS:\t") + 
                   '\t'.join('{:>12.5f}'.format(d) for d in depth_levels ) + '\n' )        
        # Write a column header:
        outf.write( '/' +  (('{:>12s}\t')*4).format( *('STATION','EASTING','NORTHING','ELEV') ) )  
        outf.write( '\t'.join('{:>12s}'.format( 'COND' + str(i+1) ) for i in xrange(len(depth_levels)) ) + '\n' )
        outf.write( '/' +  '\n' )
        
    # Always write header line with grid information:
    surface = '{:<.2f} '.format( depth_levels[0] ) 
    spacing = '{:<.2f} '.format( abs(depth_levels[1]-depth_levels[0] ))
    nlevels = '{:<d} \t'.format( len(depth_levels) )
    outf.write( surface +  spacing + nlevels + "\n")
    
    # Get the GPS coordinates corresponding to this station:
    gps_out = ('\t'.join('{:>12.5f}'.format(g) for g in gps[ xgrd[ sorter[0] ]] ))
    if WRITE_HEAD:
        # Start with writing the first station ...
        outf.write( '{:>12.5f}'.format(xgrd[ sorter[0] ]) + "\t" + gps_out + "\t")
    else:
        outf.write( gps_out + "\t" )
    
    # Cycle the sort-indices and write the rest of the file:
    for i in xrange(len(sorter)):       
        
        if i > 0:
            # Compare current with previous station:
            if xgrd[ sorter[i] ] != xgrd[ sorter[i-1] ]:
                # Get the GPS coordinates corresponding to this station:
                gps_out = ('\t'.join('{:>12.5f}'.format(g) for g in gps[ xgrd[ sorter[i] ]] ))                
                if WRITE_HEAD:
                    # Write the station (also break the line):
                    outf.write( "\n" + '{:>12.5f}'.format(xgrd[ sorter[i] ]) + "\t" + gps_out + "\t")
                else:
                    outf.write( "\n" + gps_out + "\t" )
                                           
        # Get the 'sorted' index:
        j = sorter[ i ]
                        
        # Save string with coordinates and corresponding grid values:        
        if vgrd[ j ] > INVALID:
            v_out = '{:>12s}'.format('*')  + "\t" 
        else:
            v_out = '{:>12.5f}'.format(vgrd[ j ]) + "\t"
            
        # Write to file:
        outf.write( v_out )
            

print('All Done')            
