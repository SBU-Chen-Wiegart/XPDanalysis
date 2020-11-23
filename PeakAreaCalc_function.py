# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:16:42 2020

@author: chozhao
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:45:56 2020

@author: chozhao
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#for testing purpose:
#beginAngle = 2.62  #beg_peak_Ti(1 1 0)
#endAngle = 2.72
testfile = 'C:\\Chen-WiegartGroup\\Chonghang\\Np-Ti\\Beamtime\\XPD2019Cycle2\\RawData\\FromGoogleDrive\\Ti30Cu70_460C_30min_47-1-4_3-10\\integration\\Ti30Cu70_460C_30min_47-1-4_3-10_20191125-132842_d4c4fd_0001_mean_q.chi'


def extractData(intextfile):
    I_q_data=pd.read_csv(intextfile, skiprows=8, delim_whitespace=True, encoding = 'unicode_escape', error_bad_lines=False, header= None)
    return I_q_data

def PeakAreaCal(data, roi, showplot = False, printValues = False):
 
    """
    input:
        data: float array 
            data[0] : angle or q
            data[1] : intensity
        roi: flow array, region of interest for the angle or q value
            roi [0] : lower bound of the angle or q value
            roi [1] : upper bound of the angle or q value
        showplot: plot I(q) vs. q and the roi selections, default = False
        printValues: print values used in the calculation
    
    output:
        area_sum (float): sum of the area under the peak
    """
    
    if showplot is True:
        plt.close('all')
        plt.plot(data[0],data[1])
        plt.plot([roi[0], roi[0]], [0, np.max(data[1])] )
        plt.plot([roi[1], roi[1]], [0, np.max(data[1])] )
        plt.show()
    
    
    ##readout row numbers with q value
    begin = data.iloc[(data[0]-roi[0]).abs().argsort()[:1]].index.tolist()[0]
    end = data.iloc[(data[0]-roi[1]).abs().argsort()[:1]].index.tolist()[0]
    
    #average the intensity of beginning and ending of peak
    average_begin = (data[1][begin]+data[1][begin-1]+data[1][begin-2])/3
    average_end = (data[1][end]+data[1][end+1]+data[1][end+2])/3
    
    #calculate the intensity of background
    background_sum  = (average_begin + average_end)*(roi[1]-roi[0])/2
    
    #calculate the intensity of peak
    dQ = ((data[0][end]-data[0][end-1])+(data[0][begin]-data[0][begin-1]))/2
    intensity_sum = sum(data[1][begin:end+1])*dQ
        
    #Calculate the peak area by minus background intensity from total intensity
    area_sum = intensity_sum - background_sum
    
    if printValues is True:
        print("average_begin = " + str(average_begin))
        print("average_end = " + str(average_end))
        print("background_sum = " + str(background_sum))
        print("intensity_sum = " + str(intensity_sum))
        print("area_sum = " +str(area_sum))
    
    return area_sum
    

