# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:38:29 2020

@author: chozhao
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

source = "C:\\Chen-WiegartGroup\\Chonghang\\Np-Ti\\Beamtime\\XPD2019Cycle2\\RawData\\DataAnalysis_0614_Calib1212\\Calib\\Sorted\\PeakAreaAnalysis\\ForPlot" 

### For looping through all files and ploting at one time
#file_list = next(os.walk(source))[2]
#for i in range(len(file_list)):
#    data=pd.read_excel(source + "\\" + file_list[i], delim_whitespace=True, encoding = "utf8", error_bad_lines=False, header= None, sheet_name = "Sheet1")
#    #Peak Area  
#    c = np.empty(0)
#    for j in range(2,11):
#        c = np.append(c, data[j][1:10])
#    
#    c=np.around(c.astype(np.double),3)
#    c = c*1000
#    c[c<0] = 0
#    ###########################################
#    ##data with dealloying temperature 
#    c_340 = np.split(c[0:27],3)
#    c_400 = np.split(c[27:54],3)
#    c_460 = np.split(c[54:81],3)
#    Time_point = [c_340, c_400, c_460]
#    Time_data = ['t_340', 't_400', 't_460']
#    #(e) plotting for each temperature point
#    for k in range(len(Time_point)-1):
#        if "CuMg2" in file_list[i]:
#            plt.imshow(Time_point[k], interpolation='bicubic',extent=[10, 90, 7.5, 60],origin='lower', cmap = 'cividis')   #align = 'center'
#        if "Cu2Mg" in file_list[i]:
#            plt.imshow(Time_point[k], interpolation='bicubic',extent=[10, 90, 7.5, 60],origin='lower', cmap = 'viridis')
#        if "Ti" in file_list[i]:
#            plt.imshow(Time_point[k], interpolation='bicubic',extent=[10, 90, 7.5, 60],origin='lower', cmap = 'plasma')
#        plt.colorbar()
#        plt.xlabel('Ti Composition (at.%)')
#        plt.ylabel('Dealloying Time (min)')
#        plt.axes().set_aspect(1)
#        plt.savefig(source +'\\'+file_list[i] +'_'+Time_data[k]+'_ZeroNeg'+'.jpg')
#        plt.close()
#    ####################
#    if "CuMg2" in file_list[i]:
#        plt.imshow(Time_point[2], interpolation='bicubic',extent=[10, 90, 7.5, 30],origin='lower', cmap = 'cividis')  ## Cu2Mg cmap = 'viridis'   Ti cmap = 'plasma' CuMg2 cmap= 'cividis'   #align = 'center'
#    if "Cu2Mg" in file_list[i]:
#        plt.imshow(Time_point[2], interpolation='bicubic',extent=[10, 90, 7.5, 30],origin='lower', cmap = 'viridis')  ## Cu2Mg cmap = 'viridis'   Ti cmap = 'plasma' CuMg2 cmap= 'cividis'   #align = 'center'
#    if "Ti" in file_list[i]:
#        plt.imshow(Time_point[2], interpolation='bicubic',extent=[10, 90, 7.5, 30],origin='lower', cmap = 'plasma')  ## Cu2Mg cmap = 'viridis'   Ti cmap = 'plasma' CuMg2 cmap= 'cividis'   #align = 'center'  
#    plt.colorbar()
#    plt.xlabel('Ti Composition (at.%)')
#    plt.ylabel('Dealloying Time (min)')
#    plt.axes().set_aspect(2)
#    plt.savefig(source +'\\'+file_list[i] +'_'+Time_data[2]+'_ZeroNeg'+'.jpg')
#    plt.close()


############################################
############################################

### for individual plotting
############################################
#file_name = "Output_Xpd_AreaCal_Cu2Mg(1 1 1)_673-686"
#file_name = "Output_Xpd_AreaCal_Cu2Mg(4 0 0)_1557-1574"
#file_name = "Output_Xpd_AreaCal_Cu2Mg(4 4 0)_2204-2221"
#file_name = "Output_Xpd_AreaCal_Cu2Mg(3 1 1)_1285_1309"
#file_name = "Output_Xpd_AreaCal_Cu2Mg(2 2 2)_1345_1364"
#file_name = "Output_Xpd_AreaCal_Ti(1 1 0)_1166-1184"
#file_name = "Output_Xpd_AreaCal_Ti(2 2 0)_2340-2360"
#file_name = "Output_Xpd_AreaCal_Ti(3 1 0)_2615-2641"
#file_name = "Output_Xpd_AreaCal_CuMg2(1 1 1)_620-630"
#file_name = "Output_Xpd_AreaCal_CuMg2(1 3 1)_750-760"
#file_name = "Output_Xpd_AreaCal_CuMg2(1 3 1)_750-760_BkgAverage1"
#file_name = "Output_Xpd_AreaCal_CuMg2(0 8 0)_1204-1223"
#file_name = "Output_Xpd_AreaCal_CuMg2(0 4 0)_597_610"
#file_name = "Output_Xpd_AreaCal_Ti(1 1 0)_1166-1184_5050From340-400"
#file_name = "Output_Xpd_AreaCal_Ti(2 2 0)_2340-2360_5050From340-400"
#file_name = "Output_Xpd_AreaCal_Ti(3 1 0)_2615-2641_5050From340-400"
file_name = "Output_Xpd_AreaCal_Ti(3 1 0)_2615-2641_5050From340-400"
data=pd.read_excel(source + "\\" + file_name+'.xlsx', delim_whitespace=True, encoding = "utf8", error_bad_lines=False, header= None, sheet_name = "Sheet1")
#Peak Area  
c = np.empty(0)
for j in range(2,11):
    c = np.append(c, data[j][1:10])

c=np.around(c.astype(np.double),3)
c = c*1000
c[c<0] = 0
###########################################
##data with dealloying temperature 
c_340 = np.split(c[0:27],3)
c_400 = np.split(c[27:54],3)
c_460 = np.split(c[54:81],3)
Time_point = [c_340, c_400, c_460]
Time_data = ['t_340', 't_400', 't_460']
#(e) plotting for each temperature point
for k in range(len(Time_point)-1):
    if "CuMg2" in file_name:
        plt.imshow(Time_point[k], interpolation='bicubic',extent=[10, 90, 7.5, 60],origin='lower', cmap = 'cividis')   #align = 'center'
    if "Cu2Mg" in file_name:
        plt.imshow(Time_point[k], interpolation='bicubic',extent=[10, 90, 7.5, 60],origin='lower', cmap = 'viridis')
    if "Ti" in file_name:
        plt.imshow(Time_point[k], interpolation='bicubic',extent=[10, 90, 7.5, 60],origin='lower', cmap = 'plasma')
    plt.colorbar()
    plt.xlabel('Ti Composition (at.%)')
    plt.ylabel('Dealloying Time (min)')
    plt.axes().set_aspect(1)
    plt.savefig(source +'\\'+file_name +'_'+Time_data[k]+'_ZeroNeg'+'.jpg')
    plt.close()
####################
if "CuMg2" in file_name:
    plt.imshow(Time_point[2], interpolation='bicubic',extent=[10, 90, 7.5, 30],origin='lower', cmap = 'cividis')  ## Cu2Mg cmap = 'viridis'   Ti cmap = 'plasma' CuMg2 cmap= 'cividis'   #align = 'center'
if "Cu2Mg" in file_name:
    plt.imshow(Time_point[2], interpolation='bicubic',extent=[10, 90, 7.5, 30],origin='lower', cmap = 'viridis')  ## Cu2Mg cmap = 'viridis'   Ti cmap = 'plasma' CuMg2 cmap= 'cividis'   #align = 'center'
if "Ti" in file_name:
    plt.imshow(Time_point[2], interpolation='bicubic',extent=[10, 90, 7.5, 30],origin='lower', cmap = 'plasma')  ## Cu2Mg cmap = 'viridis'   Ti cmap = 'plasma' CuMg2 cmap= 'cividis'   #align = 'center'  
plt.colorbar()
plt.xlabel('Ti Composition (at.%)')
plt.ylabel('Dealloying Time (min)')
plt.axes().set_aspect(2)
plt.savefig(source +'\\'+file_name +'_'+Time_data[2]+'_ZeroNeg'+'.jpg')
plt.close()
