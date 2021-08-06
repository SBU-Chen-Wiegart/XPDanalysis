"""
File: Autonomous experiments of the dealloyed Ti-Cu metal
Name: Cheng-Chu Chung
----------------------------------------
TODO: Simulate gpCAM autonomous data using different searching conditions
"""
import matplotlib.pyplot as plt
import numpy as np
from mayavi import mlab
from skimage import io
import pandas as pd
from dataclasses import dataclass, asdict, astuple, field
from collections import namedtuple, defaultdict
from databroker._drivers.msgpack import BlueskyMsgpackCatalog
import time
import json
from gpcam.gp_optimizer import GPOptimizer
from numpy.random import default_rng
from scipy.interpolate import CubicSpline, interpn, griddata, Rbf
import plotly.graph_objects as go
TransformPair = namedtuple("TransformPair", ["forward", "inverse"])


def main():
    strip_list = load_from_json("layout.json")
    pair = single_strip_set_transform_factory(strip_list)
    xca_db = BlueskyMsgpackCatalog(['D:/Software/Python/SSID/XPD_20201207_TiCuMg_alloy_auto_1/adaptive_reduced/xca/*msgpack'])
    gpcam_db = BlueskyMsgpackCatalog(
        ['D:/Software/Python/SSID/XPD_20201207_TiCuMg_alloy_auto_1/adaptive_reduced/gpcam/*msgpack'])
    grid_db = BlueskyMsgpackCatalog(
        ['D:/Software/Python/SSID/XPD_20201207_TiCuMg_alloy_auto_1/adaptive_reduced/grid/*msgpack'])
    grid_missing = BlueskyMsgpackCatalog(
        ['D:/Research data/SSID/202105/grid_missing/*msgpack']
    )
    # print('Scanning numbers:', len(grid_db))
    # thick_measurements = list(grid_db.search({'adaptive_step.snapped.ctrl_thickness': 1}))
    # thin_measurements = list(grid_db.search({'adaptive_step.snapped.ctrl_thickness': 0}))
    # print('Thick samples:', len(thick_measurements))
    # print('Thin samples:', len(thin_measurements))
    #################################################################### Run the data
    # check_scan_id_and_CPU_time(gpcam_db)
    # plot_gpcam(gpcam_db, pair, grid_db, grid_missing)
    # plot_temperature(gpcam_db)
    # plot_annealing_time(gpcam_db)
    # plot_roi(gpcam_db)
    plot_grid_data(grid_db, grid_missing, pair)
    # plot_xca_from_karen(grid_db, pair, grid_missing, peak_location=(2.635, 2.708))
    # grid_list(grid_db, pair, grid_missing)
    # ti_, temp_, time_, thickness_ = pair[1](68.5, 31.95)
    # print(ti_, temp_, time_, thickness_)
    # plot_xca(xca_db, pair, grid_db, grid_missing, peak_loc=(1.526, 1.588))
    # plot_xca_from_philip(grid_db, pair, grid_missing, peak_location=(2.734, 2.779))
    # grid_interpolation()
    # dic = plot_grid_data(grid_db, grid_missing, pair)
    # grid_vis = grid_3d(dic)
    """
    Cu2Mg(311): (2.925, 2.974)
    Cu2Mg(111): (1.526, 1.588)
    CuMg2(080): (2.734, 2.779)
    Cu2Mg(222): (3.047, 3.106)
    Beta Ti(110): (2.635, 2.708)
    """
    #################################################################### Simulate gpCAM
    # version_6_of_gpCAM(gpcam_db, pair)
    #################################################################### Run a specific code
    # the_last_scan = -1    # Scan from the last scan
    # result = xca_db[the_last_scan]     # Extract data from a scan_id
    # print(result.metadata['start']['batch_scan']['points'])
    # for i in result.metadata['start']:
    #     print(i, result.metadata['start'][i])
    # otime = result.metadata['start']['original_start_time']
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(otime)))
    #################################################################### Convert the coordinates based on Json file
    # print(pair[0](result.metadata['start']['adaptive_step']['snapped']['ctrl_Ti'],
    #               result.metadata['start']['adaptive_step']['snapped']['ctrl_temp'],
    #               result.metadata['start']['adaptive_step']['snapped']['ctrl_annealing_time'],
    #               result.metadata['start']['adaptive_step']['snapped']['ctrl_thickness']))
    # print(pair[1](94.5, 79.45))
    # print(pair[1](92.25, 79.45))
    # print(pair[1](90, 79.45))
    # print(pair[1](87.75, 79.45))
    # print(pair[1](28.5, 79.45))
    # print(pair[1](94.75, 60.45))
    # print(pair[1](88, 31.95))
    # print(grid_interpolation(94.5, 3.45))
    ##################################################################### Read data
    # print(pair[0](51, 340, 450, 0))
    # for i in load_from_json('layout.json'):
    #     words = str(i)
    #     print(words)
    #     index_ti = words.find(']')
    #     print(words[index_ti-2:index_ti])
    #     index_temp = words.find(',')
    #     print(words[index_temp-3:index_temp])
    #     index_time = words.find(',', index_temp + 1)
    #     index_equal = words.find('=', index_temp + 1)
    #     print(words[index_equal+1:index_time])
    #     print(words[-2])
    # number = np.linspace(1, 6, 10)
    # print(number)
    # X, Y, Z = np.mgrid[-1:1:30j, -1:1:30j, -1:1:30j]
    # X = np.linspace(-1, 1, 30)
    # Y = np.linspace(-1, 1, 30)
    # Z = np.linspace(-1, 1, 30)
    # values = np.sin(np.pi * X) * np.cos(np.pi * Z) * np.sin(np.pi * Y)
    #
    # fig = go.Figure(data=go.Volume(
    #     x=X.flatten(),
    #     y=Y.flatten(),
    #     z=Z.flatten(),
    #     value=values.flatten(),
    #     isomin=-0.1,
    #     isomax=0.8,
    #     opacity=0.1,  # needs to be small to see through all surfaces
    #     surface_count=21,  # needs to be a large number for good volume rendering
    # ))
    # fig.show()


def plot_xca(xca_db, pair, grid_db, grid_missing, peak_loc=(2.925, 2.974)):
    """
    Cu2Mg(311): (2.925, 2.974)
    Cu2Mg(111): (1.526, 1.588)
    CuMg2(080): (2.734, 2.779)
    Cu2Mg(222): (3.047, 3.106)
    Beta Ti(110): (2.635, 2.708)
    :param xca_db:
    :param pair:
    :param grid_db:
    :return:
    """
    print('Original start time, Ti, temp, time, roi, thickness, beamline x, beamline y and phase')
    time_list = []
    initial_ti = 0.0  # To sort out the XCA searching phase
    index = 0   # To sort out the XCA searching phase
    phase = 'None'  # To sort out the XCA searching phase
    inner_index = 0     # To count the inner points
    peak_location = peak_loc
    P1_P2_list = ["Ti", "Ti", "Ti", "MgCu2", "Ti", "Ti", "Ti", "Ti", "Ti", "Mg2Cu",
                  "Ti", "Mg2Cu", "Mg2Cu", "Mg2Cu", "Ti", "Ti", "MgCu2", "Ti", "MgCu2", "MgCu2",
                  "Ti", "Ti", "MgCu2", "MgCu2", "MgCu2", "Ti", "Ti",
                  "Ti", "Ti", "Ti", "Ti", "Ti", "Ti", "Mg2Cu", "Mg2Cu", "MgCu2", "MgCu2",
                  "Ti", "Ti", "Ti", "Mg2Cu", "MgCu2", "Ti", "Ti", "Ti", "MgCu2", "MgCu2",
                  "MgCu2", "MgCu2", "Ti", "MgCu2", "MgCu2", "MgCu2", "MgCu2"]
    for i in range(1, len(xca_db) + 1):  # Extract all information from metadata['start']
        result = xca_db[-i]
        points = result.metadata['start']['batch_scan']['points']
        ti_range = result.metadata['start']['batch_scan']['ti_range']
        otime = result.metadata['start']['original_start_time']  # Assign original_start_time as otime
        ti = result.metadata['start']['center_point'][0]
        if ti != initial_ti:    # Be careful this ti, it only works when the next ti is different
            phase = P1_P2_list[index]
            index += 1
            # if index % 3 == 1:
            #     phase = 'MgCu2'
            # elif index % 3 == 2:
            #     phase = 'Ti'
            # else:
            #     phase = 'Mg2Cu'
            initial_ti = ti
            inner_index = 0
        else:
            inner_index += 1
            inner_increment = np.linspace(0, 0+ti_range, points)[1] - np.linspace(0, 0+ti_range, points)[0]
            ti += inner_increment*inner_index
            # print(ti)
        temp = result.metadata['start']['center_point'][1]
        annealing_time = result.metadata['start']['center_point'][2]
        q, I = result.primary.read()['q'].mean('time'), result.primary.read()['mean'].mean('time')
        # compute_total_area(q, I)  # Compute the total area of the spectrum
        roi = compute_peak_area(q, I, *peak_location)
        roi = np.array([roi])
        thickness = result.metadata['start']['center_point'][3]
        for chs in load_from_json('layout.json'):   # To find the maximum Ti% edge and over value would be initial Ti%
            words = str(chs)    # Check the layout file and you will know the character
            index_ti = words.find(']')
            index_temp = words.find(',')
            index_time = words.find(',', index_temp + 1)
            index_equal = words.find('=', index_temp + 1)
            if temp == int(words[index_temp - 3:index_temp])\
            and annealing_time == int(words[index_equal + 1:index_time])\
            and thickness == int(words[-2]) \
            and ti > int(words[index_ti - 2:index_ti]):     # If Ti% beyond the maximum
                ti = result.metadata['start']['center_point'][0]
                # ti = result.metadata['start']['center_point'][0] + np.linspace(ti, ti + ti_range, points)[
                #     points - total_points + i - 1
                #     ] - ti
        beamline_x, beamline_y = pair[0](ti, temp, annealing_time, thickness)

        # Append (Original start start time, Ti, Temp, Annealing time, roi, thickness, beamline_x, beamline_y, phase)
        time_list.append((time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(otime)),
                          ti, temp, annealing_time, roi[0], thickness, beamline_x, beamline_y, phase))
    time_list.sort()

    y_axis = []  # Ti concentration
    y_axis_temp = []  # Annealing temperature
    z_axis_time = []  # Annealing time
    intensity_roi = []  # Region of interest
    beamline_x_axis = np.array([])  # Beamline x position
    beamline_y_axis = np.array([])  # Beamline y position
    beamline_x_axis_MgCu2 = np.array([])
    beamline_y_axis_MgCu2 = np.array([])
    x_axis_ti_MgCu2 = []
    y_axis_temp_MgCu2 = []
    y_axis_time_MgCu2 = []
    intensity_roi_MgCu2 = []
    beamline_x_axis_Ti = np.array([])
    beamline_y_axis_Ti = np.array([])
    x_axis_ti_Ti = []
    y_axis_temp_Ti = []
    y_axis_time_Ti = []
    intensity_roi_Ti = []
    beamline_x_axis_Mg2Cu = np.array([])
    beamline_y_axis_Mg2Cu = np.array([])
    x_axis_ti_Mg2Cu = []
    y_axis_temp_Mg2Cu = []
    y_axis_time_Mg2Cu = []
    intensity_roi_Mg2Cu = []

    for j in range(len(time_list)):
        # Print (Original start start time, Ti, Temp, Annealing time, roi, thickness, beamline_x, beamline_y, phase)
        print('{}, {:.1f}, {}, {:4}, {:7.4f}, {}, {:6.4f}, {:7.4f}, {}'.format(time_list[j][0], time_list[j][1],
                                                                           time_list[j][2], time_list[j][3],
                                                                           time_list[j][4], time_list[j][5],
                                                                           time_list[j][6], time_list[j][7],
                                                                               time_list[j][8]))
        # x_axis.append(j)
        y_axis.append(time_list[j][1])  # Ti concentration
        y_axis_temp.append(time_list[j][2])
        z_axis_time.append(time_list[j][3])
        intensity_roi.append(time_list[j][4])
        beamline_x_axis = np.append(beamline_x_axis, time_list[j][6])
        beamline_y_axis = np.append(beamline_y_axis, time_list[j][7])
        if time_list[j][8] == 'MgCu2':
            beamline_x_axis_MgCu2 = np.append(beamline_x_axis_MgCu2, time_list[j][6])
            beamline_y_axis_MgCu2 = np.append(beamline_y_axis_MgCu2, time_list[j][7])
            x_axis_ti_MgCu2.append(time_list[j][1])
            y_axis_temp_MgCu2.append(time_list[j][2])
            y_axis_time_MgCu2.append(time_list[j][3])
            intensity_roi_MgCu2.append(time_list[j][4])
        elif time_list[j][8] == 'Ti':
            beamline_x_axis_Ti = np.append(beamline_x_axis_Ti, time_list[j][6])
            beamline_y_axis_Ti = np.append(beamline_y_axis_Ti, time_list[j][7])
            x_axis_ti_Ti.append(time_list[j][1])
            y_axis_temp_Ti.append(time_list[j][2])
            y_axis_time_Ti.append(time_list[j][3])
            intensity_roi_Ti.append(time_list[j][4])
        elif time_list[j][8] == 'Mg2Cu':
            beamline_x_axis_Mg2Cu = np.append(beamline_x_axis_Mg2Cu, time_list[j][6])
            beamline_y_axis_Mg2Cu = np.append(beamline_y_axis_Mg2Cu, time_list[j][7])
            x_axis_ti_Mg2Cu.append(time_list[j][1])
            y_axis_temp_Mg2Cu.append(time_list[j][2])
            y_axis_time_Mg2Cu.append(time_list[j][3])
            intensity_roi_Mg2Cu.append(time_list[j][4])

    grid_array = grid_list(grid_db, pair, grid_missing, peak_loc=peak_location)     # Call the grid scan background
    # print(grid_array)
    map_thin = np.empty(
        [8, 34])  # We were supposed to have 18X34, but the last two rows didn't be measured due to beam down
    for row in range(8):
        for column in range(34):
            map_thin[row, column] = grid_array[34 * row + column][2]  # Scan from left to right, down to upper
    map_thick = np.empty(
        [9, 34])  # We were supposed to have 18X34, but the last two rows didn't be measured due to beam down
    for row in range(9):
        for column in range(34):
            map_thick[row, column] = grid_array[34 * (row + 9) + column][2]  # Scan from left to right, down to upper
    # Plot the grid scan background
    plt.imshow(map_thick, vmin=-0.07, vmax=0.37, interpolation='bicubic', extent=[94.5, 28.5, 41.45, 3.45],
               origin='lower',
               cmap='plasma')  # Y range = 41.45 to 12.95 mm, vmin=-0.07, vmax=0.37, vmin=-0.00007, vmax=0.00045,
    plt.imshow(map_thin, vmin=-0.07, vmax=0.37, interpolation='bicubic', extent=[94.5, 28.5, 84.2, 50.95],
               origin='lower',
               cmap='plasma')  # Y range = 84.2 to 50.95 mm  vmin=-0.07, vmax=0.37, vmin=-0.00007, vmax=0.00045,
    plt.text(83, 47.5,
             'Ni standard on glass slide',
             fontsize=12)
    cbar = plt.colorbar()
    cbar.set_label('Region of interest (Roi)', size=12)
    ################################################## Plot scattering plot
    """
    Cu2Mg(311): (2.925, 2.974)
    Cu2Mg(111): (1.526, 1.588)
    CuMg2(080): (2.734, 2.779)
    Cu2Mg(222): (3.047, 3.106)
    Beta Ti(110): (2.635, 2.708)
    """
    if peak_location == (1.526, 1.588):
        plt.scatter(beamline_x_axis_MgCu2, beamline_y_axis_MgCu2, marker='^', s=32, edgecolor='w', facecolor='None',
                    label='$\mathregular{Cu_2Mg}$', linestyle='-', linewidth=0.5)
        plt.title('Crystallography companion agent (XCA) scan\n'
                  '1.547 ($\mathregular{Cu_2Mg}$) (1 1 1)\n'
                  'Peak location = [1.526, 1.588]')
    if peak_location == (2.635, 2.708):
        plt.scatter(beamline_x_axis_Ti, beamline_y_axis_Ti, marker='v', s=32, edgecolor='w', facecolor='None',
                    label=r'$\mathregular{Ti}$', linestyle='-', linewidth=0.5)
        plt.title('Crystallography companion agent (XCA) scan\n'
                  r'2.665 ($\mathregular{\beta-Ti}$) (1 1 0)''\n'
                  'Peak location = [2.635, 2.708]')
    if peak_location == (2.734, 2.779):
        plt.scatter(beamline_x_axis_Mg2Cu, beamline_y_axis_Mg2Cu, marker='o', s=32, edgecolor='w', facecolor='None',
                    label='$\mathregular{CuMg_2}$', linestyle='-', linewidth=0.5)
        plt.title('Crystallography companion agent (XCA) scan\n'
                  '2.755 ($\mathregular{CuMg_2}$) (0 8 0)\n'
                  'Peak location = [2.734, 2.779]')
    if peak_location == (2.925, 2.974):
        plt.scatter(beamline_x_axis_MgCu2, beamline_y_axis_MgCu2, marker='^', s=32, edgecolor='w', facecolor='None',
                    label='$\mathregular{Cu_2Mg}$', linestyle='-', linewidth=0.5)
        plt.scatter(beamline_x_axis_Mg2Cu, beamline_y_axis_Mg2Cu, marker='o', s=32, edgecolor='w', facecolor='None',
                    label='$\mathregular{CuMg_2}$', linestyle='-', linewidth=0.5)
        plt.title('Crystallography companion agent (XCA) scan\n'
                  '2.965 ($\mathregular{Cu_2Mg}$) (3 1 1) or 2.9499 ($\mathregular{CuMg_2}$) (3 5 1)\n'
                  'Peak location = [2.925, 2.974]')
    if peak_location == (3.047, 3.106):
        plt.scatter(beamline_x_axis_MgCu2, beamline_y_axis_MgCu2, marker='^', s=32, edgecolor='w', facecolor='None',
                    label='$\mathregular{Cu_2Mg}$', linestyle='-', linewidth=0.5)
        plt.scatter(beamline_x_axis_Mg2Cu, beamline_y_axis_Mg2Cu, marker='o', s=32, edgecolor='w', facecolor='None',
                    label='$\mathregular{CuMg_2}$', linestyle='-', linewidth=0.5)
        plt.title('Crystallography companion agent (XCA) scan\n'
                  '3.093 ($\mathregular{Cu_2Mg}$) (2 2 2) or 3.095 ($\mathregular{CuMg_2}$) (4 4 0)\n'
                  'Peak location = [3.047, 3.106]')

    plt.ylim(84.2, 3.45)
    plt.xlabel('X position (mm)', fontsize=12)
    plt.ylabel('Y position (mm)', fontsize=12)
    plt.legend(loc='upper left')
    plt.show()
    ########################################## Plot 3D visualization (Ti concentration, Annealing temperature and time)
    x = np.array(y_axis)
    y = np.array(y_axis_temp)
    z = np.array(z_axis_time)
    intensity = np.array(intensity_roi)
    # print(intensity)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Ti concentration')
    ax.set_ylabel('Annealing temperature')
    ax.set_zlabel('Annealing time')
    p = ax.scatter3D(x, y, z, c=intensity, marker='o', s=32, label='Acquiring data')
    cbar = fig.colorbar(p, ax=ax, pad=0.2)
    cbar.set_label('Region of interest (Roi)', size=12)
    plt.title('Crystallography companion agent (XCA) scan\n'
              '2.965 ($\mathregular{Cu_2Mg}$) (3 1 1) or 2.9499 ($\mathregular{CuMg_2}$) (3 5 1)\n'
              'Peak location = [2.925, 2.974]')
    plt.legend()
    plt.show()


def plot_xca_from_karen(grid_db, pair, grid_missing, peak_location=(2.925, 2.974)):
    df = pd.read_csv('output_probabilities.csv')
    xca_data = np.array(df)
    P0 = np.array([])
    P1 = np.array([])
    Probability_of_mgcu2 = np.array([])
    Probability_of_ti = np.array([])
    Probability_of_mg2cu = np.array([])
    parameters = np.array([])
    for i in range(len(xca_data)):
        P0 = np.append(P0, xca_data[i][1])
        P1 = np.append(P1, xca_data[i][2])
        Probability_of_mgcu2 = np.append(Probability_of_mgcu2, xca_data[i][3])
        Probability_of_ti = np.append(Probability_of_ti, xca_data[i][4])
        Probability_of_mg2cu = np.append(Probability_of_mg2cu, xca_data[i][5])
    ################################
    """
        Cu2Mg(311): (2.925, 2.974)
        Cu2Mg(111): (1.526, 1.588)
        CuMg2(080): (2.734, 2.779)
        Cu2Mg(222): (3.047, 3.106)
        Beta Ti(110): (2.635, 2.708)
    """
    grid_array = grid_list(grid_db, pair, grid_missing, peak_loc=peak_location)  # Call the grid scan background
    print(grid_array)
    map_thin = np.empty(
        [8, 34])  # We were supposed to have 18X34, but the last two rows didn't be measured due to beam down
    for row in range(8):
        for column in range(34):
            map_thin[row, column] = grid_array[34 * row + column][2]  # Scan from left to right, down to upper
    map_thick = np.empty(
        [9, 34])  # We were supposed to have 18X34, but the last two rows didn't be measured due to beam down
    for row in range(9):
        for column in range(34):
            map_thick[row, column] = grid_array[34 * (row + 9) + column][2]  # Scan from left to right, down to upper
    # Plot the grid scan background
    plt.imshow(map_thick, vmin=-0.07, vmax=0.37, interpolation='bicubic', extent=[94.5, 28.5, 41.45, 3.45],
               origin='lower',
               cmap='plasma')  # Y range = 41.45 to 12.95 mm, vmin=-0.07, vmax=0.37, vmin=-0.00007, vmax=0.00045,
    plt.imshow(map_thin, vmin=-0.07, vmax=0.37, interpolation='bicubic', extent=[94.5, 28.5, 84.2, 50.95],
               origin='lower',
               cmap='plasma')  # Y range = 84.2 to 50.95 mm  vmin=-0.07, vmax=0.37, vmin=-0.00007, vmax=0.00045,
    plt.text(85, 47.5,
             'Ni standard on glass slide',
             fontsize=12)
    cbar = plt.colorbar(pad=0.1)
    cbar.set_label('Region of interest (Roi)', size=12)
    ################################
    if peak_location == (1.526, 1.588):
        plt.scatter(P0, P1, c=Probability_of_mgcu2, marker='o', s=32,
                    label='MgCu2')
        # plt.gca().invert_xaxis()  # Invert the x axis
        # plt.gca().invert_yaxis()  # Invert the y axis
        plt.clim(0.00, 0.6)
        plt.ylim(84.2, 3.45)
        plt.xlabel('X position (mm)', fontsize=12)
        plt.ylabel('Y position (mm)', fontsize=12)
        cbar = plt.colorbar()
        cbar.set_label('Output probabilities', size=12)
        plt.title('1.547 ($\mathregular{Cu_2Mg}$) (1 1 1)\n'
                  'Peak location = [1.526, 1.588]')
        plt.show()
    if peak_location == (2.635, 2.708):
        plt.scatter(P0, P1, c=Probability_of_ti, marker='o', s=32,
                    label='Ti')
        # plt.gca().invert_xaxis()  # Invert the x axis
        # plt.gca().invert_yaxis()  # Invert the y axis
        plt.clim(0.00, 0.6)
        plt.ylim(84.2, 3.45)
        plt.xlabel('X position (mm)', fontsize=12)
        plt.ylabel('Y position (mm)', fontsize=12)
        cbar = plt.colorbar()
        cbar.set_label('Output probabilities', size=12)
        plt.title(r'2.665 ($\mathregular{\beta-Ti}$) (1 1 0)''\n'
                  'Peak location = [2.635, 2.708]')
        plt.show()
    if peak_location == (2.734, 2.779):
        plt.scatter(P0, P1, c=Probability_of_mg2cu, marker='o', s=32,
                    label='Mg2Cu')
        # plt.gca().invert_xaxis()  # Invert the x axis
        # plt.gca().invert_yaxis()  # Invert the y axis
        plt.clim(0.00, 0.6)
        plt.ylim(84.2, 3.45)
        plt.xlabel('X position (mm)', fontsize=12)
        plt.ylabel('Y position (mm)', fontsize=12)
        cbar = plt.colorbar()
        cbar.set_label('Output probabilities', size=12)
        plt.title('2.755 ($\mathregular{CuMg_2}$) (0 8 0)\n'
                  'Peak location = [2.734, 2.779]')
        plt.show()


def plot_xca_from_philip(grid_db, pair, grid_missing, peak_location=(2.925, 2.974)):
    df = pd.read_csv('D:/Software/Python/SSID/XCA Tom re-run/fully_fixed_by Philip/output_proposals_fully_fixed_cc.csv')
    group = df.groupby("phase_of_interest")
    print(list(group))
    print('-----------------------')
    prb_mgcu2_x = np.array(list(group)[1][1]['x'])
    prb_mgcu2_y = np.array(list(group)[1][1]['y'])
    prb_mgcu2 = np.array(list(group)[1][1]['Prby MgCu2'])
    prb_ti_x = np.array(list(group)[2][1]['x'])
    prb_ti_y = np.array(list(group)[2][1]['y'])
    prb_ti = np.array(list(group)[2][1]['Prby Ti'])
    prb_mg2cu_x = np.array(list(group)[0][1]['x'])
    prb_mg2cu_y = np.array(list(group)[0][1]['y'])
    prb_mg2cu = np.array(list(group)[0][1]['Prby Mg2Cu'])
    target_phase = 0
    phase_index = 0
    phase_in_dataset = 0
    if peak_location == (1.526, 1.588):
        target_phase = prb_mgcu2_x
        phase_in_dataset = 1
    elif peak_location == (2.635, 2.708):
        target_phase = prb_ti_x
        phase_index = 1
        phase_in_dataset = 2
    elif peak_location == (2.734, 2.779):
        target_phase = prb_mg2cu_x
        phase_index = 2
        phase_in_dataset = 0
    else:
        print('Phase error')
    dict_inner_points = {'inner_x': np.array([]), 'inner_y': np.array([])}
    for i in range(len(target_phase)):
        target_ti = list(group)[phase_in_dataset][1]['Ti_frac'][i*3+phase_index]
        print('Ti_fraction', list(group)[phase_in_dataset][1]['Ti_frac'][i*3+phase_index])
        ti = np.linspace(list(group)[phase_in_dataset][1]['Ti_frac'][i*3+phase_index]-5,
                         list(group)[phase_in_dataset][1]['Ti_frac'][i*3+phase_index]+5,
                         21)
        for j in range(len(ti)):
            for chs in load_from_json('layout.json'):  # To find the maximum Ti% edge and over value would be initial Ti%
                words = str(chs)  # Check the layout file and you will know the character
                index_ti = words.find(']')
                index_ti_min = words.find('[')
                index_temp = words.find(',')
                index_time = words.find(',', index_temp + 1)
                index_equal = words.find('=', index_temp + 1)
                if list(group)[phase_in_dataset][1]['temperature'][i*3+phase_index] == \
                        int(words[index_temp - 3:index_temp]) \
                        and list(group)[phase_in_dataset][1]['annealing_time'][i*3+phase_index] == \
                        int(words[index_equal + 1:index_time]) \
                        and list(group)[phase_in_dataset][1]['thickness'][i*3+phase_index] == \
                        int(words[-2]) \
                        and int(words[index_ti_min+1:index_ti_min+3]) \
                        < ti[j] < \
                        int(words[index_ti - 2:index_ti]):  # If Ti% beyond the maximum
                    target_ti = ti[j]
                    print('Ti target', ti[j])
                elif int(words[index_ti_min+1:index_ti_min+3]) > ti[j] or ti[j] > int(words[index_ti - 2:index_ti]):
                    print('Out of max/min')
            beamline_x, beamline_y = pair[0](target_ti,
                                             list(group)[phase_in_dataset][1]['temperature'][i*3+phase_index],
                                             list(group)[phase_in_dataset][1]['annealing_time'][i*3+phase_index],
                                             list(group)[phase_in_dataset][1]['thickness'][i*3+phase_index])
            print(beamline_x, beamline_y)
            dict_inner_points['inner_x'] = np.append(dict_inner_points['inner_x'], beamline_x)
            dict_inner_points['inner_y'] = np.append(dict_inner_points['inner_y'], beamline_y)
            # print(beamline_x1, beamline_y1)

    if peak_location == (1.526, 1.588):
        plt.scatter(dict_inner_points['inner_x'], dict_inner_points['inner_y'],
                    c=None, marker='o', s=32, edgecolor='w', facecolor='None',
                    label='Inner points')
        plt.scatter(prb_mgcu2_x, prb_mgcu2_y, c=prb_mgcu2, marker='o', s=32,
                    label='$\mathregular{Cu_2Mg}$')
        plt.title('Crystallography companion agent (XCA) scan\n'
                  '1.547 ($\mathregular{Cu_2Mg}$) (1 1 1)\n'
                  'Peak location = [1.526, 1.588]')
    if peak_location == (2.635, 2.708):
        plt.scatter(dict_inner_points['inner_x'], dict_inner_points['inner_y'],
                    c=None, marker='o', s=32, edgecolor='w', facecolor='None',
                    label='Inner points')
        plt.scatter(prb_ti_x, prb_ti_y, c=prb_ti, marker='o', s=32,
                    label='Ti')
        plt.title('Crystallography companion agent (XCA) scan\n'
                  r'2.665 ($\mathregular{\beta-Ti}$) (1 1 0)''\n'
                  'Peak location = [2.635, 2.708]')
    if peak_location == (2.734, 2.779):
        plt.scatter(dict_inner_points['inner_x'], dict_inner_points['inner_y'],
                    c=None, marker='o', s=32, edgecolor='w', facecolor='None',
                    label='Inner points')
        plt.scatter(prb_mg2cu_x, prb_mg2cu_y, c=prb_mg2cu, marker='o', s=32,
                    label='$\mathregular{CuMg_2}$')
        plt.title('Crystallography companion agent (XCA) scan\n'
                  '2.755 ($\mathregular{CuMg_2}$) (0 8 0)\n'
                  'Peak location = [2.734, 2.779]')
    if peak_location == (2.925, 2.974):
        plt.title('Crystallography companion agent (XCA) scan\n'
                  '2.965 ($\mathregular{Cu_2Mg}$) (3 1 1) or 2.9499 ($\mathregular{CuMg_2}$) (3 5 1)\n'
                  'Peak location = [2.925, 2.974]')
    if peak_location == (3.047, 3.106):
        plt.title('Crystallography companion agent (XCA) scan\n'
                  '3.093 ($\mathregular{Cu_2Mg}$) (2 2 2) or 3.095 ($\mathregular{CuMg_2}$) (4 4 0)\n'
                  'Peak location = [3.047, 3.106]')
    cbar = plt.colorbar(pad=0.12)
    plt.clim(0, 1)
    cbar.set_label('Probability', size=12)

    grid_array = grid_list(grid_db, pair, grid_missing, peak_loc=peak_location)  # Call the grid scan background
    # print(grid_array)
    map_thin = np.empty(
        [8, 34])  # We were supposed to have 18X34, but the last two rows didn't be measured due to beam down
    for row in range(8):
        for column in range(34):
            map_thin[row, column] = grid_array[34 * row + column][2]  # Scan from left to right, down to upper
    map_thick = np.empty(
        [9, 34])  # We were supposed to have 18X34, but the last two rows didn't be measured due to beam down
    for row in range(9):
        for column in range(34):
            map_thick[row, column] = grid_array[34 * (row + 9) + column][2]  # Scan from left to right, down to upper
    # Plot the grid scan background
    plt.imshow(map_thick, vmin=-0.07, vmax=0.37, interpolation='bicubic', extent=[94.5, 28.5, 41.45, 3.45],
               origin='lower',
               cmap='plasma')  # Y range = 41.45 to 12.95 mm, vmin=-0.07, vmax=0.37, vmin=-0.00007, vmax=0.00045,
    plt.imshow(map_thin, vmin=-0.07, vmax=0.37, interpolation='bicubic', extent=[94.5, 28.5, 84.2, 50.95],
               origin='lower',
               cmap='plasma')  # Y range = 84.2 to 50.95 mm  vmin=-0.07, vmax=0.37, vmin=-0.00007, vmax=0.00045,
    plt.text(84.5, 47.5,
             'Ni standard on glass slide',
             fontsize=12)
    cbar = plt.colorbar()
    cbar.set_label('Region of interest (Roi)', size=12)
    plt.ylim(84.2, 3.45)
    plt.xlabel('X position (mm)', fontsize=12)
    plt.ylabel('Y position (mm)', fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1.2, 1.2), facecolor='lightgrey')    # You could consider it from the normal coordinate
    plt.show()


def plot_gpcam(gpcam_db, pair, grid_db, grid_missing):
    print('Original start time, Ti, temp, time, roi, thickness, beamline x and beamline y')
    time_list = []
    for i in range(1, len(gpcam_db) + 1):  # Extract all information from metadata['start']
        result = gpcam_db[-i]
        otime = result.metadata['start']['original_start_time']  # Assign original_start_time as otime
        ti = result.metadata['start']['adaptive_step']['snapped']['ctrl_Ti']
        temp = result.metadata['start']['adaptive_step']['snapped']['ctrl_temp']
        annealing_time = result.metadata['start']['adaptive_step']['snapped']['ctrl_annealing_time']
        peak_location = (2.925, 2.974)
        q, I, snapped, requested = extract_data(result)  # Extract q number, intensity, measured and predicted info
        compute_total_area(q, I)  # Compute the total area of the spectrum
        roi = compute_peak_area(q, I, *peak_location)  # Compute the region of interest
        roi = np.array([roi])
        thickness = result.metadata['start']['adaptive_step']['snapped']['ctrl_thickness']  # Metal agent thickness
        # Convert the physical parameters to the beamline coordinate (x, y)
        beamline_x, beamline_y = pair[0](result.metadata['start']['adaptive_step']['snapped']['ctrl_Ti'],
                                         result.metadata['start']['adaptive_step']['snapped']['ctrl_temp'],
                                         result.metadata['start']['adaptive_step']['snapped']['ctrl_annealing_time'],
                                         result.metadata['start']['adaptive_step']['snapped']['ctrl_thickness'])
        # Append (Original start start time, Ti, Temp, Annealing time, roi, thickness, beamline_x, beamline_y)
        time_list.append((time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(otime)),
                          ti, temp, annealing_time, roi[0], thickness, beamline_x, beamline_y))
    time_list.sort()
    # -------------------------------- Extract the sorted data to each list for plotting
    x_axis = []  # Run sequence
    y_axis = []  # Ti concentration
    x_axis_thin = []  # Run sequence of thin Mg
    x_axis_thick = []  # Run sequence of thick Mg
    y_axis_thin = np.array([])  # Ti concentration with thin Mg
    y_axis_thick = np.array([])  # Ti concentration with thick Mg
    y_axis_thin_temp = np.array([])    # Temperature with thin Mg
    y_axis_thick_temp = np.array([])    # Temperature with thick Mg
    z_axis_thin_time = np.array([])    # Time with thin Mg
    z_axis_thick_time = np.array([])    # Time with thick Mg
    intensity_roi_thin = np.array([])  # Roi with thin Mg
    intensity_roi_thick = np.array([])   # Roi with thick Mg
    y_axis_temp = []  # Annealing temperature
    z_axis_time = []  # Annealing time
    intensity_roi = []  # Region of interest
    beamline_x_axis = np.array([])  # Beamline x position
    beamline_y_axis = np.array([])  # Beamline y position
    dict_thickness = {'thick_ti': np.array([]), 'thin_ti': np.array([])}
    for j in range(len(time_list)):
        if time_list[j][0] > '2020-12-11 13:15:00':  # Extract the data meausured after this due to calibration
            # Print (Original start start time, Ti, Temp, Annealing time, roi, thickness, beamline_x, beamline_y)
            print('{}, {:.1f}, {}, {:4}, {:7.4f}, {}, {:6.4f}, {:7.4f}'.format(time_list[j][0], time_list[j][1],
                                                                             time_list[j][2], time_list[j][3],
                                                                             time_list[j][4], time_list[j][5],
                                                                             time_list[j][6], time_list[j][7]))
            x_axis.append(j)  # Run sequence
            y_axis.append(time_list[j][1])  # Ti concentration
            y_axis_temp.append(time_list[j][2])
            z_axis_time.append(time_list[j][3])
            intensity_roi.append(time_list[j][4])
            beamline_x_axis = np.append(beamline_x_axis, time_list[j][6])
            beamline_y_axis = np.append(beamline_y_axis, time_list[j][7])
            if time_list[j][5] == 0:    # If Mg agent is thick
                y_axis_thick = np.append(y_axis_thick, time_list[j][1])
                y_axis_thick_temp = np.append(y_axis_thick_temp, time_list[j][2])
                z_axis_thick_time = np.append(z_axis_thick_time, time_list[j][3])
                intensity_roi_thick = np.append(intensity_roi_thick, time_list[j][4])
            if time_list[j][5] == 1:    # If the Mg agent is thin
                y_axis_thin = np.append(y_axis_thin, time_list[j][1])
                y_axis_thin_temp = np.append(y_axis_thin_temp, time_list[j][2])
                z_axis_thin_time = np.append(z_axis_thin_time, time_list[j][3])
                intensity_roi_thin = np.append(intensity_roi_thin, time_list[j][4])
    ###################################################### Plot 2D stacking map (grid and gpCAM data)
    # ----------------------------------------------- From plot_grid_data function
    grid_array = grid_list(grid_db, pair, grid_missing)  # From grid_list function
    map = np.empty([16, 34])  # Create a empty array to store the data point for the 2D mapping
    time_and_temp_axis = np.array([])  # Store the y information
    for row in range(16):
        ti_axis = np.array([])  # Store the x position
        roi_axis = np.array([])  # Store the roi intensity
        for column in range(34):
            ti_axis = np.append(ti_axis, grid_array[34 * row + column][0])
            time_and_temp_axis = np.append(time_and_temp_axis, grid_array[34 * row + column][1])
            roi_axis = np.append(roi_axis, grid_array[34 * row + column][2])
            map[row, column] = grid_array[34 * row + column][2]  # Scan from left to right, down to upper
        ti_axis = np.flip(ti_axis)  # Make the x position low to high for the interpolation
        time_and_temp_axis = np.flip(time_and_temp_axis)  # Make the y position changing with the x position
        roi_axis = np.flip(roi_axis)  # Make the roi changing with the x position
        print(ti_axis[6], time_and_temp_axis[6])  # Show the x, y position for check
        if time_and_temp_axis[17] == 46.2:  # Skip the Ni standard sample between upper and down sample
            print(roi_axis[17])
            continue
        # Convert the beamline coordinate (x, y) to the physical parameters
        ti_, temp_, time_, thickness_ = pair[1](ti_axis[6], time_and_temp_axis[6])
        print(ti_)
        f = CubicSpline(ti_axis, roi_axis, bc_type='natural')  # Do the interpolation
        xplt = np.linspace(ti_axis[0], ti_axis[-1])
        plt.plot(xplt, f(xplt), 'b', ti_axis, roi_axis, 'ro')
        plt.title('Annealing temperature = {}$^o$C, \n'
                  'Annealing time = {}s, thickness = {}'.format(temp_, time_, thickness_))
        plt.ylim(-0.05, 0.35)
        plt.gca().invert_xaxis()
        plt.xlabel('X position (mm)')
        plt.ylabel('Region of interest (Roi)')
        plt.legend(['CubicSpline interpolation', 'Experimental results'])
        # plt.show()
    # #################################################### Plot the grid scan background
    map_thin = np.empty(
        [8, 34])  # We were supposed to have 18X34, but the last two rows didn't be measured due to beam down
    for row in range(8):
        for column in range(34):
            map_thin[row, column] = grid_array[34 * row + column][2]  # Scan from left to right, down to upper
    map_thick = np.empty(
        [9, 34])  # We were supposed to have 18X34, but the last two rows didn't be measured due to beam down
    for row in range(9):
        for column in range(34):
            map_thick[row, column] = grid_array[34 * (row + 9) + column][2]  # Scan from left to right, down to upper
    ################################################## Heat map
    plt.imshow(map_thick, vmin=-0.07, vmax=0.37, interpolation='bicubic', extent=[94.5, 28.5, 41.45, 3.45],
               origin='lower',
               cmap='plasma')  # Y range = 41.45 to 12.95 mm, vmin=-0.07, vmax=0.37, vmin=-0.00007, vmax=0.00045,
    plt.imshow(map_thin, vmin=-0.07, vmax=0.37, interpolation='bicubic', extent=[94.5, 28.5, 84.2, 50.95],
               origin='lower',
               cmap='plasma')  # Y range = 84.2 to 50.95 mm  vmin=-0.07, vmax=0.37, vmin=-0.00007, vmax=0.00045,
    plt.text(84, 47.5,
             'Ni standard on glass slide',
             fontsize=12)
    cbar = plt.colorbar(pad=0.13)
    cbar.set_label('Region of interest (Roi)', size=12)
    # plt.clim(-0.07, 0.37)
    plt.ylim(84.2, 3.45)
    plt.xlabel('X position (mm)')
    plt.ylabel('Y position (mm)')
    ################################################### Plot gpCAM data
    plt.plot(beamline_x_axis, beamline_y_axis, color='k',
             label='Trajectory', linestyle='--', linewidth=0.5)
    plt.scatter(beamline_x_axis, beamline_y_axis, c=intensity_roi, marker='o', s=32,
                label='Acquiring data', linestyle='-', linewidth=0.5)   # edgecolor='black', facecolor='None',
    cbar = plt.colorbar()
    cbar.set_label('Region of interest (Roi)', size=12)
    plt.clim(-0.07, 0.37)
    # plt.scatter(beamline_x_axis, beamline_y_axis, c=x_axis_color, marker='o', s=32,
    #             label='Region of interest', linestyle='-', linewidth=1)
    # text_x = str(beamline_x_axis.tolist())
    # text_y = str(beamline_y_axis.tolist())
    # plt.text(text_x, text_y, x_axis, fontsize=9)
    plt.xlabel('X position (mm)', fontsize=12)
    plt.ylabel('Y position (mm)', fontsize=12)
    # plt.ylim(84.2, 3.45)
    # plt.gca().invert_xaxis()    # Invert the x axis
    plt.title('gpCAM run number after 1:15 pm\n'
              '2.965 ($\mathregular{Cu_2Mg}$) (3 1 1) or 2.9499 ($\mathregular{CuMg_2}$) (3 5 1)\n'
              'Peak location = [2.925, 2.974]')
    plt.legend(loc='upper left')
    plt.show()
    ########################################## Plot 3D visualization (Ti concentration, Annealing temperature and time)
    x = np.array(y_axis)    # Ti%
    y = np.array(y_axis_temp)   # Temperature
    z = np.array(z_axis_time)   # Time
    sequence = np.array(x_axis)
    intensity = np.array(intensity_roi)
    print('---------------------------------gpCAM data number', len(x))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Ti concentration (at.%)')
    ax.set_ylabel('Annealing temperature ($\mathregular{^oC}$)')
    ax.set_zlabel('Annealing time (s)')
    p = ax.scatter3D(x, y, z, c=intensity, marker='o', s=(sequence - 140) * 5, label='Acquiring data')
    ax.plot3D(x, y, z, 'black', linestyle='--', linewidth=0.5, label='Trajectory')
    cbar = fig.colorbar(p, ax=ax, pad=0.2)
    cbar.set_label('Region of interest (Roi)', size=12)
    plt.title('gpCAM run number after 1:15 pm\n'
              '2.965 ($\mathregular{Cu_2Mg}$) (3 1 1) or 2.9499 ($\mathregular{CuMg_2}$) (3 5 1)\n'
              'Peak location = [2.925, 2.974]')
    plt.legend()
    plt.show()
    print('---------------------------------------------Axis below')
    # ti_library = sorted(set(x))
    ti_library = np.arange(0, 101, 1)
    # temp_library = sorted(set(y))
    # temp_library = np.arange(340, 470, 10)
    temp_library = np.arange(340, 462, 2)
    # time_library = sorted(set(z))
    # time_library = np.arange(450, 4050, 450)
    time_library = np.arange(450, 3645, 45)

    print(ti_library)
    print(temp_library)
    print(time_library)
    visual_matrix = np.zeros([len(ti_library), len(temp_library), len(time_library)])
    for gpcam_point in range(len(y_axis_thin)):    # Change length if using different array
        # volumn_x = x[gpcam_point]
        # volumn_y = y[gpcam_point]
        # volumn_z = z[gpcam_point]
        volumn_x = np.round(y_axis_thin[gpcam_point], 0)    # Check thick or thin
        volumn_y = y_axis_thin_temp[gpcam_point]    # Check thick or thin
        volumn_z = z_axis_thin_time[gpcam_point]    # Check thick or thin
        # y_axis_thin Ti concentration with thin Mg
        # y_axis_thick Ti concentration with thick Mg
        # y_axis_thin_temp Temperature with thin Mg
        # y_axis_thick_temp Temperature with thick Mg
        # z_axis_thin_time Time with thin Mg
        # z_axis_thick_time Time with thick Mg
        visual_matrix[np.where(ti_library == volumn_x)[0][0],
                      np.where(temp_library == volumn_y)[0][0],
                      np.where(time_library == volumn_z)[0][0]] \
            = intensity_roi_thin[gpcam_point]    # [0][0] to remove column # Check thick or thin

    print('----------------------------------Interpolation below')
    data_grid = list(zip(y_axis_thin, y_axis_thin_temp, z_axis_thin_time))    # Check thick or thin
    grid_x, grid_y, grid_z = np.mgrid[0:101:1, 340:462:2, 450:3645:45]
    roi_values = intensity_roi_thin    # Check thick or thin
    grid_visualization = griddata(np.array(data_grid), np.array(roi_values), (grid_x, grid_y, grid_z), method='linear')
    print(grid_visualization)
    print('----------------------Data number', len(y_axis_thin))    # Check thick or thin
    """
        Cheng-Hung's suggestion for masking values < 0 and assigning them to nan
        Then normalize them
        """
    grid_visualization[grid_visualization < 0] = np.nan  # Set the 0 to nan
    grid_visualization_norm = grid_visualization / np.nanmax(grid_visualization)  # Do the normalization
    print(np.nanmax(grid_visualization_norm))
    tomviz = np.float32(grid_visualization_norm)
    io.imsave('D:/Software/Python/SSID/gpCAM_visualization311_norm_thin_08050722.tif', tomviz)
    print('Saved!')


def version_6_of_gpCAM(gpcam_db, pair):
    time_list = sorted_timelist(gpcam_db, pair)
    simulation_map = np.array([])
    x_ti = []
    y_temp = []
    z_time = []
    roi = []
    final_list = []
    for j in range(len(time_list)):
        if time_list[j][0] > '2020-12-11 13:15:00' and time_list[j][5] == 0:
            # Print(Original start start time, Ti, Temp, Annealing time, roi, thickness, beamline_x, beamline_y)
            # print('{}, {:.1f}, {}, {:4}, {:7.4f}, {}, {:6.4f}, {:7.4f}'.format(time_list[j][0], time_list[j][1],
            #                                                                  time_list[j][2], time_list[j][3],
            #                                                                  time_list[j][4], time_list[j][5],
            #                                                                  time_list[j][6], time_list[j][7]))
            simulation_map = np.append(simulation_map, [[time_list[j][1]], [time_list[j][2]], [time_list[j][3]],
                                       [time_list[j][4]]])  # Append ([Ti, temp, time, roi])
            final_list.append(time_list[j])
            x_ti.append(time_list[j][1])
            y_temp.append((time_list[j][2]))
            z_time.append((time_list[j][3]))
            roi.append(time_list[j][4])
    a = simulation_map
    a = simulation_map.reshape(-1, 4)   # Reshape to be [x, 4] array (Ti, temp, time, roi)
    b = np.load("us_topo.npy")
    # print(len(a))
    rng = default_rng()
    ind = rng.choice(len(a) - 1, size=10, replace=False)    # len size should be larger than the size
    points = a[ind, 0:3]    # Ti, temp, time
    values = a[ind, 3:4]    # roi
    # # index_set_bounds = np.array([[0, 99], [0, 248]])
    index_set_bounds = np.array([[0, len(x_ti)], [0, len(y_temp)], [0, len(z_time)]])    # Set bounds
    hyperparameter_bounds = np.array([[0.001, 1e9], [1, 1000], [1, 1000]])
    hps_guess = np.array([4.71907062e+06, 4.07439017e+02, 3.59068120e+02])
    ###################################################################################
    gp = GPOptimizer(3, 1, 1, index_set_bounds)     # 3, 1, 1 for our system
    gp.tell(points, values)
    gp.init_gp(hps_guess)

    next = gp.ask(position=None, n=1, objective_function="covariance", optimization_bounds=None,
                  optimization_method="global", optimization_pop_size=50, optimization_max_iter=20,
                  optimization_tol=10e-6, dask_client=False)
    print(next)


def sorted_timelist(gpcam_db, pair):
    """
    :param gpcam_db: dataset, gpcam_db data
    :param pair: function, convert the coordinates (forward(data --> beamline), backward(beamline --> data))
    :return: list, sorted time list
    """
    print('Original start time, Ti, temp, time, roi, thickness, beamline x and beamline y')
    time_list = []
    for i in range(1, len(gpcam_db) + 1):  # Extract all information from metadata['start']
        result = gpcam_db[-i]
        otime = result.metadata['start']['original_start_time']  # Assign original_start_time as otime
        ti = result.metadata['start']['adaptive_step']['snapped']['ctrl_Ti']
        temp = result.metadata['start']['adaptive_step']['snapped']['ctrl_temp']
        annealing_time = result.metadata['start']['adaptive_step']['snapped']['ctrl_annealing_time']
        peak_location = (2.925, 2.974)
        q, I, snapped, requested = extract_data(result)
        roi = compute_peak_area(q, I, *peak_location)
        roi = np.array([roi])
        thickness = result.metadata['start']['adaptive_step']['snapped']['ctrl_thickness']
        beamline_x, beamline_y = pair[0](result.metadata['start']['adaptive_step']['snapped']['ctrl_Ti'],
                                         result.metadata['start']['adaptive_step']['snapped']['ctrl_temp'],
                                         result.metadata['start']['adaptive_step']['snapped']['ctrl_annealing_time'],
                                         result.metadata['start']['adaptive_step']['snapped']['ctrl_thickness'])
        # Append (Original start start time, Ti, Temp, Annealing time, roi, thickness, beamline_x, beamline_y)
        time_list.append((time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(otime)),
                          ti, temp, annealing_time, roi[0], thickness, beamline_x, beamline_y))

    time_list.sort()
    return time_list


def plot_grid_data(grid_db, grid_missing, pair, peak_loc=(2.925, 2.974)):

    """
        Cu2Mg(311): (2.925, 2.974)
        Cu2Mg(111): (1.526, 1.588)
        Cu2Mg(080): (2.734, 2.779)
        Cu2Mg(222): (3.047, 3.106)
        Beta Ti(110): (2.635, 2.708)
        :param grid_db: dataframe, XPD data
        :param pair: function, convert between beamline coordinate and physical parameters
        :param peak_loc: tuple, the range of a specific phase formation
        :return: None
        """
    print('Original start time, plan name and shape')
    ################################################## Print the grid data
    time_list = []
    for i in range(1, len(grid_db) + 1):  # Extract all information from metadata['start']
        result = grid_db[-i]
        otime = result.metadata['start']['original_start_time']  # Assign original_start_time as otime
        plan_name = result.metadata['start']['plan_name']
        if 'shape' in result.metadata['start']:
            shape = result.metadata['start']['shape']
        else:
            shape = None
        time_list.append(
            (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(otime)), plan_name, shape))
    time_list.sort()  # Print grid scan time
    for j in range(len(time_list)):
        print(time_list[j])
    ################################################## Plot Intensity vs q space
    result = grid_db[-3]
    d = result.primary.read()
    otime = result.metadata['start']['original_start_time']
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(otime)))
    print('Number of points, spectrum')
    print(np.shape(d['q']))
    peak_location = peak_loc
    roi_array = np.array([])
    grid_array = np.array([])
    for i in range(len(d['sample_x'])):  # All acquiring data
        # plt.plot(d['q'][i], d['mean'][i])
        roi = compute_peak_area(d['q'][i], d['mean'][i], *peak_location)  # Compute the roi
        roi = np.array([roi][0])  # Clean the format to become an int
        # total_area = compute_total_area(np.array(d['q'][i]), np.array(d['mean'][i]))
        # roi = roi/total_area
        roi_array = np.append(roi_array, roi)  # Collect the roi of the a phase
        # Collect beamline x, y and roi information
        grid_array = np.append(grid_array, [np.array(d['sample_x'][i]), np.array(d['ss_stg2_y'][i]), roi])
    # plt.title('Grid Scan')
    # plt.ylabel('Intensity (a.u.)')
    # plt.xlabel('q')
    # plt.show()
    ################################################## Import grid data
    grid_array = grid_list(grid_db, pair, grid_missing, peak_loc=peak_location)  # Call the grid scan background
    print(grid_array)
    map_thin = np.empty(
        [8, 34])  # We were supposed to have 18X34, but the last two rows didn't be measured due to beam down
    for row in range(8):
        for column in range(34):
            map_thin[row, column] = grid_array[34 * row + column][2]  # Scan from left to right, down to upper
    map_thick = np.empty(
        [9, 34])  # We were supposed to have 18X34, but the last two rows didn't be measured due to beam down
    for row in range(9):
        for column in range(34):
            map_thick[row, column] = grid_array[34 * (row + 9) + column][2]  # Scan from left to right, down to upper
    # Plot the grid scan background
    plt.imshow(map_thick, vmin=-0.07, vmax=0.37, interpolation='bicubic', extent=[94.5, 28.5, 41.45, 3.45],
               origin='lower',
               cmap='plasma')  # Y range = 41.45 to 12.95 mm, vmin=-0.07, vmax=0.37, vmin=-0.00007, vmax=0.00045,
    plt.imshow(map_thin, vmin=-0.07, vmax=0.37, interpolation='bicubic', extent=[94.5, 28.5, 84.2, 50.95],
               origin='lower',
               cmap='plasma')  # Y range = 84.2 to 50.95 mm  vmin=-0.07, vmax=0.37, vmin=-0.00007, vmax=0.00045,
    plt.text(84, 47.5,
             'Ni standard on glass slide',
             fontsize=12)
    cbar = plt.colorbar()
    cbar.set_label('Region of interest (Roi)', size=12)

    # plt.clim(-0.07, 0.37)
    plt.ylim(84.2, 3.45)
    plt.xlabel('X position (mm)', fontsize=12)
    plt.ylabel('Y position (mm)', fontsize=12)
    if peak_loc == (2.925, 2.974):
        plt.title('Grid scan\n'
                  '2.965 ($\mathregular{Cu_2Mg}$) (3 1 1) or 2.9499 ($\mathregular{CuMg_2}$) (3 5 1)\n'
                  'Peak location = [2.925, 2.974]')
    elif peak_loc == (1.526, 1.588):
        plt.title('Grid scan\n'
                  '1.547 ($\mathregular{Cu_2Mg}$) (1 1 1)\n'
                  'Peak location = [1.526, 1.588]')
    elif peak_loc == (2.734, 2.779):
        plt.title('Grid scan\n'
                  '2.755 ($\mathregular{CuMg_2}$) (0 8 0)\n'
                  'Peak location = [2.734, 2.779]')
    elif peak_loc == (3.047, 3.106):
        plt.title('Grid scan\n'
                  '3.093 ($\mathregular{Cu_2Mg}$) (2 2 2) or 3.095 ($\mathregular{CuMg_2}$) (4 4 0)\n'
                  'Peak location = [3.047, 3.106]')
    elif peak_loc == (2.635, 2.708):
        plt.title('Grid scan\n'
                  r'2.665 ($\mathregular{\beta-Ti}$) (1 1 0)''\n'
                  'Peak location = [2.635, 2.708]')
    plt.show()
    ############################################################# Scattering points
    # plt.scatter(np.array(d['sample_x']), np.array(d['ss_stg2_y']), c=roi_array, marker='o', s=32,
    #             label='Acquiring data', linestyle='--', linewidth=1)
    # # plt.gca().invert_xaxis()  # Invert the x axis
    # # plt.gca().invert_yaxis()  # Invert the y axis
    # cbar = plt.colorbar()
    # cbar.set_label('Region of interest (Roi)', size=12)
    # plt.xlabel('X position (mm)')
    # plt.ylabel('Y position (mm)')
    # plt.title('Grid scan\n'
    #           '2.965 ($\mathregular{Cu_2Mg}$) (3 1 1) or 2.9499 ($\mathregular{CuMg_2}$) (3 5 1)\n'
    #           'Peak location = [2.925, 2.974]')
    # plt.legend()
    # plt.show()
    ############################################################## For 3D visualization
    dic = {"x": np.array([]), "y": np.array([]), "z": np.array([]), 'roi': np.array([]),
           'thin_x': np.array([]), 'thin_y': np.array([])}
    for para in range(18):
        print(grid_array[para*34+17][0], grid_array[para*34+17][1])    # Print one (x, y) on each stripe
        if grid_array[para*34+17][1] == 46.2:  # Skip the Ni standard sample between upper and down sample
            print('This is Ni standard')
            continue
        ti_, temp_, time_, thickness_ = pair[1](grid_array[para*34+17][0], grid_array[para*34+17][1])   # Convert
        print(ti_, temp_, time_, thickness_)
    ti_, temp_, time_, thick_ = pair[1](grid_array[17][0], grid_array[17][1])   # Initial point
    for i in range(len(grid_array)):
        a, b = grid_array[i][0], grid_array[i][1]
        if b == 46.2:   # Skip the Ni standard sample between upper and down sample
            # ti_, temp_, time_, thick_ = pair[1](grid_array[170][0], grid_array[170][1])
            print('This is Ni standard')
            continue
        if i % 34 == 17:
            ti_, temp_, time_, thick_ = pair[1](grid_array[i // 34+17][0], b)   # Print one parameter on each stripe
        print(i)
        print(b)
        print(ti_)
        print(grid_interpolation(a, b))
        """
        b > 47 means thin Mg
        b < 46 means thick Mg
        """
        if grid_interpolation(a, b) < 0 and b < 46:    # Check thick or thin
            dic['x'] = np.append(dic['x'], 0)   # Set any negative value to 0
        elif grid_interpolation(a, b) >= 0 and b < 46:     # Check thick or thin
            dic['x'] = np.append(dic['x'], np.round(grid_interpolation(a, b), 0))   # Append a reasonable Ti at.%
        if b < 46:     # Check thick or thin
            dic['y'] = np.append(dic['y'], temp_)
            dic['z'] = np.append(dic['z'], time_)
            dic['roi'] = np.append(dic['roi'], grid_array[i][2])
    print('------------------Data number:', len(dic['x']))
    # ti_library = sorted(set(x))
    ti_library = np.arange(0, 101, 1)
    # temp_library = sorted(set(y))
    temp_library = np.arange(340, 462, 2)
    # time_library = sorted(set(z))
    time_library = np.arange(450, 3645, 45)
    print(ti_library)
    print(temp_library)
    print(time_library)
    visual_matrix = np.zeros([len(ti_library), len(temp_library), len(time_library)])
    for grid_point in range(len(dic['x'])):  # Change length if using different array
        volumn_x = dic['x'][grid_point]
        volumn_y = dic['y'][grid_point]
        volumn_z = dic['z'][grid_point]
        visual_matrix[np.where(ti_library == volumn_x)[0][0],
                      np.where(temp_library == volumn_y)[0][0],
                      np.where(time_library == volumn_z)[0][0]] \
            = dic['roi'][grid_point]  # [0][0] to remove column

    print(visual_matrix)
    print('----------------------------------Interpolation below')
    data_grid = list(zip(dic['x'], dic['y'], dic['z']))
    grid_x, grid_y, grid_z = np.mgrid[0:101:1, 340:462:2, 450:3645:45]
    roi_values = dic['roi']
    grid_visualization = griddata(np.array(data_grid), np.array(roi_values), (grid_x, grid_y, grid_z), method='linear')
    """
    Cheng-Hung's suggestion for masking values < 0 and assigning them to nan
    Then normalize them
    """
    grid_visualization[grid_visualization < 0] = np.nan    # Set the 0 to nan
    grid_visualization_norm = grid_visualization / np.nanmax(grid_visualization)    # Do the normalization
    print(np.nanmax(grid_visualization_norm))
    tomviz = np.float32(grid_visualization_norm)
    io.imsave('D:/Software/Python/SSID/grid_visualization311_thick_norm_08050731.tif', tomviz)
    print('Saved!')
    # return dic


def grid_3d(dic):
    data_grid = list(zip(dic['x'], dic['y'], dic['z']))
    grid_x, grid_y, grid_z = np.mgrid[0:101:1, 340:462:2, 450:3645:45]
    roi_values = dic['roi']
    grid_visualization = griddata(np.array(data_grid), np.array(roi_values), (grid_x, grid_y, grid_z), method='linear')
    print(grid_visualization)

    # return grid_visualization


def grid_list(grid_db, pair, grid_missing, peak_loc=(1.526, 1.588)):
    """
    :param grid_db: dataset, databroker data
    :param pair: function, convert the parameters between beamline coordinates and physical parameters
    :param peak_loc: tuple, Cu2Mg or CuMg2
    :return: array, grid data 2D matrix [-1, 3]
    """
    result = grid_db[-3]
    d = result.primary.read()
    otime = result.metadata['start']['original_start_time']
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(otime)))
    print('------------------------')
    peak_location = peak_loc
    roi_array = np.array([])
    grid_array = np.array([])
    for i in range(len(d['sample_x'])-24):
        roi = compute_peak_area(d['q'][i], d['mean'][i], *peak_location)
        roi = np.array([roi][0])
        # total_area = compute_total_area(np.array(d['q'][i]), np.array(d['mean'][i]))
        # roi = roi/total_area
        roi_array = np.append(roi_array, roi)
        # Collect beamline x, y and roi information
        grid_array = np.append(grid_array, [np.array(d['sample_x'][i]), np.array(d['ss_stg2_y'][i]), roi])
    for missing in [-1, -2]:    # For missing grid data (the last two rows)
        result = grid_missing[missing]
        peak_location = peak_loc
        d = result.primary.read()
        for i in range(len(d['sample_x'])):  # All acquiring data
            roi = compute_peak_area(d['q'][i], d['mean'][i], *peak_location)  # Compute the roi
            roi = np.array([roi][0])  # Clean the format to become an int
            # total_area = compute_total_area(np.array(d['q'][i]), np.array(d['mean'][i]))
            # roi = roi/total_area
            roi_array = np.append(roi_array, roi)  # Collect the roi of the a phase
            # Collect beamline x, y and roi information
            grid_array = np.append(grid_array, [np.array(d['sample_x'][i]), np.array(d['ss_stg2_y'][i]), roi])
    grid_array = grid_array.reshape(-1, 3)
    # print(grid_array)
    return grid_array


def grid_interpolation(x, y):
    grid_dict = {
        'f1x': np.array([28.5, 33, 37.5, 42, 46.5, 51, 55.5, 60, 64.5, 69, 73.5, 78, 82.5, 87, 91.5]),
        'f1ti': np.array([75, 71, 67, 63, 58, 53, 48, 43, 38, 33, 28, 25, 21, 18, 15]),
        'f2x': np.array([28.5, 33, 37.5, 42, 46.5, 51, 55.5, 60, 64.5, 69, 73.5, 78, 82.5, 87, 91.5]),
        'f2ti': np.array([76, 73, 69, 65, 61, 56, 51, 46, 40, 35, 31, 26, 22, 19, 17]),
        'f3x': np.array([28.5, 33, 37.5, 42, 46.5, 51, 55.5, 60, 64.5, 69, 73.5, 78, 82.5, 87, 91.5]),
        'f3ti': np.array([78, 75, 71, 67, 63, 58, 52, 48, 43, 37, 32, 28, 24, 20, 17]),
        'f4x': np.array([28.5, 33, 37.5, 42, 46.5, 51, 55.5, 60, 64.5, 69, 73.5, 78, 82.5, 87, 91.5]),
        'f4ti': np.array([79, 77, 73, 69, 65, 60, 55, 50, 45, 39, 35, 30, 25, 22, 19]),
        'f5x': np.array([28.5, 33, 37.5, 42, 46.5, 51, 55.5, 60, 64.5, 69, 73.5, 78, 82.5, 87, 91.5]),
        'f5ti': np.array([81, 78, 75, 71, 67, 63, 57, 51, 47, 42, 37, 32, 27, 23, 20]),
        'f6x': np.array([33, 37.5, 42, 46.5, 51, 55.5, 60, 64.5, 69, 73.5, 78, 82.5, 87]),
        'f6ti': np.array([69, 65, 61, 56, 51, 46, 41, 36, 31, 27, 23, 20, 17]),
        'f7x': np.array([33, 37.5, 42, 46.5, 51, 55.5, 60, 64.5, 69, 73.5, 78, 82.5, 87]),
        'f7ti': np.array([67, 62, 58, 53, 49, 43, 39, 34, 29, 25, 22, 18, 16]),
        'f8x': np.array([37.5, 42, 46.5, 51, 55.5, 60, 64.5, 69, 73.5, 78, 82.5]),
        'f8ti': np.array([60, 56, 51, 46, 42, 37, 32, 28, 23, 20, 19]),
        'f10x': np.array([28.5, 33, 37.5, 42, 46.5, 51, 55.5, 60, 64.5, 69, 73.5, 78, 82.5, 87, 91.5]),
        'f10ti': np.array([75, 71, 67, 63, 58, 53, 48, 43, 38, 33, 28, 25, 21, 18, 15]),
        'f11x': np.array([28.5, 33, 37.5, 42, 46.5, 51, 55.5, 60, 64.5, 69, 73.5, 78, 82.5, 87, 91.5]),
        'f11ti': np.array([76, 73, 69, 65, 61, 56, 51, 46, 40, 35, 31, 26, 22, 19, 17]),
        'f12x': np.array([28.5, 33, 37.5, 42, 46.5, 51, 55.5, 60, 64.5, 69, 73.5, 78, 82.5, 87, 91.5]),
        'f12ti': np.array([78, 75, 71, 67, 63, 58, 52, 48, 43, 37, 32, 28, 24, 20, 17]),
        'f13x': np.array([28.5, 33, 37.5, 42, 46.5, 51, 55.5, 60, 64.5, 69, 73.5, 78, 82.5, 87, 91.5]),
        'f13ti': np.array([79, 77, 73, 69, 65, 60, 55, 50, 45, 39, 35, 30, 25, 22, 19]),
        'f14x': np.array([28.5, 33, 37.5, 42, 46.5, 51, 55.5, 60, 64.5, 69, 73.5, 78, 82.5, 87, 91.5]),
        'f14ti': np.array([81, 78, 75, 71, 67, 63, 57, 51, 47, 42, 37, 32, 27, 23, 20]),
        'f15x': np.array([33, 37.5, 42, 46.5, 51, 55.5, 60, 64.5, 69, 73.5, 78, 82.5, 87]),
        'f15ti': np.array([69, 65, 61, 56, 51, 46, 41, 36, 31, 27, 23, 20, 17]),
        'f16x': np.array([33, 37.5, 42, 46.5, 51, 55.5, 60, 64.5, 69, 73.5, 78, 82.5, 87]),
        'f16ti': np.array([67, 62, 58, 53, 49, 43, 39, 34, 29, 25, 22, 18, 16]),
        'f17x': np.array([37.5, 42, 46.5, 51, 55.5, 60, 64.5, 69, 73.5, 78, 82.5]),
        'f17ti': np.array([60, 56, 51, 46, 42, 37, 32, 28, 23, 20, 19]),
        'f18x': np.array([42, 46.5, 51, 55.5, 60, 64.5, 69, 73.5, 78]),
        'f18ti': np.array([53, 49, 44, 40, 35, 30, 27, 22, 19])
    }
    f1 = CubicSpline(grid_dict['f1x'], grid_dict['f1ti'], bc_type='natural')   # Do the interpolation
    f2 = CubicSpline(grid_dict['f2x'], grid_dict['f2ti'], bc_type='natural')  # Do the interpolation
    f3 = CubicSpline(grid_dict['f3x'], grid_dict['f3ti'], bc_type='natural')  # Do the interpolation
    f4 = CubicSpline(grid_dict['f4x'], grid_dict['f4ti'], bc_type='natural')  # Do the interpolation
    f5 = CubicSpline(grid_dict['f5x'], grid_dict['f5ti'], bc_type='natural')  # Do the interpolation
    f6 = CubicSpline(grid_dict['f6x'], grid_dict['f6ti'], bc_type='natural')  # Do the interpolation
    f7 = CubicSpline(grid_dict['f7x'], grid_dict['f7ti'], bc_type='natural')  # Do the interpolation
    f8 = CubicSpline(grid_dict['f8x'], grid_dict['f8ti'], bc_type='natural')  # Do the interpolation
    f18 = CubicSpline(grid_dict['f18x'], grid_dict['f18ti'], bc_type='natural')  # Do the interpolation
    if y == 84.2 or y == 41.45:
        return f1(x)
    elif y == 79.45 or y == 36.7:
        return f2(x)
    elif y == 74.7 or 31.95 < y < 31.951:
        return f3(x)
    elif y == 69.95 or 27.2 < y < 27.3:
        return f4(x)
    elif y == 65.2 or 22.45 < y < 22.46:
        return f5(x)
    elif y == 60.45 or y == 17.7:
        return f6(x)
    elif y == 55.7 or 12.95 < y < 12.96:
        return f7(x)
    elif y == 50.95 or 8.2 < y < 8.3:
        return f8(x)
    elif y == 3.45:
        return f18(x)


def plot_roi(gpcam_db, peak_loc=(2.925, 2.974)):
    print('Original start time, roi and thickness')
    # Compute the roi
    peak_location = peak_loc  # region of interest (roi) (351) CuMg2
    time_list = []
    for i in range(1, len(gpcam_db) + 1):  # Extract all information from metadata['start']
        result = gpcam_db[-i]
        q, I, snapped, requested = extract_data(result)
        roi = compute_peak_area(q, I, *peak_location)
        roi = np.array([roi])
        otime = result.metadata['start']['original_start_time']  # Assign original_start_time as otime
        thickness = result.metadata['start']['adaptive_step']['snapped']['ctrl_thickness']
        time_list.append(
            (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(otime)), roi[0], thickness)) # Append (time, roi, thickness)

    time_list.sort()
    x_axis = []
    y_axis = []
    x_axis_thin = []
    x_axis_thick = []
    y_axis_thin = []
    y_axis_thick = []
    for j in range(len(time_list)):
        if time_list[j][0] > '2020-12-11 13:15:00':
            print('{}, {}, {}'.format(time_list[j][0], time_list[j][1], time_list[j][2]))
            x_axis.append(j)
            y_axis.append(time_list[j][1])
            if time_list[j][2] == 0:
                x_axis_thin.append(j)
                y_axis_thin.append(time_list[j][1])
            if time_list[j][2] == 1:
                x_axis_thick.append(j)
                y_axis_thick.append(time_list[j][1])
    plt.plot(x_axis_thin, y_axis_thin, color='b', marker='o', markersize=5, markerfacecolor='b',
             label='Thin Mg', linestyle='-', linewidth=2)
    plt.plot(x_axis_thick, y_axis_thick, color='r', marker='o', markersize=5, markerfacecolor='r',
             label='Thick Mg', linestyle='-', linewidth=2)
    plt.plot(x_axis, y_axis, color='k', label='Run sequence', linestyle='--', linewidth=1)
    plt.xlabel('Run number after 1:15 pm')
    plt.ylabel('Region of interest (roi), q = (2.925, 2.974)')
    plt.title('gpCAM')
    plt.legend(loc='upper left')
    # plt.show()


def plot_annealing_time(gpcam_db):
    print('Original start time, annealing time and thickness')
    time_list = []
    for i in range(1, len(gpcam_db) + 1):  # Extract all information from metadata['start']
        result = gpcam_db[-i]
        otime = result.metadata['start']['original_start_time']  # Assign original_start_time as otime
        annealing_time = result.metadata['start']['adaptive_step']['snapped']['ctrl_annealing_time']
        thickness = result.metadata['start']['adaptive_step']['snapped']['ctrl_thickness']
        time_list.append(
            (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(otime)), annealing_time, thickness))  # Append (time, ti)

    time_list.sort()
    x_axis = []
    y_axis = []
    x_axis_thin = []
    x_axis_thick = []
    y_axis_thin = []
    y_axis_thick = []
    for j in range(len(time_list)):
        if time_list[j][0] > '2020-12-11 13:15:00':
            print('{}, {}, {}'.format(time_list[j][0], time_list[j][1], time_list[j][2]))
            x_axis.append(j)
            y_axis.append(time_list[j][1])
            if time_list[j][2] == 0:
                x_axis_thin.append(j)
                y_axis_thin.append(time_list[j][1])
            if time_list[j][2] == 1:
                x_axis_thick.append(j)
                y_axis_thick.append(time_list[j][1])
    plt.plot(x_axis_thin, y_axis_thin, color='b', marker='o', markersize=5, markerfacecolor='b',
             label='Thin Mg', linestyle='-', linewidth=2)
    plt.plot(x_axis_thick, y_axis_thick, color='r', marker='o', markersize=5, markerfacecolor='r',
             label='Thick Mg', linestyle='-', linewidth=2)
    plt.plot(x_axis, y_axis, color='k', label='Run sequence', linestyle='--', linewidth=1)
    plt.xlabel('Run number after 1:15 pm')
    plt.ylabel('Dealloying time (s)')
    plt.title('gpCAM')
    # plt.ylim(300, 500)
    plt.legend(loc='lower right')
    # plt.show()


def plot_temperature(gpcam_db):
    print('Original start time, temperature and thickness')
    time_list = []
    for i in range(1, len(gpcam_db) + 1):  # Extract all information from metadata['start']
        result = gpcam_db[-i]
        otime = result.metadata['start']['original_start_time']  # Assign original_start_time as otime
        temp = result.metadata['start']['adaptive_step']['snapped']['ctrl_temp']
        thickness = result.metadata['start']['adaptive_step']['snapped']['ctrl_thickness']
        time_list.append(
            (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(otime)), temp, thickness))  # Append (time, ti)

    time_list.sort()
    x_axis = []
    y_axis = []
    x_axis_thin = []
    x_axis_thick = []
    y_axis_thin = []
    y_axis_thick = []
    for j in range(len(time_list)):
        if time_list[j][0] > '2020-12-11 13:15:00':
            print('{}, {}, {}'.format(time_list[j][0], time_list[j][1], time_list[j][2]))
            x_axis.append(j)
            y_axis.append(time_list[j][1])
            if time_list[j][2] == 0:
                x_axis_thin.append(j)
                y_axis_thin.append(time_list[j][1])
            if time_list[j][2] == 1:
                x_axis_thick.append(j)
                y_axis_thick.append(time_list[j][1])
    plt.plot(x_axis_thin, y_axis_thin, color='b', marker='o', markersize=5, markerfacecolor='b',
             label='Thin Mg', linestyle='-', linewidth=2)
    plt.plot(x_axis_thick, y_axis_thick, color='r', marker='o', markersize=5, markerfacecolor='r',
             label='Thick Mg', linestyle='-', linewidth=2)
    plt.plot(x_axis, y_axis, color='k', label='Run sequence', linestyle='--', linewidth=1)
    plt.xlabel('Run number after 1:15 pm')
    plt.ylabel('Dealloying temperature ' + r"($^\circ$C)")
    plt.ylim(300, 500)
    plt.title('gpCAM')
    plt.legend(loc='upper left')
    # plt.show()


def check_scan_id_and_CPU_time(gpcam_db):
    print('Show Original time and Scan ID')
    time_list = []
    for i in range(1, len(gpcam_db) + 1):   # Extract all information from metadata['start']
        result = gpcam_db[-i]
        otime = result.metadata['start']['original_start_time']     # Assign original_start_time as otime
        time_list.append((time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(otime)),
                          result.metadata['start']['scan_id']))     # Append (time, scan id)
    time_list.sort()
    for j in range(len(time_list)):
        if time_list[j][0] > '2020-12-11 13:15:00':
            print(time_list[j])


def single_strip_transform_factory(
        temperature,
        annealing_time,
        ti_fractions,
        reference_x,
        reference_y,
        start_distance,
        angle,
        thickness,
        *,
        cell_size=4.5,
):
    """
    Generate the forward and reverse transforms for a given strip.
    This assumes that the strips are mounted parallel to one of the
    real motor axes.  This only handles a single strip which has a
    fixed annealing time and temperature.
    Parameters
    ----------
    temperature : int
       The annealing temperature in degree C
    annealing_time : int
       The annealing time in seconds
    ti_fractions : Iterable
       The fraction of Ti in each cell (floats in range [0, 100])
       Assume that the values are for the center of the cells.
    reference_x, reference_y : float
       The position of the reference point on the left edge of the
       sample (looking upstream into the beam) and on the center line
       of the sample strip.
    angle : float
       The angle in radians of the tilt.  The rotation point is the
       reference point.
    start_distance : float
       Distance along the strip from the reference point to the center
       of the first cell in mm.
    cell_size : float, optional
       The size of each cell along the gradient where the Ti fraction
       is measured in mm.
    Returns
    -------
    transform_pair
       forward (data -> bl)
       inverse (bl -> data)
    """
    _temperature = int(temperature)
    _annealing_time = int(annealing_time)
    _thickness = int(thickness)

    cell_positions = np.arange(len(ti_fractions)) * cell_size

    def to_bl_coords(Ti_frac, temperature, annealing_time, thickness):
        if (
                _temperature != temperature
                or annealing_time != _annealing_time
                or _thickness != thickness
        ):
            raise ValueError

        if Ti_frac > np.max(ti_fractions) or Ti_frac < np.min(ti_fractions):
            raise ValueError

        d = (
                np.interp(Ti_frac, ti_fractions, cell_positions)
                - start_distance
                + (cell_size / 2)
        )

        # minus because we index the cells backwards
        return reference_x - np.cos(angle) * d, reference_y - np.sin(angle) * d

    def to_data_coords(x, y):
        # negative because we index the cells backwards
        x_rel = -(x - reference_x)
        y_rel = y - reference_y

        r = np.hypot(x_rel, y_rel)

        d_angle = -np.arctan2(y_rel, x_rel)

        from_center_angle = d_angle - angle
        if x_rel > 0 and r < cell_size/2:  # I add
            d = np.cos(from_center_angle) * (r + start_distance)    # I add
        else:   # I add
            d = np.cos(from_center_angle) * (r + start_distance - (cell_size / 2))  # Tom's
        if d > np.max(cell_positions):  # I add
            d = np.cos(from_center_angle) * np.max(cell_positions)  # I add
        h = -np.sin(from_center_angle) * r

        if not (np.min(cell_positions) < d < np.max(cell_positions)):
            print('------------------')
            print(f'xy: {(x, y)}')
            print(f'x_rel: {x_rel}')
            print(f'y_rel: {y_rel}')
            print(f'r: {r}')
            print(f'd_angle: {d_angle}')
            print(f'angle: {angle}')
            print(f'cell_positions: {cell_positions}')
            print(f'np.cos(from_center_angle): {np.cos(from_center_angle)}')
            print(f'r + start_distance - (cell_size / 2): {r + start_distance - (cell_size / 2)}')
            print(
                f'[np.min(cell_positions), d, np.max(cell_positions)]: '
                f'{[np.min(cell_positions), d, np.max(cell_positions)]}')
            print('-----------------')
            raise ValueError

        if not (-cell_size / 2) < h < (cell_size / 2):
            raise ValueError

        ti_frac = np.interp(d, cell_positions, ti_fractions)

        return ti_frac, _temperature, _annealing_time, _thickness

    return TransformPair(to_bl_coords, to_data_coords)


def single_strip_set_transform_factory(strips, *, cell_size=4.5):
    """
    Generate the forward and reverse transforms for set of strips.
    This assumes that the strips are mounted parallel to one of the
    real motor axes.
    This assumes that the temperature and annealing time have been
    pre-snapped.
    Parameters
    ----------
    strips : List[StripInfo]
    cell_size : float, optional
       The size of each cell along the gradient where the Ti fraction
       is measured in mm.
    Returns
    -------
    to_data_coords, to_bl_coords
    """
    by_annealing = defaultdict(list)
    by_strip = {}

    for strip in strips:
        pair = single_strip_transform_factory(*astuple(strip))
        by_annealing[(strip.temperature, strip.annealing_time, strip.thickness)].append(
            (strip, pair)
        )
        by_strip[strip] = pair

    def forward(Ti_frac, temperature, annealing_time, thickness):
        candidates = by_annealing[(temperature, annealing_time, thickness)]

        # we need to find a strip that has the right Ti_frac available
        for strip, pair in candidates:
            if strip.ti_min <= Ti_frac <= strip.ti_max:
                return pair.forward(Ti_frac, temperature, annealing_time, thickness)
        else:
            # get here if we don't find a valid strip!
            raise ValueError

    def inverse(x, y):
        # the y value fully determines what strip we are in
        for strip, pair in by_strip.items():
            if (
                    strip.reference_y - cell_size / 2
                    < y
                    < strip.reference_y + cell_size / 2
            ):
                return pair.inverse(x, y)

        else:
            raise ValueError

    return TransformPair(forward, inverse)


@dataclass(frozen=True)
class StripInfo:
    """Container for strip information."""

    temperature: int
    annealing_time: int
    # exclude the ti_fraction from the hash
    ti_fractions: list = field(hash=False)
    reference_x: float
    reference_y: float
    start_distance: float
    angle: float
    # treat this as a categorical
    thickness: int

    # helpers to get the min/max of the ti fraction range.
    @property
    def ti_min(self):
        return min(self.ti_fractions)

    @property
    def ti_max(self):
        return max(self.ti_fractions)


def load_from_json(fname):
    """
    Load strip info from a json file.
    Parameters
    ----------
    fname : str or Path
        File to write
    Returns
    -------
    list[StripInfo]
    """
    # TODO make this take a file-like as well
    with open(fname, "r") as fin:
        data = json.load(fin)

    return [StripInfo(**d) for d in data]


def compute_peak_area(Q, I, q_start, q_stop):
    """
    Integrated area under a peak with estimated background removed.
    Estimates the background by averaging the 3 values on either side
    of the peak and subtracting that as a constant from I before
    integrating.
    Parameters
    ----------
    Q, I : array
        The q-values and binned intensity.  Assumed to be same length.
    q_start, q_stop : float
        The region of q to integrate.  Must be in same units as the Q.
    Returns
    -------
    peak_area : float
    """

    # figure out the index of the start and stop of the q
    # region of interest
    start, stop = np.searchsorted(Q, (q_start, q_stop))
    # add one to stop because we want the index after the end
    # value not the one before
    stop += 1
    # pull out the region of interest from I.
    data_section = I[start:stop]
    # pull out one more q value than I because we want the bin widths.
    q_section = Q[start : stop + 1]
    # compute width of each of the Q bins.
    dQ = np.diff(q_section)
    # estimate the background level by averaging the 3 and and 3 I(q) outside of
    # our ROI in either direction.
    background = (np.mean(I[start - 3 : start]) + np.mean(I[stop : stop + 3])) / 2
    # do the integration!
    return np.sum((data_section - background) * dQ)


def extract_data(h):
    d = h.primary.read()  # this is an xarray
    step = h.metadata['start']['adaptive_step']
    # Average the Q, I(Q) in time (we have 3 measurements at different ys), the snapped ctrl and requested ctrl
    return d['q'].mean('time'), d['mean'].mean('time'), step['snapped'], step['requested']
    # requested: the last one predicted

def compute_total_area(Q, I):
    """
    Trapezoidal Rule
    :param Q: array, q space
    :param I: array, spectrum intensity
    :return: int, total area
    """
    start = 0
    stop = -1
    h = Q[1]-Q[0]   # Step
    s = 0.5*(I[start]+I[stop])
    for i in range(1, len(Q)-1):
        s += I[i]
    integral = h*s
    return integral


if __name__ == '__main__':
    main()