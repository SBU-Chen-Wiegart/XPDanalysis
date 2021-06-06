"""
File: Autonomous experiments of the dealloyed Ti-Cu metal
Name: Cheng-Chu Chung
----------------------------------------
TODO: Simulate gpCAM autonomous data using different searching conditions
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict, astuple, field
from collections import namedtuple, defaultdict
from databroker._drivers.msgpack import BlueskyMsgpackCatalog
import time
import json
from gpcam.gp_optimizer import GPOptimizer
from numpy.random import default_rng
from scipy.interpolate import CubicSpline
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
    # plot_Ti(gpcam_db, pair, grid_db)
    # plot_temperature(gpcam_db)
    # plot_annealing_time(gpcam_db)
    # plot_roi(gpcam_db)
    # plot_grid_data(grid_db, pair)
    # plot_xca_from_karen()
    # grid_list(grid_db, pair)
    # ti_, temp_, time_, thickness_ = pair[1](68.5, 31.95)
    # print(ti_, temp_, time_, thickness_)
    # plot_xca(xca_db, pair, grid_db, peak_loc=(2.925, 2.974))
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
    # the_last_scan = -3    # Scan from the last scan
    # result = grid_db[the_last_scan]     # Extract data from a scan_id
    the_last_scan1 = -2
    result1 = grid_missing[the_last_scan1]
    print(result1.primary.read())
    # for center in range(1, len(xca_db)):
    #     print(xca_db[-center].metadata['start']['center_point'])
    # for i in result1.metadata['start']:
    #     print(i, result1.metadata['start'][i])
    # otime = result.metadata['start']['original_start_time']
    # otime1 = result1.metadata['start']['original_start_time']
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(otime)))
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(otime1)))
    #################################################################### Load Grid data
    # map = np.empty([16, 34])
    # for y1 in range(1, np.shape(CuMg_311_351_norm)[0]):
    #     for x1 in range(1, np.shape(CuMg_311_351_norm)[1]):
    #         map[y1-1, x1-1] = CuMg_311_351_norm[y1, x1]
    # plt.imshow(map, interpolation='bicubic', extent=[28.5, 94.5, 84.2, 12.95], origin='lower',
    #            cmap='plasma')
    # plt.colorbar()
    # # dx, dy = 0.015, 0.05
    # # y, x = np.mgrid[slice(-4, 4 + dy, dy),
    # #                 slice(-4, 4 + dx, dx)]
    # # z = (1 - x / 3. + x ** 6 + y ** 3) * np.exp(-x ** 2 - y ** 2)
    # # z = z[:-1, :-1]
    # # z_min, z_max = -np.abs(z).max(), np.abs(z).max()
    # # c = plt.pcolormesh(x, y, z, cmap='Greens', vmin=z_min, vmax=z_max)
    # # plt.colorbar(c)
    # plt.title('2.965 (Cu2Mg) (3 1 1) or 2.9499 (CuMg2) (3 5 1) \n peak location = [2.925, 2.974]')
    # # fig, ax = plt.subplots()
    # # ax.pcolormesh(x, y, z)
    # plt.xlabel('X location - Ti Composition (at.%)')
    # plt.ylabel('Y location - Dealloying Time and Temperature')
    # plt.show()
    #################################################################### Convert the coordinates based on Json file
    # print(pair[0](result.metadata['start']['adaptive_step']['snapped']['ctrl_Ti'],
    #               result.metadata['start']['adaptive_step']['snapped']['ctrl_temp'],
    #               result.metadata['start']['adaptive_step']['snapped']['ctrl_annealing_time'],
    #               result.metadata['start']['adaptive_step']['snapped']['ctrl_thickness']))
    ##################################################################### Read data
    # d = result.primary.read()
    # print(d)
    ##################################################################### Compute the roi
    # Compute the roi
    # peak_location = (2.925, 2.974)  # region of interest (roi) (351) CuMg2
    # q, I, snapped, requested = extract_data(result)
    # roi = np.array([compute_peak_area(q, I, *peak_location)])
    # # roi = np.array([roi])
    # print(roi[0])
    ##################################################################### Make a plot of a single measurement
    # fig, ax = plt.subplots()
    # ax.plot(q, I, label=str(snapped.values()))
    # # Label shows: 'ctrl_Ti', 'ctrl_annealing_time', 'ctrl_temp', 'ctrl_thickness'
    # ax.legend()
    # ax.set_xlabel('q')
    # ax.set_ylabel('I')
    # plt.show()


def plot_xca(xca_db, pair, grid_db, peak_loc=(2.925, 2.974)):
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
    initial_ti = 0.0
    index = 0
    phase = 'None'
    peak_location = peak_loc
    for i in range(1, len(xca_db) + 1):  # Extract all information from metadata['start']
        result = xca_db[-i]
        otime = result.metadata['start']['original_start_time']  # Assign original_start_time as otime
        ti = result.metadata['start']['center_point'][0]
        temp = result.metadata['start']['center_point'][1]
        annealing_time = result.metadata['start']['center_point'][2]
        peak_location = peak_location
        q, I = result.primary.read()['q'].mean('time'), result.primary.read()['mean'].mean('time')
        # compute_total_area(q, I)  # Compute the total area of the spectrum
        roi = compute_peak_area(q, I, *peak_location)
        roi = np.array([roi])
        thickness = result.metadata['start']['center_point'][3]
        beamline_x, beamline_y = pair[0](ti, temp, annealing_time, thickness)
        if ti != initial_ti:
            index += 1
            if index % 3 == 1:
                phase = 'MgCu2'
            elif index % 3 == 2:
                phase = 'Ti'
            else:
                phase = 'Mg2Cu'
            initial_ti = ti

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

    grid_array = grid_list(grid_db, pair, peak_loc=peak_location)
    map_thin = np.empty(
        [8, 34])  # We were supposed to have 18X34, but the last two rows didn't be measured due to beam down
    for row in range(8):
        for column in range(34):
            map_thin[row, column] = grid_array[34 * row + column][2]  # Scan from left to right, down to upper
    map_thick = np.empty(
        [7, 34])  # We were supposed to have 18X34, but the last two rows didn't be measured due to beam down
    for row in range(7):
        for column in range(34):
            map_thick[row, column] = grid_array[34 * (row + 9) + column][2]  # Scan from left to right, down to upper
    # Plot the grid scan background
    plt.imshow(map_thick, vmin=-0.07, vmax=0.37, interpolation='bicubic', extent=[94.5, 28.5, 41.45, 12.95],
               origin='lower',
               cmap='plasma')  # Y range = 41.45 to 12.95 mm, vmin=-0.07, vmax=0.37, vmin=-0.00007, vmax=0.00045,
    plt.imshow(map_thin, vmin=-0.07, vmax=0.37, interpolation='bicubic', extent=[94.5, 28.5, 84.2, 50.95],
               origin='lower',
               cmap='plasma')  # Y range = 84.2 to 50.95 mm  vmin=-0.07, vmax=0.37, vmin=-0.00007, vmax=0.00045,
    plt.text(82, 47.5,
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
        plt.scatter(beamline_x_axis_MgCu2, beamline_y_axis_MgCu2, marker='^', s=32, edgecolor='black', facecolor='None',
                    label='Acquiring data', linestyle='-', linewidth=0.5)
        plt.title('Crystallography companion agent (XCA) scan\n'
                  '1.547 ($\mathregular{Cu_2Mg}$) (1 1 1)\n'
                  'Peak location = [1.526, 1.588]')
    if peak_location == (2.635, 2.708):
        plt.scatter(beamline_x_axis_Ti, beamline_y_axis_Ti, marker='o', s=32, edgecolor='black', facecolor='None',
                    label='Acquiring data', linestyle='-', linewidth=0.5)
        plt.title('Crystallography companion agent (XCA) scan\n'
                  r'2.665 ($\mathregular{\beta-Ti}$) (1 1 0)''\n'
                  'Peak location = [2.635, 2.708]')
    if peak_location == (2.734, 2.779):
        plt.scatter(beamline_x_axis_Mg2Cu, beamline_y_axis_Mg2Cu, marker='o', s=32, edgecolor='black', facecolor='None',
                    label='Acquiring data', linestyle='-', linewidth=0.5)
        plt.title('Crystallography companion agent (XCA) scan\n'
                  '2.755 ($\mathregular{CuMg_2}$) (0 8 0)\n'
                  'Peak location = [2.734, 2.779]')
    if peak_location == (2.925, 2.974):
        plt.scatter(beamline_x_axis_MgCu2, beamline_y_axis_MgCu2, marker='^', s=32, edgecolor='black', facecolor='None',
                    label='$\mathregular{Cu_2Mg}$ (3 1 1)', linestyle='-', linewidth=0.5)
        plt.scatter(beamline_x_axis_Mg2Cu, beamline_y_axis_Mg2Cu, marker='o', s=32, edgecolor='black', facecolor='None',
                    label='$\mathregular{CuMg_2}$ (3 5 1)', linestyle='-', linewidth=0.5)
        plt.title('Crystallography companion agent (XCA) scan\n'
                  '2.965 ($\mathregular{Cu_2Mg}$) (3 1 1) or 2.9499 ($\mathregular{CuMg_2}$) (3 5 1)\n'
                  'Peak location = [2.925, 2.974]')
    if peak_location == (3.047, 3.106):
        plt.scatter(beamline_x_axis_MgCu2, beamline_y_axis_MgCu2, marker='^', s=32, edgecolor='black', facecolor='None',
                    label='$\mathregular{Cu_2Mg}$ (2 2 2)', linestyle='-', linewidth=0.5)
        plt.scatter(beamline_x_axis_Mg2Cu, beamline_y_axis_Mg2Cu, marker='o', s=32, edgecolor='black', facecolor='None',
                    label='$\mathregular{CuMg_2}$ (4 4 0)', linestyle='-', linewidth=0.5)
        plt.title('Crystallography companion agent (XCA) scan\n'
                  '3.093 ($\mathregular{Cu_2Mg}$) (2 2 2) or 3.095 ($\mathregular{CuMg_2}$) (4 4 0)\n'
                  'Peak location = [3.047, 3.106]')

    plt.ylim(84.2, 3.45)
    plt.xlabel('X position (mm)')
    plt.ylabel('Y position (mm)')
    plt.legend(loc='upper left')
    plt.show()
    ########################################## Plot 3D visualization (Ti concentration, Annealing temperature and time)
    x = np.array(y_axis)
    y = np.array(y_axis_temp)
    z = np.array(z_axis_time)
    intensity = np.array(intensity_roi)
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


def plot_xca_from_karen():
    df = pd.read_csv('output_probabilities.csv')
    xca_data = np.array(df)
    P0 = np.array([])
    P1 = np.array([])
    Probability_of_mgcu2 = np.array([])
    Probability_of_ti = np.array([])
    Probability_of_mg2cu = np.array([])
    for i in range(len(xca_data)):
        P0 = np.append(P0, xca_data[i][1])
        P1 = np.append(P1, xca_data[i][2])
        Probability_of_mgcu2 = np.append(Probability_of_mgcu2, xca_data[i][3])
        Probability_of_ti = np.append(Probability_of_ti, xca_data[i][4])
        Probability_of_mg2cu = np.append(Probability_of_mg2cu, xca_data[i][5])
    # print(Probability_of_mgcu2)
    # X, Y = np.meshgrid(P0, P1)
    # Z = Probability_of_mgcu2
    # plt.pcolormesh(X, Y, Z)
    # heatmap, xedges, yedges = np.histogram2d(P0, P1, bins=(len(P0), len(P1)))
    # heatmap = gaussian_filter(heatmap, sigma=32)
    # img = heatmap.T
    # plt.imshow(img, interpolation='bicubic', origin='lower',
    #            extent=[28.5, 94.5, 84.2, 3.45], cmap='plasma')
    # map = np.empty([16, 34])
    # for y1 in range(1, np.shape(CuMg_311_351_norm)[0]):
    #     for x1 in range(1, np.shape(CuMg_311_351_norm)[1]):
    #         map[y1 - 1, x1 - 1] = CuMg_311_351_norm[y1, x1]
    # plt.imshow(map, interpolation='bicubic', extent=[28.5, 94.5, 84.2, 3.45], origin='lower',
    #            cmap='plasma')  # Y range = 84.2 to 12.95 mm
    plt.scatter(P0, P1, c=Probability_of_mgcu2, marker='o', s=32,
                label='MgCu2')
    # plt.gca().invert_xaxis()  # Invert the x axis
    plt.gca().invert_yaxis()  # Invert the y axis
    plt.clim(0.05, 0.5)
    plt.colorbar()
    plt.title('$\mathregular{MgCu_2}$')
    plt.show()
    plt.scatter(P0, P1, c=Probability_of_ti, marker='o', s=32,
                label='Ti')
    # plt.gca().invert_xaxis()  # Invert the x axis
    plt.gca().invert_yaxis()  # Invert the y axis
    plt.clim(0.05, 0.5)
    plt.colorbar()
    plt.title('$\mathregular{Ti}$')
    plt.show()
    plt.scatter(P0, P1, c=Probability_of_mg2cu, marker='o', s=32,
                label='Mg2Cu')
    # plt.gca().invert_xaxis()  # Invert the x axis
    plt.gca().invert_yaxis()  # Invert the y axis
    plt.clim(0.05, 0.5)
    plt.colorbar()
    plt.title('$\mathregular{Mg_2Cu}$')
    plt.show()


def plot_Ti(gpcam_db, pair, grid_db):
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
    y_axis_thin = []  # Ti concentration with thin Mg
    y_axis_thick = []  # Ti concentration with thick Mg
    y_axis_temp = []  # Annealing temperature
    z_axis_time = []  # Annealing time
    intensity_roi = []  # Region of interest
    beamline_x_axis = np.array([])  # Beamline x position
    beamline_y_axis = np.array([])  # Beamline y position
    for j in range(len(time_list)):
        if time_list[j][0] > '2020-12-11 13:15:00':  # Extract the data meausured after this due to calibration
            # Print (Original start start time, Ti, Temp, Annealing time, roi, thickness, beamline_x, beamline_y)
            # print('{}, {:.1f}, {}, {:4}, {:7.4f}, {}, {:6.4f}, {:7.4f}'.format(time_list[j][0], time_list[j][1],
            #                                                                  time_list[j][2], time_list[j][3],
            #                                                                  time_list[j][4], time_list[j][5],
            #                                                                  time_list[j][6], time_list[j][7]))
            x_axis.append(j)  # Run sequence
            y_axis.append(time_list[j][1])  # Ti concentration
            y_axis_temp.append(time_list[j][2])
            z_axis_time.append(time_list[j][3])
            intensity_roi.append(time_list[j][4])
            beamline_x_axis = np.append(beamline_x_axis, time_list[j][6])
            beamline_y_axis = np.append(beamline_y_axis, time_list[j][7])
            if time_list[j][5] == 0:  # If Mg agent is thick
                x_axis_thin.append(j)
                y_axis_thin.append(time_list[j][1])  # If the Mg agent is thin
            if time_list[j][5] == 1:
                x_axis_thick.append(j)
                y_axis_thick.append(time_list[j][1])
    ###################################################### Plot 2D stacking map (grid and gpCAM data)
    # ----------------------------------------------- From plot_grid_data function
    grid_array = grid_list(grid_db, pair)  # From grid_list function
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
        plt.show()
    # #################################################### Plot the grid scan background
    map_thin = np.empty(
        [8, 34])  # We were supposed to have 18X34, but the last two rows didn't be measured due to beam down
    for row in range(8):
        for column in range(34):
            map_thin[row, column] = grid_array[34 * row + column][2]  # Scan from left to right, down to upper
    map_thick = np.empty(
        [7, 34])  # We were supposed to have 18X34, but the last two rows didn't be measured due to beam down
    for row in range(7):
        for column in range(34):
            map_thick[row, column] = grid_array[34 * (row + 9) + column][2]  # Scan from left to right, down to upper
    ################################################## Heat map
    plt.imshow(map_thick, vmin=-0.07, vmax=0.37, interpolation='bicubic', extent=[94.5, 28.5, 41.45, 12.95],
               origin='lower',
               cmap='plasma')  # Y range = 41.45 to 12.95 mm, vmin=-0.07, vmax=0.37, vmin=-0.00007, vmax=0.00045,
    plt.imshow(map_thin, vmin=-0.07, vmax=0.37, interpolation='bicubic', extent=[94.5, 28.5, 84.2, 50.95],
               origin='lower',
               cmap='plasma')  # Y range = 84.2 to 50.95 mm  vmin=-0.07, vmax=0.37, vmin=-0.00007, vmax=0.00045,
    plt.text(82, 47.5,
             'Ni standard on glass slide',
             fontsize=12)
    cbar = plt.colorbar()
    cbar.set_label('Region of interest (Roi)', size=12)
    # plt.clim(-0.07, 0.37)
    plt.ylim(84.2, 3.45)
    plt.xlabel('X position (mm)')
    plt.ylabel('Y position (mm)')
    # ------------------------------------------------ From plot grid data above
    plt.plot(beamline_x_axis, beamline_y_axis, color='k',
             label='Trajectory', linestyle='--', linewidth=0.5)
    plt.scatter(beamline_x_axis, beamline_y_axis, marker='o', s=32, edgecolor='black', facecolor='None',
                label='Acquiring data', linestyle='-', linewidth=0.5)
    # plt.scatter(beamline_x_axis, beamline_y_axis, c=x_axis_color, marker='o', s=32,
    #             label='Region of interest', linestyle='-', linewidth=1)
    # text_x = str(beamline_x_axis.tolist())
    # text_y = str(beamline_y_axis.tolist())
    # plt.text(text_x, text_y, x_axis, fontsize=9)
    plt.xlabel('X position (mm)')
    plt.ylabel('Y position (mm)')
    # plt.ylim(84.2, 3.45)
    # plt.gca().invert_xaxis()    # Invert the x axis
    plt.title('gpCAM run number after 1:15 pm\n'
              '2.965 ($\mathregular{Cu_2Mg}$) (3 1 1) or 2.9499 ($\mathregular{CuMg_2}$) (3 5 1)\n'
              'Peak location = [2.925, 2.974]')
    plt.legend(loc='upper left')
    plt.show()
    ########################################## Plot 3D visualization (Ti concentration, Annealing temperature and time)
    x = np.array(y_axis)
    y = np.array(y_axis_temp)
    z = np.array(z_axis_time)
    sequence = np.array(x_axis)
    intensity = np.array(intensity_roi)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Ti concentration')
    ax.set_ylabel('Annealing temperature')
    ax.set_zlabel('Annealing time')
    p = ax.scatter3D(x, y, z, c=intensity, marker='o', s=(sequence - 140) * 5, label='Acquiring data')
    ax.plot3D(x, y, z, 'black', linestyle='--', linewidth=0.5, label='Trajectory')
    cbar = fig.colorbar(p, ax=ax, pad=0.2)
    cbar.set_label('Region of interest (Roi)', size=12)
    plt.title('gpCAM run number after 1:15 pm\n'
              '2.965 ($\mathregular{Cu_2Mg}$) (3 1 1) or 2.9499 ($\mathregular{CuMg_2}$) (3 5 1)\n'
              'Peak location = [2.925, 2.974]')
    plt.legend()
    plt.show()


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


def plot_grid_data(grid_db, pair, peak_loc=(2.925, 2.974)):
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
    ################################################## Extract beamline x, y and roi information
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
        plt.plot(d['q'][i], d['mean'][i])
        roi = compute_peak_area(d['q'][i], d['mean'][i], *peak_location)  # Compute the roi
        roi = np.array([roi][0])  # Clean the format to become an int
        # total_area = compute_total_area(np.array(d['q'][i]), np.array(d['mean'][i]))
        # roi = roi/total_area
        roi_array = np.append(roi_array, roi)  # Collect the roi of the a phase
        # Collect beamline x, y and roi information
        grid_array = np.append(grid_array, [np.array(d['sample_x'][i]), np.array(d['ss_stg2_y'][i]), roi])
    plt.title('Grid Scan')
    plt.ylabel('Intensity (a.u.)')
    plt.xlabel('q')
    plt.show()
    grid_array = grid_array.reshape(-1, 3)  # Make it easier to be processed
    map_thin = np.empty(
        [8, 34])  # We were supposed to have 18X34, but the last two rows didn't be measured due to beam down
    for row in range(8):
        for column in range(34):
            map_thin[row, column] = grid_array[34 * row + column][2]  # Scan from left to right, down to upper
    map_thick = np.empty(
        [7, 34])  # We were supposed to have 18X34, but the last two rows didn't be measured due to beam down
    for row in range(7):
        for column in range(34):
            map_thick[row, column] = grid_array[34 * (row + 9) + column][2]  # Scan from left to right, down to upper
    ################################################## Heat map
    plt.imshow(map_thick, vmin=-0.07, vmax=0.37, interpolation='bicubic', extent=[94.5, 28.5, 41.45, 12.95],
               origin='lower',
               cmap='plasma')  # Y range = 41.45 to 12.95 mm, vmin=-0.07, vmax=0.37, vmin=-0.00007, vmax=0.00033,
    plt.imshow(map_thin, vmin=-0.07, vmax=0.37, interpolation='bicubic', extent=[94.5, 28.5, 84.2, 50.95],
               origin='lower',
               cmap='plasma')  # Y range = 84.2 to 50.95 mm  vmin=-0.07, vmax=0.37
    plt.text(82, 47.5,
             'Ni standard on glass slide',
             fontsize=12)
    cbar = plt.colorbar()
    cbar.set_label('Region of interest (Roi)', size=12)
    # plt.clim(-0.07, 0.37)
    plt.ylim(84.2, 12.95)
    plt.xlabel('X position (mm)')
    plt.ylabel('Y position (mm)')
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
    plt.scatter(np.array(d['sample_x']), np.array(d['ss_stg2_y']), c=roi_array, marker='o', s=32,
                label='Acquiring data', linestyle='--', linewidth=1)
    plt.gca().invert_xaxis()  # Invert the x axis
    plt.gca().invert_yaxis()  # Invert the y axis
    cbar = plt.colorbar()
    cbar.set_label('Region of interest (Roi)', size=12)
    plt.xlabel('X position (mm)')
    plt.ylabel('Y position (mm)')
    plt.title('Grid scan\n'
              '2.965 ($\mathregular{Cu_2Mg}$) (3 1 1) or 2.9499 ($\mathregular{CuMg_2}$) (3 5 1)\n'
              'Peak location = [2.925, 2.974]')
    plt.legend()
    plt.show()


def grid_list(grid_db, pair, peak_loc=(2.925, 2.974)):
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
    for i in range(len(d['sample_x'])):
        roi = compute_peak_area(d['q'][i], d['mean'][i], *peak_location)
        roi = np.array([roi][0])
        roi_array = np.append(roi_array, roi)
        grid_array = np.append(grid_array, [np.array(d['sample_x'][i]), np.array(d['ss_stg2_y'][i]), roi])
    grid_array = grid_array.reshape(-1, 3)
    # print(grid_array)
    return grid_array


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
        d = np.cos(from_center_angle) * (r + start_distance - (cell_size / 2))
        h = -np.sin(from_center_angle) * r

        if not (np.min(cell_positions) < d < np.max(cell_positions)):
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