"""
File: Autonomous experiments of the dealloyed Ti-Cu metal
Name: Cheng-Chu Chung
----------------------------------------
TODO: Simulate gaCAM autonomous data using different searching conditions
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass, asdict, astuple, field
from collections import namedtuple, defaultdict
from databroker._drivers.msgpack import BlueskyMsgpackCatalog
import time
import json
from gpcam.gp_optimizer import GPOptimizer
from numpy.random import default_rng
TransformPair = namedtuple("TransformPair", ["forward", "inverse"])


def main():
    strip_list = load_from_json("layout.json")
    pair = single_strip_set_transform_factory(strip_list)
    xca_db = BlueskyMsgpackCatalog(['/mnt/data/bnl/2020-12_ae/adaptive_reduced/xca/*msgpack'])
    gpcam_db = BlueskyMsgpackCatalog(
        ['D:/Software/Python/SSID/XPD_20201207_TiCuMg_alloy_auto_1/adaptive_reduced/gpcam/*msgpack'])
    grid_db = BlueskyMsgpackCatalog(
        ['D:/Software/Python/SSID/XPD_20201207_TiCuMg_alloy_auto_1/adaptive_reduced/grid/*msgpack'])
    # print('Scanning numbers:', len(grid))
    # thick_measurements = list(grid_db.search({'adaptive_step.snapped.ctrl_thickness': 1}))
    # thin_measurements = list(grid_db.search({'adaptive_step.snapped.ctrl_thickness': 0}))
    # print('Thick samples:', len(thick_measurements))
    # print('Thin samples:', len(thin_measurements))
    #################################################################### Run the data
    # check_scan_id_and_CPU_time(gpcam_db)
    plot_Ti(gpcam_db, pair)
    # plot_temperature(gpcam_db)
    # plot_annealing_time(gpcam_db)
    # plot_roi(gpcam_db)
    #################################################################### Simulate gpCAM
    # version_6_of_gpCAM(gpcam_db, pair)
    #################################################################### Run a specific code
    # the_last_scan = -1    # Scan from the last scan
    # result = grid_db[the_last_scan]     # Extract data from a scan_id
    # the_last_scan1 = -3
    # result1 = grid_db[the_last_scan1]
    # # print(result.metadata['start'])
    # # print(result.metadata['start']['hints']['dimensions'][0])
    # # for i in result.metadata['start']:
    # #     print(i, result.metadata['start'][i])
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
    # print(result1.primary.read())    # Information for each scan, a xarray dataset
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


def plot_Ti(gpcam_db, pair):
    print('Original start time, Ti, temp, time, roi, thickness, beamline x and beamline y')
    time_list = []
    for i in range(1, len(gpcam_db) + 1):   # Extract all information from metadata['start']
        result = gpcam_db[-i]
        otime = result.metadata['start']['original_start_time']     # Assign original_start_time as otime
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
    x_axis = []     # Run sequence
    y_axis = []     # Ti concentration
    x_axis_thin = []    # Run sequence of thin Mg
    x_axis_thick = []   # Run sequence of thick Mg
    y_axis_thin = []    # Ti concentration with thin Mg
    y_axis_thick = []   # Ti concentration with thick Mg
    y_axis_temp = []    # Annealing temperature
    z_axis_time = []    # Annealing time
    intensity_roi = []
    beamline_x_axis = np.array([])
    beamline_y_axis = np.array([])
    for j in range(len(time_list)):
        if time_list[j][0] > '2020-12-11 13:15:00':
            # Print (Original start start time, Ti, Temp, Annealing time, roi, thickness, beamline_x, beamline_y)
            # print('{}, {:.1f}, {}, {:4}, {:7.4f}, {}, {:6.4f}, {:7.4f}'.format(time_list[j][0], time_list[j][1],
            #                                                                  time_list[j][2], time_list[j][3],
            #                                                                  time_list[j][4], time_list[j][5],
            #                                                                  time_list[j][6], time_list[j][7]))
            x_axis.append(j)
            y_axis.append(time_list[j][1])
            y_axis_temp.append(time_list[j][2])
            z_axis_time.append(time_list[j][3])
            intensity_roi.append(time_list[j][4])
            beamline_x_axis = np.append(beamline_x_axis, time_list[j][6])
            beamline_y_axis = np.append(beamline_y_axis, time_list[j][7])
            if time_list[j][5] == 0:
                x_axis_thin.append(j)
                y_axis_thin.append(time_list[j][1])
            if time_list[j][5] == 1:
                x_axis_thick.append(j)
                y_axis_thick.append(time_list[j][1])
    ###################################################### Plot 1D curve
    # plt.plot(x_axis_thin, y_axis_thin, color='b', marker='o', markersize=5, markerfacecolor='b',
    #          label='Thin Mg', linestyle='-', linewidth=2)
    # plt.plot(x_axis_thick, y_axis_thick, color='r', marker='o', markersize=5, markerfacecolor='r',
    #          label='Thick Mg', linestyle='-', linewidth=2)
    # plt.plot(x_axis, y_axis, color='k', label='Run sequence', linestyle='--', linewidth=1)
    # plt.xlabel('Run number after 1:15 pm')
    # plt.ylabel('Ti percentage (%)')
    # plt.ylim(0, 100)
    # plt.title('gpCAM')
    # plt.legend(loc='upper left')
    ###################################################### Plot 2D map
    df = pd.read_excel('CuMg_311_351_full_norm.xlsx')
    CuMg_311_351_norm = np.array(df)

    map = np.empty([16, 34])
    for y1 in range(1, np.shape(CuMg_311_351_norm)[0]):
        for x1 in range(1, np.shape(CuMg_311_351_norm)[1]):
            map[y1-1, x1-1] = CuMg_311_351_norm[y1, x1]
    plt.imshow(map, interpolation='bicubic', extent=[28.5, 94.5, 84.2, 3.45], origin='lower',
               cmap='plasma')   # Y range = 84.2 to 12.95 mm
    plt.colorbar()
    x_axis_color = np.array(intensity_roi)
    plt.plot(beamline_x_axis, beamline_y_axis, color='k',
             label='Trajectory', linestyle='--', linewidth=0.5)
    plt.scatter(beamline_x_axis, beamline_y_axis, c=x_axis_color*10, marker='o', s=32,
                label='Region of interest', linestyle='--', linewidth=1)
    plt.xlabel('X position (mm)')
    plt.ylabel('Y position (mm)')
    # plt.ylim(84.2, 3.45)
    plt.gca().invert_xaxis()    # Invert the x axis
    plt.title('gpCAM run number after 1:15 pm\n'
              '2.965 ($\mathregular{Cu_2Mg}$) (3 1 1) or 2.9499 ($\mathregular{CuMg_2}$) (3 5 1)\n'
              'Peak location = [2.925, 2.974]')
    plt.legend(loc='upper left')
    plt.show()
    ###################################################### Plot 3D
    # x = np.array(y_axis)
    # y = np.array(y_axis_temp)
    # z = np.array(z_axis_time)
    # sequence = np.array(x_axis)
    # intensity = np.array(intensity_roi)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax = Axes3D(fig)
    # ax.set_xlabel('Ti concentration')
    # ax.set_ylabel('Annealing temperature')
    # ax.set_zlabel('Annealing time')
    # ax.scatter3D(x, y, z, c=intensity*100, marker='o', s=(sequence-140)*5)
    # ax.plot3D(x, y, z, 'black', linestyle='--', linewidth=0.5)
    # plt.show()


def version_6_of_gpCAM(gpcam_db, pair):
    # time_list = sorted_timelist(gpcam_db, pair)

    a = np.load("us_topo.npy")
    rng = default_rng()
    ind = rng.choice(len(a) - 1, size=3000, replace=False)
    points = a[ind, 0:2]
    values = a[ind, 2:3]
    print("x_min ", np.min(points[:, 0]), " x_max ", np.max(points[:, 0]))
    print("y_min ", np.min(points[:, 1]), " y_max ", np.max(points[:, 1]))
    print("length of data set: ", len(points))
    index_set_bounds = np.array([[0, 99], [0, 248]])
    hyperparameter_bounds = np.array([[0.001, 1e9], [1, 1000], [1, 1000]])
    hps_guess = np.array([4.71907062e+06, 4.07439017e+02, 3.59068120e+02])
    ###################################################################################
    gp = GPOptimizer(2, 1, 1, index_set_bounds)     # 3, 1, 1 for our system
    gp.tell(points, values)
    gp.init_gp(hps_guess)
    gp.train_gp(hyperparameter_bounds, likelihood_optimization_pop_size=20,
                likelihood_optimization_tolerance=1e-6, likelihood_optimization_max_iter=2)
    x_pred = np.empty((10000, 2))
    counter = 0
    x = np.linspace(0, 99, 100)
    y = np.linspace(0, 248, 100)
    for i in x:
        for j in y:
            x_pred[counter] = np.array([i, j])
            counter += 1
    res1 = gp.gp.posterior_mean(x_pred)
    res2 = gp.gp.posterior_covariance(x_pred)
    # res3 = gp.gp.shannon_information_gain(x_pred)
    X, Y = np.meshgrid(x, y)
    PM = np.reshape(res1["f(x)"], (100, 100))
    PV = np.reshape(res2["v(x)"], (100, 100))
    plt.figure(figsize=(10, 10))
    plt.pcolormesh(X, Y, PM)
    plt.figure(figsize=(10, 10))
    plt.pcolormesh(X, Y, PV)
    plt.show()
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


if __name__ == '__main__':
    main()