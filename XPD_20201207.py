"""
File: Autonomous experiments of the dealloyed Ti-Cu metal
Name: Cheng-Chu Chung
----------------------------------------
TODO: plot Ti vs. run(first, second scan...)
"""
import matplotlib.pyplot as plt
import numpy as np
from databroker._drivers.msgpack import BlueskyMsgpackCatalog
import time


def main():
    # xca_db = BlueskyMsgpackCatalog(['/mnt/data/bnl/2020-12_ae/adaptive_reduced/xca/*msgpack'])
    gpcam_db = BlueskyMsgpackCatalog(
        ['D:/Software/Python/SSID/XPD_20201207_TiCuMg_alloy_auto_1/adaptive_reduced/gpcam/*msgpack'])
    # grid_db = BlueskyMsgpackCatalog(
    #     ['D:/Software/Python/SSID/XPD_20201207_TiCuMg_alloy_auto_1/adaptive_reduced/grid/*msgpack'])
    # print('Scanning numbers:', len(grid_db))
    thick_measurements = list(gpcam_db.search({'adaptive_step.snapped.ctrl_thickness': 1}))
    thin_measurements = list(gpcam_db.search({'adaptive_step.snapped.ctrl_thickness': 0}))
    # print('Thick samples:', len(thick_measurements))
    # print('Thin samples:', len(thin_measurements))
    # check_scan_id_and_CPU_time(gpcam_db)
    # plot_Ti(gpcam_db)
    plot_temperature(gpcam_db)
    plot_annealing_time(gpcam_db)
    plot_roi(gpcam_db)

    # the_last_scan = -1    # Scan from the last scan
    # result = gpcam_db[the_last_scan]     # Extract data from a scan_id
    # print(result.metadata['start'])

    # print(result.primary.read())    # Information for each scan, a xarray dataset
    # Compute the roi
    # peak_location = (2.925, 2.974)  # region of interest (roi) (351) CuMg2
    # q, I, snapped, requested = extract_data(result)
    # roi = np.array([compute_peak_area(q, I, *peak_location)])
    # # roi = np.array([roi])
    # print(roi[0])

    # # Make a plot of a single measurement
    # fig, ax = plt.subplots()
    # ax.plot(q, I, label=str(snapped.values()))
    # # Label shows: 'ctrl_Ti', 'ctrl_annealing_time', 'ctrl_temp', 'ctrl_thickness'
    # ax.legend()
    # ax.set_xlabel('q')
    # ax.set_ylabel('I')
    # plt.show()


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
    plt.ylabel('Region of interest (roi)')
    plt.title('gpCAM')
    plt.legend(loc='upper left')
    plt.show()


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
    plt.show()


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
    plt.show()


def plot_Ti(gpcam_db):
    print('Original start time, Ti and thickness')
    time_list = []
    for i in range(1, len(gpcam_db) + 1):   # Extract all information from metadata['start']
        result = gpcam_db[-i]
        otime = result.metadata['start']['original_start_time']     # Assign original_start_time as otime
        ti = result.metadata['start']['adaptive_step']['snapped']['ctrl_Ti']
        thickness = result.metadata['start']['adaptive_step']['snapped']['ctrl_thickness']
        time_list.append((time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(otime)), ti, thickness))     # Append (time, ti)

    time_list.sort()
    x_axis = []
    y_axis = []
    x_axis_thin = []
    x_axis_thick = []
    y_axis_thin = []
    y_axis_thick = []
    for j in range(len(time_list)):
        if time_list[j][0] > '2020-12-11 13:15:00':
            print('{}, {}, {}'.format(time_list[j][0], round(time_list[j][1], 1), time_list[j][2]))
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
    plt.ylabel('Ti percentage (%)')
    plt.ylim(0, 100)
    plt.title('gpCAM')
    plt.legend(loc='upper left')
    plt.show()


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