"""
Functions for plotting various CA's vs. time:
    -plottingMaxima_And_Minima_vsTime():    extract min and maximum CA found in entire CA dataset vs. time
        -adv.: quick & easy.       -neg.: prone to errors in CA anlaysis (e.g. near pinning sites), yielding too high/low CA's
    -plottingTop_And_BottomCA_vsTime():     extract CA's at 12&6 'oclock.

    -selectCA_minmax():     select regions where to find a min/maximum CA.

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
import glob
import re

import pandas as pd
from matplotlib.animation import FFMpegWriter
import cv2
from matplotlib.widgets import RectangleSelector
from scipy.signal import savgol_filter
import traceback
import pickle
from natsort import natsorted

from shapely.predicates import is_empty


def path_in_use():
    """
    Write path to folder in which the analyzed images (and subsequent analysis) are
    :return:
    """
    path = "G:\\2024_05_07_PLMA_Basler15uc_Zeiss5x_dodecane_Xp1_31_S2_WEDGE_2coverslip_spacer_V3"
    #path = "D:\\2024-09-04 PLMA dodecane Xp1_31_2 ZeissBasler15uc 5x M3 tilted drop"
    #path = "F:\\2023_11_13_PLMA_Dodecane_Basler5x_Xp_1_24S11los_misschien_WEDGE_v2"


    metadata = dict(title='Movie', artist='Sjendieee')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    return path

right_clicks = list()
def click_event(event, x, y, flags, params):
    '''
    Click event for the setMouseCallback cv2 function. Allows to select 2 points on the image and return it coordinates.
    '''
    if event == cv2.EVENT_LBUTTONDOWN:
        global right_clicks
        right_clicks.append([x, y])
    if len(right_clicks) == 2:
        cv2.destroyAllWindows()
class Highlighter(object):
    def __init__(self, ax, x, y):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.x, self.y = x, y
        self.mask = np.zeros(x.shape, dtype=bool)
        self._highlight = ax.scatter([], [], s=200, color='yellow', zorder=10)
        self.selector = RectangleSelector(ax, self, useblit=True)

    def __call__(self, event1, event2):
        self.mask |= self.inside(event1, event2)
        xy = np.column_stack([self.x[self.mask], self.y[self.mask]])
        self._highlight.set_offsets(xy)
        self.canvas.draw()

    def inside(self, event1, event2):
        """Returns a boolean mask of the points inside the rectangle defined by
        event1 and event2."""
        # Note: Could use points_inside_poly, as well
        x0, x1 = sorted([event1.xdata, event2.xdata])
        y0, y1 = sorted([event1.ydata, event2.ydata])
        mask = ((self.x > x0) & (self.x < x1) &
                (self.y > y0) & (self.y < y1))
        return mask

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def ffmpegPath():
    paths_ffmpeg = ['C:\\Users\\ReuvekampSW\\Desktop\\ffmpeg-7.1-essentials_build\\bin\\ffmpeg.exe',  # UT pc & laptop
                    'C:\\Users\\Sander PC\\Desktop\\ffmpeg-7.1-essentials_build\\bin\\ffmpeg.exe'  # thuis pc
                    ]
    for ffmpeg_path in paths_ffmpeg:
        if os.path.exists(ffmpeg_path):
            plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path        #set path to ffmpeg file.
            break
    if not os.path.exists(ffmpeg_path):
        logging.critical("No good path to ffmpeg.exe.\n Correct path, or install from e.g. https://www.gyan.dev/ffmpeg/builds/#git-master-builds")
    return ffmpeg_path

def videoMakerOfImges(imgList, analysisFolder, videoname, fps = 1, compression = 100):
    #Read in size of original image for compression later
    referenceFrame = cv2.imread(imgList[0])

    (inputHeight, inputWidth, referenceLayers) = referenceFrame.shape
    outputHeight = round(inputHeight * (compression / 100))
    outputWidth = round(inputWidth * (compression / 100))

    #https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python
    ffmpeg_path = ffmpegPath()
    output_folder = analysisFolder
    video_name = os.path.join(analysisFolder, videoname)
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')       #tried for mp4 - didn't work: https://stackoverflow.com/questions/30103077/what-is-the-codec-for-mp4-videos-in-python-opencv/55596396
    video = cv2.VideoWriter(video_name, 0, fps, (outputWidth, outputHeight))      #output name, codec used, FPS, tuple of dimensions

    for n, img in enumerate(imgList):
        logging.info(f"Processing image {n}/{len(imgList)}")
        img = cv2.resize(cv2.imread(img), (outputWidth, outputHeight), interpolation=cv2.INTER_AREA)
        video.write(img)
    cv2.destroyAllWindows()
    video.release()
    logging.info(f'Finished turning images into a video')

def coordsToPhi(xArrFinal, yArrFinal, medianmiddleX, medianmiddleY):
    """
    :return phi: range [-pi : pi]
    :return rArray: distance from the middle to the coordinate. UNIT= same as input units (so probably pixel, or e.g. mm)

    phi = 0 at right side -> 0.5pi at top -> 1pi at left -> -1pi at left -> -0.5pi at bottom
    example how phi evolves: https://stackoverflow.com/questions/17574424/how-to-use-atan2-in-combination-with-other-radian-angle-systems
    """
    dx = np.subtract(xArrFinal, medianmiddleX)
    dy = np.subtract(yArrFinal, medianmiddleY)
    phi = np.arctan2(dy, dx)  # angle of contour coordinate w/ respect to 12o'clock (radians)
    rArray = np.sqrt(np.square(dx) + np.square(dy))
    return phi, rArray


def plottingMaxima_And_Minima_vsTime(csv_data_list, analysisFolder, videoname, nrList, timeList, neighborhood_size=20):
    """
    Reads data from a list of CSV files and extracts the minimum and maximum values.
    Parameters:
        csv_data_list (list): List of paths to the CSV files.
    Returns:
        list:
    """
    fig1, ax1 = plt.subplots(figsize=(9,6))
    CA_min = []
    CA_max = []
    phi_min = []
    phi_max = []
    for n, csv_file in enumerate(csv_data_list):
        try:

            #extract nr from filepath
            json_data = json.load(open(os.path.join(analysisFolder, f"Analyzed Data\\{nrList[n]}_analysed_data.json"), 'r'))
            middleOfArea = (json_data['middleCoords-surfaceArea'])

            # Read the CSV file into a DataFrame
            data = pd.read_csv(csv_file)

            # Assuming the CSV has a single column or a specific column name, adapt accordingly
            # If the data column is unnamed, it'll be automatically indexed as data.iloc[:, 0]
            values = data.iloc[:, 2]  # Select the first column
            xvals = data.iloc[:, 0]  # Select the first column
            yvals = data.iloc[:, 1]  # Select the first column

            # # Extract & append minimum and maximum values
            # CA_min.append(values.min())
            # CA_max.append(values.max())

            #TODO show both min & max from both sides, not only abs value

            # Find the max value and its position (index)
            max_value = values.max()
            max_position = values.idxmax()  # Get the index of the max value
            # Define the range for neighborhood averaging
            start = max(0, max_position - neighborhood_size)
            end = min(len(values), max_position + neighborhood_size + 1)
            # Calculate the average of the nearby values
            CA_max.append(values[start:end].mean())
            xmax, ymax = xvals[max_position], yvals[max_position]
            phimax, _ = coordsToPhi(xmax, ymax, middleOfArea[0], middleOfArea[1])
            phi_max.append(phimax)

            # Find the min value and its position (index)
            min_value = values.min()
            min_position = values.idxmin()  # Get the index of the min value
            # Define the range for min neighborhood averaging
            min_start = max(0, min_position - neighborhood_size)
            min_end = min(len(values), min_position + neighborhood_size + 1)
            # Calculate the average of the nearby values for min
            CA_min.append(values[min_start:min_end].mean())
            xmin, ymin = xvals[min_position], yvals[min_position]
            phimin, _ = coordsToPhi(xmin, ymin, middleOfArea[0], middleOfArea[1])
            phi_min.append(phimin)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            print(traceback.format_exc())

    ax1.plot(np.array(timeList)/60, CA_min, '.', markersize=8, label=f'CA min')
    ax1.plot(np.array(timeList)/60, CA_max, '.', markersize=8, label=f'CA max')
    ax1.set(xlabel = 'time (min)', ylabel = 'Contact Angle (deg)', title = 'Min & Max Contact Angle (averaged 40 points) over time')
    ax1.legend(loc='best')
    #fig1.savefig(f"C:\\Downloads\\CAPlot", dpi=600)

    fig2, ax2 = plt.subplots(figsize = (9,6))
    ax2.plot(np.array(timeList)/60, np.absolute(phi_min) / np.pi * 180, '.', markersize=8, label=f'Angle of CA min')
    ax2.plot(np.array(timeList)/60, np.absolute(phi_max) / np.pi * 180, '.', markersize=8, label=f'Angle of CA max')
    ax2.set(xlabel = 'time (min)', ylabel = 'Angle (deg)', title = 'Angle at which CA_min and CA_max are found\n Note: both top & bottom half combined (abs value)')
    ax2.legend(loc='best')

    plt.show()
    return



def plottingTop_And_BottomCA_vsTime(csv_data_list, analysisFolder, videoname, nrList, timeList, neighborhood_size=20):
    """
    Reads data from a list of CSV files and extracts the minimum and maximum values.
    Parameters:
        csv_data_list (list): List of paths to the CSV files.
    Returns:
        list:
    """
    fig1, ax1 = plt.subplots(figsize=(9,6))
    CA_min = []
    CA_max = []

    for csv_file in csv_data_list:
        try:
            # Read the CSV file into a DataFrame
            data = pd.read_csv(csv_file)

            # Assuming the CSV has a single column or a specific column name, adapt accordingly
            # If the data column is unnamed, it'll be automatically indexed as data.iloc[:, 0]
            valuesY = data.iloc[:, 1]  # Select the first column
            valuesCA = data.iloc[:, 2]  # Select the first column

            # # Extract & append minimum and maximum values
            # CA_min.append(values.min())
            # CA_max.append(values.max())

            # Find the max value OF Y and its position (index)
            max_value = valuesY.max()
            max_position = valuesY.idxmax()  # Get the index of the max value
            # Define the range for neighborhood averaging
            start = max(0, max_position - neighborhood_size)
            end = min(len(valuesCA), max_position + neighborhood_size + 1)
            # Calculate the average of the nearby values
            CA_max.append(valuesCA[start:end].mean())

            # Find the min value and its position (index)
            min_value = valuesY.min()
            min_position = valuesY.idxmin()  # Get the index of the min value
            # Define the range for min neighborhood averaging
            min_start = max(0, min_position - neighborhood_size)
            min_end = min(len(valuesCA), min_position + neighborhood_size + 1)
            # Calculate the average of the nearby values for min
            CA_min.append(valuesCA[min_start:min_end].mean())

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")



    ax1.plot(np.array(timeList)/60, CA_min, '.', markersize=8, label='CA top')       #TODO mind you: Y as a coordinate increases when going down in the figure. SO be sure CA in X-Y plot is corresponding with what we call top&bottom here
    ax1.plot(np.array(timeList)/60, CA_max, '.', markersize=8, label='CA bottom')      #TODO ^ invert top & bottom for min & max
    ax1.set(xlabel = 'time (min)', ylabel = 'Contact Angle (deg)', title = 'Top & Bottom Contact Angle (averaged 40 points) over time')
    ax1.legend(loc='best')
    fig1.savefig(f"C:\\Downloads\\CAPlot", dpi=600)
    plt.show()
    return

def plottingOuterAdvRecCA_vsTime(csv_data_list, analysisFolder, videoname, nrList, timeList, neighborhood_size=20):
    """
    Reads data from a list of CSV files and extracts the minimum and maximum values.
    Parameters:
        csv_data_list (list): List of paths to the CSV files.
    Returns:
        list:
    """
    fig1, ax1 = plt.subplots(figsize=(9,6))
    CA_adv = []
    CA_rec = []

    for csv_file in csv_data_list:
        try:
            # Read the CSV file into a DataFrame
            data = pd.read_csv(csv_file)

            # Assuming the CSV has a single column or a specific column name, adapt accordingly
            # If the data column is unnamed, it'll be automatically indexed as data.iloc[:, 0]
            valuesX = data.iloc[:, 0]  # Select the first column
            valuesCA = data.iloc[:, 2]  # Select the first column

            # Find the max value OF X and its position (index)
            max_value = valuesX.max()
            max_position = valuesX.idxmax()  # Get the index of the max value
            # Define the range for neighborhood averaging
            start = max(0, max_position - neighborhood_size)
            end = min(len(valuesCA), max_position + neighborhood_size + 1)
            # Calculate the average of the nearby values
            CA_adv.append(valuesCA[start:end].mean())

            # Find the min value and its position (index)
            min_value = valuesX.min()
            min_position = valuesX.idxmin()  # Get the index of the min value
            # Define the range for min neighborhood averaging
            min_start = max(0, min_position - neighborhood_size)
            min_end = min(len(valuesCA), min_position + neighborhood_size + 1)
            # Calculate the average of the nearby values for min
            CA_rec.append(valuesCA[min_start:min_end].mean())

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    ax1.plot(np.array(timeList) / 60, CA_adv, '.', markersize=8,
             label='CA_adv')  # TODO mind you: Y as a coordinate increases when going down in the figure. SO be sure CA in X-Y plot is corresponding with what we call top&bottom here
    ax1.plot(np.array(timeList) / 60, CA_rec, '.', markersize=8,
             label='CA_rec')  # TODO ^ invert top & bottom for min & max
    ax1.set(xlabel='time (min)', ylabel='Contact Angle (deg)',
            title='Outer Advancing & Receding Contact Angle (averaged 40 points) in time')
    ax1.legend(loc='best')
    fig1.savefig(os.path.join(analysisFolder, "A_advrec_CA_vsTime"), dpi=600)
    plt.show()
    return


#USABLE!#
def selectCA_minmax(csv_data_list, analysisFolder, videoname, nrList, timeList, neighborhood_size=20):
    """
    Reads data from a list of CSV files, plots the CA profile, and allows the user to select MANUALLY IN A REGION where a min or max is to be found.
    Then, plots this info vs. time.

    Parameters:
        csv_data_list (list): List of paths to the CSV files.
    Returns:
        list:
    """

    CA_min = []
    CA_max = []
    phi_min = []
    phi_max = []
    infotuple = []      #will contain tuples of (time(int), 'min' (str), CA (flt), phi(flt))
    if os.path.exists(os.path.join(analysisFolder, "pickle dumps\\CA analysis.p")):
        infotuple = pickle.load(open(os.path.join(analysisFolder, "pickle dumps\\CA analysis.p"), "rb"))
    else:
        backup_file = os.path.join(analysisFolder, "pickle dumps\\backup_data.pkl")       #backup tuple save file for data
        backup_nrs = []
        if os.path.exists(backup_file): #if a backup file exist already, load its data
            with open(backup_file, "rb") as f:
                backup_results = pickle.load(f)
            for datapoint in backup_results:
                backup_nrs.append(datapoint[0])

        for n, csv_file in enumerate(csv_data_list):
            print(f"Extracting data from file nr {n}/{len(csv_data_list)}")

            if nrList[n] in backup_nrs:
                print(f"Loading data from backup file")
                infotuple.append(backup_results[np.where(np.array(backup_nrs)) == nrList[n][0][0]][1:])

            else:
                try:
                    #extract nr from filepath
                    json_data = json.load(open(os.path.join(analysisFolder, f"Analyzed Data\\{nrList[n]}_analysed_data.json"), 'r'))
                    middleOfArea = (json_data['middleCoords-surfaceArea'])

                    # Read the CSV file into a DataFrame
                    data = pd.read_csv(csv_file)

                    # Assuming the CSV has a single column or a specific column name, adapt accordingly
                    # If the data column is unnamed, it'll be automatically indexed as data.iloc[:, 0]
                    values = np.array(data.iloc[:, 2])  # Select the first column
                    xvals = data.iloc[:, 0]  # Select the first column
                    yvals = data.iloc[:, 1]  # Select the first column

                    phi_total, _ = coordsToPhi(xvals, yvals, middleOfArea[0], middleOfArea[1])

                    #plot CA in colorplot & allow for selection where to find min & maxima
                    #FIRST MAXIMA
                    fig1, ax1 = plt.subplots(figsize=(9,6))
                    im1 = ax1.scatter(xvals, yvals, c=values, cmap='jet', vmin=min(values), vmax=max(values), label=f'Local contact angles')
                    highlighter = Highlighter(ax1, np.array(xvals), np.array(yvals))
                    ax1.set_xlabel("X-coord"); ax1.set_ylabel("Y-Coord"); ax1.set_title(f"Spatial Contact Angle Colormap\n SELECT MAXIMA")
                    fig1.colorbar(im1, format="%.3f")
                    plt.show()
                    #determine selected areas and what to do with them: TODO
                    selected_regions_MAXIMA = highlighter.mask
                    plt.close()

                    #THEN MINIMA
                    fig1, ax1 = plt.subplots(figsize=(9,6))
                    im1 = ax1.scatter(xvals, yvals, c=values, cmap='jet', vmin=min(values), vmax=max(values), label=f'Local contact angles')
                    highlighter = Highlighter(ax1, np.array(xvals), np.array(yvals))
                    ax1.set_xlabel("X-coord"); ax1.set_ylabel("Y-Coord"); ax1.set_title(f"Spatial Contact Angle Colormap\n SELECT MINIMA")
                    fig1.colorbar(im1, format="%.3f")
                    plt.show()
                    #determine selected areas and what to do with them: TODO
                    selected_regions_MINIMA = highlighter.mask

                    def findRanges(selected_regions):
                        # Indices are found here
                        a_true_ranges = np.argwhere(np.diff(selected_regions, prepend=False, append=False))
                        # # Conversion into list of 2-lists
                        a_true_ranges = a_true_ranges.reshape(len(a_true_ranges) // 2, 2)

                        if len(a_true_ranges) > 1:  #if multiple regions are 'selected':
                            if (a_true_ranges[0][0] == 0 or a_true_ranges[0][0] == 1) and a_true_ranges[-1][1] == len(selected_regions):  #selected regions wrap around end-of-list back to beginning
                                a_true_ranges[0][0] = a_true_ranges[-1][0]      #add 'last range' starting point into 'first range'
                                a_true_ranges = a_true_ranges[0:-1]     #remove last 'selected range'

                        a_true_ranges_final = []
                        for range in a_true_ranges:     #only add index 'ranges' into list if more than 10 datapoints were selected (to ftiler off weird short ranges of e.g. 1 point)
                            if abs(range[1] - range[0]) > 10:
                                a_true_ranges_final.append(range)
                        return a_true_ranges_final

                    a_true_ranges_maxima = findRanges(selected_regions_MAXIMA)
                    a_true_ranges_minima = findRanges(selected_regions_MINIMA)

                    print(f"nr. of ranges selected MAXIMA: {len(a_true_ranges_maxima)}.\n Selected ranges are: {a_true_ranges_maxima}")
                    print(f"nr. of ranges selected MINIMA: {len(a_true_ranges_minima)}.\n Selected ranges are: {a_true_ranges_minima}")

                    for range in a_true_ranges_maxima:      #for all selected ranges, find maximum (w/ some averaging around it)
                        #if first value>second value, it wrapped around end-of-list: For proper analysis split up again and
                        #do analysis on both ranges 'combined'
                        if range[0] > range[1]:
                            i_range = np.arange(0, range[1], 1) + np.arange(range[0], len(values), 1)        #index numbers of range
                        else:   #else do analysis on single range
                            i_range = np.arange(range[0], range[1], 1)        #index numbers of range
                        i_max = i_range[np.argmax(values[i_range])]                  #find local max, and return its index in entire list
                        CA = np.mean(           #average over 20left&20right nearby for CA
                            np.roll(values, -i_max+20)[0:(20*2+1)]  #shift array so no issues arise when taking datapoints near the end-of-list
                        )
                        CA_max.append(CA)
                        phi = phi_total[i_max]
                        phi_max.append(phi)
                        infotuple.append((timeList[n], 'max', CA, phi))

                    for range in a_true_ranges_minima:      #for all selected ranges, find maximum (w/ some averaging around it)
                        #if first value>second value, it wrapped around end-of-list: For proper analysis split up again and
                        #do analysis on both ranges 'combined'
                        if range[0] > range[1]:
                            i_range = np.arange(0, range[1], 1) + np.arange(range[0], len(values), 1)        #index numbers of range
                        else:   #else do analysis on single range
                            i_range = np.arange(range[0], range[1], 1)        #index numbers of range
                        i_min = i_range[np.argmin(values[i_range])]                  #find local min, and return its index in entire list
                        CA = np.mean(           #average over 20left&20right nearby for CA
                            np.roll(values, -i_min+20)[0:(20*2+1)]  #shift array so no issues arise when taking datapoints near the end-of-list
                        )
                        CA_min.append(CA)
                        phi = phi_total[i_min]
                        phi_min.append(phi)
                        infotuple.append((timeList[n], 'min', CA, phi))
                        backup_results.append((nrList[n], timeList[n], 'min', CA, phi))
                    # #Determine min & max when 2 regions are selected
                    # inverted_selected_regions = [not elem for elem in selected_regions]  # invert booleans to 'deselect' the selected regions
                    # xrange1, yrange1 = np.array(xvals)[inverted_selected_regions], np.array(yvals)[inverted_selected_regions]

                    # #show something TODO
                    # filtered_angleDegArr = np.array(values)[inverted_selected_regions]
                    # fig3, ax3 = plt.subplots(figsize=(15, 9.6))
                    # im3 = ax3.scatter(xrange1, abs(np.subtract(4608, yrange1)), c=filtered_angleDegArr, cmap='jet', vmin=min(filtered_angleDegArr), vmax=max(filtered_angleDegArr))
                    # ax3.set_xlabel("X-coord"); ax3.set_ylabel("Y-Coord"); ax3.set_title( f"FILTERED Spatial Contact Angles Colormap")
                    # fig3.colorbar(im3)
                    # plt.show()

                except Exception as e:
                    print(f"Error processing {csv_file}: {e}")
                    with open(backup_file, "wb") as f:
                        pickle.dump(backup_results, f)
                    print(f"Saved data up till now sucesfully to {backup_file}")
                    print(traceback.format_exc())


        pickle.dump(infotuple, open(os.path.join(analysisFolder, "pickle dumps\\CA analysis.p"), "wb"))

    fig1, ax1 = plt.subplots(figsize=(9, 6))
    xdatamax = []; ydatamax = []
    xdatamin = []; ydatamin = []
    infotuple = sorted(infotuple)   #to sort the times from low-high (for plotting purposes; lines between points)

    for datapoint in infotuple:
        if datapoint[1] == 'max':       #maximum
            xdatamax.append(datapoint[0])
            ydatamax.append(datapoint[2])
        elif datapoint[1] == 'min':       #minimum
            xdatamin.append(datapoint[0])
            ydatamin.append(datapoint[2])
    ax1.plot(np.array(xdatamax) / 60, ydatamax, 'r.-', markersize=8, label=f'CA max')
    ax1.plot(np.array(xdatamin) / 60 , ydatamin, 'b.-', markersize=8, label=f'CA min')
    ax1.set(xlabel = 'time (min)', ylabel = 'Contact Angle (deg)', title = 'Min & Max Contact Angle (averaged 40 points) over time')
    ax1.legend(loc='best')
    fig1.savefig(os.path.join(analysisFolder, "plot-minMaxCA_vs_time_manSelected.png"), dpi=600)

    fig2, ax2 = plt.subplots(figsize = (9,6))
    xdatamax = []; ydatamax = []
    xdatamin = []; ydatamin = []
    for datapoint in infotuple:
        if datapoint[1] == 'max':       #maximum
            xdatamax.append(datapoint[0])
            ydatamax.append(datapoint[3])
        elif datapoint[1] == 'min':       #minimum
            xdatamin.append(datapoint[0])
            ydatamin.append(datapoint[3])
    ax2.plot(np.array(xdatamax) / 60, np.array(ydatamax) / np.pi * 180, 'r.-', markersize=8, label=f'Angle of CA max')
    ax2.plot(np.array(xdatamin) / 60 , np.array(ydatamin) / np.pi * 180, 'b.-', markersize=8, label=f'Angle of CA min')

    # ax2.plot(np.array(timeList)/60, np.absolute(phi_min) / np.pi * 180, '.', markersize=8, label=f'Angle of CA min')
    # ax2.plot(np.array(timeList)/60, np.absolute(phi_max) / np.pi * 180, '.', markersize=8, label=f'Angle of CA max')
    ax2.set(xlabel = 'time (min)', ylabel = 'Angle (deg)', title = 'Angle at which CA_min and CA_max are found')
    ax2.legend(loc='best')
    fig2.savefig(os.path.join(analysisFolder, "plot-angle of minmaxCA_vs_time.png"), dpi=600)
    plt.show()
    return


def main():
    path_images = path_in_use()
    analysisFolder = os.path.join(path_images, "Analysis CA Spatial")  # name of output folder of Spatial Contact Analysis

    outputname = f'ColorplotMovie.mp4v'
    json_dataList = [f for f in glob.glob(os.path.join(analysisFolder, f"Analyzed Data\\*.json"))]        #grab all json files
    nr = []
    time = []

    # Extract numbers (imgnr) & time from JSON files
    for jsonPath in json_dataList:
        json_data = json.load(open(jsonPath, 'r'))
        nr.append(json_data['imgnr'])
        time.append(json_data['timeFromStart'])     #in seconds

    csv_dataList = natsorted([f for f in glob.glob(os.path.join(analysisFolder, f"ContactAngleData*.csv"))])  # grab all json files

    # Filter CSV paths to retain only those containing an imgnr
    filtered_csv_dataList = [
        csv_path
        for csv_path in csv_dataList
        if any(str(number) in os.path.basename(csv_path) for number in nr)  # Check only the filename
    ]

    # Remove unused entries from `nr` and `time` to match filtered CSVs
    valid_nrs = [
        number
        for number in nr
        if any(str(number) in os.path.basename(csv_path) for csv_path in csv_dataList)
    ]
    # Rebuild `nr` and `time` lists with only valid entries
    nr_time_mapping = dict(sorted(zip(nr, time)))  # Map `nr` to `time`
    nrList = [number for number in nr_time_mapping]
    timeList = [nr_time_mapping[number] for number in nr_time_mapping]

    #final check the timelist & csv list are sorted in same order:
    for n, csv_path in enumerate(csv_dataList):
        if not str(nrList[n]) in os.path.basename(csv_path):
            logging.critical(f"number is sorted csv file does not match number is sorted nrList. That means either list is incorrectly sorted:\n"
                             f"n={n}, nrList[n] ={nrList[n]}, csvpath = {csv_path}")
            break


    #Functions for plotting

    plottingOuterAdvRecCA_vsTime(filtered_csv_dataList, analysisFolder, outputname, nrList, timeList)
    selectCA_minmax(filtered_csv_dataList, analysisFolder, outputname, nrList, timeList)

    plottingMaxima_And_Minima_vsTime(filtered_csv_dataList, analysisFolder, outputname, nrList, timeList)
    #plottingTop_And_BottomCA_vsTime(filtered_csv_dataList, analysisFolder, outputname, nrList, timeList)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')     #configuration for printing logging messages. Can be removed safely
    main()
    exit()
