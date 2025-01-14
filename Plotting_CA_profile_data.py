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

def path_in_use():
    """
    Write path to folder in which the analyzed images (and subsequent analysis) are
    :return:
    """
    path = "H:\\2024_05_07_PLMA_Basler15uc_Zeiss5x_dodecane_Xp1_31_S2_WEDGE_2coverslip_spacer_V3"
    #path = "D:\\2024-09-04 PLMA dodecane Xp1_31_2 ZeissBasler15uc 5x M3 tilted drop"

    metadata = dict(title='Movie', artist='Sjendieee')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    return path


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



    ax1.plot(np.array(timeList)/60, CA_min, '.', markersize=8, label=f'CA min')
    ax1.plot(np.array(timeList)/60, CA_max, '.', markersize=8, label=f'CA max')
    ax1.set(xlabel = 'time (min)', ylabel = 'Contact Angle (deg)', title = 'Min & Max Contact Angle (averaged 40 points) over time')
    ax1.legend(loc='best')
    fig1.savefig(f"C:\\Downloads\\CAPlot", dpi=600)

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

    csv_dataList = [f for f in glob.glob(os.path.join(analysisFolder, f"ContactAngleData*.csv"))]  # grab all json files

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
    nr_time_mapping = dict(zip(nr, time))  # Map `nr` to `time`
    nrList = [number for number in valid_nrs]
    timeList = [nr_time_mapping[number] for number in nr]


    #Functions for plotting
    plottingMaxima_And_Minima_vsTime(filtered_csv_dataList, analysisFolder, outputname, nrList, timeList)
    #plottingTop_And_BottomCA_vsTime(filtered_csv_dataList, analysisFolder, outputname, nrList, timeList)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')     #configuration for printing logging messages. Can be removed safely
    main()
    exit()
