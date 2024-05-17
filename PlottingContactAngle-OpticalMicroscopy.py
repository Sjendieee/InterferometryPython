import matplotlib.pyplot as plt
import csv
import glob
import numpy as np
import os
from pathlib import Path
import logging

def extractContourNumbersFromFile(lines):
    importedData1 = []
    importedData2 = []
    importedData3 = []
    importedData4 = []
    importedData5 = []
    importedData6 = []
    importedData7 = []

    for line in lines[8:]:
        cleaned_string = line.replace(',', '.')
        importedData1.append(float(cleaned_string.split('\t')[0]))
        importedData2.append(float(cleaned_string.split('\t')[1]))
        importedData3.append(float(cleaned_string.split('\t')[2]))
        importedData4.append(float(cleaned_string.split('\t')[3]))
        importedData5.append(int(cleaned_string.split('\t')[4]))
        importedData6.append(float(cleaned_string.split('\t')[5]))
        importedData7.append(float(cleaned_string.split('\t')[6]))

    return importedData1, importedData5

def importData(filePath):
    if os.path.exists(
            filePath):  # read in all contourline data from existing file (filenr ,+ i for obtaining contour location)
        f = open(filePath, 'r')
        lines = f.readlines()
        CA_mean, run_nr = extractContourNumbersFromFile(lines)
    return CA_mean, run_nr

def runToTime(run_nr, framerate, outputformat):
    if outputformat == 's':
        time = np.divide(run_nr, framerate)
    elif outputformat == 'min':
        time = np.divide(run_nr, framerate * 60)
    elif outputformat == 'h':
        time = np.divide(run_nr, framerate *60*60)
    else:
        logging.error("No correct output format given. Please select eithe rseconds ('s'), minutes ('min') or hours ('h').")
    return time

def main():
    path = "D:\\2024_05_07 Contact Angle Microscopy hexadecane PODMA 2_8_S7 Temp variation\\29C.txt"
    #path = "D:\\2024_05_07 Contact Angle Microscopy hexadecane PODMA 2_8_S7 Temp variation\\32_5C continuation after 5min.txt"

    outputFormat = 's'          #s, min, h
    framerate = 3.4     #frames (runs) / second
    CA_mean, run_nr = importData(path)
    time = runToTime(run_nr, framerate, 's')
    fig1, ax1 = plt.subplots()
    ax1.plot(time, CA_mean, '.')
    ax1.set(xlabel=f'Time ({outputFormat})', ylabel='Contact Angle (deg)', title=f"Contact angle evolution at {Path(path).stem}")
    ax1.set_ylim([30, 40])
    plt.show()
    fig1.savefig(os.path.join(os.path.dirname(path), f'Plot {Path(path).stem} - CA vs time .png'), dpi=600)

if __name__ == "__main__":
    main()
    exit()

