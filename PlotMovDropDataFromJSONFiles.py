import os
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
import glob
import re

def path_in_use():
    path = "H:\\2024_05_07_PLMA_Basler15uc_Zeiss5x_dodecane_Xp1_31_S2_WEDGE_2coverslip_spacer_V3"
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


def analyzeForcevsTime(JSON_folder):
    time = []
    force_quad = []
    force_trapz_function = []
    force_trapz_data = []
    for filename in glob.glob(os.path.join(JSON_folder, f"*json")):
        with open(filename, 'r') as file:
            json_data = json.load(file)
        time.append(json_data['timeFromStart'])
        force_quad.append(json_data['F_hor-quad-fphi'][0])              #TODO: check waarom dit 2 waardes zijn
        force_trapz_function.append(json_data['F-hor-trapz-fphi'])
        force_trapz_data.append(json_data['F-hor-trapz-data'])

    fig1, ax1 = plt.subplots()
    ax1.plot(np.array(time) / 60, force_quad, '.', label="quad integration")
    ax1.set(xlabel = 'time (min)', ylabel = 'Force (microN)', title = 'Horizontal force over time')
    ax1.legend(loc='best')

    fig2, ax2 = plt.subplots()
    ax2.plot(np.array(time) / 60, force_trapz_function, '.', label="trapz integration, function")
    ax2.set(xlabel = 'time (min)', ylabel = 'Force (microN)', title = 'Horizontal force over time')
    ax2.legend(loc='best')

    fig3, ax3 = plt.subplots()
    ax3.plot(np.array(time) / 60, force_trapz_data, '.', label="trapz integration, data")
    ax3.set(xlabel = 'time (min)', ylabel = 'Force (microN)', title = 'Horizontal force over time')
    ax3.legend(loc='best')
    return

def analyzeVelocityProfile_middleSurfaceArea(JSON_folder, path_images):
    time = []
    middleCoord_surfaceArea = []
    velocity = [0]

    for filename in glob.glob(os.path.join(path_images, f"*json")):
        with open(filename, 'r') as file:
            json_measurement_data = json.load(file)
    conversionFactorXY = json_measurement_data['conversionFactorXY']        #mm/pixel
    if json_measurement_data['unitXY'] != 'mm':
        logging.critical(f"UnitXY is not in mm: make sure to adjust code!")
        exit()

    analyzedJsonFiles = glob.glob(os.path.join(JSON_folder, f"*json"))
    analyzedJsonFiles.sort(key=alphanum_key)
    for filename in analyzedJsonFiles:
        with open(filename, 'r') as file:
            json_data = json.load(file)
        time.append(json_data['timeFromStart'])
        middleCoord_surfaceArea.append(json_data['middleCoords-surfaceArea'])

    for n, coord in enumerate(middleCoord_surfaceArea):
        if n == 0:
            pass
        else:
            dx = abs(middleCoord_surfaceArea[n-1][0] - coord[0])
            dy = abs(middleCoord_surfaceArea[n-1][1] - coord[1])
            dxy = (dx**2 + dy**2)**0.5                              #covered distance in pixels
            dxy_units = dxy * conversionFactorXY                    #units (mm)
            dt = time[n] - time[n-1]                                #difference in time (s)

            velocity.append(dxy_units / dt)

    fig1, ax1 = plt.subplots()
    ax1.plot(np.array(time) / 60, np.array(velocity) * 60, '.', label="middlecoord surface area")
    ax1.set(xlabel='time (min)', ylabel='velocity (mm/min)', title='Velocity profile')
    ax1.legend(loc='best')
    return

def analyzeVelocityProfile_adv_rec(JSON_folder, path_images):
    time = []
    coord_right = []
    coord_left = []
    velocity_left = [0]
    velocity_right = [0]

    for filename in glob.glob(os.path.join(path_images, f"*json")):
        with open(filename, 'r') as file:
            json_measurement_data = json.load(file)
    conversionFactorXY = json_measurement_data['conversionFactorXY']        #mm/pixel
    if json_measurement_data['unitXY'] != 'mm':
        logging.critical(f"UnitXY is not in mm: make sure to adjust code!")
        exit()

    analyzedJsonFiles = glob.glob(os.path.join(JSON_folder, f"*json"))
    analyzedJsonFiles.sort(key=alphanum_key)
    for filename in analyzedJsonFiles:
        with open(filename, 'r') as file:
            json_data = json.load(file)
        time.append(json_data['timeFromStart'])
        coord_right.append(json_data['OuterRightPixel'])
        coord_left.append(json_data['OuterLeftPixel'])

    for n, coord in enumerate(coord_right):
        if n == 0:
            pass
        else:
            dx_right = abs(coord_right[n-1][0] - coord[0])
            dy_right = abs(coord_right[n-1][1] - coord[1])
            dxy_right = (dx_right**2 + dy_right**2)**0.5                              #covered distance in pixels
            dxy_units_right = dxy_right * conversionFactorXY                    #units (mm)

            dx_left = abs(coord_left[n-1][0] - coord_left[n][0])
            dy_left = abs(coord_right[n-1][1] - coord_left[n][1])
            dxy_left = (dx_left**2 + dy_left**2)**0.5                              #covered distance in pixels
            dxy_units_left = dxy_left * conversionFactorXY                    #units (mm)

            dt = time[n] - time[n-1]                                #difference in time (s)

            velocity_right.append(dxy_units_right / dt)
            velocity_left.append(dxy_units_left / dt)

    fig1, ax1 = plt.subplots()
    ax1.plot(np.array(time) / 60, np.array(velocity_right) * 60, '.', label="velocity right")
    ax1.plot(np.array(time) / 60, np.array(velocity_left) * 60, '.', label="velocity left")
    ax1.set(xlabel='time (min)', ylabel='velocity (mm/min)', title='Velocity profile')
    ax1.legend(loc='best')
    return

def main():
    path_images = path_in_use()
    analysisFolder = os.path.join(path_images, "Analysis CA Spatial") #name of output folder of Spatial Contact Analysis
    JSON_folder = os.path.join(analysisFolder, "Analyzed Data")

    #analyzeForcevsTime(JSON_folder)
    try:
        #analyzeVelocityProfile_middleSurfaceArea(JSON_folder, path_images)
        analyzeVelocityProfile_adv_rec(JSON_folder, path_images)
    except:
        pass

    plt.show()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')     #configuration for printing logging messages. Can be removed safely
    main()
    exit()