"""
File for plotting of various kinds of data (Force_horizontal, velocity vs time) from previously analysed images
Data is extracted from JSON files.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
import glob
import re
import traceback

def path_in_use():
    """
    Write path to folder in which the analyzed images (and subsequent analysis) are
    :return:
    """
    path = "G:\\2024_05_07_PLMA_Basler15uc_Zeiss5x_dodecane_Xp1_31_S2_WEDGE_2coverslip_spacer_V3"
    filter_images = list(np.arange(0, 21)) + [48, 72] + [96, 100] + [88, 92, 96, 100, 104, 108]

    # path = "G:\\2024_02_05_PLMA 160nm_Basler17uc_Zeiss5x_dodecane_FULLCOVER_v3"
    # filter_images = []
    #
    # path = "D:\\2024-09-04 PLMA dodecane Xp1_31_2 ZeissBasler15uc 5x M3 tilted drop"
    # filter_images = [63, 67]
    return path, filter_images


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


def analyzeForcevsTime(JSON_folder, path_images, filter_images, analysisFolder):
    """
    Plot horizontal force calculated in 3 ways vs time.
    -from quad integration on smooth fourier fit function   (imported data 2 values: First value = force. Second value = error)
    -trapz integration on raw data
    -trapz integration on smooth fourier fit function
    :param JSON_folder:
    :return:
    """
    time = []
    time_used = []
    imgnr = []
    imgnr_used = []
    force_quad = []
    force_quad_used = []
    force_trapz_function = []
    force_trapz_function_used = []
    force_trapz_data = []
    force_trapz_data_used = []

    for filename in glob.glob(os.path.join(JSON_folder, f"*json")):
        with open(filename, 'r') as file:
            json_data = json.load(file)
        time.append(json_data['timeFromStart'])
        imgnr.append(json_data['imgnr'])
        force_quad.append(json_data['F_hor-quad-fphi'][0])              #First value = force. Second value = error
        force_trapz_function.append(json_data['F-hor-trapz-fphi'])
        force_trapz_data.append(json_data['F-hor-trapz-data'])

    for n in range(0,len(time)):
        if imgnr[n] in filter_images:
            pass        #if to be filtered, don't add info into lists
        else:
            time_used.append(time[n])
            imgnr_used.append(imgnr[n])
            force_quad_used.append(force_quad[n])
            force_trapz_function_used.append(force_trapz_function[n])
            force_trapz_data_used.append(force_trapz_data[n])
    fig1, ax1 = plt.subplots()
    ax2 = ax1.twiny()  # create double x-axis
    ax1.plot(np.array(time_used) / 60, np.array(force_quad_used) * 1000, '.', markersize=7, label="quad integration")
    ax1.set(xlabel = 'time (min)', ylabel = r'Force ($\mu$N)', title = 'Horizontal force over time')
    #ax1.legend(loc='best')
    #fig1.tight_layout()

    #fig2, ax2 = plt.subplots()
    ax1.plot(np.array(time_used) / 60, np.array(force_trapz_function_used) * 1000, '.', markersize=7,  label="trapz integration, function")
    #ax2.set(xlabel = 'time (min)', ylabel = 'Force (uN)', title = 'Horizontal force over time')
    #ax2.legend(loc='best')
    #fig2.tight_layout()

    #fig3, ax3 = plt.subplots()
    ax1.plot(np.array(time_used) / 60, np.array(force_trapz_data_used) * 1000, '.', markersize=7,  label="trapz integration, data")
    #ax3.set(xlabel = 'time (min)', ylabel = 'Force (uN)', title = 'Horizontal force over time')

    ax2.set_xlabel('img nr [-]')
    lines = ax2.plot(imgnr_used, force_quad_used) #create dummy plot to set x-axis
    lines[0].remove()  # remove 'dummy' data but retain axis

    ax1.legend(loc='best')
    fig1.tight_layout()
    fig1.savefig(os.path.join(analysisFolder, 'A_AllHorizontalForces.png'), dpi=600)
    return

def analyzeVelocityProfile_middleSurfaceArea(JSON_folder, path_images, filter_images, analysisFolder):
    """
    Plot middle of droplet pixel velocity, obtained from mean surfacearea calculation:
    import pixel coordinate location & calculate how much distance it moves in between frames for velocity
    :param JSON_folder:
    :param path_images:
    :return:
    """
    time = []
    time_used = []      #for plotting of non-filtered times
    middleCoord_surfaceArea = []
    velocity = []
    imgnr = []
    imgnr_used = []

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
        imgnr.append(json_data["imgnr"])

    for n, coord in enumerate(middleCoord_surfaceArea):
        if imgnr[n] in filter_images:
            pass        #if to be filtered, don't add info into lists
        else:
            if n == 0:
                velocity.append(0)
                time_used.append(time[n])
                imgnr_used.append(imgnr[n])
                pass
            else:
                dx = abs(middleCoord_surfaceArea[n-1][0] - coord[0])
                dy = abs(middleCoord_surfaceArea[n-1][1] - coord[1])
                dxy = (dx**2 + dy**2)**0.5                              #covered distance in pixels
                dxy_units = dxy * conversionFactorXY                    #units (mm)
                dt = time[n] - time[n-1]                                #difference in time (s)

                velocity.append(dxy_units / dt)
                time_used.append(time[n])
                imgnr_used.append(imgnr[n])
    fig1, ax1 = plt.subplots()
    ax1.plot(np.array(time_used) / 60, np.array(velocity) * 60 * 1000, '.', label="middlecoord surface area")
    ax1.set(xlabel='time (min)', ylabel='velocity ($\mu$m/min)', title='Velocity profile: Middle of droplet')

    #ax1.plot(imgnr_used, np.array(velocity) * 60, '.', label="middlecoord surface area")
    #ax1.set(xlabel='frame number (-)', ylabel='velocity (mm/min)', title='Velocity profile: Middle of droplet')
    ax1.legend(loc='best')
    fig1.tight_layout()
    fig1.savefig(os.path.join(analysisFolder, 'A_middleOfDropVSTime.png'), dpi=600)
    return fig1, ax1, time_used, velocity, velocity, imgnr_used

def analyzeVelocityProfile_adv_rec(JSON_folder, path_images, filter_images, analysisFolder, **kwargs):
    """
    Plot outer left & outer right pixel velocity:
    import pixel coordinate location & calculate how much distance it moves in between frames for velocity
    :param JSON_folder:
    :param path_images:
    :param kwargs:
    :return:
    """
    time_factor = 60                        #1 = second, 60 = minutes, etc..
    velocity_factor = 60*1000             #1 = mm/second, 60 = mm/min, etc..
    ylim = []
    fig1, ax1 = plt.subplots()

    for keyword, value in kwargs.items():
        if keyword == 'ylim':
            ylim = value
        elif keyword == 'fig':
            fig1 = value
        elif keyword == 'ax':
            ax1 = value
        else:
            logging.error(f"Incorrect keyword inputted: {keyword} is not known")

    time = []
    time_used = []
    coord_right = []
    coord_left = []
    velocity_left = []
    velocity_right = []
    imgnr = []
    imgnr_used = []

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
        imgnr.append(json_data["imgnr"])

    for n, coord in enumerate(coord_right):
        if imgnr[n] in filter_images:
            pass        #if to be filtered, don't add info into lists
        else:
            if n == 0:
                velocity_right.append(0)
                velocity_left.append(0)
                time_used.append(time[n])
                imgnr_used.append(imgnr[n])
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
                time_used.append(time[n])
                imgnr_used.append(imgnr[n])

    ax1.plot(np.array(time_used) / time_factor, np.array(velocity_right) * velocity_factor, '.', label="velocity right")
    ax1.plot(np.array(time_used) / time_factor, np.array(velocity_left) * velocity_factor, '*', label="velocity left")
    ax1.set(xlabel='time (min)', ylabel='velocity ($\mu$m/min)', title='Velocity profile')
    if ylim:
        ax1.set(ylim=ylim)
    ax1.legend(loc='best')
    fig1.tight_layout()
    fig1.savefig(os.path.join(analysisFolder, 'A_adv_rec_velocityVStime.png'), dpi=600)
    return time_used, velocity_left, velocity_right, imgnr_used

def main():
    path_images, filter_images = path_in_use()
    analysisFolder = os.path.join(path_images, "Analysis CA Spatial") #name of output folder of Spatial Contact Analysis
    JSON_folder = os.path.join(analysisFolder, "Analyzed Data")


    try:
        analyzeForcevsTime(JSON_folder, path_images, filter_images, analysisFolder)
    except:
        print(traceback.format_exc())

    plt.show()

    try:
        _,_, time_used, velocity, velocity, imgnr_used = analyzeVelocityProfile_middleSurfaceArea(JSON_folder, path_images, filter_images, analysisFolder)
        time_used, velocity_left, velocity_right, imgnr_used = analyzeVelocityProfile_adv_rec(JSON_folder, path_images, filter_images, analysisFolder, ylim=[0, 200])#, fig=fig1, ax=ax1)
        time_used = time_used[1:]
        velocity = velocity[1:]
        velocity_left = velocity_left[1:]
        velocity_right = velocity_right[1:]
        imgnr_used = imgnr_used[1:]

        fig1, ax1 = plt.subplots()
        ax2 = ax1.twiny()  # create double x-axis
        ax1.plot(np.array(time_used) / 60, np.array(velocity_left) * 60 * 1000, 'm*', label="velocity left")
        ax1.plot(np.array(time_used) / 60, np.array(velocity) * 60 * 1000, 'kx', label="middlecoord surface area")
        ax1.plot(np.array(time_used) / 60, np.array(velocity_right) * 60 * 1000, 'k.', markersize=9, )
        ax1.plot(np.array(time_used) / 60, np.array(velocity_right) * 60*1000, '.', markersize=7, color='#FFFF14', label="velocity right")
        ax1.set(xlabel='time (min)', ylabel='velocity ($\mu$m/min)', title='Velocity profiles')
        ax1.legend(loc='best')

        ax2.set_xlabel('img nr [-]')
        lines = ax2.plot(imgnr_used, velocity_left) #create dummy plot to set x-axis
        lines[0].remove()       #remove 'dummy' data but retain axis

        fig1.tight_layout()
        fig1.savefig(os.path.join(analysisFolder, 'A_all_velocityVStime.png'), dpi=600)

    except:
        print(traceback.format_exc())

    plt.show()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')     #configuration for printing logging messages. Can be removed safely
    main()
    exit()