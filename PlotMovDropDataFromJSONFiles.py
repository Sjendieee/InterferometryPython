import os
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
import glob

def path_in_use():
    path = "G:\\2024_05_07_PLMA_Basler15uc_Zeiss5x_dodecane_Xp1_31_S2_WEDGE_2coverslip_spacer_V3"

    return path


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
    plt.show()
    return

def main():
    path_images = path_in_use()
    analysisFolder = os.path.join(path_images, "Analysis CA Spatial") #name of output folder of Spatial Contact Analysis
    JSON_folder = os.path.join(analysisFolder, "Analyzed Data")

    analyzeForcevsTime(JSON_folder)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')     #configuration for printing logging messages. Can be removed safely
    main()
    exit()