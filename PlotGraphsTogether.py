import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import csv
import re

def extractTimeFromName(name):
    nr = re.findall(r'(\d+)' , name)
    time = re.findall(r'(s|min|hrs)', name)[0]
    return nr[0] + time


def main():
    #source = 'E:\\2023_04_06_PLMA_HexaDecane_Basler2x_Xp1_24_s11_split____GOODHALO-DidntReachSplit\\D_analysisv4\\PROC_20230913122145_condensOnly'  # hexadecane, condens only
    #source2 ='E:\\2023_04_06_PLMA_HexaDecane_Basler2x_Xp1_24_s11_split____GOODHALO-DidntReachSplit\\D_analysisv4\\PROC_20230724185238'
    source = 'F:\\2023_02_17_PLMA_DoDecane_Basler2x_Xp1_24_S9_splitv2____DECENT_movedCameraEarly\\B_Analysis_V2\\PROC_20230829105238'
    #source2 = 'E:\\2023_08_30_PLMA_Basler2x_dodecane_1_29_S2_ClosedCell\\B_Analysis2\\PROC_20230905134930'

    colorscheme = 'plasma'     #colorscheme for matplotlib.  Can be any of the schemes, https://matplotlib.org/stable/users/explain/colors/colormaps.html#
    csvList = [f for f in glob.glob(os.path.join(source, f"Swellingimages\\data*minPureIntensity.csv"))]
    [csvList.append(f) for f in glob.glob(os.path.join(source, f"Swellingimages\\data*hrsPureIntensity.csv"))]
    #[csvList.append(f) for f in glob.glob(os.path.join(source2, f"Swellingimages\\data*minPureIntensity.csv"))]
    #[csvList.append(f) for f in glob.glob(os.path.join(source2, f"Swellingimages\\data*hrsPureIntensity.csv"))]
    nrofFiles = len(csvList)
    gradient = np.linspace(0, 1, nrofFiles)
    cmap = plt.get_cmap(colorscheme)

    fig1, ax1 = plt.subplots()
    rowsToImport = np.subtract([3 ,5], 1)
    for i, n in enumerate(csvList):
        filename = os.path.splitext(os.path.basename(n))[0]
        file = open(n)
        csvreader = csv.reader(file)
        xdata = []
        ydata = []
        for row in csvreader:
            try:
                xdata.append(float(row[rowsToImport[0]]))
                ydata.append(float(row[rowsToImport[1]]))
            except:
                print("!Some value could not be casted to a float. Whether that is an issue or not is up to the user.!")

        file.close()
        linecolor = cmap(gradient[i])
        ax1.plot(xdata, ydata, label=f'{extractTimeFromName(filename)}', color=linecolor)

    ax1.set_xlabel('Distance from CL (mm)')
    ax1.set_ylabel('Swelling ratio (h/h$_{0}$)')
    ax1.legend(loc='upper right')
    #fig1.show()
    fig1.savefig(os.path.join(source, "SwellingImages\\CombinedFigures.png"), dpi=300)

if __name__ == "__main__":
    main()
    exit()