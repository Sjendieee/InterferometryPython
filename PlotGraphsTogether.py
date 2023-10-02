import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import csv
import re

def extractTimeFromName(name):          #extract the time (e.g. 5min, or 10 hrs) from the filename
    nr = re.findall(r'(\d+)' , name)        #look for the number in the name (5, 10, 60, etc..)
    tformat = re.findall(r'(s|min|hrs)', name)[0]  #check for the time-unit: s, min, or hrs
    return nr[0], tformat

def sortCSVListAscendingTime(list):
    timesUnordered = np.ones(len(list))
    for i, n in enumerate(list):
        filename = os.path.splitext(os.path.basename(n))[0]
        nr, tformat = extractTimeFromName(filename)
        if tformat == 's':
            timesUnordered[i] = float(nr)
        elif tformat == 'min':
            timesUnordered[i] = float(nr)*60
        elif tformat == 'hrs':
            timesUnordered[i] = float(nr)*60*60
    return [x for _, x in sorted(zip(timesUnordered, list))]        #return ordered csv list, based on times in ascending order

def main():
    #source = 'D:\\2023_04_06_PLMA_HexaDecane_Basler2x_Xp1_24_s11_split____GOODHALO-DidntReachSplit\\D_analysisv4\\PROC_20230913122145_condensOnly'  # hexadecane, condens only
    source ='D:\\2023_04_06_PLMA_HexaDecane_Basler2x_Xp1_24_s11_split____GOODHALO-DidntReachSplit\\D_analysisv4\\PROC_20230724185238'
    #source = 'F:\\2023_02_17_PLMA_DoDecane_Basler2x_Xp1_24_S9_splitv2____DECENT_movedCameraEarly\\B_Analysis_V2\\PROC_20230829105238'
    source2 = 'E:\\2023_09_22_PLMA_Basler2x_hexadecane_1_29S2_split\\B_Analysis\\PROC_20230927135916_imbed'
    #source2 = 'E:\\2023_08_30_PLMA_Basler2x_dodecane_1_29_S2_ClosedCell\\B_Analysis2\\PROC_20230905134930'

    #source = 'D:\\2023_09_21_PLMA_Basler2x_tetradecane_1_29S2_split_ClosedCell\\B_Analysis\\PROC_20230922150617_imbed'
    #source2 = 'D:\\2023_09_21_PLMA_Basler2x_tetradecane_1_29S2_split_ClosedCell\\B_Analysis\\PROC_20230922150617_imbed'

    firstData = "Source1"; secondData = "Source2"
    colorscheme1 = 'plasma'; colorscheme2 = 'plasma'     #colorscheme for matplotlib.  Can be any of the schemes, https://matplotlib.org/stable/users/explain/colors/colormaps.html#
    csvList = [f for f in glob.glob(os.path.join(source, f"Swellingimages\\data*minPureIntensity.csv"))]
    [csvList.append(f) for f in glob.glob(os.path.join(source, f"Swellingimages\\data*hrsPureIntensity.csv"))]
    nrofFilesList1 = len(csvList)
    OrderedList1 = sortCSVListAscendingTime(csvList)
    csvList2 = [f for f in glob.glob(os.path.join(source2, f"Swellingimages\\data*minPureIntensity.csv"))]
    [csvList2.append(f) for f in glob.glob(os.path.join(source2, f"Swellingimages\\data*hrsPureIntensity.csv"))]
    OrderedList2 = sortCSVListAscendingTime(csvList2)
    csvList = OrderedList1 + OrderedList2
    xoffset = [0.1, 0.1, 0.1, 0.1, 0.1, 0.23, 0.23, 0.27]
    #xoffset = np.multiply(np.ones(len(csvList)), 0.0)
    nrofFiles = len(csvList)
    gradient1 = np.linspace(0, 1, nrofFilesList1)
    gradient2 = np.linspace(0, 1, nrofFiles-nrofFilesList1)
    cmap1 = plt.get_cmap(colorscheme1)
    cmap2 = plt.get_cmap(colorscheme2)

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
                xdata.append(float(row[rowsToImport[0]]) - xoffset[i])
                ydata.append(float(row[rowsToImport[1]]))
            except:
                print("!Some value could not be casted to a float. Whether that is an issue or not is up to the user.!")
        file.close()
        t, tformat = extractTimeFromName(filename)
        if i < nrofFilesList1:
            linecolor = cmap1(gradient1[i])
            ax1.plot(xdata, ydata, label=f'{firstData}: {t+tformat}', color=linecolor)
        else:
            linecolor = cmap2(gradient2[i - nrofFilesList1])
            ax1.plot(xdata, ydata, label=f'{secondData}: {t+tformat}', color=linecolor, linestyle='--')

    ax1.set_xlabel('Distance from CL (mm)')
    ax1.set_ylabel('Swelling ratio (h/h$_{0}$)')
    ax1.legend(loc='upper right')
    ax1.axvline(0, color='black')
    fig1.savefig(os.path.join(source, "SwellingImages\\CombinedFigures.png"), dpi=600)

if __name__ == "__main__":
    main()
    exit()