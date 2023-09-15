import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import csv

def main():
    source = 'E:\\2023_04_06_PLMA_HexaDecane_Basler2x_Xp1_24_s11_split____GOODHALO-DidntReachSplit\\D_analysisv4\\PROC_20230913122145_condensOnly'  # hexadecane, condens only
    source2 ='E:\\2023_04_06_PLMA_HexaDecane_Basler2x_Xp1_24_s11_split____GOODHALO-DidntReachSplit\\D_analysisv4\\PROC_20230724185238'
    csvList = [f for f in glob.glob(os.path.join(source, f"Swellingimages\\data*hrsPureIntensity.csv"))]
    [csvList.append(f) for f in glob.glob(os.path.join(source2, f"Swellingimages\\data*hrsPureIntensity.csv"))]


    fig1, ax1 = plt.subplots()
    rowsToImport = np.subtract([3 ,5], 1)
    for n in csvList:
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
        ax1.plot(xdata, ydata, label=f'{filename[4:8]}')

    ax1.set_xlabel('Distance from CL (mm)')
    ax1.set_ylabel('Swelling ratio (h/h$_{0}$)')
    ax1.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    main()
    exit()