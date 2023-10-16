import math

import numpy as np
from matplotlib import pyplot as plt
import json
import os
import csv
from matplotlib.widgets import RectangleSelector
from datetime import datetime
import glob


'''
IMPORTANT

THIS CODE IS STILL UNDOCUMENTED. USE WITH CARE.

'''


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

#TODO check if this works still. I put everything below in a main function, whereas before it was 'standalone'
def main():
    #procStatsJsonPath = r'C:\Users\ReuvekampSW\PycharmProjects\InterferometryPython\export\PROC_20230808171326\PROC_20230808171326_statistics.json'        #laptop
    #procStatsJsonPath = r'C:\Users\ReuvekampSW\Documents\InterferometryPython\export\PROC_20230809115938\PROC_20230809115938_statistics.json'                    #chickencoop workstation
    #procStatsJsonPath = r'I:\2023_08_09CA_analysis_dodecane_glassSide\PROC_20230809115938\PROC_20230809115938_statistics.json'
    procStatsJsonPath = r'E:\2023_08_011_PLMA_Basler5x_dodecane_1_28_S5_OpenAir\Analysis_v1\PROC_20230811121032\PROC_20230811121032_statistics.json'

    #PROC_PROC_20230810183031                       air side                106 datapoints              #nr1            till frame 60 movement
    #PROC_20230809115938                            glass side v1, all      253 datapoints              #nr2
    #ROC_20230810204448_do wedge glass v2 f1_58     glass side v2, 1-58     60                          @nr3
    #PROC_20230810210250_do wedge glass v2 f58_end  glass side v2, 58-end   16                          #nr4
    #PROC_20230811121032                            open air                77                          #nr5

    print(os.path.join(os.path.dirname(procStatsJsonPath), f"angleFittingData.csv"))
    originalPath = os.path.dirname(procStatsJsonPath)

    csvPathAppend = r'csv'
    flipData = False

    #analyzeImages = np.concatenate((np.arange(0, 60, 5), np.arange(70, 86, 10)))                                       #nr1
    #analyzeImages = np.concatenate((np.arange(0, 40, 4), np.arange(50, 100, 10), np.arange(120, 250, 20)))             #nr2
    #analyzeImages = np.concatenate((np.arange(0, 15, 2), np.arange(20, 60, 5)))                                         #nr3
    #analyzeImages = np.arange(0, 16, 1)                                                                                 #nr4
    analyzeImages = np.arange(2, 77, 4)

    #analyzeImages = np.array([5])

    # 1 slice: Contact angle = -1.6494950309356011 degrees.
    # 11 slices: -1.650786783947852 degrees.

    with open(procStatsJsonPath, 'r') as f:
        procStats = json.load(f)


    '''
    20 periods = 226 pix = 134um
    1pi = lambda / 4n = 532 / (4*1.434) = 92.74 um
    20 periods = 20*2pi = 40pi = 40*92.74 = 3709.6nm
    1.53
    
    237pix = 237/1687 = 0.1405 mm
    '''
    data = {}
    data['jsonPath'] = procStatsJsonPath
    data['processDatetime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data['analyzeImages'] = ','.join(analyzeImages.astype(str))
    data['csvPathAppend'] = csvPathAppend
    data['data'] = {}

    deltaTime = procStats["deltatime"]
    timeFromStart = np.cumsum(deltaTime)

    # # Some extra stuff below here for fixing timestamps in post
    # my_str = ','.join(str(item) for item in timeFromStart[analyzeImages])
    # print(my_str)
    # timestamps = np.array(procStats['timestamps'])
    # timestamp_reference = datetime.strptime(timestamps[6], '%Y-%m-%d %H:%M:%S')
    # print(f"{timestamp_reference=}")
    # timestamps = timestamps[analyzeImages]
    # timestamps = [datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S') for timestamp in timestamps]
    # dt = [(timestamp - timestamp_reference).total_seconds() for timestamp in timestamps]
    # print(dt)
    # exit()

    angleDegAll = np.zeros_like(timeFromStart, dtype='float')

    try:
        conversionZ = procStats["conversionFactorZ"]
        conversionXY = procStats["conversionFactorXY"]
        unitZ = procStats["unitZ"]
        unitXY = procStats["unitXY"]
        #TODO implement in code that conversion is good also for different units of input (factors 1000) belowfor x & y = ...
        if unitZ != "nm" or unitXY != "mm":
            raise Exception("One of either units is not correct for good conversion. Fix manually or implement in code")

        print(f"unitZ: {unitZ}, conversionZ = {conversionZ}. unitXY: {unitXY},  converzionXY = {conversionXY}")

        for idx, imageNumber in enumerate(analyzeImages):
            print(f'Analyzing image {idx}/{len(analyzeImages)}.')
            csvList = [f for f in glob.glob(os.path.join(originalPath, csvPathAppend, f"*.csv"))]
            print(csvList[imageNumber])

            y = np.loadtxt(csvList[imageNumber], delimiter=",") * conversionZ / 1000  # y is in um
            if flipData:
                y = -y + max(y)

            x = np.arange(0, len(y)) * conversionXY * 1000  # x is now in um

            ##TODO: removed first datapoint because for whatever reason it was spuerfar outside the range, making it hard to select the good range in the plot
            x = x[1:]
            y = y[1:]
            fig, ax = plt.subplots(figsize=(10,10))
            ax.scatter(x, y)
            highlighter = Highlighter(ax, x, y)
            plt.show()
            # plt.draw()
            # plt.waitforbuttonpress(0)  # this will wait for indefinite time
            # plt.close(fig)
            selected_regions = highlighter.mask
            xrange1, yrange1 = x[selected_regions], y[selected_regions]
            print(f"x ranges from: [{xrange1[0]} - {xrange1[-1]}]\n"
                  f"y ranges from: [{yrange1[0]} - {yrange1[-1]}]\n"
                  f"Therefore dy/dx = {yrange1[-1] - yrange1[0]} / {xrange1[-1]-xrange1[0]} = {(yrange1[-1] - yrange1[0])/(xrange1[-1]-xrange1[0])}")
            # fig, ax = plt.subplots()
            # ax.scatter(x, y)
            # highlighter = Highlighter(ax, x, y)
            # plt.show()
            # selected_regions = highlighter.mask
            # xrange2, yrange2 = x[selected_regions], y[selected_regions]

            # print(xrange1, yrange1)
            # print(xrange2, yrange2)

            coef1 = np.polyfit(xrange1, yrange1, 1)
            poly1d_fn1 = np.poly1d(coef1)

            # coef2 = np.polyfit(xrange2, yrange2, 1)
            # poly1d_fn2 = np.poly1d(coef2)

            # print(coef1, coef2)

            a_horizontal = 0


            angleRad = math.atan((coef1[0]-a_horizontal)/(1+coef1[0]*a_horizontal))

            angleDeg = math.degrees(angleRad)

            #Flip measured CA degree if higher than 45.
            if angleDeg > 45:
                angleDeg = 90 - angleDeg

            # print(f"{angleRad=}")
            # print(f"{angleDeg=}")

            fig, ax = plt.subplots()
            ax.scatter(x, y, label=f'Raw data {os.path.basename(originalPath)}')
            ax.scatter(xrange1, yrange1, color='green', label='Selected data line 1')
            # ax.scatter(xrange2, yrange2, color='green', label='Selected data line 2')
            ax.plot(x, poly1d_fn1(x), color='red', linewidth=3, label='Linear fit 1')
            # ax.plot(x, poly1d_fn2(x), color='red', linewidth=3, label='Linear fit 2')
            ax.set_title(f"{angleDeg=}")
            ax.set_xlabel("[um]")
            ax.set_ylabel("[um]")
            ax.set_xlim([x[0], x[-1]])
            ax.set_ylim([y[0], y[-1]])

            foldername = "CA_analysis"
            newfolder = os.path.join(os.path.dirname(procStatsJsonPath), foldername)
            if not os.path.exists(newfolder):
                os.mkdir(newfolder)
                print('created path: ', newfolder)
            print(os.path.abspath(foldername))
            #fig.savefig(os.path.join(os.path.dirname(originalPath),  f"angleFitting_{os.path.splitext(os.path.basename(originalPath))[0]}.png"), dpi=300)
            fig.savefig(os.path.join(newfolder, f"angleFitting_{idx}_{os.path.splitext(os.path.basename(originalPath))[0]}.png"), dpi=300)

            data['data'][idx] = {}
            data['data'][idx]['timeFromStart'] = timeFromStart[imageNumber]
            data['data'][idx]['xrange1'] = xrange1.tolist()
            data['data'][idx]['yrange1'] = yrange1.tolist()
            # data['data'][idx]['xrange2'] = xrange2
            # data['data'][idx]['yrange2'] = yrange2
            data['data'][idx]['coef1'] = coef1.tolist()
            # data['data'][idx]['coef2'] = coef2
            data['data'][idx]['angleDeg'] = angleDeg
            data['data'][idx]['angleRad'] = angleRad

            angleDegAll[idx] = angleDeg
            print(f'Contact angle = {angleDeg} degrees.')


            # plt.show()
            plt.close('all')
    except:
        print("Something went wrong, still saving data.")

    with open(os.path.join(newfolder, f"angleFittingData.json"), 'w') as f:
        json.dump(data, f, indent=4)

    timeFromStart = np.array([data['data'][i]['timeFromStart'] for i in data['data']], dtype='float')
    angleDeg = np.array([data['data'][i]['angleDeg'] for i in data['data']], dtype='float')
    print(f"TimefromStart: {timeFromStart}")
    print(f"Angle in deg.: {angleDeg}")

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(np.divide(timeFromStart, 60), angleDeg, '.-')
    ax.set_xlabel(f'[Time from drop creation [min]')
    ax.set_ylabel(f'[Contact angle [deg]')
    fig.tight_layout()
    fig.savefig(os.path.join(newfolder, f"angleFittingData.png"), dpi=300)

    np.savetxt(os.path.join(newfolder, f"angleFittingData{os.path.splitext(os.path.basename(originalPath))[0]}.csv"), np.vstack((timeFromStart, angleDeg)),
               delimiter=',', fmt='%f', header=f'Dataset: {os.path.basename(originalPath)}, row 1 = Time from start '
                                               f'(depositing drop) [s], row 2 = contact angle [deg] ')
    exit()



