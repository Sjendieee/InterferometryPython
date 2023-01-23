import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
from general_functions import image_resize
from line_method import coordinates_on_line
import numpy as np
import logging


right_clicks = list()
def click_eventSingle(event, x, y, flags, params):
    '''
    Click event for the setMouseCallback cv2 function. Allows to select 2 points on the image and return it coordiantes.
    '''
    if event == cv2.EVENT_LBUTTONDOWN:
        global right_clicks
        right_clicks.append([x, y])
    if len(right_clicks) == 1:
        cv2.destroyAllWindows()

def main():
    #Required changeables
    pixelLoc1 = 3200
    pixelLoc2 = 3201
    pixelIV = 100
    cvsFilenr1 = 236
    cvsFilenr2 = 244
    cvsFileIV = 4
    source = "C:\\Users\\ReuvekampSW\\Documents\\InterferometryPython\\export\\PROC_20230123120524"
    csvName = "*"
    csvList = [f for f in glob.glob(os.path.join(source, f"process\\*.csv"))]
    #Probably fine
    rangeLength = 40

    #Get images to see where you chose your pixel
    rawImgList = [f for f in glob.glob(os.path.join(source, f"rawslicesimage\\*.png"))]
    im_raw = cv2.imread(rawImgList[0])
    im_temp = image_resize(im_raw, height=1200)
    resize_factor = 1200 / im_raw.shape[0]
    cv2.imshow('image', im_temp)
    cv2.setWindowTitle("image", "Slice selection window. Select 2 points for the slice.")
    cv2.setMouseCallback('image', click_eventSingle)
    cv2.waitKey(0)
    global right_clicks

    P1 = np.array(right_clicks[0]) / resize_factor
    print(f"Selected coordinates: P1 = [{P1[0]:.0f}, {P1[1]:.0f}]")

    #Read in from config file (selected points on which the line was drawn)
    pointa = 5272, 1701
    pointb = 430, 1843
    x_coords, y_coords = zip(*[pointa, pointb])  # unzip coordinates to x and y
    a = (y_coords[1] - y_coords[0]) / (x_coords[1] - x_coords[0])
    b = y_coords[0] - a * x_coords[0]

    bn = b


    #Obtain scaling factor to correspond chosen pixellocation to new position in raw image
    #Scaling factor = factor by which raw image was made smaller in new image
    P1arr = np.array(pointa)
    P2arr = np.array(pointb)
    BLarr = np.array([im_raw.shape[1], im_raw.shape[0]])

    adjP1 = np.subtract(P1arr, P2arr)
    scaling = np.divide(adjP1, BLarr)
    print(f"Scaling fators are: {scaling}")



    coordinates = coordinates_on_line(a, bn, [0, im_raw.shape[1], 0, im_raw.shape[0]])



    print(f"The coordinates are: {coordinates}")


    #With this loop, different pixel locations can be chosen to plot for
    for i in range(pixelLoc1, pixelLoc2, pixelIV):
        meanIntensity = []
        elapsedtime = []
        #This loop takes data van intensity csv files at different moments in time (n), and calculates a mean value for a given rangeLength at the chosen pixelLocation
        for n in csvList: #range(cvsFilenr1, cvsFilenr2, cvsFileIV):
            #file = open(os.path.join(source, f"process\\{csvName}{str(n).zfill(4)}_analyzed__real.csv"))
            file = open(n)
            csvreader = csv.reader(file)
            header = ['Real intensity value']
            rows = []
            for row in csvreader:
                rows.append(row[0])
            file.close()
            elapsedtime.append(float(rows[0]))

            pixelLocation = i
            range1 = pixelLocation - rangeLength
            range2 = pixelLocation + rangeLength
            if (range1 < 1) or (range2>len(rows)):
                raise Exception(f"There were not enough values to average over. Either lower mean-range, or choose different pixel location")

            total = 0

            for idx in range(range1, range2):
                total = total + float(rows[idx]) + 100
            meanIntensity.append(total / (range2 - range1))
        plt.plot(elapsedtime, meanIntensity)
        plt.xlabel('Time (s)')
        plt.ylabel('Mean intensity')
        plt.title(f'pixellocation = {pixelLocation}')
        #plt.show()
        plt.draw()

    # img_path = os.path.join(source, f"rawslicesimage\\rawslicesimage_Basler_a2A5328-15ucBAS__40087133__20230110_175829604_{str(n).zfill(4)}_analyzed_.png")
    # image = mpimg.imread(img_path)
    # plt.imshow(image)
    # plt.show()
    plt.show()

if __name__ == "__main__":
    main()
    exit()