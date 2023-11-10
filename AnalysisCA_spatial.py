import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shapely
import math
from datetime import datetime
import warnings
import logging
import json
import glob
import scipy.signal
import pandas as pd
import easygui
import matplotlib as mpl
import git

#I'm pretty sure this does not work as it's completely supposed to:
#The limits (right now) should be from [0 - some value], and does not work for e.g. [200 - 500]
def intersection_imageedge(a, b, limits):
    '''
    For a given image dimension (limits) and linear line (y=ax+b) it determines which borders of the image the line
    intersects and the length (in pixels) of the slice.
    :param a: slope of the linear line (y=ax+b)
    :param b: intersect with the y-axis of the linear line (y=ax+b)
    :param limits: image dimensions (x0, width, y0, height)
    :return: absolute length of slice (l) and border intersects booleans (top, bot, left, right).

    Note: line MUST intersects the image edges, otherwise it throws an error.
    '''
    # define booleans for the 4 image limit intersections.
    top, bot, left, right = False, False, False, False
    w, h = limits[1], limits[3]   # w = width, h = height

    '''
    Determining the intersections (assuming x,y=0,0 is bottom right).
    Bottom: for y=0: 0<=x<=w --> y=0, 0<=(y-b)/a<=w --> 0<=-b/a<=w
    Top: for y=h: 0<=x<=w --> y=h, 0<=(y-b)/a<=w --> 0<=(y-h)/a<=w
    Left: for x=0: 0<=y<=h --> x=0, 0<=ax+b<=h --> 0<=b<=h
    Right: for x=w: 0<=y<=h --> x=0, 0<=ax+b<=h --> 0<=a*w+b<=h
    '''
    if -b / a >= 0 and -b / a <= w:
        bot = True
    if (h - b) / a >= 0 and (h - b) / a <= w:
        top = True
    if b >= 0 and b <= h:
        left = True
    if (a * w) + b >= 0 and (a * w) + b <= h:
        right = True

    if top + bot + left + right != 2:  # we must always have 2 intersects for a linear line
        logging.error("The profile should always intersect 2 image edges.")
        exit()

    '''
    Determining the slice length.
    There are 6 possible intersections of the linear line with a rectangle.
    '''
    if top & bot:  # case 1
        # l = imageheight / sin(theta), where theta = atan(a)
        # == > l = (sqrt(a ^ 2 + 1) * imageheight) / a
        l = np.round((np.sqrt(a ** 2 + 1) * h) / a)  # l = profile length (density=1px)
    if left & right:  # case 2
        # l = imagewidth / cos(theta), where theta=atan(a)
        # ==> l = sqrt(a^2+1)*imagewidth
        l = np.round(np.sqrt(a ** 2 + 1) * w)  # l = profile length (density=1px)
    if left & top:  # case 3
        # dx = x(y=h), dy = h-y(x=0) --> l = sqrt(dx^2+dy^2)
        dx = (h - b) / a
        dy = h - b
        l = np.sqrt(dx**2 + dy**2)
    if top & right:  # case 4
        # dx = w-x(y=h), dy = h-y(x=0) --> l = sqrt(dx^2+dy^2)
        dx = w - ((h - b) / a)
        dy = h - b
        l = np.sqrt(dx ** 2 + dy ** 2)
    if bot & right:  # case 5
        # dx = w-x(y=h), dy = y(x=0) --> l = sqrt(dx^2+dy^2)
        dx = w - ((h - b) / a)
        dy = b
        l = np.sqrt(dx ** 2 + dy ** 2)
    if bot & left:  # case 6
        # dx = x(y=h), dy = y(x=0) --> l = sqrt(dx^2+dy^2)
        dx = (h - b) / a
        dy = b
        l = np.sqrt(dx ** 2 + dy ** 2)

    return np.abs(l), (top, bot, left, right)

def coordinates_on_line(a, b, limits):
    '''
    For a given image, a slope 'a' and offset 'b', this function gives the coordinates of all points on this
    linear line, bound by limits.
    y = a * x + b

    :param a: slope of line y=ax+b
    :param b: intersection of line with y-axis y=ax+b
    :param limits: (min x, max x, min y, max y)
    :return: zipped list of (x, y) coordinates on this line
    :return: length of line in coordinate (aka pixels)-units
    '''

    # Determine the slice length of the linear line with the image edges. This is needed because we need to know how
    # many x coordinates we need to generate
    #l, _ = intersection_imageedge(a, b, limits)
    l = 1
    dx = abs(limits[1]-limits[0])
    dy = abs(limits[3]-limits[2])
    # generate x coordinates, keep as floats (we might have floats x-coordinates), we later look for the closest x value
    # in the image. For now we keep them as floats to determine the exact y-value.
    # x should have length l. x_start and x_end are determined by where the line intersects the image border. There are
    # 4 options for the xbounds: x=limits[0], x=limits[1], x=x(y=limits[2]) --> (x=limits[2]-b)/a, x=x(y=limits[3]) -->
    # (x=limits[3]-b)/a.
    # we calculate all 4 possible bounds, then check whether they are valid (within image limits).
    # xbounds = np.array([limits[0], limits[1], (limits[2]-b)/a, (limits[3]-b)/a])
    # # filter so that 0<=x<=w and 0<=y(x)<=h. we round to prevent strange errors (27-10-22)
    # xbounds = np.delete(xbounds, (np.where(
    #     (xbounds < limits[0]) |
    #     (xbounds > limits[1]) |
    #     (np.round(a * xbounds + b) < limits[2]) |
    #     (np.round(a * xbounds + b) > limits[3])
    # )))
    # if len(xbounds) != 2:
    #     logging.error(f"Something went wrong in determining the x-coordinates for the image slice. {xbounds=}")
    #     exit()
    #x = np.linspace(np.min(xbounds)+1, np.max(xbounds)-1, int(l))
    if dx > dy:
        if limits[0]<limits[1]:
            x = np.array(range(limits[0], limits[1], 1))
        else:
            x = np.array(range(limits[0], limits[1], -1))
        # calculate corresponding y coordinates based on y=ax+b, keep as floats
        y = (a * x + b)
    else:
        if limits[2]<limits[3]:
            y = np.array(range(limits[2], limits[3], 1))
        else:
            y = np.array(range(limits[2], limits[3], -1))
            # calculate corresponding x coordinates based on y=ax+b, keep as floats
        x = (y - b) / a
    lineLengthPixels = (dx**2 + dy**2)**0.5
    # return a zipped list of coordinates, thus integers
    return list(zip(x.astype(int), y.astype(int))), lineLengthPixels

""""
Input: 
x = xcoords of contour
y = ycoords of contour
L = desired length of normal vector (determines how many fringes will be accounted for later on)
"""
def get_normalsV4(x, y, L): #TODO attempting to get the normals to work on the left side / entire droplet
    # For each coordinate, fit with nearby points to a polynomial to better estimate the dx dy -> tangent
    # Take derivative of the polynomial to obtain tangent and use that one.
    x0arr = []; dyarr = []; y0arr = []; dxarr = []
    window_size = 25            #!!! window size to use. 17 seems to be good for most normal vectors
    k = round((window_size+1)/2)

    middleCoordinate = [(max(x) + min(x))/2, (max(y) + min(y))/2]   #estimate for "middle coordinate" of contour. Will be used to direct normal vector to inside contour

    for idx in range(k, len(x) - k):
        if idx == 8331:
            print("we pausin")
        xarr = x[(idx-k):(idx+k)] # define x'es to use for polynomial fitting
        yarr = y[(idx-k):(idx+k)] # define y's ...
        switchedxy = False
        #TODO attemting to rotate x&y coords for polynomial fitting when at left or right side, where the fits are very poor: check if good normals are obtained
        if (max(xarr)-min(xarr)) < (max(yarr)-min(yarr)):   #if dx < dy = left or right side droplet -> rotate cooridnate system by 90 degrees clockwise (x,y)->(y,-x))
            xarrtemp = xarr
            xarr = yarr
            yarr = np.multiply(xarrtemp, -1)
            switchedxy = True
        x0 = xarr[k]
        ft = np.polyfit(xarr, yarr, 2) # fit with second order polynomial
        fit = np.poly1d(ft)  # fit with second order polynomial
        y0 = fit(x0)
        ffit = lambda xcoord: 2 * fit[2] * xcoord + fit[1]  # derivative of a second order polynomial

        if np.sum(np.abs(np.array(xarr) - x0)) == 0:        #if all x-coords are the same in given array
            nx1 = - L; nx2 = L #consider both normal vector possibilities, check below which one is pointing inwards
            ny1 = 0; ny2 = 0
            xrange = np.ones(100) * x0
            yrange = np.linspace(yarr[0], yarr[-1], 100)
        elif np.sum(np.abs(np.array(yarr) - yarr[0])) == 0:     #if all y-coords are the same in given array
            nx1 = 0; nx2 = 0    #consider both normal vector possibilities, check below which one is pointing inwards
            ny1 = L; ny2 = -L
            xrange = np.linspace(xarr[0], xarr[-1], 100)
            yrange = np.ones(100) * y0
        else:   #continue as normal
            dx = 1
            dy = ffit(x0)  #ffit(x0)

            normalisation = L / np.sqrt(1+dy**2)        #normalise by desired length vs length of vector

            #Determine direction of normal vector by calculating of both direction the Root-square-mean to the "middle cooridnate" of the contour, and take the smallest one
            nx1 = -dy * normalisation; ny1 = dx * normalisation #normal vector 1
            nx2 = +dy * normalisation; ny2 = -dx * normalisation #normal vector 2

        if switchedxy:  #turn coordinate system back 90 degrees counterclockwise (x,y) -> (-y, x)
            x0temp = x0
            x0 = np.multiply(y0, -1)
            y0 = x0temp
            tempnx1 = nx1
            nx1 = np.multiply(ny1, -1)
            ny1 = tempnx1
            tempnx2 = nx2
            nx2 = np.multiply(ny2, -1)
            ny2 = tempnx2

        if ((middleCoordinate[0]-(x0+nx1))**2 + (middleCoordinate[1]-(y0+ny1))**2)**0.5 > ((middleCoordinate[0]-(x0+nx2))**2 + (middleCoordinate[1]-(y0+ny2))**2)**0.5:
            nx = nx2
            ny = ny2
        else:
            nx = nx1
            ny = ny1

            xrange = np.linspace(xarr[0], xarr[-1], 100)
            yrange = fit(xrange)
        x0arr.append(round(x0))
        y0arr.append(round(y0))
        dxarr.append(round(x0+nx))
        dyarr.append(round(y0+ny))
        #Below: for plotting & showing of a few polynomial fits - to investigate how good the fit is
        # if idx - k == 101:
        #     plt.plot(xarr, np.array(yarr), '.', label="coordinates")
        #     plt.plot(xrange, np.array(yrange), label="fit")
        #     plt.plot([x0,x0+nx], np.array([y0, y0+ny]), '-', label="normal to contour")
        #     plt.title(f"Zoomed-in coordinates of contour. idx = {idx}")
        #     plt.xlabel("x - coords")
        #     plt.ylabel("y - coords")
        #     plt.legend()
        #     plt.show()
        #     plt.close()
        #     print(f"idx: {idx} - {fit}")
    return x0arr, dxarr, y0arr, dyarr  # return the original data points x0&y0, and the coords of the normals dx&dy

# Attempting to get a contour from the full-sized HQ image, and using resizefactor only for showing a copmressed image so it fits in the screen
# Parses all 'outer' coordinates, not only on right side of droplet
#With working popup box for checking and changing contour
def getContourCoordsV4(imgPath, resizeFactor, analysisFolder, contouri, thresholdSensitivity, contourCoords = 0):
    minimalDifferenceValue = 100    #Value of minimal difference in x1 and x2 at same y-coord to check for when differentiating between entire droplet & partial contour & fish-hook-like contour
    img = cv2.imread(imgPath)  # read in image
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to greyscale
    # Apply adaptive thresholding to the blurred frame
    thresh = cv2.adaptiveThreshold(grayImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, thresholdSensitivity[0], thresholdSensitivity[1])
    #cv2.imwrite(os.path.join(analysisFolder, f"threshImage_contourLine.png"), thresh)

    # TODO not nice now, but for the code to work
    greyresizedimg = grayImg
    resizedimg = img

    # Find contours in the thresholded frame
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    list_of_pts = []
    contourList = []
    maxxlist = []
    # Iterate over the contours
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # Set a minimum threshold for contour area
        if area > 10000:  # OG = 2000
            # Check if the contour has at least 5 points
            if len(contour) >= 5:
                list_of_pts += [pt[0] for pt in contour]
                print(f"length of contours={len(contour)}")

                maxxlist.append(max([elem[0][0] for elem in contour]))  # create a list with the max-x coord of a contour, to know which contour is the furthest right, 1 from furthest etc..
                contourList.append(contour)
    if len(contourList) > 5:
        nrOfContoursToShow = 5      #nr of contours allow to be to checked
    else:
        nrOfContoursToShow = len(contourList)
    FurthestRightContours = sorted(zip(maxxlist, contourList), reverse=True)[:nrOfContoursToShow]  # Sort contours, and zip the ones of which the furthest right x coords are found
    contourList = [elem[1] for elem in FurthestRightContours]

    #If contour is not known yet from imported file:
    # Generally, not the furthest one out (halo), but the one before that is the contour of the CL. i can be changed with a popup box
    if contouri < 0:    #contour not known from file
        #TODO implement here: check current contours with good one of previous k iteration, and select the one that is
        # most similar. If big differences for all, select manually

        if len(contourCoords) > 1:  #if coord of previous iteration are given, check with those for best contour
            ylistToCheckCoords = np.array([elem[1] for elem in contourCoords])
            xlistToCheckCoords = [elem[0] for elem in contourCoords]
            ysToCheck = np.linspace(min(ylistToCheckCoords), max(ylistToCheckCoords), 50).round()    #only compare 50 values of y
            xsToCheck = []
            for y in ysToCheck: #obtain corresponding x's of the to-be-investigated y's
                xsToCheck.append(np.where(ylistToCheckCoords == y))

            bestCorrespondingContour = np.zeros(len(contourList))
            for i, yToCheck in enumerate(ysToCheck):    #compare x-values at various y-heights
                xlistPerY = []
                for contour in contourList:
                    xAtyIndices = np.where([elem[0][1] for elem in contour] == yToCheck)
                    xaty = [contour[0][k][0] for k in xAtyIndices]
                    #TODO blijkbaar hier iets aan het doen
                    xlistPerY.append(xAty)
                bestCorrespondingCoordinateIndex = np.argmin(abs(np.subtract(xlistPerY, xsToCheck[i])))  #find best corresponding x of xsToCheck[i] xlistPerY
                bestCorrespondingContour[bestCorrespondingCoordinateIndex] += 1
            i = np.argmax(bestCorrespondingContour)
            contouri = i
            contour = contourList[i]
            print(f"BestCorrespondingContour list= {bestCorrespondingContour}")
            goodContour = True

        else: #else, give i a value & have the user manually check for contour correctness later on
            #if all big differences (or the img number is =0), set i manually to a value and allow user input to decide the contour
            if nrOfContoursToShow > 1:
                i = 1
            else:
                i = 0
            goodContour = False

    else:   #if contouri has a value, it is an imported value, chosen in a previous iteration & should already be good
        i = contouri
        contour = contourList[i]
        goodContour = True

    #show img w/ contour to check if it is the correct one
    #make popup box to show next contour (or previous) if desired
    while goodContour == False:
        tempimg = []
        copyImg = resizedimg.copy()
        tempimg = cv2.polylines(copyImg, np.array([contourList[i]]), False, (255, 0, 0), 8)  # draws 1 blue contour with the x0&y0arrs obtained from get_normals
        tempimg = cv2.resize(tempimg, [round(5328 / 5), round(4608 / 5)], interpolation=cv2.INTER_AREA)  # resize image
        cv2.imshow(f"Contour img with current selection of contour {i+1} out of {nrOfContoursToShow}", tempimg)
        choices = ["One contour outwards (-i)", "Current contour is fine", "One contour inwards (+1)"]
        myvar = easygui.choicebox("Is this a desired contour?", choices=choices)
        #cv2.waitKey(0)
        cv2.destroyAllWindows()

        if myvar == choices[0]:
            if i == 0:
                out = easygui.msgbox("i is already 1, cannot go further out")
            else:
                i -= 1
        elif myvar == choices[1]:
            goodContour = True
            contour = contourList[i]
        elif myvar == choices[2]:
            if i == len(contourList)-1:
                out = easygui.msgbox("i is already maximum value, cannot go further inwards")
            else:
                i += 1


    if goodContour == True:     #i == 1  # generally, not the furthest one out (halo), but the one before that is the contour of the CL
        # check per y value first, then compare x's to find largest x
        usableContour = []
        usableContourMax = []
        usableContourMin = []
        ylist = np.array([elem[0][1] for elem in contour])
        xlist = [elem[0][0] for elem in contour]

        # TODO temp only for plotting the contour of which the vectors are taken
        # tempContourImg = cv2.polylines(resizedimg, np.array([contour]), False, (0, 0, 255), 2)  # draws 1 good contour around the outer halo fringe
        # tempContourImg = cv2.resize(tempContourImg, [round(5328 / 5), round(4608 / 5)], interpolation=cv2.INTER_AREA)  # resize image
        # cv2.imshow(f"Contour img of i={i} out of {len(contourList)}", tempContourImg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # iterate from min y to max y
        # if (1) the x values at +- the middle y are not spaced far apart, we are only looking at a small contour at either the left or right side of the droplet
        # if (2) the x values are far apart, probably the contour of an entire droplet is investigated

        calcMiddleYVal = round((max(ylist) + min(ylist))/2) #calculated middle Y value; might be that it does not exist, so look for closest real Y value below
        diffMiddleYval = abs(ylist-calcMiddleYVal)     #
        realMiddleYIndex = diffMiddleYval.argmin()    #find index of a real Y is closest to the calculated middle Y
        realMiddleYVal = ylist[realMiddleYIndex]        #find 1 value of real Y
        allYindicesAtMiddle = np.where(ylist == realMiddleYVal)[0]         #all indices of the same real y's
        if len(allYindicesAtMiddle) > 0:        #if there's more than 1 y-value, compare the min and max X at that value; if the difference is big -> entire droplet
            allXValuesMiddle = [xlist[i] for i in allYindicesAtMiddle]

            if abs(max(allXValuesMiddle) - min(allXValuesMiddle)) < minimalDifferenceValue: #X-values close together -> weird contour at only a part (primarily left or right) of the droplet
                case = 1
                for j in range(min(ylist), max(ylist)):     #iterate over all y-coordinates form lowest to highest
                    indexesJ = np.where(ylist == j)[0]      #find all x-es at 1 y-coord
                    xListpery = [xlist[x] for x in indexesJ]
                    if len(indexesJ) > 2 and max(xListpery) - min(xListpery) > minimalDifferenceValue:  #if more than 2 x-coords at 1 y-coord, take the min & max as useable contour (contour is fish-hook shaped)
                        usableContourMin.append([min(xListpery), j])
                        usableContourMax.append([max(xListpery), j])

                    # else: left or right side of drop: check by comparing current x-coord with x-coord of minimum-y
                    elif max(xListpery) > xlist[ylist.argmin()] : #xcoord(current) > xcoord(@y_min) = right side droplet -> take max (x)
                        usableContourMax.append([max(xListpery), j])

                    else:
                        usableContourMin.append([min(xListpery), j])
                usableContour = usableContourMax + usableContourMin[::-1]
                # TODO not sure if this works properly: meant to concate the coords of a partial contour such that the coords are on a 'smooth partial ellips' without a gap
                for ii in range(0, len(usableContour) - 1):
                        if abs(usableContour[ii][1] - usableContour[ii + 1][1]) > 20:  # if difference in y between 2 appending coords is large, a gap in the contour is formed
                            usableContour = usableContour[ii:] + usableContour[0:ii]
                            print("The order of coords has been changed")

                useableylist = np.array([elem[1] for elem in usableContour])
                useablexlist = [elem[0] for elem in usableContour]

            else:   #far spaced x-values: probaly contour of an entire droplet: take the min & max at every y-coordinate
                case = 2
                for j in range(min(ylist), max(ylist)):  # iterate over all y-coordinates form lowest to highest
                    indexesJ = np.where(ylist == j)[0]  # find all x-es at 1 y-coord

                    xListpery = [xlist[x] for x in indexesJ]    #list all x'es at that y
                    usableContourMax.append([max(xListpery), j])    #add the max [x,y] into a list
                    usableContourMin.append([min(xListpery), j])    #add the min [x,y] into another list
                usableContour = usableContourMax + usableContourMin[::-1]   #combine lists, such that the coordinates are listed counter-clockwise

                # TODO not sure if this works properly: meant to concate the coords of a partial contour such that the coords are on a 'smooth partial ellips' without a gap
                for ii in range(0, len(usableContour)-1):
                    if abs(usableContour[ii][1] - usableContour[ii+1][1]) > 200:       #if difference in y between 2 appending coords is large, a gap in the contour is formed
                        usableContour = usableContour[ii:] + usableContour[0:ii]
                useableylist = np.array([elem[1] for elem in usableContour])
                useablexlist = [elem[0] for elem in usableContour]


        else:   #if only 1 value
            print(f"For now something seems to be weird. Either bad contour, or accidentally only 1 real Y at that level."
                  f"Might therefore be a min/max in image, or the contour at the other side somehow skipped 1 Y-value (if this happens, implement new code)")
            exit()

        # for j in range(min(ylist), max(ylist)):     #iterate over all y-coordinates form lowest to highest
        #     indexesJ = np.where(ylist == j)[0]      #find all x-es at 1 y-coord
        #     if len(indexesJ) > 1:                   #if more than 2 x-coords at 1 y-coord, take the min & max as useable contour (contour is fish-hook shaped)
        #         xListpery = [xlist[x] for x in indexesJ]
        #         usableContour.append([max(xListpery), j])
        #     elif len(indexesJ) > 0:
        #         xListpery = [xlist[x] for x in indexesJ]
        #         usableContour.append([max(xListpery), j])
        # useableylist = np.array([elem[1] for elem in usableContour])
        # useablexlist = [elem[0] for elem in usableContour]



    return useablexlist, useableylist, usableContour, resizedimg, greyresizedimg, thresh, i





def importConversionFactors(procStatsJsonPath):
    with open(procStatsJsonPath, 'r') as f:
        procStats = json.load(f)
    conversionZ = procStats["conversionFactorZ"]
    conversionXY = procStats["conversionFactorXY"]
    unitZ = procStats["unitZ"]
    unitXY = procStats["unitXY"]
    return conversionZ, conversionXY, unitZ, unitXY

def filePathsFunction(path):
    if "json" in path:
        procStatsJsonPath = path
        imgFolderPath = os.path.dirname(os.path.dirname(os.path.dirname(procStatsJsonPath)))
        conversionZ, conversionXY, unitZ, unitXY = importConversionFactors(procStatsJsonPath)
    elif os.path.exists(os.path.join(path,f"Measurement_Info.json")):
        procStatsJsonPath = os.path.join(path,f"Measurement_Info.json")
        imgFolderPath = path
        conversionZ, conversionXY, unitZ, unitXY = importConversionFactors(procStatsJsonPath)
    else:   #make a json file with correct measurement conversion factors
        imgFolderPath = path
        conversionZ, conversionXY, unitZ, unitXY = determineLensPresets(path)
    return imgFolderPath, conversionZ, conversionXY, unitZ, unitXY

def convertunitsToEqual(unit):
    units = ['nm', 'um', 'mm', 'm', 'pixels']
    conversionsXY = [1e6, 1e3, 1, 1e-3, 1]  # standard unit is um
    conversionsZ = [1, 1e-3, 1e-6, 1e-9, 1]  # standard unit is nm

    return units.index(unit)

def determineLensPresets(imgFolderPath, wavelength_laser=520, refr_index=1.434):
    units = ['nm', 'um', 'mm', 'm', 'pixels']
    conversionsXY = [1e6, 1e3, 1, 1e-3, 1]  # standard unit is um
    conversionsZ = [1, 1e-3, 1e-6, 1e-9, 1]  # standard unit is nm

    choices = ["ZEISS_OLYMPUSX2", "ZEISS_ZEISSX5", "ZEISS_ZEISSX10"]
    answer = easygui.choicebox("What lens preset was used?", choices=choices)
    if answer == choices[0]:
        preset = 672
    elif answer == choices[1]:
        preset = 1836
    elif answer == choices[2]:
        preset = 3695
    choices_outputUnits = ['nm', 'um', 'mm', 'm', 'pixels']
    unitZ = easygui.choicebox("What output z-unit (nm recommended)?", choices=choices_outputUnits)
    unitXY = easygui.choicebox("What output xy-unit (mm recommended)?", choices=choices_outputUnits)

    if unitXY == "pixels":
        preset = 1
    conversionFactorXY = 1 / preset * conversionsXY[units.index(unitXY)]  # apply unit conversion
    conversionFactorZ = (wavelength_laser) / (2 * refr_index) / (2 * np.pi)  # 1 period of 2pi = lambda / (4n). /2pi since our wrapped space is in absolute units, not pi
    conversionFactorZ = conversionFactorZ * conversionsZ[units.index(unitZ)]  # apply unit conversion

    stats = {}  # save statistics of this measurement
    stats['About'] = {}
    stats['About']['__author__'] = "Sander Reuvekamp"
    stats['About']['repo'] = str(git.Repo(search_parent_directories=True))
    stats['About']['sha'] = git.Repo(search_parent_directories=True).head.object.hexsha
    stats['startDateTime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f %z')
    stats['conversionFactorXY'] = conversionFactorXY
    stats['conversionFactorZ'] = conversionFactorZ
    stats['unitXY'] = unitXY
    stats['unitZ'] = unitZ

    # Save statistics
    with open(os.path.join(imgFolderPath, f"Measurement_Info.json"), 'w') as f:
        json.dump(stats, f, indent=4)

    return conversionFactorZ, conversionFactorXY, unitZ, unitXY

def main():
    #procStatsJsonPath = os.path.join("D:\\2023_08_07_PLMA_Basler5x_dodecane_1_28_S5_WEDGE_1coverslip spacer_COVERED_SIDE\Analysis_1\PROC_20230809115938\PROC_20230809115938_statistics.json")
    #procStatsJsonPath = os.path.join("D:\\2023_09_22_PLMA_Basler2x_hexadecane_1_29S2_split\\B_Analysis\\PROC_20230927135916_imbed", "PROC_20230927135916_statistics.json")
    #imgFolderPath = os.path.dirname(os.path.dirname(os.path.dirname(procStatsJsonPath)))
    path = os.path.join("D:\\2023_08_07_PLMA_Basler5x_dodecane_1_28_S5_WEDGE_1coverslip spacer_COVERED_SIDE\Analysis_1\PROC_20230809115938\PROC_20230809115938_statistics.json")
    #path = "D:\\2023_08_07_PLMA_Basler5x_dodecane_1_28_S5_WEDGE_1coverslip spacer_AIR_SIDE"
    #path = "E:\\2023_10_31_PLMA_Dodecane_Basler5x_Xp_1_28_S5_WEDGE"
    #path = "E:\\2023_10_31_PLMA_Dodecane_Basler5x_Xp_1_29_S1_FullDropletInFocus"

    thresholdSensitivity = [11 * 3, 2 * 5]          #OG: 11 * 5, 2 * 5
    imgFolderPath, conversionZ, conversionXY, unitZ, unitXY = filePathsFunction(path)

    imgList = [f for f in glob.glob(os.path.join(imgFolderPath, f"*tiff"))]
    usedImages = [200, 201]   #np.arange(10,len(imgList), 5)              #200 is the working one
    analysisFolder = os.path.join(imgFolderPath,"Analysis CA Spatial")
    resizeFactor = 1            #=5 makes the image fit in your screen, but also has less data points when analysing
    lengthVector = 200      #225 length of normal vector over which intensity profile data is taken
    FLIPDATA = True
    if not os.path.exists(analysisFolder):
        os.mkdir(analysisFolder)
        print('created path: ', analysisFolder)
    contourListFilePath = os.path.join(analysisFolder, "ContourListFile.txt")
    if os.path.exists(contourListFilePath): #read in all contourline data from existing file (filenr ,+ i for obtaining contour location)
        f = open(contourListFilePath, 'r')
        lines = f.readlines()
        importedContourListData_n = []
        importedContourListData_i = []
        for line in lines:
            importedContourListData_n.append(int(line.split(',')[0]))
            importedContourListData_i.append(int(line.split(',')[1]))
        #importedContourListData = np.loadtxt(contourListFilePath, dtype='int', delimiter=',',  usecols=(0,1))
    else:
        f = open(contourListFilePath, 'w')
        f.close()
        print("Created contour list file.txt")
        importedContourListData_n = []
        importedContourListData_i = []


    if unitZ != "nm" or unitXY != "mm":
        raise Exception("One of either units is not correct for good conversion. Fix manually or implement in code")
    for n, img in enumerate(imgList):
        if n in usedImages:
            if n in importedContourListData_n:
                contouri = importedContourListData_i[importedContourListData_n.index(n)]
            else:
                contouri = -1

            #One of the main functions: outputs the coordiates of the desired contour of current image
            if n == usedImages[0]:  #on first iteration, don't parse previous coords (because there are none)
                useablexlist, useableylist, usableContour, resizedimg, greyresizedimg, thresh, outputi = getContourCoordsV4(img, resizeFactor, analysisFolder, contouri, thresholdSensitivity)
            elif n - usedImages[usedImages.index(n)-1] == 1:   #if using sequential images, use coordinates of previous image
                useablexlist, useableylist, usableContour, resizedimg, greyresizedimg, thresh, outputi = getContourCoordsV4(img, resizeFactor, analysisFolder, contouri, thresholdSensitivity, usableContour)
            else: #else, don't parse coordinates (let user define them themselves)
                useablexlist, useableylist, usableContour, resizedimg, greyresizedimg, thresh, outputi = getContourCoordsV4(img, resizeFactor, analysisFolder, contouri, thresholdSensitivity)

            if contouri < 0:    #Save which contour is used to a txt file for this n, for ease of further iterations
                file = open(contourListFilePath, 'a')
                file.write(f"{n}, {outputi}\n")
                file.close()

            try:
                resizedimg = cv2.polylines(resizedimg, np.array([usableContour]), False, (0, 0, 255), 2)  # draws 1 red good contour around the outer halo fringe

                #One of the main functions:
                #Should yield the normal for every point: output is original x&y, and corresponding normal x,y (defined as dx and dy)
                x0arr, dxarr, y0arr, dyarr = get_normalsV4(useablexlist, useableylist, lengthVector)

                tempcoords = [[x0arr[k], y0arr[k]] for k in range(0, len(x0arr))]
                for k in x0arr:
                    if k < 0:
                        print(f"xval: {k}, index: {x0arr.index(k)}")
                for k in y0arr:
                    if k < 0:
                        print(f"yval {k}, index: {y0arr.index(k)}")

                tempimg = []
                tempimg = cv2.polylines(resizedimg, np.array([tempcoords]), False, (0, 255, 0), 20)  # draws 1 blue contour with the x0&y0arrs obtained from get_normals
                tempimg = cv2.resize(tempimg, [round(5328 / 5), round(4608 / 5)], interpolation=cv2.INTER_AREA)  # resize image
                cv2.imshow(f"Contour img with coordinates being used by get_normals", tempimg)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
                cv2.imwrite(os.path.join(analysisFolder, f"rawImage_x0y0Arr_blue{n}.png"), tempimg)

                # #TODO, temp: export coordinates of contour to csv file
                # wrappedPath = os.path.join(analysisFolder, f"ContourLineData.csv")
                # d = dict({'x-coords': useablexlist, 'y-coords': useableylist})
                # df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))  # pad shorter colums with NaN's
                # df.to_csv(wrappedPath, index=False)
                angleDegArr = []

                for k in range(0, len(x0arr)):
                    # if k == 101:
                    #     print(f"at this k we break={k}")
                    #TODO trying to get this to work: plotting normals obtained with above function get_normals
                    #resizedimg = cv2.polylines(resizedimg, np.array([[x0arr[k], y0arr[k]], [dxarr[k], dyarr[k]]]), False, (0, 255, 0), 2)  # draws 1 good contour around the outer halo fringe#
                    if k % 25 == 0: #only plot 1/25th of the vectors to not overcrowd the image
                        resizedimg = cv2.line(resizedimg, ([x0arr[k], y0arr[k]]), ([dxarr[k], dyarr[k]]), (0, 255, 0), 2)  # draws 1 good contour around the outer halo fringe

                    # if k == 101:  # TODO remove, only for plotting a k value which does not work at n=10
                    #     resizedimg = cv2.line(resizedimg, ([x0arr[k], y0arr[k]]), ([dxarr[k], dyarr[k]]), (0, 255, 0), 8)  # draws 1 good contour around the outer halo fringe
                    #     cv2.imshow("Point of breaking", resizedimg)
                    #     cv2.waitKey(0)
                    #     cv2.destroyAllWindows()

                    if dxarr[k] - x0arr[k] == 0:        #constant x, variable y
                        xarr = np.ones(lengthVector) * x0arr[k]
                        if y0arr[k] > dyarr[k]:
                            yarr = np.arange(dyarr[k], y0arr[k]+1)
                        else:
                            yarr = np.arange(y0arr[k], dyarr[k]+1)
                        coords = list(zip(xarr.astype(int), yarr.astype(int)))
                    else:
                        a = (dyarr[k] - y0arr[k])/(dxarr[k] - x0arr[k])
                        b = y0arr[k] - a*x0arr[k]
                        coords, lineLengthPixels = coordinates_on_line(a, b, [x0arr[k], dxarr[k], y0arr[k], dyarr[k]])     #
                    profile = [np.transpose(greyresizedimg)[pnt] for pnt in coords]

                    profile_fft = np.fft.fft(profile)  # transform to fourier space
                    mask = np.ones_like(profile).astype(float)
                    # NOTE: lowpass seems most important for a good sawtooth profile. Filtering half of the data off seems fine
                    lowpass = round(len(profile) / 2); highpass = 2
                    mask[0:lowpass] = 0; mask[-highpass:] = 0
                    profile_fft = profile_fft * mask
                    profile_filtered = np.fft.ifft(profile_fft)
                    # plt.plot(profile_fft)
                    # plt.title("Fourier plot")
                    # plt.show()

                    # calculate the wrapped space
                    wrapped = np.arctan2(profile_filtered.imag, profile_filtered.real)
                    peaks, _ = scipy.signal.find_peaks(wrapped, height=0.4)  # obtain indeces om maxima

                    unwrapped = np.unwrap(wrapped)
                    if FLIPDATA:
                        unwrapped = -unwrapped + max(unwrapped)


                    unwrapped *= conversionZ / 1000         #if unwapped is in um: TODO fix so this can be used for different kinds of Z-unit
                    #x = np.arange(0, len(unwrapped)) * conversionXY * 1000 #TODO same ^
                    x = np.linspace(0, lineLengthPixels, len(unwrapped)) * conversionXY * 1000
                    coef1 = np.polyfit(x, unwrapped, 1)


                    # plt.show()

                    # TODO temp, only to show the profile in a plot
                    if k == round(len(x0arr)/2):
                        fig1, ax1 = plt.subplots(2, 2)
                        ax1[0,0].plot(profile); ax1[1,0].plot(wrapped); ax1[1,0].plot(peaks, wrapped[peaks], '.')
                        ax1[0,0].set_title(f"Intensity profile"); ax1[1,0].set_title("Wrapped profile")
                        ax1[0,1].plot(x, unwrapped * 1000); ax1[0,1].set_title("Drop height vs distance (unwrapped profile)")
                        ax1[0,1].plot(x, np.poly1d(coef1)(x) * 1000, linewidth=0.5)
                        ax1[0, 1].set_xlabel("Distance (um)"); ax1[0, 1].set_ylabel("Height profile (nm)")
                        plt.show()

                    a_horizontal = 0
                    angleRad = math.atan((coef1[0] - a_horizontal) / (1 + coef1[0] * a_horizontal))
                    angleDeg = math.degrees(angleRad)
                    # Flip measured CA degree if higher than 45.
                    if angleDeg > 45:
                        angleDeg = 90 - angleDeg
                    #print(f"Length of studied array is= {len(unwrapped)}")
                    #print(f"Calculated angle: {angleDeg} deg")

                    angleDegArr.append(angleDeg)

                # Fit an ellipse around the contour
                #ellipse = cv2.fitEllipse(contour)
                # Draw the ellipse on the original frame
                #resizedimg = cv2.ellipse(resizedimg, ellipse, (0, 255, 0), 2)                #draws an ellips, which fits poorly
                # get the middlepoint of the contour and draw a circle in it
                #M = cv2.moments(contour)
                #cx = int(M["m10"] / M["m00"])
                #cy = int(M["m01"] / M["m00"])
                #resizedimg = cv2.circle(resizedimg, (cx, cy), 13, (255, 0, 0), -1)           #draws a white circle
                #cx_save.append(cx)
                #cy_save.append(cy)
                plt.plot(y0arr, angleDegArr, '.')
                plt.xlabel("Y-coord"); plt.ylabel("Calculated Contact Angle (deg)"); plt.title("Calculated Contact angles")
                plt.savefig(os.path.join(analysisFolder, f'CA vs Ycoord {n}.png'), dpi=600)
                plt.close()

                plt.scatter(x0arr, abs(np.subtract(4608,y0arr)), c=angleDegArr, cmap='jet', vmin=min(angleDegArr), vmax=max(angleDegArr))
                #plt.scatter(x0arr, abs(np.subtract(4608, y0arr)), c=angleDegArr, cmap='jet', vmin=3, vmax=max(angleDegArr))
                plt.xlabel("X-coord"); plt.ylabel("Y-Coord"); plt.title("Spatial Contact Angles Colormap")
                plt.colorbar()
                plt.savefig(os.path.join(analysisFolder, f'Colorplot XYcoord-CA {n}.png'), dpi=600)
                plt.show()

                #Export Contact Angles to a csv file
                wrappedPath = os.path.join(analysisFolder, f"ContactAngleData {n}.csv")
                d = dict({'x-coords': x0arr, 'y-coords': y0arr, 'contactAngle' : angleDegArr})
                df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))  # pad shorter colums with NaN's
                df.to_csv(wrappedPath, index=False)
            except:
                print(f"Some error occured. Still plotting obtained contour")
            tstring = str(datetime.now().strftime("%Y_%m_%d")) #day&timestring, to put into a filename    "%Y_%m_%d_%H_%M_%S"
            cv2.imwrite(os.path.join(analysisFolder, f"rawImage_contourLine_{tstring}_{n}.png") , resizedimg)
            #cv2.imwrite(os.path.join(analysisFolder, f"threshImage_contourLine{tstring}.png") , thresh)
if __name__ == "__main__":
    main()
    exit()