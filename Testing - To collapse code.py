import itertools
import os.path
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import matplotlib.animation as animation
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
import git
import statistics

from matplotlib.widgets import RectangleSelector
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import integrate
from scipy.interpolate import interpolate
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from SwellingFromAbsoluteIntensity import heightFromIntensityProfileV2
import traceback

right_clicks = list()
def click_event(event, x, y, flags, params):
    '''
    Click event for the setMouseCallback cv2 function. Allows to select 2 points on the image and return it coordinates.
    '''
    if event == cv2.EVENT_LBUTTONDOWN:
        global right_clicks
        right_clicks.append([x, y])
    if len(right_clicks) == 2:
        cv2.destroyAllWindows()
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
def get_normalsV4(x, y, L, L2 = 0, L3 = 0):
    # For each coordinate, fit with nearby points to a polynomial to better estimate the dx dy -> tangent
    # Take derivative of the polynomial to obtain tangent and use that one.
    x0arr = []; dyarr = []; y0arr = []; dxarr = []; dxnegarr = []; dynegarr = []; dxL3arr = []; dyL3arr = []
    window_size = round(len(x) / 100)            #!!! window size to use. 25 seems to be good for most normal vectors
    k = round((window_size+1)/2)

    middleCoordinate = [(max(x) + min(x))/2, (max(y) + min(y))/2]   #estimate for "middle coordinate" of contour. Will be used to direct normal vector to inside contour

    connectingCoordinatesDyDx = 30  #if coordinates are within 30 pixels of each other, probably they were connecting
    if abs(y[0]-y[-1]) < connectingCoordinatesDyDx and abs(x[0]-x[-1]) < connectingCoordinatesDyDx:   #if ends of contours are connecting, allow all the points also from the other end to be used for fitting
        x = x[-k:] + x + x[:k]
        y = np.hstack((y[-k:], y, y[:k]))

    for idx in range(k, len(x) - k):
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

            #Determine direction of normal vector by calculating of both direction the Root-square-mean to the "middle coordinate" of the contour, and take the smallest one
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

        if L2 != 0:
            dxnegarr.append(round(x0 - (L2*(nx/L))))   #for normals pointing outwards of the droplet
            dynegarr.append(round(y0 - (L2*(ny/L))))

        if L3 != 0:
            dxL3arr.append(round(x0 - (L3*(nx/L))))   #for normals pointing outwards of the droplet
            dyL3arr.append(round(y0 - (L3*(ny/L))))

    vector = [[dxarr[i] - x0arr[i], dyarr[i] - y0arr[i]] for i in range(0, len(x0arr))]   #vector [dx, dy] for each coordinate
    return x0arr, dxarr, y0arr, dyarr, vector, dxnegarr, dynegarr, dxL3arr, dyL3arr  # return the original data points x0&y0, and the coords of the inwards normals dx&dy, and the outwards normals dxneg&dyneg


def getContourList(grayImg, thresholdSensitivity):
    contourArea = 3000  #usually 5000
    WORKINGTRESH = False
    while not WORKINGTRESH:
        try:
            thresh = cv2.adaptiveThreshold(grayImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, thresholdSensitivity[0], thresholdSensitivity[1])
            WORKINGTRESH = True
        except:
            print(f"Tresh didn't work. Choose different sensitivity")
            title = "Inputted thresh didn't work: New threshold input"
            msg = f"Inputted threshold sensitivity didn't work! Input new.\nCurrent threshold sensitivity is: {thresholdSensitivity}. Change to (comma seperated):"
            out = easygui.enterbox(msg, title)
            thresholdSensitivity = list(map(int, out.split(',')))

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
        if area > contourArea:  # OG = 2000
            # Check if the contour has at least 5 points
            if len(contour) >= 5:
                list_of_pts.append(len(contour))     #list_of_pts += [pt[0] for pt in contour]
                print(f"length of contours={len(contour)}")

                maxxlist.append(max([elem[0][0] for elem in contour]))  # create a list with the max-x coord of a contour, to know which contour is the furthest right, 1 from furthest etc..
                contourList.append(contour)

    #TODO this number determines how many contours will be passed&shown to the user! Should be coupled to the later chosen img number, otherwise further iterations pick the wrong one from the save-file
    #TODO for most runs, I used 20
    nrOfContoursToShow = 20
    if len(contourList) < nrOfContoursToShow:# nr of contours allow to be to checked
        nrOfContoursToShow = len(contourList)

    unfinished = True       #while loop below = for making sure the code doesn't break when multiple x's have same value.
    # while unfinished:
    #     changedList = False
    #     for i, val in enumerate(maxxlist):          #This function is required because maxxList must have unique values.
    #         if maxxlist.count(val) > 1:             #Check for all values if it is unique
    #             maxxlist[i] = val+0.1                 #if not, change the value to itself+0.1, in order
    #             changedList = True
    #     if not changedList:
    #         unfinished = False
    while unfinished:
        changedList = False
        for i, val in enumerate(list_of_pts):          #This function is required because maxxList must have unique values.
            if list_of_pts.count(val) > 1:             #Check for all values if it is unique
                list_of_pts[i] = val+0.1                 #if not, change the value to itself+0.1, in order
                changedList = True
        if not changedList:
            unfinished = False

    #FurthestRightContours = sorted(zip(maxxlist, contourList), reverse=True)[:nrOfContoursToShow]  # Sort contours, and zip the ones of which the furthest right x coords are found
    FurthestRightContours = sorted(zip(list_of_pts, contourList), reverse=True)[:nrOfContoursToShow]  # Sort contours, and zip the ones of which the furthest right x coords are found

    #cv2.imwrite(os.path.join(analysisFolder, f"threshImage_contourLine{tstring}.png"), thresh)     #if you want to save the threshold image
    return [elem[1] for elem in FurthestRightContours], nrOfContoursToShow, thresholdSensitivity


def tryVariousThreshholdSensitivities(grayImg, thresholdSensitivityrange1, thresholdSensitivityrange2):
    workingThreshes = []
    threshesToShow = 15
    contourArea = 5000
    totalcounter = 0
    for thresh1 in thresholdSensitivityrange1:
        counter = 0
        for thresh2 in thresholdSensitivityrange2:
            try:
                if counter < 3 and totalcounter < threshesToShow: #only try for 3 thresh2-s at the same thresh1
                    thresh = cv2.adaptiveThreshold(grayImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, thresh1, thresh2)
                    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    for contour in contours:
                        # Calculate the area of the contour
                        area = cv2.contourArea(contour)
                        # Set a minimum threshold for contour area
                        if area > contourArea:  # OG = 2000
                            # Check if the contour has at least 5 points
                            if len(contour) >= 5:
                                 workingThreshes.append([thresh1, thresh2])
                                 print(f"{thresh1, thresh2}")
                                 break
                    counter=+1
                    totalcounter=+1
            except:
                pass
    if totalcounter == threshesToShow:
        logging.warning(f"Stopped displaying working threshold sensitivities after {threshesToShow} were found.\n "
                        f"If for whatever reason you want more next time, change 'threshesToShow' in the code manually."  )
    return workingThreshes


#TODO Purely for manually selecting a part of an image in which a contour must be found
def selectAreaAndFindContour(resizedimg, thresholdSensitivity, resizeFactor):
    tempimg = []
    copyImg = resizedimg.copy()
    #tempimg = cv2.polylines(copyImg, np.array([contourList[i]]), False, (255, 0, 0), 8)  # draws 1 blue contour with the x0&y0arrs obtained from get_normals
    #if combineMultipleContours:
    #    tempimg = cv2.polylines(tempimg, np.array([contour]), False, (255, 0, 0), 8)  # draws 1 blue contour with the x0&y0arrs obtained from get_normals

    tempimg = cv2.resize(copyImg, resizeFactor, interpolation=cv2.INTER_AREA)  # resize image
    cv2.imshow('Grey image', tempimg)
    cv2.setWindowTitle("Grey image", "Square selection window. Click 2 times to select box in which contour is to be found.")
    cv2.setMouseCallback('Grey image', click_event)
    cv2.waitKey(0)
    global right_clicks

    P1 = np.array(right_clicks[0]) * 5      #point 1 [x,y]
    P2 = np.array(right_clicks[1]) * 5
    selectionOfInterest = copyImg[P1[1]:P2[1], P1[0]:P2[0]]     #img[y1:y2, x1:x2]
    tempimg2 = cv2.resize(selectionOfInterest, [round(selectionOfInterest.shape[0]/5), round(selectionOfInterest.shape[1]/5)], interpolation=cv2.INTER_AREA)  # resize image
    cv2.imshow('Partial image', tempimg2)
    cv2.waitKey(0)
    contourList, nrOfContoursToShow, thresholdSensitivity = getContourList(selectionOfInterest, thresholdSensitivity)  # obtain new contours with new thresholldSensitivity
    if len(contourList) == 0:
        logging.warning(f"INFO: no contours found in selected region!")
    adjustedContourList = []
    for contour in contourList:
        adjustedContourList.append([np.array([[elem[0][0] + P1[0], elem[0][1]+P1[1]]]) for elem in contour])

    right_clicks = list()
    return adjustedContourList


# Attempting to get a contour from the full-sized HQ image, and using resizefactor only for showing a copmressed image so it fits in the screen
# Parses all 'outer' coordinates, not only on right side of droplet
#With working popup box for checking and changing contour
def getContourCoordsV4(imgPath, contourListFilePath, n, contouri, thresholdSensitivity, MANUALPICKING, **kwargs):
    contourCoords = 0
    FITGAPS_POLYOMIAL = True
    saveCoordinates = False     #save obtained [x,y] coordinates to a .txt file
    coordinatesListFilePath = os.path.join(os.path.dirname(contourListFilePath), f"ContourCoords\\coordinatesListFilePath_{n}.txt")
    for keyword, value in kwargs.items():
        if keyword == "contourcoords":
            contourCoords = value
        elif keyword == "fitgapspolynomial":
            FITGAPS_POLYOMIAL = value
        elif keyword == "saveCoordinates":
            saveCoordinates = value
        elif keyword == "contourCoordsFolderFilePath":
            coordinatesListFilePath = value
        else:
            logging.error(f"Incorrect keyword inputted: {keyword} is not known")

    minimalDifferenceValue = 100    #Value of minimal difference in x1 and x2 at same y-coord to check for when differentiating between entire droplet & partial contour & fish-hook-like contour
    img = cv2.imread(imgPath)  # read in image
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to greyscale
    imgshape = img.shape  #tuple (height, width, channel)
    # Apply adaptive thresholding to the blurred frame
    # TODO not nice now, but for the code to work
    greyresizedimg = grayImg
    resizedimg = img

    contourList, nrOfContoursToShow, thresholdSensitivity = getContourList(grayImg, thresholdSensitivity)

    if MANUALPICKING == 0 or (MANUALPICKING == 1 and contouri[0] < 0):
        i = 0
        goodContour = False
        combineMultipleContours = False
        contour = []
    elif MANUALPICKING == 3:  #always use the second most outer halo, if available.
        if len(contourList) > 1:
            contouri = [1]
        else:
            contouri = [0]

        #TODO attempting different way of checking which contour is the correct one.
        #Maybe iterate for all contours, starting at the most outer one to obtain a few intensity profiles: determine if peak periodicity is very constant.
        if not isinstance(contourCoords, int):
            ylistToCheckCoords = np.array([elem[1] for elem in contourCoords])
            xlistToCheckCoords = [elem[0] for elem in contourCoords]
            lenOfCoordinatesToCheck = 50 # only compare 50 values of y
            ysToCheck = np.linspace(min(ylistToCheckCoords) + 5, max(ylistToCheckCoords) - 5,
                                    lenOfCoordinatesToCheck).round()
            xsToCheck = []
            for y in ysToCheck:  # obtain corresponding x's of the to-be-investigated y's
                xsToCheck.append(xlistToCheckCoords[np.where(ylistToCheckCoords == y)[0][0]])

            bestCorrespondingContour = np.zeros(len(contourList))
            absError = 0
            for i, yToCheck in enumerate(ysToCheck):  # compare x-values at various y-heights
                xlistPerY = []
                for contour in contourList:
                    xAtyIndices = np.where([elem[0][1] for elem in contour] == yToCheck)
                    xaty = [contour[k][0][0] for k in xAtyIndices[0]]
                    xlistPerY.append(np.mean(xaty))  # TODO temp max() or np.mean : xaty are multiple values, and we only want to compare 1. But it is unknown which one would be 'best' to compare. For now try max()
                absErrorArray = abs(np.subtract(xlistPerY, xsToCheck[i]))
                bestCorrespondingCoordinateIndex = np.argmin(
                    absErrorArray)  # find best corresponding x of xsToCheck[i] xlistPerY
                absError += absErrorArray[bestCorrespondingCoordinateIndex]
                bestCorrespondingContour[bestCorrespondingCoordinateIndex] += 1
            i = np.argmax(bestCorrespondingContour)
            contouri = [i]
            contour = contourList[i]
            print(f"BestCorrespondingContour list= {bestCorrespondingContour}")
            if absError / lenOfCoordinatesToCheck > 100:  # if error for every y-coord is large, probably the image was shifted. Therefore previous image should not be compared with this one, and a manual contour should be picked
                goodContour = False
                print(
                    f"Manually picking contour because absolute error was determined to be: {absError}, meaning error per y = {absError / lenOfCoordinatesToCheck}")
                combineMultipleContours = False
                contour = []
            else:
                goodContour = True
                print(
                    f"Automatically picked contour looks good from error per y ({absError / lenOfCoordinatesToCheck}/ycoord)")


        else: #else, give i a value & have the user manually check for contour correctness later on
            #if all big differences (or the img number is =0), set i manually to a value and allow user input to decide the contour
            if nrOfContoursToShow > 1:
                i = 1
            else:
                i = 0
            goodContour = False
            print(f"Picking manual contour, because no previous image (and therefore no contour) was given")
            combineMultipleContours = False
            contour = []
    elif MANUALPICKING == 1:   #if contouri has a value, it is an imported value, chosen in a previous iteration & should already be good
        if len(contouri) > 1:
            contour = []
            for ii in contouri:
                for elem in contourList[ii]:
                    contour.append(list(elem))
        else:
            contour = contourList[contouri[0]]
        goodContour = True
        print(f"Using a contour which was determined in a previous iteration from .txt file. ")



    iout = []
    firstTimeNoContours = True
    #show img w/ contour to check if it is the correct one
    #make popup box to show next contour (or previous) if desired
    while goodContour == False:
        if len(contourList) > 0:
            if i >= len(contourList):    #to make sure list index cannot be out of range
                logging.warning(f"INFO: i ({i})was smaller than the len of contourlist {len(contourList)} and has therefore be changed to {len(contourList)-1}")
                i = len(contourList)-1
            tempimg = []
            copyImg = resizedimg.copy()
            tempimg = cv2.polylines(copyImg, np.array([contourList[i]]), False, (255, 0, 0), 8)  # draws 1 blue contour with the x0&y0arrs obtained from get_normals
            if combineMultipleContours:
                tempimg = cv2.polylines(tempimg, np.array([contour]), False, (255, 0, 0), 8)  # draws 1 blue contour with the x0&y0arrs obtained from get_normals
            tempimg = cv2.resize(tempimg, [round(imgshape[1] / 5), round(imgshape[0] / 5)], interpolation=cv2.INTER_AREA)  # resize image
            cv2.imshow(f"Contour img with current selection of contour {i+1} out of {nrOfContoursToShow}", tempimg)
            choices = ["One contour outwards (-i)", "Current contour is fine", "One contour inwards (+1)",
                       "Stitch multiple contours together: first selection",
                       "No good contours: Ajdust threshold sensitivities", "No good contours: quit programn",
                       "EXPERIMENTAL: Drawing box in which contour MUST be found (in case it never finds it there)"]
            myvar = easygui.choicebox("Is this a desired contour?", choices=choices)
        else:
            tempimg = []
            copyImg = resizedimg.copy()
            tempimg = cv2.resize(copyImg, [round(imgshape[1] / 5), round(imgshape[0] / 5)], interpolation=cv2.INTER_AREA)  # resize image
            #TODO iterate over a bunch of thresholds, and suggest a few working ones
            cv2.imshow(f"Contour img with current selection of contour {i + 1} out of {nrOfContoursToShow}", tempimg)

            choices = ["One contour outwards (-i)",
                       "Current contour is fine",
                       "One contour inwards (+1)",
                       "Stitch multiple contours together: first selection",
                       "No good contours: Ajdust threshold sensitivities",
                       "No good contours: quit programn",
                       "EXPERIMENTAL: Drawing box in which contour MUST be found (in case it never finds it there)",
                       "Suggest working thresholds. This might take some time."]
            myvar = easygui.choicebox("From the get-go, no contours were obtained with this threshold sensitivity. Choose option 5 to change this.\n"
                                      "Working thresholds are suggested in the terminal. This might take some time", choices=choices)
            if myvar == choices[7]:
                if firstTimeNoContours:
                    thresholdSensitivityrange1 = np.arange(1, 80, 3)
                    thresholdSensitivityrange2 = np.arange(1, 80, 3)
                    workingthreshholds = tryVariousThreshholdSensitivities(grayImg, thresholdSensitivityrange1, thresholdSensitivityrange2)
                    print(f"{workingthreshholds}")
                    firstTimeNoContours = False
                else:
                    logging.warning("Already printed working contours previously to terminal. Not doing that again.")

        #cv2.waitKey(0)

        if myvar == choices[0]: #picks different i-1, if possible
            if i == 0:
                out = easygui.msgbox("i is already 1, cannot go further out")
            else:
                i -= 1
        elif myvar == choices[1]:   #confirms this i, saves contour
            goodContour = True
            if combineMultipleContours:
                contour = list(contour)
                for elem in contourList[i]:
                    contour.append(list(elem))
            else:
                contour = contourList[i]
            iout.append(i)
        elif myvar == choices[2]:       #pick different i+1, if possible
            if i == len(contourList)-1:
                out = easygui.msgbox("i is already maximum value, cannot go further inwards")
            else:
                i += 1
        elif myvar == choices[3]:   #Stitch together multiple contours with current settings:
            if combineMultipleContours:
                contour = list(contour)
                for elem in contourList[i]:
                    contour.append(list(elem))
            else:
                contour = contourList[i]
            combineMultipleContours = True
            iout.append(i)
        elif myvar == choices[4]:   #Redo entire loop with different sensitivity threshold
            title = "New threshold input"
            msg = f"Current threshold sensitivity is: {thresholdSensitivity}. Change to (comma seperated):"
            out = easygui.enterbox(msg, title)
            thresholdSensitivity = list(map(int, out.split(',')))
            contourList, nrOfContoursToShow, thresholdSensitivity = getContourList(grayImg, thresholdSensitivity)         #obtain new contours with new thresholldSensitivity
            i = 0
            iout = []       #reset previously chosen i values
            combineMultipleContours = False
        elif myvar == choices[5]:   #Quit programn altogether
            out = easygui.msgbox("Closing loop, the programn will probably break down")
            break
        #TODO select box for finding contour
        elif myvar == choices[6]:   #experimental: drawing a box in which the contour MUST be found
            resizeFactor = [round(imgshape[1] / 5), round(imgshape[0] / 5)]
            contourList = selectAreaAndFindContour(grayImg, thresholdSensitivity, resizeFactor)
        contour = np.array(contour)
        cv2.destroyAllWindows()


    if goodContour == True:     #i == 1  # generally, not the furthest one out (halo), but the one before that is the contour of the CL
        # check per y value first, then compare x's to find largest x
        usableContour = []
        usableContourMax = []
        usableContourMin = []
        ylist = np.array([elem[0][1] for elem in contour])
        xlist = [elem[0][0] for elem in contour]

        # TODO temp only for plotting the contour of which the vectors are taken
        # tempContourImg = cv2.polylines(resizedimg, np.array([contour]), False, (0, 0, 255), 2)  # draws 1 good contour around the outer halo fringe
        # tempContourImg = cv2.resize(tempContourImg, [round(imgshape[1] / 5), round(imgshape[0] / 5)], interpolation=cv2.INTER_AREA)  # resize image
        # cv2.imshow(f"Contour img of i={i} out of {len(contourList)}", tempContourImg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        ########################################################################################
        # iterate from min y to max y
        # if (1) the x values at +- the middle y are not spaced far apart, we are only looking at a small contour at either the left or right side of the droplet
        # if (2) the x values are far apart, probably the contour of an entire droplet is investigated
        ########################################################################################

        calcMiddleYVal = round((max(ylist) + min(ylist))/2) #calculated middle Y value; might be that it does not exist, so look for closest real Y value below
        diffMiddleYval = abs(ylist-calcMiddleYVal)     #
        realMiddleYIndex = diffMiddleYval.argmin()    #find index of a real Y is closest to the calculated middle Y
        realMiddleYVal = ylist[realMiddleYIndex]        #find 1 value of real Y
        allYindicesAtMiddle = np.where(ylist == realMiddleYVal)[0]         #all indices of the same real y's
        if len(allYindicesAtMiddle) > 0:        #if there's more than 1 y-value, compare the min and max X at that value; if the difference is big -> entire droplet
            allXValuesMiddle = [xlist[i] for i in allYindicesAtMiddle]
            if abs(max(allXValuesMiddle) - min(allXValuesMiddle)) < minimalDifferenceValue: #X-values close together -> weird contour at only a part (primarily left or right) of the droplet
                case = 1
                logging.warning("PARTIAL DROPLET - as determined by getCountourCoords() code ")
                for j in range(min(ylist), max(ylist)):     #iterate over all y-coordinates from lowest to highest
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
                usableContour = usableContourMax + usableContourMin[::-1]   #combined list 1 and reversed list2
                # TODO not sure if this works properly: meant to concate the coords of a partial contour such that the coords are on a 'smooth partial ellips' without a gap
                for ii in range(0, len(usableContour) - 1):
                        if abs(usableContour[ii][1] - usableContour[ii + 1][1]) > 20:  # if difference in y between 2 appending coords is large, a gap in the contour is formed
                            usableContour = usableContour[ii:] + usableContour[0:ii] #combined list 1 and reversed list2
                            print("The order of coords has been changed")

                useableylist = np.array([elem[1] for elem in usableContour])
                useablexlist = [elem[0] for elem in usableContour]

            else:   #far spaced x-values: probably contour of an entire droplet: take the min & max at every y-coordinate
                case = 2
                logging.warning("WHOLE DROPLET - as determined by getCountourCoords() code ")
                #for j in range(min(ylist), max(ylist)):  # iterate over all y-coordinates form lowest to highest
                for j in sorted(set(ylist)):        #iterate through all values in ylist (with duplicates removed)
                    indexesJ = np.where(ylist == j)[0]  # find all x-es at 1 y-coord

                    xListpery = [xlist[x] for x in indexesJ]    #list all x'es at that y
                    if (max(xListpery) - min(xListpery)) < 30: #if pixel difference is less than some value, even here something is kinda wrong. So only parse 'close' values to previous good ones
                        #left or right side of drop: check by comparing current x-coord with x-coord of minimum-y
                        if max(xListpery) > xlist[ylist.argmin()]:  # xcoord(current) > xcoord(@y_min) = right side droplet -> take max (x)
                            usableContourMax.append([max(xListpery), j])
                        else:
                            usableContourMin.append([min(xListpery), j])
                    else: #if pixel are far enough spaced apart: probably good outer contour of droplet. Parse min & max [x,y]
                        usableContourMax.append([max(xListpery), j])    #add the max [x,y] into a list
                        usableContourMin.append([min(xListpery), j])    #add the min [x,y] into another list
                usableContour = usableContourMax + usableContourMin[::-1]   #combine lists, such that the coordinates are listed counter-clockwise
                useableylist = np.array([elem[1] for elem in usableContour])
                useablexlist = [elem[0] for elem in usableContour]


                usableContourCopy = np.array(usableContour)
                windowSizePolynomialCheck = 40  #nr of values to check left and right for fitting polynomial, if distance between 2 values is 'too large'
                usableContourCopy = np.concatenate([usableContourCopy[-windowSizePolynomialCheck:], usableContourCopy, usableContourCopy[:windowSizePolynomialCheck]])      #add values (periodic BC) for checking beginning& end of array
                usableContourCopy_instertion = usableContourCopy    #copy into which coords from polynomial fits are inserted
                ii_inserted = 0     #counter for index offset if multiple insertions have to be performed
                if FITGAPS_POLYOMIAL:
                    # TODO not sure if this works properly: meant to concate the coords of a partial contour such that the coords are on a 'smooth partial ellips' without a gap
                    for ii in range(windowSizePolynomialCheck, len(usableContourCopy)-(windowSizePolynomialCheck)+1):   #TODO check deze +1: ik wil ook een fit @top droplet, maar werkt nog niet
                        #TODO try to implement function that fits an ellips between gaps:
                        #OG CODE START
                        # if abs(usableContour[ii][1] - usableContour[ii+1][1]) > 200:       #if difference in y between 2 appending coords is large, a gap in the contour is formed
                        #     usableContour = usableContour[ii:] + usableContour[0:ii]        #shift coordinates in list such that the coordinates are sequential neighbouring
                        #OG CODE END

                        # TODO check for gaps in x axis
                        if abs(usableContourCopy[ii][0] - usableContourCopy[ii + 1][0]) > 20:      #if difference in x between 2 appending coords is large, a horizontal gap in the contour is formed
                            xrange_for_fitting = [i[0] for i in usableContourCopy[(ii-windowSizePolynomialCheck):(ii+windowSizePolynomialCheck)]]
                            yrange_for_fitting = [i[1] for i in usableContourCopy[(ii - windowSizePolynomialCheck):(ii + windowSizePolynomialCheck)]]
                            # xrange_for_fitting = usableContour[(ii-windowSizePolynomialCheck):(ii+windowSizePolynomialCheck)][0] #to fit polynomial with 30 points on both sides of the gap   #todo gaat fout als ii<30 of > (len()-30)
                            # yrange_for_fitting = usableContour[ii-windowSizePolynomialCheck:ii+windowSizePolynomialCheck][1]
                            if usableContourCopy[ii][0] < usableContourCopy[ii + 1][0]:    #ind if x is increasing
                                x_values_to_be_fitted = np.arange(usableContourCopy[ii][0]+1, usableContourCopy[ii + 1][0]-1, 1)
                            else:
                                x_values_to_be_fitted = np.arange(usableContourCopy[ii + 1][0]+1, usableContourCopy[ii][0]-1, 1)
                            localfit = np.polyfit(xrange_for_fitting, yrange_for_fitting, 2)    #horizontal gap: fit y(x)
                            y_fitted = np.poly1d(localfit)(x_values_to_be_fitted).astype(int)
                            usableContourCopy_instertion = np.insert(usableContourCopy_instertion, ii+ii_inserted+1, list(map(list, zip(x_values_to_be_fitted, y_fitted))), axis=0)
                            ii_inserted+=len(x_values_to_be_fitted) #offset index of insertion by length of previous arrays which were inserted
                            plt.plot(xrange_for_fitting, yrange_for_fitting, '.', label='x-gap data')
                            plt.plot(x_values_to_be_fitted, y_fitted, label='x-gap fit')
                            plt.legend(loc='best')
                            #plt.show()

                        #TODO then check for gaps in y-direction
                        #TODO THIS IS STILL WRONG !!!
                        elif abs(usableContourCopy[ii][1] - usableContourCopy[ii + 1][1]) > 20:      #if difference in y between 2 appending coords is large, a vertical gap in the contour is formed
                            # xrange_for_fitting = usableContour[ii-windowSizePolynomialCheck:ii+windowSizePolynomialCheck][0] #to fit polynomial with 30 points on both sides of the gap   #todo gaat fout als ii<30 of > (len()-30)
                            # yrange_for_fitting = usableContour[ii-windowSizePolynomialCheck:ii+windowSizePolynomialCheck][1]
                            xrange_for_fitting = [i[0] for i in usableContourCopy[(ii-windowSizePolynomialCheck):(ii+windowSizePolynomialCheck)]]
                            yrange_for_fitting = [i[1] for i in usableContourCopy[(ii - windowSizePolynomialCheck):(ii + windowSizePolynomialCheck)]]
                            if usableContourCopy[ii][1] < usableContourCopy[ii + 1][1]:    #find if y is increasing
                                y_values_to_be_fitted = np.arange(usableContourCopy[ii][1]+1, usableContourCopy[ii+1][1]-1, 1)
                            else:
                                y_values_to_be_fitted = np.arange(usableContourCopy[ii+1][1]+1, usableContourCopy[ii][1]-1, 1)
                            localfit = np.polyfit(yrange_for_fitting, xrange_for_fitting, 2)    #horizontal gap: fit x(y)
                            x_fitted = np.poly1d(localfit)(y_values_to_be_fitted).astype(int)
                            usableContourCopy_instertion = np.insert(usableContourCopy_instertion, ii+ii_inserted+1, list(map(list, zip(x_fitted, yrange_for_fitting))), axis=0)
                            ii_inserted+=len(y_values_to_be_fitted) #offset index of insertion by length of array which was just inserted
                            plt.plot(xrange_for_fitting, yrange_for_fitting, '.', label='y-gap data')
                            plt.plot(x_fitted, y_values_to_be_fitted, label='y-gap fit')
                            plt.legend(loc='best')
                            plt.show()
                    plt.plot([elem[0] for elem in usableContour], [elem[1] for elem in usableContour], '.', color = 'b', label='total contour')
                    plt.legend(loc='best')
                    plt.show()

                    #TODO show suggested image with interpolated contour points & allow user to verify correctness
                    if ii_inserted>0:
                        usableContourCopy_instertion = usableContourCopy_instertion[windowSizePolynomialCheck:-windowSizePolynomialCheck]
                        tempimg = []
                        copyImg = resizedimg.copy()
                        tempimg = cv2.polylines(copyImg, np.array([usableContourCopy_instertion]), False, (255, 0, 0), 8)  # draws 1 blue contour with the x0&y0arrs obtained from get_normals
                        tempimg = cv2.resize(tempimg, [round(imgshape[1] / 5), round(imgshape[0] / 5)], interpolation=cv2.INTER_AREA)  # resize image
                        cv2.imshow(f"IS THIS POLYNOMIAL FITTED GOOD???", tempimg)
                        choices = ["Yes (continue)", "No (don't use fitted polynomial)"]
                        myvar = easygui.choicebox("IS THIS POLYNOMIAL FITTED GOOD?", choices=choices)
                        if myvar == choices[1]:
                            logging.warning("Polynomial did not fit as desired. NOT using the fitted polynomial.")
                        else:
                            #usableContour = list(list(usableContourCopy)) #if good poly fits, use those
                            usableContour = [list(i) for i in usableContourCopy_instertion]
                        cv2.destroyAllWindows()
                    useableylist = np.array([elem[1] for elem in usableContour])
                    useablexlist = [elem[0] for elem in usableContour]



        else:   #if only 1 value
            logging.error(f"For now something seems to be weird. Either bad contour, or accidentally only 1 real Y at that level."
                  f"Might therefore be a min/max in image, or the contour at the other side somehow skipped 1 Y-value (if this happens, implement new code)")
            exit()

        if contouri[0] < 0:  # Save which contour & thresholdSensitivity is used to a txt file for this n, for ease of further iterations
            file = open(contourListFilePath, 'a')
            if len(iout) > 1:
                file.write(f"{n}; {','.join(map(str, iout))}; {thresholdSensitivity[0]}; {thresholdSensitivity[1]}\n")
            else:
                file.write(f"{n}; {iout[0]}; {thresholdSensitivity[0]}; {thresholdSensitivity[1]}\n")
            file.close()

        if saveCoordinates == True:
            if os.path.exists(os.path.dirname(coordinatesListFilePath)):
                with open(coordinatesListFilePath, 'wb') as internal_filename:
                    pickle.dump(usableContour, internal_filename)
            else:
                logging.error("Path to folder in which the contour coordinates file is to be saved DOES NOT exist.\n"
                              "When parsing 'saveCoordinates' = True, make sure 'coordinatesListFilePath' is parsed (correctly) as well")

    return useablexlist, useableylist, usableContour, resizedimg, greyresizedimg

def getContourCoordsFromDatafile(imgPath, coordinatesListFilePath):
    with open(coordinatesListFilePath, 'rb') as new_filename:
        usableContour = pickle.load(new_filename)
    useableylist = np.array([elem[1] for elem in usableContour])
    useablexlist = [elem[0] for elem in usableContour]

    #TODO Not not nice for now, but for the code to work (same as in getContourCoordsV4())
    img = cv2.imread(imgPath)  # read in image
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to greyscale
    greyresizedimg = grayImg
    resizedimg = img
    imgshape = img.shape  # tuple (height, width, channel)


    ###TODO trial for fixing polynomial fitting
    FITGAPS_POLYOMIAL = True
    usableContourCopy = np.array(usableContour)
    windowSizePolynomialCheck = 40  # nr of values to check left and right for fitting polynomial, if distance between 2 values is 'too large'
    usableContourCopy = np.concatenate([usableContourCopy[-windowSizePolynomialCheck:], usableContourCopy,
                                        usableContourCopy[
                                        :windowSizePolynomialCheck]])  # add values (periodic BC) for checking beginning& end of array
    usableContourCopy_instertion = usableContourCopy  # copy into which coords from polynomial fits are inserted
    ii_inserted = 0  # counter for index offset if multiple insertions have to be performed
    if FITGAPS_POLYOMIAL:
        # TODO not sure if this works properly: meant to concate the coords of a partial contour such that the coords are on a 'smooth partial ellips' without a gap
        for ii in range(windowSizePolynomialCheck, len(usableContourCopy) - (
        windowSizePolynomialCheck) + 1):  # TODO check deze +1: ik wil ook een fit @top droplet, maar werkt nog niet
            # TODO try to implement function that fits an ellips between gaps:
            # OG CODE START
            # if abs(usableContour[ii][1] - usableContour[ii+1][1]) > 200:       #if difference in y between 2 appending coords is large, a gap in the contour is formed
            #     usableContour = usableContour[ii:] + usableContour[0:ii]        #shift coordinates in list such that the coordinates are sequential neighbouring
            # OG CODE END

            # TODO check for gaps in x axis
            if abs(usableContourCopy[ii][0] - usableContourCopy[ii + 1][
                0]) > 20:  # if difference in x between 2 appending coords is large, a horizontal gap in the contour is formed
                xrange_for_fitting = [i[0] for i in usableContourCopy[
                                                    (ii - windowSizePolynomialCheck):(ii + windowSizePolynomialCheck)]]
                yrange_for_fitting = [i[1] for i in usableContourCopy[
                                                    (ii - windowSizePolynomialCheck):(ii + windowSizePolynomialCheck)]]
                # xrange_for_fitting = usableContour[(ii-windowSizePolynomialCheck):(ii+windowSizePolynomialCheck)][0] #to fit polynomial with 30 points on both sides of the gap   #todo gaat fout als ii<30 of > (len()-30)
                # yrange_for_fitting = usableContour[ii-windowSizePolynomialCheck:ii+windowSizePolynomialCheck][1]
                if usableContourCopy[ii][0] < usableContourCopy[ii + 1][0]:  # ind if x is increasing
                    x_values_to_be_fitted = np.arange(usableContourCopy[ii][0] + 1, usableContourCopy[ii + 1][0] - 1, 1)
                else:
                    x_values_to_be_fitted = np.arange(usableContourCopy[ii + 1][0] + 1, usableContourCopy[ii][0] - 1, 1)
                localfit = np.polyfit(xrange_for_fitting, yrange_for_fitting, 2)  # horizontal gap: fit y(x)
                y_fitted = np.poly1d(localfit)(x_values_to_be_fitted).astype(int)
                usableContourCopy_instertion = np.insert(usableContourCopy_instertion, ii + ii_inserted + 1,
                                                         list(map(list, zip(x_values_to_be_fitted, y_fitted))), axis=0)
                ii_inserted += len(
                    x_values_to_be_fitted)  # offset index of insertion by length of previous arrays which were inserted
                plt.plot(xrange_for_fitting, yrange_for_fitting, '.', label='x-gap data')
                plt.plot(x_values_to_be_fitted, y_fitted, label='x-gap fit')
                plt.legend(loc='best')
                # plt.show()

            # TODO then check for gaps in y-direction
            # TODO THIS IS STILL WRONG !!!
            elif abs(usableContourCopy[ii][1] - usableContourCopy[ii + 1][
                1]) > 20:  # if difference in y between 2 appending coords is large, a vertical gap in the contour is formed
                # xrange_for_fitting = usableContour[ii-windowSizePolynomialCheck:ii+windowSizePolynomialCheck][0] #to fit polynomial with 30 points on both sides of the gap   #todo gaat fout als ii<30 of > (len()-30)
                # yrange_for_fitting = usableContour[ii-windowSizePolynomialCheck:ii+windowSizePolynomialCheck][1]
                xrange_for_fitting = [i[0] for i in usableContourCopy[
                                                    (ii - windowSizePolynomialCheck):(ii + windowSizePolynomialCheck)]]
                yrange_for_fitting = [i[1] for i in usableContourCopy[
                                                    (ii - windowSizePolynomialCheck):(ii + windowSizePolynomialCheck)]]
                if usableContourCopy[ii][1] < usableContourCopy[ii + 1][1]:  # find if y is increasing
                    y_values_to_be_fitted = np.arange(usableContourCopy[ii][1] + 1, usableContourCopy[ii + 1][1] - 1, 1)
                else:
                    y_values_to_be_fitted = np.arange(usableContourCopy[ii + 1][1] + 1, usableContourCopy[ii][1] - 1, 1)
                localfit = np.polyfit(yrange_for_fitting, xrange_for_fitting, 2)  # horizontal gap: fit x(y)
                x_fitted = np.poly1d(localfit)(y_values_to_be_fitted).astype(int)
                usableContourCopy_instertion = np.insert(usableContourCopy_instertion, ii + ii_inserted + 1,
                                                         list(map(list, zip(x_fitted, yrange_for_fitting))), axis=0)
                ii_inserted += len(
                    y_values_to_be_fitted)  # offset index of insertion by length of array which was just inserted
                plt.plot(xrange_for_fitting, yrange_for_fitting, '.', label='y-gap data')
                plt.plot(x_fitted, y_values_to_be_fitted, label='y-gap fit')
                plt.legend(loc='best')
                plt.show()
        plt.plot([elem[0] for elem in usableContour], [elem[1] for elem in usableContour], '.', color='b',
                 label='total contour')
        plt.legend(loc='best')
        plt.show()

        # TODO show suggested image with interpolated contour points & allow user to verify correctness
        if ii_inserted > 0:
            usableContourCopy_instertion = usableContourCopy_instertion[
                                           windowSizePolynomialCheck:-windowSizePolynomialCheck]
            # temp_useableylist = np.array([elem[1] for elem in usableContourCopy_instertion])
            # temp_useablexlist = [elem[0] for elem in usableContourCopy_instertion]
            # sorted_temp_useablexlist, sorted_temp_useableylist = [list(a) for a in zip(*sorted(zip(temp_useablexlist, temp_useableylist)))]

            tempimg = []
            copyImg = resizedimg.copy()
            #tempimg = cv2.polylines(copyImg, np.array([usableContourCopy_instertion]), False, (255, 0, 0),
            #                        8)  # draws 1 blue contour with the x0&y0arrs obtained from get_normals
            # tempimg = cv2.resize(tempimg, [round(imgshape[1] / 5), round(imgshape[0] / 5)],
            #                      interpolation=cv2.INTER_AREA)  # resize image
            makersizeImg = 4
            for xc, yc in usableContourCopy_instertion:
                #yc = imgshape[0] - yc
                copyImg[yc-makersizeImg:yc+makersizeImg, xc-makersizeImg:xc+makersizeImg] = [255,0,0]

            tempimg = cv2.resize(copyImg, [round(imgshape[1] / 5), round(imgshape[0] / 5)],
                                 interpolation=cv2.INTER_AREA)  # resize image
            cv2.imshow(f"IS THIS POLYNOMIAL FITTED GOOD???", tempimg)
            choices = ["Yes (continue)", "No (don't use fitted polynomial)"]
            myvar = easygui.choicebox("IS THIS POLYNOMIAL FITTED GOOD?", choices=choices)
            if myvar == choices[1]:
                logging.warning("Polynomial did not fit as desired. NOT using the fitted polynomial.")
            else:
                # usableContour = list(list(usableContourCopy)) #if good poly fits, use those
                usableContour = [list(i) for i in usableContourCopy_instertion]
            cv2.destroyAllWindows()
        useableylist = np.array([elem[1] for elem in usableContour])
        useablexlist = [elem[0] for elem in usableContour]


    return useablexlist, useableylist, usableContour, resizedimg, greyresizedimg


def getfilteredContourCoordsFromDatafile(imgPath, coordinatesListFilePath):
    with open(coordinatesListFilePath, 'rb') as new_filename:
        data = pickle.load(new_filename)
    usableContour = data[0]
    vectorsFinal = data[1]
    angleDegArr = data[2]
    useableylist = np.array([elem[1] for elem in usableContour])
    useablexlist = [elem[0] for elem in usableContour]

    #TODO Not not nice for now, but for the code to work (same as in getContourCoordsV4())
    img = cv2.imread(imgPath)  # read in image
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to greyscale
    greyresizedimg = grayImg
    resizedimg = img
    imgshape = img.shape  # tuple (height, width, channel)

    return useablexlist, useableylist, usableContour, resizedimg, greyresizedimg, vectorsFinal, angleDegArr


def importConversionFactors(procStatsJsonPath):
    with open(procStatsJsonPath, 'r') as f:
        procStats = json.load(f)
    conversionZ = procStats["conversionFactorZ"]
    conversionXY = procStats["conversionFactorXY"]
    unitZ = procStats["unitZ"]
    unitXY = procStats["unitXY"]
    #lensUsed = procStats['UsedLens']
    return conversionZ, conversionXY, unitZ, unitXY

def filePathsFunction(path, wavelength_laser=520, refr_index=1.434):
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
        conversionZ, conversionXY, unitZ, unitXY = determineLensPresets(path, wavelength_laser, refr_index)
    return imgFolderPath, conversionZ, conversionXY, unitZ, unitXY

def convertunitsToEqual(unit):
    units = ['nm', 'um', 'mm', 'm', 'pixels']
    conversionsXY = [1e6, 1e3, 1, 1e-3, 1]  # standard unit is um
    conversionsZ = [1, 1e-3, 1e-6, 1e-9, 1]  # standard unit is nm

    return units.index(unit)

def determineLensPresets(imgFolderPath, wavelength_laser, refr_index):
    units = ['nm', 'um', 'mm', 'm', 'pixels']
    conversionsXY = [1e6, 1e3, 1, 1e-3, 1]  # standard unit is um
    conversionsZ = [1, 1e-3, 1e-6, 1e-9, 1]  # standard unit is nm

    choices = ["ZEISS_OLYMPUSX2", "ZEISS_ZEISSX5", "ZEISS_ZEISSX10", "SR_NIKON_NIKONX10_PIEZO"]
    answer = easygui.choicebox("What lens preset was used?", choices=choices)
    if answer == choices[0]:
        preset = 672
    elif answer == choices[1]:
        preset = 1836
    elif answer == choices[2]:
        preset = 3695
    elif answer == choices[3]:
        preset = 3662
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
    stats['UsedLens'] = answer
    stats['conversionFactorXY'] = conversionFactorXY
    stats['conversionFactorZ'] = conversionFactorZ
    stats['unitXY'] = unitXY
    stats['unitZ'] = unitZ

    # Save statistics
    with open(os.path.join(imgFolderPath, f"Measurement_Info.json"), 'w') as f:
        json.dump(stats, f, indent=4)

    return conversionFactorZ, conversionFactorXY, unitZ, unitXY

def getTimeStamps(filenames_fullpath):
    """
    :param filenames_fullpath: list with full path the filenames of which the times are to be obtained.
    :return timestamps: list with absolute timestamps
    :return deltat: list with difference in time (seconds, float) between two sequential images. First image = 0
    :return deltatFromZero: list with difference in time (seconds, float) between first image and image(n).
    """
    # read from creation date file property
    timestamps = []
    for f in filenames_fullpath:
        timestamps.append(datetime.fromtimestamp(os.path.getmtime(f)))
    deltat = [0]
    deltatFromZero = [0]
    for idx in range(1, len(timestamps)):
        deltat.append((timestamps[idx] - timestamps[idx - 1]).total_seconds())
        deltatFromZero.append(deltatFromZero[-1] + deltat[-1])
    return timestamps, deltat, deltatFromZero

def timeFormat(time):
    out = []
    for t in time:
        if t < 90:
            out.append(f"{round(t)}s")
        elif t < 3600:
            out.append(f"{round(t / 60)}min")
        else:
            out.append(f"{round(t / 3600)}hrs")
    return out

def linearFitLinearRegimeOnly(xarr, yarr, sensitivityR2, k):
    """
    Try fitting in linear regime of droplet shape for contact angle analysis
    :param xarr: array of distance in x-direction
    :param yarr: array of height
    :param sensitivityR2: desired minimum R^2 of fit to judge its validity
    :param k: nr of vector being analysed
    :return startLinRegimeIndex: index nr from which point forwards the linear regime is taken.
    :return coef1: calculated a & b values of linear fit
    :return r2: calculated R^2 of fit
    """
    #TODO: make it so that if error is too large for linear fit, a NaN is return instead of a completely bogus CA
    #TODO currently, all values are just returned and the R^2 is checked in a next function
    # minimalnNrOfDatapoints = round(len(yarr) * (2/4))
    # residualLastTime = 10000        #some arbitrary high value to have the first iteration work
    # for i in range(0, len(yarr)-minimalnNrOfDatapoints):    #iterate over the first 2/4 of the datapoints as a starting value
    #     coef1, residuals, _, _, _ = np.polyfit(xarr[i:-4], yarr[i:-4], 1, full=True)        #omit last 5 datapoints (since these are generally not great due to wrapped function)
    #     startLinRegimeIndex = i
    #     if not (residualLastTime/(len(yarr)-i+1) - residuals[0]/(len(yarr)-i)) / (residuals[0]/(len(yarr)-i)) > 0.05:    #if difference between
    #         break
    #     residualLastTime = residuals
    # if i == (len(yarr)-minimalnNrOfDatapoints-1):
    #     print(f"Apparently no the difference in squared sum differs a lot for all 2/4th first datapoints. "
    #           f"\nTherefore the data is fitted from 2/4th of the data onwards.")
    # r2 = r2_score(yarr, np.poly1d(coef1)(xarr))
    # #if r2 < sensitivityR2:
    # #    print(f"{k}: R^2 of fit is worse than {sensitivityR2}, namely: {r2:.4f}. This fit is not to be trusted")

    #
    minimalnNrOfDatapoints = round(len(yarr) * (2/4))   #
    residualLastTime = 10000        #some arbitrary high value to have the first iteration work
    for i in range(0, len(yarr)-minimalnNrOfDatapoints):    #iterate over the first 2/4 of the datapoints as a starting value
        coef1, residuals, _, _, _ = np.polyfit(xarr[i:-4], yarr[i:-4], 1, full=True)        #omit last 5 datapoints (since these are generally not great due to wrapped function)
        startLinRegimeIndex = i
        r2 = r2_score(yarr[i:-4], np.poly1d(coef1)(xarr[i:-4]))
        if r2 > sensitivityR2:    #stop iterating when desired R2 is achieved
            break
        residualLastTime = residuals
    if i == (len(yarr)-minimalnNrOfDatapoints-1):
        print(f"Apparently the difference in squared sum differs a lot for all 2/4th first datapoints. "
              f"\nTherefore the data is fitted from 2/4th of the data onwards.")

    return startLinRegimeIndex, coef1, r2

def linearFitLinearRegimeOnly_wPointsOutsideDrop(xarr, yarr, sensitivityR2, k, lenArrOutsideDrop):
    """
    Version 2 of 'linearFitLinearRegimeOnly()'
    Try fitting in linear regime of droplet shape for contact angle analysis
    :param xarr: array of distance in x-direction
    :param yarr: array of height
    :param sensitivityR2: desired minimum R^2 of fit to judge its validity
    :param k: nr of vector being analysed
    :param lenArrOutsideDrop: pixellength of vector pointing outside of drop for extra margin on CL position
    :return startLinRegimeIndex: index nr from which point forwards the linear regime is taken.
    :return coef1: calculated a & b values of linear fit
    :return r2: calculated R^2 of fit
    """
    #
    minimalnNrOfDatapointsi = round((len(yarr)-lenArrOutsideDrop) * (2 / 4))  #minim nr of datapoints over which should be fitted = half the array inside droplet

    residualLastTime = 10000  # some arbitrary high value to have the first iteration work
    rangeStarti = round(lenArrOutsideDrop/2) # start somewhere outside the CL, but not super far (So here, at half the length outside drop)
    rangeEndi = round((len(yarr)-lenArrOutsideDrop) / 4) + lenArrOutsideDrop #only try to fit up untill from the first 1/4th of datapoints inside the drop onwards #len(yarr) - minimalnNrOfDatapointsi
    rangeEndk = -round((len(yarr)-lenArrOutsideDrop) * (2 / 4)) # iterate till at most 2/4th back from the end of the array.
    GoodFit = False
    #Vary linear fit range of dataset. Always start by changing the 'first' index in array and fit till end of array.
    #However since sometimes, due to artifacts, the end of array gives poor height values, also allow to omit datapoints from there.
    stepj = round(minimalnNrOfDatapointsi/10)
    for j in range(-4, rangeEndk, -stepj): # omit last 5 datapoints (since these are generally not great due to wrapped function). Make steps of 10 lengths
        if GoodFit: #if fit is good, break out of loop
            break
        for i in range(rangeStarti, rangeEndi):  # iterate over half of the outside points and then the first 2/4 of the inside datapoints as a starting value
            coef1, residuals, _, _, _ = np.polyfit(xarr[i:j], yarr[i:j], 1, full=True)
            startLinRegimeIndex = i
            r2 = r2_score(yarr[i:j], np.poly1d(coef1)(xarr[i:j]))
            if r2 > sensitivityR2:  # stop iterating when desired R2 is achieved
                GoodFit = True
                break
            residualLastTime = residuals


    if not GoodFit:
        print(f"No good linear fit was found inside the droplet. Skipping this vector {k}")
              #f"\nTherefore the data is fitted from 2/4th of the data onwards.")

    return startLinRegimeIndex+rangeStarti, coef1, r2, GoodFit

def extractContourNumbersFromFile(lines):
    importedContourListData_n = []
    importedContourListData_i = []
    importedContourListData_thresh = []
    for line in lines[1:]:
        importedContourListData_n.append(int(line.split(';')[0]))
        idata = (line.split(';')[1])
        ilist = []
        for i in idata.split(','):
            ilist.append(int(i))
        importedContourListData_i.append(ilist)
        importedContourListData_thresh.append([int(line.split(';')[2]), int(line.split(';')[3])])
    return importedContourListData_n, importedContourListData_i, importedContourListData_thresh


def CA_And_Coords_To_Force(xcoords, ycoords, vectors, CAs, analysisFolderPath, surfaceTension):
    unitST = "mN/m"     #units of the Surface Tension filled in
    nettforces = []
    tangentforces = []
    for i in range(0, len(CAs)):
        correction = 0
        if vectors[i][0] > 0:    #correct angle by 180 deg or 1pi if dx of normal is positive (left side of droplet)
            correction = np.pi
        nettforce = math.cos(CAs[i] * np.pi / 180) * surfaceTension #unit=unitST #horizontal projection of g-l surface tension for force balance
        if vectors[i][0] == 0:          #if dx = 0
            vectors[i][0] = 0.0001       #make dx arbitrary small, but non-zero  (otherwise next division breaks)
        localVector_dydx = vectors[i][1]/vectors[i][0]
        localTangentAngle = math.atan(localVector_dydx) + correction                 #beta = tan-1 (dy/dx)
        localTangentForce = nettforce * math.cos(localTangentAngle) #unit=unitST
        nettforces.append(nettforce)
        tangentforces.append(localTangentForce)

    fig10, ax10 = plt.subplots()
    im10 = ax10.scatter(xcoords, ycoords, c=nettforces, cmap='jet', vmin=min(nettforces), vmax=max(nettforces), label=f'Line Force ({unitST})')
    ax10.set_xlabel("X-coord"); ax10.set_ylabel("Y-Coord"); ax10.set_title(f"Spatial Perpendicular Forces ({unitST}) Colormap")
    ax10.legend([f"Median Nett Force: {(statistics.median(nettforces)):.2f} {unitST}"], loc='center left')
    fig10.colorbar(im10, format="%.5f")
    #plt.show()
    fig10.savefig(os.path.join(analysisFolderPath, 'Spatial Perpendicular Force.png'), dpi=600)
    plt.close(fig10)

    fig11, ax11 = plt.subplots()
    im11 = ax11.scatter(xcoords, ycoords, c=tangentforces, cmap='jet', vmin=min(tangentforces), vmax=max(tangentforces), label=f'Horizontal Line Force ({unitST})')
    ax11.set_xlabel("X-coord"); ax11.set_ylabel("Y-Coord"); ax11.set_title(f"Spatial Horizontal Force Components ({unitST}) Colormap")
    ax11.legend([f"Median Horizontal Component Force: {(statistics.median(tangentforces)):.2f} {unitST}"], loc='center left')
    fig11.colorbar(im11)
    #plt.show()
    fig11.savefig(os.path.join(analysisFolderPath, 'Spatial Horizontal Force.png'), dpi=600)
    plt.close(fig11)

    print(f"Sum of Horizontal Components forces = {sum(tangentforces)} (compare with total (abs horizontal) = {sum(abs(np.array(tangentforces)))}")
    return tangentforces

#for 2 linear lines; y = ax + c & y= bx + d, their intersect is at (x,y) = {(d-c)/(a-b), (a*(d-c)/(a-b))+c}
def approxMiddlePointDroplet(coords, vectors):
    intersectCoordsX = []
    intersectCoordsY = []
    for i in itertools.chain(range(0, round(len(vectors)/4)), range(round(len(vectors)/2), round(len(vectors)*3/4))):
        if vectors[i][0] == 0 or vectors[i+round(len(vectors)/4)][0] == 0:  #would give a division by 0, so just skip
            pass
        else:
            a = vectors[i][1] / vectors[i][0]           #a=dy/dx
            b = vectors[i+round(len(vectors)/4)][1] / vectors[i+round(len(vectors)/4)][0]  # c=dy/dx    (of vector 1 quarter away from current one)
            c = coords[i][1] - (coords[i][0] * a)
            d = coords[i+round(len(vectors)/4)][1] - (coords[i+round(len(vectors)/4)][0] * b)
            if (a - b) == 0:
                pass
            else:
                intersectCoordsX.append(round((d - c) / (a - b)))
                intersectCoordsY.append(round((a * (d - c) / (a - b)) + c))
    meanmiddleX = np.mean(intersectCoordsX)
    meanmiddleY = np.mean(intersectCoordsY)
    medianmiddleX = np.median(intersectCoordsX)
    medianmiddleY = np.median(intersectCoordsY)
    return intersectCoordsX, intersectCoordsY, meanmiddleX, meanmiddleY, medianmiddleX, medianmiddleY

#TODO get this to work? : fitting CA = (x,y) to interpolate for missing datapoints & get total contour length for good force calculation
#part of OG code: https://www.geeksforgeeks.org/3d-curve-fitting-with-python/
def givemeZ(xin, yin, zin, xout, yout, conversionXY, analysisFolder, n, imgshape):
    #tck = interpolate.bisplrep(xin, yin, zin, s=0)
    #f = scipy.interpolate.interp2d(xin, yin, zin, kind="cubic")
    # Define mathematical function for curve fitting
    yin = abs(np.subtract(imgshape[0], yin))       #flip y's for good plotting of data
    yout = abs(np.subtract(imgshape[0], yout))
    def func(xy, a, b, c, d, e, f):
        x, y = xy
        return a + b * x + c * y + d * x ** 2 + e * y ** 2 + f * x * y

    popt, pcov = curve_fit(func, (xin, yin), zin)

    #ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #Below: attempt at interpolating between (x,y) cooridnates to make a 'full oval-oid'. Doesn't work because x must be strictly increasing.
    #cs = scipy.interpolate.CubicSpline(xin, yin, bc_type='periodic')

    localContourLength = [] #in pixels
    totalContourLength = 0  #in pixels
    for i in range(0,len(xout)-1):
        localContourLength.append(np.sqrt((xout[i+1]-xout[i])**2 + (yout[i+1]-yout[i])**2))
        totalContourLength += localContourLength[-1]
    localContourLength.append(np.sqrt((xout[0] - xout[i+1]) ** 2 + (yout[0] - yout[i+1]) ** 2))
    totalContourLength += localContourLength[-1]
    print(f"Total contour length={totalContourLength} (in coordinates (i.e. pixel) units) (estimate from contour x&y coords, by pythagoras)")
    print(f"Total contour length={totalContourLength*conversionXY} mm")
    print(f"length of xmax-xmin & ymax-ymin (mm)= {(max(xout) - min(xout)) * conversionXY} & {(max(yout) - min(yout)) * conversionXY}")

    #TODO split up droplet in upper & lower half, iterate over increasing x to interpolate for missing values
    #for now: assume entire droplet is investigated. Code prob. doesn't work for partial droplets.
    upperhalf=[]
    lowerhalf=[]
    n_xmin = (xin); n_xmax = max(xin)
    yatminx = yin[np.argmin(xin)]; yatmaxx = yin[np.argmax(xin)]
    #linearly interpolate y value between xmin & xmax for comparison above or below half droplet
    # for xval in xin:

    # x_arranged = np.arrange(min(xin), max(xin))
    # fig1, ax1 = plt.subplots()
    # ax1.plot(xin, yin, 'b')
    # ax1.plot(x_arranged, cs(x_arranged), 'r')
    # ax1.plot()
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xin, yin, zin, color='blue', label='input data')
    Z = func((xout, yout), *popt)
    ax.scatter(xout, yout, Z, color='red', label='fit data')
    ax.set_xlabel('X coords')
    ax.set_ylabel('Y coords')
    ax.set_zlabel('Horizontal component Force (mN)')
    ax.legend(loc='best')
    #plt.show()
    fig.savefig(os.path.join(analysisFolder, f'Spatial CA with fit {n:04}.png'), dpi=600)
    plt.close()

    weighedZ = np.multiply(Z, localContourLength)   #multiply Z value with the length of the contour (pixels) at that coordinate to weigh the relevance.
    #Also ^ basically converts Z (mN/m) to (mN * pixels/m), which can be converted to mN (below)
    totalZ = sum(weighedZ) * (conversionXY/1000)   #in mN

    print(f"meanTotalZ = {totalZ} mN ")
    return Z, totalZ

#TODO probably the path is not working as intended
#TODO Seems to be working just fine?
def swellingRatioNearCL(xdata, ydata, elapsedtime, path, imgNumber, vectorNumber, outwardsLengthVector):
    """
    :param xdata: np.aray of x-position data (unit=pixels)
    :param ydata: np.array of y-Intensity data
    :param elapsedtime: elapsed time with respect to t=0 (unit=seconds, format = int or float)
    :param imgNumber: image number of analysed tiff file in folder. Required for saving & importing peak-picking data
    :param vectorNumber: vector number k analysed. Required for saving & importing peak-picking data
    :return height: np.array with absolute heights.
    :return h_ratio: np.array with swelling ratios.
    """
    EVALUATERIGHTTOLEFT = True         #evaluate from left to right, or the other way around    (required for correct conversion of intensity to height profile)
    if EVALUATERIGHTTOLEFT:
        list(xdata).reverse()
        list(ydata).reverse()

    FLIP = False                 #True=flip data after h analysis to have the height increase at the left
    MANUALPEAKSELECTION = True
    PLOTSWELLINGRATIO = False
    SAVEFIG = True
    SEPERATEPLOTTING = True
    USESAVEDPEAKS = True
    figswel0, axswel0 = plt.subplots()
    figswel1, axswel1, = plt.subplots()
    knownHeightArr = [165]  #TODO: for now, set the known height of all profiles to 165. Hopefully outwardsLengthVector is long enough that it is hardly swelling at the end
    colorscheme = 'plasma'; cmap = plt.get_cmap(colorscheme); colorGradient = np.linspace(0, 1, len(knownHeightArr))
    dryBrushThickness = 160 #TODO: for now, set the known dry height of all profiles to 160. Hopefully outwardsLengthVector is long enough that it is hardly swelling at the end
    idx = imgNumber             #idx is the image number being investigated
    idxx = 0    #in the OG code, idxx is used for the knownheight array and only incremented when an desired idx is investigated. Such that for e.g. 4 idx's to be investigated, knownHeigthArr[idxx] is used
    intensityProfileZoomConverted = ydata
    knownPixelPosition = 30     #TODO random pixel near end of outwardsLengthVector for now, just to test if entire function works
    normalizeFactor = 1
    range1 = 0
    range2 = len(ydata)
    source = path
    xshifted = xdata

    height, h_ratio = heightFromIntensityProfileV2(FLIP, MANUALPEAKSELECTION, PLOTSWELLINGRATIO, SAVEFIG, SEPERATEPLOTTING, USESAVEDPEAKS,
                                 axswel0, axswel1, cmap, colorGradient, dryBrushThickness, elapsedtime, figswel0, figswel1, idx, idxx,
                                 intensityProfileZoomConverted, knownHeightArr, knownPixelPosition, normalizeFactor,
                                 range1, range2, source, xshifted, vectorNumber, outwardsLengthVector, unitXY="pixels")
    return height, h_ratio

#TODO ik denk dat constant x, var y nog niet goed kan werken: Output geen lineLength pixel & lengthVector moet langer zijn dan aantal punten van np.arrange (vanwege eerdere normalisatie)?
def profileFromVectorCoords(x0arrcoord, y0arrcoord, dxarrcoord, dyarrcoord, lengthVector, greyresizedimg):
    """
    :param x0arrcoord:
    :param y0arrcoord:
    :param dxarrcoord:
    :param dyarrcoord:
    :param lengthVector:    desired vector length over which the intensity profile will be taken (unit = pixels)
    :param greyresizedimg:
    :return profile: intensity profile over pixels in image between 2 coordinates
    :return lineLengthPixels: amount of pixels the vector spans. Can be used to calculate the length of the vector in e.g. mm
            ^NOTE/TODO: this one should be almost the same for all vectors right?
    :return fitInside: boolean to show whether to line fits inside the given image. If not, return a linelength of 0 and empty profile
    """
    fitInside = True
    profile = []
    if dxarrcoord - x0arrcoord == 0:  # 'flat' vector in x-dir: constant x, variable y
        xarr = np.ones(lengthVector) * x0arrcoord
        if y0arrcoord > dyarrcoord:
            yarr = np.arange(dyarrcoord, y0arrcoord)
        else:
            yarr = np.arange(y0arrcoord, dyarrcoord)
        coords = list(zip(xarr.astype(int), yarr.astype(int))) #list of [(x,y), (x,y), ....]
        lineLengthPixels = lengthVector
    else:   #for any other vector orientation
        a = (dyarrcoord - y0arrcoord) / (dxarrcoord - x0arrcoord)
        b = y0arrcoord - a * x0arrcoord
        coords, lineLengthPixels = coordinates_on_line(a, b, [x0arrcoord, dxarrcoord, y0arrcoord, dyarrcoord])


    #Check if line exceeds image boundaries: if so, set bool to false. Else, obtain intensity profile from coordinates
    sy, sx = greyresizedimg.shape
    if coords[0][0] < 0 or coords[0][1] < 0 or coords[-1][0] >= sx or coords[-1][1] >= sy:          #x1<0, y1<0, xn>=sx, yn>=sy
        fitInside = False
        logging.warning(f"Trying to extract intensity data from outside the given image. Skipping this vector.")
        lineLengthPixels = 0
    else:
        profile = [np.transpose(greyresizedimg)[pnt] for pnt in coords]

    return profile, lineLengthPixels, fitInside

def intensityToHeightProfile(profile, lineLengthPixels, conversionXY, conversionZ, FLIPDATA):
    """
    Convert an intensity profile to a relative height profile by using monochromatic interferometry.
    Best applied when many interference fringes are visible. Not suitable for e.g. less than 5 fringes.
    Intensity profile with fringes is converted to a 'wrapped' or 'sawtooth' profile after filtering in fourier space.
    This allows us to distinguish full phases present in the intensity profile, which can then be converted
    to a smooth height profile by 'unwrapping' (stacking the phases).

    Lowpass & highpass values for filtering in frequency domain are important w/ respect to the obtain results.
    So far, I found generally 'lowpass = 1/2 profile length' & 'highpass = 2' work fine for CA fringes.

    :param profile: intensity profile
    :param lineLengthPixels: length of the
    :param conversionXY: conversion factor for pixels -> um in xy-plane
    :param conversionZ: conversion factor for pixels -> nm in z-plane
    :param FLIPDATA: boolean to reverse data in xy-plane (only for plotting purposes)
    :return unwrapped: calculated height profile (units = what was specified by conversionZ)
    :return x: calculated x-values on corresponding with the unwrapped height profile (distance. units = what was specified by conversionXY)
    :return wrapped: wrapped profile
    :return peaks: indices of calculated maxima
    """
    # transform to fourier space
    profile_fft = np.fft.fft(profile)
    mask = np.ones_like(profile).astype(float)
    lowpass = round(len(profile) / 2);
    highpass = 2  # NOTE: lowpass seems most important for a good sawtooth profile. Filtering half of the data off seems fine
    mask[0:lowpass] = 0;
    mask[-highpass:] = 0
    profile_fft = profile_fft * mask
    profile_filtered = np.fft.ifft(profile_fft)

    # calculate the wrapped space
    wrapped = np.arctan2(profile_filtered.imag, profile_filtered.real)
    peaks, _ = scipy.signal.find_peaks(wrapped, height=0.4)  # obtain indeces om maxima

    unwrapped = np.unwrap(wrapped)
    if FLIPDATA:
        unwrapped = -unwrapped + max(unwrapped)
    # TODO conversionZ generally in nm, so /1000 -> in um
    unwrapped *= conversionZ / 1000  # if unwapped is in um: TODO fix so this can be used for different kinds of Z-unit
    # x = np.arange(0, len(unwrapped)) * conversionXY * 1000 #TODO same ^
    # TODO conversionXY generally in mm, so *1000 -> unit in um.
    x = np.linspace(0, lineLengthPixels, len(unwrapped)) * conversionXY * 1000  # converts pixels to desired unit (prob. um)

    # fig1, ax1 = plt.subplots(2, 2)
    # ax1[0, 0].plot(profile);
    # ax1[1, 0].plot(wrapped);
    # ax1[1, 0].plot(peaks, wrapped[peaks], '.')
    # ax1[0, 0].set_title(f"Intensity profile with TEMP = {temp}, meaning LOWPASS = {lowpass}");
    # ax1[1, 0].set_title("Wrapped profile")
    # # TODO unit unwrapped was in um, *1000 -> back in nm. unit x in um
    # ax1[0, 1].plot(x, unwrapped * 1000);
    # ax1[0, 1].set_title("Drop height vs distance (unwrapped profile)")
    # ax1[0, 1].legend(loc='best')
    # ax1[0, 0].set_xlabel("Distance (nr.of datapoints)");
    # ax1[0, 0].set_ylabel("Intensity (a.u.)")
    # ax1[1, 0].set_xlabel("Distance (nr.of datapoints)");
    # ax1[1, 0].set_ylabel("Amplitude (a.u.)")
    # ax1[0, 1].set_xlabel("Distance (um)");
    # ax1[0, 1].set_ylabel("Height profile (nm)")
    # fig1.set_size_inches(12.8, 9.6)
    # plt.show()

    return unwrapped, x, wrapped, peaks


def non_uniform_savgol(x, y, window, polynom, mode = 'interp'):
    """
    Applies a Savitzky-Golay filter to y with non-uniform spacing
    as defined in x
    Returns a smoothened y-array of same size as input

    This is based on https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
    The borders are interpolated like scipy.signal.savgol_filter would do

    Parameters
    ----------
    x : array_like
        List of floats representing the x values of the data
    y : array_like
        List of floats representing the y values. Must have same length
        as x
    window : int (odd)
        Window length of datapoints. Must be odd and smaller than x
    polynom : int
        The order of polynom used. Must be smaller than the window size

    Returns
    -------
    np.array of float
        The smoothed y values
    """
    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')

    if len(x) < window:
        raise ValueError('The data size must be larger than the window size')

    if type(window) is not int:
        raise TypeError('"window" must be an integer')

    if window % 2 == 0:
        raise ValueError('The "window" must be an odd integer')

    if type(polynom) is not int:
        raise TypeError('"polynom" must be an integer')

    if polynom >= window:
        raise ValueError('"polynom" must be less than "window"')

    half_window = window // 2
    polynom += 1

    startLenghtX = len(x)
    if mode == 'periodic':
        x = np.concatenate((x[-half_window:], x, x[0:half_window]))
        y = np.concatenate((y[-half_window:], y, y[0:half_window]))

    # Initialize variables
    A = np.empty((window, polynom))     # Matrix
    tA = np.empty((polynom, window))    # Transposed matrix
    t = np.empty(window)                # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Start smoothing
    for i in range(half_window, len(x) - half_window, 1):
        # Center a window of x values on x[i]
        for j in range(0, window, 1):
            t[j] = x[i + j - half_window] - x[i]

        # Create the initial matrix A and its transposed form tA
        for j in range(0, window, 1):
            r = 1.0
            for k in range(0, polynom, 1):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]

        # Multiply the two matrices
        tAA = np.matmul(tA, A)

        # Invert the product of the matrices
        tAA = np.linalg.inv(tAA)

        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA, tA)

        # Calculate c0 which is also the y value for y[i]
        y_smoothed[i] = 0
        for j in range(0, window, 1):
            y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]

        # If at the end or beginning, store all coefficients for the polynom
        if i == half_window:
            first_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    first_coeffs[k] += coeffs[k, j] * y[j]
        elif i == len(x) - half_window - 1:
            last_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]
    if mode == 'interp':
        # Interpolate the result at the left border
        for i in range(0, half_window, 1):
            y_smoothed[i] = 0
            x_i = 1
            for j in range(0, polynom, 1):
                y_smoothed[i] += first_coeffs[j] * x_i
                x_i *= x[i] - x[half_window]

        # Interpolate the result at the right border
        for i in range(len(x) - half_window, len(x), 1):
            y_smoothed[i] = 0
            x_i = 1
            for j in range(0, polynom, 1):
                y_smoothed[i] += last_coeffs[j] * x_i
                x_i *= x[i] - x[-half_window - 1]
    elif mode == 'periodic':
        #by extending the x & y manually in beginning, the periodic boundary condition is already fulfilled
        y_smoothed = y_smoothed[half_window:-half_window]
        if startLenghtX == len(y_smoothed):
            logging.warning("nice, everything went well")
        else:
            logging.error("help, something is wrong")
    return y_smoothed


def convertPhiToazimuthal(phi):
    """
    :param phi: np.array of atan2 radians   [-pi, pi]
    :return np.sin(phi2): np.array of azimuthal angle [-1, 1] rotating clockwise, starting at with 0 at top
    :return phi2: np.array of normal radians [0, 2pi] rotating clockwise starting at top.
    """
    phi1 = 0.5 * np.pi - phi
    phi2 = np.mod(phi1, (2.0 * np.pi))  # converting atan2 to normal radians: https://stackoverflow.com/questions/17574424/how-to-use-atan2-in-combination-with-other-radian-angle-system
    return np.sin(phi2), phi2


def determineTopAndBottomOfDropletCoords(vectors, x0arr, y0arr):  #x0arr, y0arr, dxarr, dyarr
    """"
    :param x0arr: input array with x coordinate values
    :param y0arr: input array with y coordinate values
    :param dxarr: input array with x coordinate values, at end of vector
    :param dyarr: input array with y coordinate values, at end of vector
    :return coords: [x, y] values of coordinates at inflection point at bottom and top of droplet
    """
    #previously I defined the dx as dxarr-x0arr, so i'm flipping the sign here. Same for dy
    dx = np.array([i[0] for i in vectors])
    dy = np.array([i[1] for i in vectors])
    #calculate all the vectors if they are pointing left or right
    x0arr = np.array(x0arr)
    #dxarr = np.array(dxarr)
    y0arr = np.array(y0arr)
    # dyarr = np.array(dyarr)

    #dx = x0arr - dxarr      #if dx<0, they are pointing right, so on left side of droplet
    negative_dx_indices = np.where(dx < 0)[0]

    #For those values, check their dy value
    #dy = y0arr - dyarr

    # if dy<0, the vector is pointing downwards = top of droplet
    negative_dy_indices = np.where(dy < 0)[0]
    # if dy>0, the vector is pointing upwards = bottom of droplet
    positive_dy_indices = np.where(dy > 0)[0]

    upwards_vecIndices = set(negative_dx_indices).intersection(negative_dy_indices)
    downwards_vecIndices = set(negative_dx_indices).intersection(positive_dy_indices)

    #upwards_Index = np.argmax(y0arr[list(upwards_vecIndices)])
    #downwards_Index = np.argmin(y0arr[list(downwards_vecIndices)])

    upwards_Index = np.where(y0arr == max(y0arr[list(upwards_vecIndices)]))[0][0]
    downwards_Index = np.where(y0arr == min(y0arr[list(downwards_vecIndices)]))[0][0]

    #return x and y at the bottom & top respectively as seperate lists
    return [x0arr[upwards_Index], y0arr[upwards_Index]], [x0arr[downwards_Index], y0arr[downwards_Index]]


def determineTopAndBottomOfDropletCoords_SIMPLEMINMAX(vectors, x0arr, y0arr):
    """"
    :param x0arr: input array with x coordinate values
    :param y0arr: input array with y coordinate values
    :param dxarr: input array with x coordinate values, at end of vector
    :param dyarr: input array with y coordinate values, at end of vector
    :return coords: [x, y] values of coordinates at minimum & maximum y-value
    """
    miny_index = np.argmin(y0arr)
    maxy_index = np.argmax(y0arr)
    return [x0arr[maxy_index], y0arr[maxy_index]], [x0arr[miny_index], y0arr[miny_index]]

def coordsToPhi(xArrFinal, yArrFinal, medianmiddleX, medianmiddleY):
    """
    :return phi: range [-pi : pi]
    :return rArray: distance from the middle to the coordinate. UNIT= same as input units (so probably pixel, or e.g. mm)

    phi = 0 at right side -> 0.5pi at top -> 1pi at left -> -1pi at left -> -0.5pi at bottom
    example how phi evolves: https://stackoverflow.com/questions/17574424/how-to-use-atan2-in-combination-with-other-radian-angle-systems
    """
    dx = np.subtract(xArrFinal, medianmiddleX)
    dy = np.subtract(yArrFinal, medianmiddleY)
    phi = np.arctan2(dy, dx)  # angle of contour coordinate w/ respect to 12o'clock (radians)
    rArray = np.sqrt(np.square(dx) + np.square(dy))
    return phi, rArray


def calculateForceOnDroplet(phi_Force_Function, phi_r_Function, boundaryPhi1, boundaryPhi2, analysisFolder, phi_sorted, rFromMiddle_savgol_sorted, phi_tangentF_savgol_sorted):
    """"
    input: cubicSplineFunction
    :param boundaryPhi1: positive phi value
    :param boundaryPhi2: negative phi value
    such that boundaryPhi2:boundaryPhi1 is the right side of droplet
    """
    print(f"BoundaryPhi 1 = {boundaryPhi1}, BoundaryPhi 2 = {boundaryPhi2}")
    #step = 0.05
    for step in [0.001]:   #, 0.005, 0.01, 0.05, 0.1, 0.5
        phi_range = np.arange(-np.pi, np.pi, step)
        phi_range1 = np.arange(boundaryPhi2, boundaryPhi1, step)
        phi_range2 = np.concatenate([np.arange(-np.pi, boundaryPhi2, step), np.arange(boundaryPhi1, np.pi, step)])

        #required ideally: a function that defines the (nett) force / CA as a function of cartesian coordinates
        #TODO proper integration doesn't work currently, probably because the cubicSpline fits are REALLY off between too large intervals of data. (When plotted with a too small interval a lot of noise is introduced, even though the original data is really smooth)
        Ftot_func = lambda phi: phi_r_Function(phi) * phi_Force_Function(phi)
        total_force_quad, error_quad = integrate.quad(Ftot_func, -np.pi, np.pi, limit=1200)    #integrate over entire phi range

        #Simposon integration
        # force_simpson = integrate.simpson(Ftot_func(phi_range), phi_range)
        # print(f"Force simpson integration = {force_simpson*1000:.7f} microN - step={step}")

        # seperate integrations for the left & right part of droplet.
        # total_force1, error = integrate.quad(Ftot_func, boundaryPhi1, boundaryPhi2, limit=900)    #integrate between phiTop & phiBottom
        # total_force2_1, error = integrate.quad(Ftot_func, 0, boundaryPhi1, limit=900)
        # total_force2_2, error = integrate.quad(Ftot_func, boundaryPhi2, np.pi, limit=900)
        # total_force2 = total_force2_1 + total_force2_2
        print(f"Quad integration:\n"
              f"-Total, whole range = {total_force_quad * 1000} microN\n\n")
        #       f"-From Top(1) to Bot(2) = {total_force1 * 1000} microN\n"
        #       f"-From 0 to top(1) = {total_force2_1 * 1000} microN, From bot(2) to pi = {total_force2_2 * 1000} microN\n"
        #       f"-Resulting sum of line above = {total_force2 * 1000} microN\n"
        #       f"-Sum left&right parts = {(total_force1 + total_force2)*1000} microN\n\n")

        #TODO attempting to manually integrate force vs phi
        forceArr = phi_r_Function(phi_range) * phi_Force_Function(phi_range)
        forceArr1 = phi_r_Function(phi_range1) * phi_Force_Function(phi_range1)
        forceArr2 = phi_r_Function(phi_range2) * phi_Force_Function(phi_range2)
        trapz_intForce_function = np.trapz(forceArr, phi_range)
        print(f"Trapz integrated force (from function) = {trapz_intForce_function*1000} microN - step={step}\n")
        trapz_intForce_data = np.trapz(np.array(rFromMiddle_savgol_sorted) * np.array(phi_tangentF_savgol_sorted), phi_sorted)
        print(f"Trapz integrated force (from pure data) = {trapz_intForce_data*1000} microN")
        fig6, ax6 = plt.subplots()
        # ax6.plot(phi_range, forceArr, label='phi vs r*tangent Force')
        # ax6.plot(boundaryPhi1, phi_r_Function(boundaryPhi1) * phi_Force_Function(boundaryPhi1), '.', label='Top')
        # ax6.plot(boundaryPhi2, phi_r_Function(boundaryPhi2) * phi_Force_Function(boundaryPhi2), '.', label='Bottom')
        ax6.plot(phi_range1, forceArr1, '.', label='right side droplet')
        ax6.plot(phi_range2, forceArr2, '.', label='left side droplet')
        ax6.plot(boundaryPhi1, phi_r_Function(boundaryPhi1) * phi_Force_Function(boundaryPhi1), '.', label='Top')
        ax6.plot(boundaryPhi2, phi_r_Function(boundaryPhi2) * phi_Force_Function(boundaryPhi2), '.', label='Bottom')
        ax6.plot(phi_sorted, np.array(rFromMiddle_savgol_sorted) * np.array(phi_tangentF_savgol_sorted), label='data')
        ax6.set(title=f"[Horizontal force * r] as a function of phi", xlabel=f'$\phi$', ylabel='Force (mN/m (!?))')
        ax6.legend(loc='best')
        fig6.tight_layout()
        fig6.savefig(os.path.join(analysisFolder, f'Force vs Phi.png _ step = {step}.png'), dpi=600)
        #plt.show()

    #print(f"Total whatever calculated is ={total_force}")

    # localContourLength = []  # in pixels
    # totalContourLength = 0  # in pixels
    # for i in range(0, len(xCartesian)):
    #     localContourLength.append(np.sqrt((xCartesian[i + 1] - xCartesian[i]) ** 2 + (yCartesian[i + 1] - yCartesian[i]) ** 2))
    #     totalContourLength += localContourLength[-1]
    # localContourLength.append(np.sqrt((xCartesian[0] - xCartesian[i + 1]) ** 2 + (yCartesian[0] - yCartesian[i + 1]) ** 2))
    # totalContourLength += localContourLength[-1]
    # print(
    #     f"Total contour length={totalContourLength} (in coordinates (i.e. pixel) units) (estimate from contour x&y coords, by pythagoras)")
    # print(f"Total contour length={totalContourLength * conversionXY} mm")
    # print(
    #     f"length of xmax-xmin & ymax-ymin (mm)= {(max(xCartesian) - min(xCartesian)) * conversionXY} & {(max(yCartesian) - min(yCartesian)) * conversionXY}")
    #
    return total_force_quad, error_quad, trapz_intForce_function, trapz_intForce_data

#TODO trying to get this to work: dirk sin cos fitting scheme
def manualFitting_1(inputX, inputY):
    """
    Goal: fit radial data by sin&cos functions. Tune N for more or less influence of noise
    :param inputX:
    :param inputY:
    :return:
    """
    print(f"In manualFitting(): min & max inputX = {min(inputX)}, {max(inputX)}. If this is not -pi to pi, something's up...\n")
    integratedY_trapz = scipy.integrate.trapz(inputY, inputX)
    print(f"calculated trapz Y of phi: {integratedY_trapz}")

    function_s = lambda Y_phi, k, phi: Y_phi*np.sin(k*phi)
    function_c = lambda Y_phi, k, phi: Y_phi*np.cos(k*phi)

    sigma_k_s = [0]     #sigma_k_s=0  at n=0
    sigma_k_c = [(2 / np.pi) * scipy.integrate.quad(function_c, min(inputX), max(inputX), args=(integratedY_trapz, 0,))[0]]
    R_phi_func = lambda phi, n, sigma_n_s, sigma_n_c: sum(sigma_n_s * np.sin(n*phi)) + sum(sigma_n_s * np.cos(n*phi))

    N = [1,2,3,4,5]
    for k in N:
        sigma_k_s.append((1/np.pi) * scipy.integrate.quad(function_s, min(inputX), max(inputX), args=(integratedY_trapz, k,))[0])
        sigma_k_c.append((1/np.pi) * scipy.integrate.quad(function_c, min(inputX), max(inputX), args=(integratedY_trapz, k,))[0])
    N = np.array([0] + N)
    X_range = np.linspace(min(inputX), max(inputX))

    Y_range = [R_phi_func(Xval, N, sigma_k_s, sigma_k_c) for Xval in X_range]

    fig1, ax1 = plt.subplots()
    ax1.plot(inputX, inputY, '.', label='raw data')
    ax1.plot(X_range, Y_range, '-', label=f'function order N={N[-1]}')
    ax1.set(xlabel='Phi (rad)', ylabel='whatever y (?)', title="radial fitting of x vs y with sin and cos")
    ax1.legend(loc='best')
    plt.show()

    return N, sigma_k_s, sigma_k_c

#TODO trying to get this to work: dirk sin cos fitting scheme
#for now, seemingly the working one
def manualFitting(inputX, inputY, path, Ylabel, N, SHOWPLOTS_SHORT):
    """
    Goal: fit radial data by sin&cos functions. Tune N for more or less influence of noise
    :param inputX: array with radial angles
    :param inputY: array with data corresponding to the inputX. Must be periodic for this fitting to make sense
    :return:
    """
    #######
    I_k__c_j = lambda f_j1, f_j, phi_j1, phi_j, k:  f_j1 * (np.sin(k*phi_j1) / k +
                                                            (np.cos(k*phi_j1) - np.cos(k*phi_j)) / (k**2 * (phi_j1 - phi_j))) - \
                                                    f_j * (np.sin(k*phi_j)/k +
                                                            (np.cos(k*phi_j1) - np.cos(k*phi_j)) / (k**2 * (phi_j1 - phi_j)))
    f_k__c = lambda I_k__c, k, phi, f_phi : (1/np.pi) * sum([I_k__c_j(f_phi[j+1], f_phi[j], phi[j+1], phi[j], k) for j in range(0, len(f_phi)-1)])

    I_k__s_j = lambda f_j1, f_j, phi_j1, phi_j, k: f_j1 * (-np.cos(k * phi_j1) / k +
                                                           (np.sin(k*phi_j1) - np.sin(k * phi_j)) /
                                                           (k**2 * (phi_j1 - phi_j))) + \
                                                   f_j * (np.cos(k * phi_j) / k -
                                                          (np.sin(k*phi_j1) - np.sin(k * phi_j)) /
                                                          (k**2 * (phi_j1 - phi_j)))

    f_k__s = lambda I_k__s, k, phi, f_phi: (1 / np.pi) * sum([I_k__s_j(f_phi[j + 1], f_phi[j], phi[j + 1], phi[j], k) for j in range(0, len(f_phi)-1)])
    ##########
    #x = 1 value for phi to calculate the corresponding y for. k = 1 number, the max order to calculate the fit with.
    #f_c & f_k = sigma's for sin & cos: array of numbers with (at least) as many numbers as the desired order k
    f_phi = lambda x, k, f_c, f_s: sum([f_c[i] * np.cos(i*x) + f_s[i] * np.sin(i*x) for i in range(0, k+1)])
    ##########

    sigma_k_s = [0]     #sigma_k_s=0  at n=0
    sigma_k_c = [(1 / (2*np.pi)) * scipy.integrate.trapz(inputY, inputX)]

    #N = [7]
    for k in range(1, N[-1]+1): #for all orders in range 1 to N, determine the sigma's sin & cos.
        sigma_k_s.append(f_k__s(I_k__s_j, k, inputX, inputY))
        sigma_k_c.append(f_k__c(I_k__c_j, k, inputX, inputY))
    N = np.array([0] + N)
    X_range = np.linspace(min(inputX), max(inputX), 500)

    fig1, ax1 = plt.subplots()
    if len(N)>2:
        colorscheme = 'plasma'
        cmap = plt.get_cmap(colorscheme)
        colorGradient = np.linspace(0, 1, len(N))
    else:
        colorscheme = 'hsv'
        cmap = plt.get_cmap(colorscheme)
        colorGradient = [0.66, 0]

    func_range = lambda x_range: [f_phi(x, N[-1], sigma_k_c, sigma_k_s) for x in x_range]
    func_single = lambda x: f_phi(x, N[-1], sigma_k_c, sigma_k_s)

    for i, n in enumerate(N[1:]):
        Y_range = [f_phi(Xval, n, sigma_k_c, sigma_k_s) for Xval in X_range]
        ax1.plot(X_range, Y_range, '-', label=f'N={n}', linewidth=3,  color=cmap(colorGradient[i+1]))
    ax1.plot(inputX, inputY, '.', label='raw data',  color=cmap(colorGradient[0]), markersize=2)
    #TODO clean this up (messing with plot titles etc) for figure making
    #ax1.set(xlabel='Angle Phi (rad)', ylabel=f'{Ylabel[0]} {Ylabel[1]}', title=f"{Ylabel[0]} vs radial angle with fourier fitting")
    ax1.set(xlabel='Angle Phi (rad)', ylabel=f'{Ylabel[0]} {Ylabel[1]}',
            title=f"{Ylabel[0]} vs radial angle with fourier fitting\n"
                  f"Influence of function order parameter")
    ax1.legend(loc='best')
    fig1.savefig(os.path.join(path, f"{Ylabel[0]} Fourier fitted.png"), dpi=300)
    showPlot(SHOWPLOTS_SHORT, [fig1])

    return func_range, func_single, N, sigma_k_s, sigma_k_c,


def determineMiddleCoord(xArrFinal, yArrFinal):
    """
    Determine middle coordinate from surface area coordinate counting
    :param xArrFinal:
    :param yArrFinal:
    :return:    middle coordinate(?)
    """
    yArrFinal = np.array(yArrFinal)
    #iterate over all values between min and max y
    minY = min(yArrFinal)
    maxY = max(yArrFinal)
    counter = np.array([0], dtype='float64')
    xtot = np.array([0], dtype='float64')
    ytot = np.array([0], dtype='float64')
    for i in range(minY, maxY):
        indices = np.where(yArrFinal == i)[0]
        if len(indices) > 0:
            xatY = [xArrFinal[index] for index in indices]
            x1 = min(xatY)
            x2 = max(xatY)
            xtot += sum(np.arange(x1, x2+1))
            ytot += i * ((x2+1) - x1)
            counter += ((x2+1) - x1)
    middleX = xtot // counter
    middleY = ytot // counter
    return [int(middleX), int(middleY)]


def coordsToIntensity_CAv2(FLIPDATA, analysisFolder, angleDegArr, ax_heightsCombined, conversionXY, conversionZ,
                         deltatFromZeroSeconds, dxarr, dxnegarr, dyarr, dynegarr, greyresizedimg, heightPlottedCounter,
                         lengthVector, n, omittedVectorCounter, outwardsLengthVector, path, plotHeightCondition,
                         resizedimg, sensitivityR2, vectors, vectorsFinal, x0arr, xArrFinal, y0arr, yArrFinal, IMPORTEDCOORDS, SHOWPLOTS_SHORT, dxExtraOutarr, dyExtraOutarr, smallExtraOutwardsVector):
    """

    :param FLIPDATA:
    :param analysisFolder:
    :param angleDegArr:
    :param ax_heightsCombined:
    :param conversionXY:
    :param conversionZ:
    :param deltatFromZeroSeconds:
    :param dxarr:
    :param dxnegarr:
    :param dyarr:
    :param dynegarr:
    :param greyresizedimg:
    :param heightPlottedCounter:
    :param lengthVector:
    :param n:
    :param omittedVectorCounter:
    :param outwardsLengthVector:
    :param path:
    :param plotHeightCondition:
    :param resizedimg:
    :param sensitivityR2:
    :param vectors:
    :param vectorsFinal:
    :param x0arr:
    :param xArrFinal:
    :param y0arr:
    :param yArrFinal:
    :return:
    """
    x_ax_heightsCombined = []
    y_ax_heightsCombined = []
    x_ks = []
    y_ks = []

    # [4000, round(len(x0arr) / 2)]:#
    for k in range(0, len(x0arr)):  # for every contour-coordinate value; plot the normal, determine intensity profile & calculate CA from the height profile
        try:
            xOutwards = [0]     #x length pointing outwards of droplet, for possible swelling analysis
            profileOutwards = []
            if outwardsLengthVector != 0:
                profileOutwards, lineLengthPixelsOutwards, fitInside = profileFromVectorCoords(x0arr[k], y0arr[k], dxnegarr[k],
                                                                                    dynegarr[k], outwardsLengthVector,
                                                                                    greyresizedimg)

                # If intensities fit inside profile & are obtained as desired, fill an array with x-positions.
                # If not keep list empty and act as if we don't want the outside vector
                if fitInside:
                    xOutwards = np.linspace(0, lineLengthPixelsOutwards,
                                        len(profileOutwards)) * conversionXY * 1000  # converts pixels to desired unit (prob. um)
                profileOutwards.reverse()  # correct stitching of in-&outwards profiles requires reversing of the outwards profile

            # resizedimg = cv2.polylines(resizedimg, np.array([[x0arr[k], y0arr[k]], [dxarr[k], dyarr[k]]]), False, (0, 255, 0), 2)  # draws 1 good contour around the outer halo fringe#
            if k % 25 == 0 or k in plotHeightCondition:  # only plot 1/25th of the vectors to not overcrowd the image
                if k in plotHeightCondition:
                    colorInwards = (255, 0, 0)
                    colorOutwards = (255, 0, 0)
                else:
                    colorInwards = (0, 255, 0)
                    colorOutwards = (0, 0, 255)
                resizedimg = cv2.line(resizedimg, ([x0arr[k], y0arr[k]]), ([dxarr[k], dyarr[k]]), colorInwards,
                                      2)  # draws 1 good contour around the outer halo fringe
                if outwardsLengthVector != 0:  # if a swelling profile is desired, also plot it in the image
                    resizedimg = cv2.line(resizedimg, ([x0arr[k], y0arr[k]]), ([dxnegarr[k], dynegarr[k]]),
                                          colorOutwards, 2)  # draws 1 good contour around the outer halo fringe
            # intensity profile between x0,y0 & inwards vector coordinate (dx,dy)
            profile, lineLengthPixels, _ = profileFromVectorCoords(x0arr[k], y0arr[k], dxarr[k], dyarr[k], lengthVector,
                                                                greyresizedimg)

            #TODO incoorp. functionality profile + bit outside drop to check for correctness of CA & finding the linear regime
            profileExtraOut = []
            lineLengthPixelsExtraOut = 0
            if smallExtraOutwardsVector != 0:
                profileExtraOut, lineLengthPixelsExtraOut, _ = profileFromVectorCoords(x0arr[k], y0arr[k], dxExtraOutarr[k], dyExtraOutarr[k],
                                                                              smallExtraOutwardsVector, greyresizedimg)

            profileExtraOut.reverse()
            profileExtraOut = profileExtraOut[:-1]  #remove the last datapoint, as it's the same as the start of the CA profile
            # Converts intensity profile to height profile by unwrapping fourier transform wrapping & unwrapping of interferometry peaks
            unwrapped, x, wrapped, peaks = intensityToHeightProfile(profileExtraOut + profile, lineLengthPixelsExtraOut + lineLengthPixels, conversionXY,
                                                                    conversionZ, FLIPDATA)

            x += xOutwards[-1]
            # finds linear fit over most linear regime (read:excludes halo if contour was not picked ideally).
            # startIndex, coef1, r2 = linearFitLinearRegimeOnly(x[len(profileOutwards):], unwrapped[len(profileOutwards):], sensitivityR2, k)
            startIndex, coef1, r2, GoodFit = linearFitLinearRegimeOnly_wPointsOutsideDrop(x, unwrapped, sensitivityR2, k, smallExtraOutwardsVector)

            if not GoodFit: #if the linear fit is not good, skip this vector and continue w/ next
                omittedVectorCounter += 1  # TEMP: to check how many vectors should not be taken into account because the r2 value is too low
                logging.warning(f"Fit inside drop was not good - skipping vector {k}")
                if k == round(len(x0arr) / 2):
                    logging.critical("skipping the vector that would be plotted. This will break the programn for sure.")
                continue
            else:
                a_horizontal = 0
                angleRad = math.atan((coef1[0] - a_horizontal) / (1 + coef1[0] * a_horizontal))
                angleDeg = math.degrees(angleRad)
                if angleDeg > 45:  # Flip measured CA degree if higher than 45.
                    angleDeg = 90 - angleDeg
                xArrFinal.append(x0arr[k])
                yArrFinal.append(y0arr[k])
                vectorsFinal.append(vectors[k])
                angleDegArr.append(angleDeg)

            # plot 1 profile of each image with intensity, wrapped, height & resulting CA
            if k == round(len(x0arr) / 2):
                offsetDropHeight = 0
                heightNearCL = []   #empty list, which is filled when determining the swelling profile outside droplet.
                # TODO WIP: swelling or height profile outside droplet
                if xOutwards[-1] != 0:
                    extraPartIndroplet = 50  # extra datapoints from interference fringes inside droplet for calculating swelling profile outside droplet
                    if extraPartIndroplet >= outwardsLengthVector:
                        logging.critical(f'This will break. OutwardsLengthVector ({outwardsLengthVector}) must be longer than extraPartInDroplet ({extraPartIndroplet}).')
                    heightNearCL, heightRatioNearCL = swellingRatioNearCL(np.arange(0, len(profileOutwards) + extraPartIndroplet),
                        profileOutwards + profile[0:extraPartIndroplet], deltatFromZeroSeconds[n], path, n, k,
                        outwardsLengthVector)

                    # Determine difference in h between brush & droplet profile at 'profileExtraOut' distance from contour
                    offsetDropHeight = heightNearCL[-1 - extraPartIndroplet] / 1000  # height at start of droplet, in relation to the swollen height of PB
                    offsetDropHeight = (unwrapped[len(profileExtraOut)] - heightNearCL[len(profileOutwards)] / 1000)
                    offsetDropHeight = (unwrapped[0] - heightNearCL[-smallExtraOutwardsVector] / 1000)

                #unwrapped = offsetDropHeight + unwrapped

                # set equal height of swelling profile & droplet
                unwrapped = unwrapped - offsetDropHeight

                #remove overlapping datapoints from 'xOutwards' to do proper plotting later
                if xOutwards[-1] != 0:
                    xOutwards = xOutwards[:-smallExtraOutwardsVector]; profileOutwards = profileOutwards[:-smallExtraOutwardsVector]
                #Also, shift x-axis of 'x' to stitch with 'xOutwards properly'
                x = np.array(x) - (x[len(profileExtraOut)] - x[0])

                # # TODO temp
                # figtemp, axtemp = plt.subplots()
                # axtemp.plot(xOutwards, heightNearCL[:len(profileOutwards)],
                #          label="Swelling fringe calculation",
                #          color='C0');  # plot the swelling ratio outside droplet
                # axtemp.plot(np.array(x) - (x[len(profileExtraOut)] - x[0]), (unwrapped * 1000) - (unwrapped[len(profileExtraOut)] * 1000 - heightNearCL[len(profileOutwards)]), label="Interference fringe calculation",
                #          color='C1')
                # axtemp.set(xlabel=r'Distance ($\mu$m)', ylabel = 'Height (nm)')
                # axtemp.legend(loc='best')
                # plt.show()
                # # TODO temp

                #big function for 4-panel plot: Intensity, height, wrapped profile, CA colormap
                ax1, fig1 = plotPanelFig_I_h_wrapped_CAmap(coef1, heightNearCL,
                                                           offsetDropHeight, peaks, profile,
                                                           profileOutwards, r2, startIndex, unwrapped,
                                                           wrapped, x, xOutwards)

            # plot various height profiles in a seperate figure
            # every 1/th of the image, an image is plotted
            # TODO WIP: swelling or height profile outside droplet
            # TODO this part below allows for anchoring at a set distance
            # if xOutwards[-1] != 0 and k in plotHeightCondition:
            #     extraPartIndroplet = 50  # extra datapoints from interference fringes inside droplet for calculating swelling profile outside droplet
            #     heightNearCL, heightRatioNearCL = swellingRatioNearCL(
            #         np.arange(0, len(profileOutwards) + extraPartIndroplet),
            #         profileOutwards + profile[0:extraPartIndroplet], deltatFromZeroSeconds[n], path, n, k)
            #     heightNearCL = scipy.signal.savgol_filter(heightNearCL, len(heightNearCL) // 10, 3) #apply a savgol filter for data smoothing
            #     if heightPlottedCounter == 0:
            #         distanceOfEqualHeight = 10         #can be changed: distance at which the profiles must overlap. xOutwards[-1]
            #         indexOfEqualHeight = np.argmin(abs(xOutwards - distanceOfEqualHeight))
            #         equalHeight = heightNearCL[indexOfEqualHeight]
            #         x_ax_heightsCombined = []
            #         y_ax_heightsCombined = []
            #         x_ks = []
            #         y_ks = []
            #
            #         ax_heightsCombined.plot(distanceOfEqualHeight, equalHeight, '.', markersize = 15, zorder = len(x0arr), label=f'Anchor at = {distanceOfEqualHeight:.2f} um, {equalHeight:.2f} nm')
            #         ax_heightsCombined.axvspan(0, xOutwards[-1], facecolor='orange', alpha=0.3)
            #         ax_heightsCombined.axvspan(xOutwards[-1], x[extraPartIndroplet-1], facecolor='blue', alpha=0.3)
            #     else:
            #         indexOfEqualHeight = np.argmin(abs(xOutwards - distanceOfEqualHeight))
            #         heightNearCL = heightNearCL - (heightNearCL[indexOfEqualHeight] - equalHeight)  #to set all height profiles at some index to the same height
            #     x_ks.append(x0arr[k])
            #     y_ks.append(y0arr[k])
            #     x_ax_heightsCombined.append(np.concatenate([xOutwards, x[:(extraPartIndroplet-1)]]))
            #     y_ax_heightsCombined.append(heightNearCL)
            #     heightPlottedCounter += 1  # increment counter

            # TODO WIP: swelling or height profile outside droplet
            # TODO this part below sets the anchor at some index within the droplet regime
            if xOutwards[-1] != 0 and k in plotHeightCondition:
                extraPartIndroplet = 50  # extra datapoints from interference fringes inside droplet for calculating swelling profile outside droplet
                xBrushAndDroplet = np.arange(0,
                                             len(profileOutwards) + extraPartIndroplet)  # distance (nr of datapoints (NOT pixels!))
                yBrushAndDroplet = profileOutwards + profile[
                                                     0:extraPartIndroplet]  # intensity data of brush & some datapoints within droplet
                # Big function below: for calculating the height profile manually outside droplet by peak selection from intensity profile
                heightNearCL, heightRatioNearCL = swellingRatioNearCL(xBrushAndDroplet, yBrushAndDroplet,
                                                                      deltatFromZeroSeconds[n], path, n, k,
                                                                      outwardsLengthVector)
                heightNearCL = scipy.signal.savgol_filter(heightNearCL, len(heightNearCL) // 10,
                                                          3)  # apply a savgol filter for data smoothing

                if heightPlottedCounter == 0:
                    distanceOfEqualHeight = 10  # can be changed: distance at which the profiles must overlap. xOutwards[-1]
                    indexOfEqualHeight = np.argmin(abs(xOutwards - distanceOfEqualHeight))
                    equalHeight = heightNearCL[indexOfEqualHeight]


                    ax_heightsCombined.plot(distanceOfEqualHeight, equalHeight, '.', markersize=15, zorder=len(x0arr),
                                            label=f'Anchor at = {distanceOfEqualHeight:.2f} um, {equalHeight:.2f} nm')
                    ax_heightsCombined.axvspan(0, xOutwards[-1], facecolor='orange', alpha=0.3)
                    ax_heightsCombined.axvspan(xOutwards[-1], x[extraPartIndroplet - 1], facecolor='blue', alpha=0.3)
                else:
                    indexOfEqualHeight = np.argmin(abs(xOutwards - distanceOfEqualHeight))
                    heightNearCL = heightNearCL - (heightNearCL[
                                                       indexOfEqualHeight] - equalHeight)  # to set all height profiles at some index to the same height
                x_ks.append(x0arr[k])
                y_ks.append(y0arr[k])
                x_ax_heightsCombined.append(np.concatenate([xOutwards, x[:(extraPartIndroplet - 1)]]))
                y_ax_heightsCombined.append(heightNearCL)
                heightPlottedCounter += 1  # increment counter

                # Stitching together swelling height & droplet CA height
                # heightNearCL = heightNearCL - (heightNearCL[(-1-extraPartIndroplet)] - (unwrapped[len(profileOutwards)] * 1000))
                # heightNearCL = heightNearCL - (heightNearCL[(- 1 - extraPartIndroplet)] - (unwrapped[0] * 1000))
                offsetDropHeight = heightNearCL[
                                       -1 - extraPartIndroplet] / 1000  # height at start of droplet, in relation to the swollen height of PB
                unwrapped = offsetDropHeight + unwrapped

                ax10, fig10 = plotPanelFig_I_h_wrapped_CAmap(coef1, heightNearCL,
                                                           offsetDropHeight, peaks, profile,
                                                           profileOutwards, r2, startIndex, unwrapped,
                                                           wrapped, x, xOutwards)

                fig10.suptitle(f"Data profiles: imageNr {n}, vectorNr {k}", size=14)
                fig10.tight_layout()
                fig10.subplots_adjust(top=0.88)
                fig10.savefig(os.path.join(analysisFolder, f"Height profiles - imageNr {n}, vectorNr {k}.png"), dpi=600)
                plt.close(fig10)
        except:
            logging.error(f"!{k}: Analysing each coordinate & normal vector broke!")
            print(traceback.format_exc())
    return ax1, fig1, omittedVectorCounter, resizedimg, xOutwards, x_ax_heightsCombined, x_ks, y_ax_heightsCombined, y_ks


def showPlot(display_mode, figures):
    """
    Display one or more plots with the specified display mode.
    Parameters:
    - display_mode: A string that specifies the display mode. It can be:
        - 'none': Do not display the plots.
        - 'timed': Display the plots for 3 seconds.
        - 'manual': Display the plots until manually closed.
    - figures: A list of matplotlib figure objects to be displayed.
    """
    if display_mode == 'none':
        return

    for fig in figures:
        fig.show()

    if display_mode == 'timed':
        plt.pause(3)
        for fig in figures:
            plt.close(fig)
    elif display_mode == 'manual':
        plt.show()
    else:
        raise ValueError("Invalid display_mode. Use 'none', 'timed', or 'manual'.")


def primaryObtainCARoutine(path, wavelength_laser=520, outwardsLengthVector=0):
    """
    Main routine to analyse the contact angle around the entire contour of a droplet.
    Optionally, also the swelling ratio around the contour can be determined by changing the "outwardsLengthVector" from 0 to e.g. 400

    :param path: Complete path to the folder with images to be analysed.
    :param wavelength_laser: wavelength of used light (unit=nm)
    :param outwardsLengthVector: length of normal vector over which intensity profile data is taken    (pointing OUTWARDS of droplet, so for swelling ratio analysis)
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')     #configuration for printing logging messages. Can be removed safely

    #blockSize	Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
    #C Constant subtracted from the mean or weighted mean.
    #thresholdSensitivityStandard = [11 * 3, 3 * 5]  # [blocksize, C].   OG: 11 * 5, 2 * 5;     working better now = 11 * 3, 2 * 5
    #thresholdSensitivityStandard = [25, 4]  # [blocksize, C].
    #usedImages = np.arange(12, 70, everyHowManyImages)  # len(imgList)
    usedImages = [38]       #36, 57
    thresholdSensitivityStandard = [5,3]# [13, 5]      #typical [13, 5]     [5,3] for higher CA's or closed contours

    imgFolderPath, conversionZ, conversionXY, unitZ, unitXY = filePathsFunction(path, wavelength_laser)

    imgList = [f for f in glob.glob(os.path.join(imgFolderPath, f"*tiff"))]
    everyHowManyImages = 4  # when a range of image analysis is specified, analyse each n-th image
    analysisFolder = os.path.join(imgFolderPath, "Analysis CA Spatial") #name of output folder of Spatial Contact Analysis
    lengthVector = 200  # 200 length of normal vector over which intensity profile data is taken    (pointing into droplet, so for CA analysis)
    outwardsLengthVector = 0      #0 if no swelling profile to be measured., 590
    smallExtraOutwardsVector = 0    #small vector, pointing outwards from CL. Goal: overlap some height fitting from CA analysis inside w/ swelling profile outside. #TODO working code, but profile  outside CL has lower frequency than fringes inside, and this seems to mess with the phase wrapping & unwrapping. So end of height profile is flat-ish..

    FLIPDATA = True
    SHOWPLOTS_SHORT = 'timed'  # 'none' Don't show plots&images at all; 'timed' = show images for only 3 seconds; 'manual' = remain open untill clicked away manually
    sensitivityR2 = 0.997    #sensitivity for the R^2 linear fit for calculating the CA. Generally, it is very good fitting (R^2>0.99)
    FITGAPS_POLYOMIAL = True    #between gaps in CL coordinates, especially when manually picked multiple, fit with a 2d order polynomial to obtain coordinates in between
    saveCoordinates = True  #for saving the actual pixel coordinates for each file analyzed.
    MANUAL_FILTERING = True     #Manually remove certain coordinates from the contour e.g. at pinning sites

    # MANUALPICKING:Manual (0/1):  0 = always pick manually. 1 = only manual picking if 'correct' contour has not been picked & saved manually before.
    # All Automatical(2/3): 2 = let programn pick contour after 1st manual pick (TODO: not advised, doesn't work properly yet). 3 = use known contour IF available, else automatically use the second most outer contour
    MANUALPICKING = 1
    lg_surfaceTension = 27     #surface tension hexadecane liquid-gas (N/m)

    # A list of vector numbers, for which an outwardsVector (if desired) will be shown & heights can be plotted
    #plotHeightCondition = lambda xlist: [round(len(xlist) / 4), round(len(xlist) * 3 / 2)]                  #[300, 581, 4067, 4300]
    plotHeightCondition = lambda xlist: [300, 4000]        #don't use 'round(len(xlist)', as this one always used automatically

    # Order of Fourier fitting: e.g. 8 is fine for little noise/movement. 20 for more noise (can be multiple values: all are shown in plot - highest is used for analysis)
    N_for_fitting = [20]  # TODO fix dit zodat het niet manually moet // order of fitting data with fourier. Higher = describes data more accurately. Useful for noisy data.


    """"End primary changeables"""

    if not os.path.exists(analysisFolder):
        os.mkdir(analysisFolder)
        print('created path: ', analysisFolder)
    contourListFilePath = os.path.join(analysisFolder, "ContourListFile.txt")       #for saving the settings how the contour was obtained (but fails when the experimental box is drawn manually for getting contour)
    contourCoordsFolderFilePath = os.path.join(analysisFolder, "ContourCoords")     #folder for saving individual .txt files containing contour coordinates
    if not os.path.exists(contourCoordsFolderFilePath):
        os.mkdir(contourCoordsFolderFilePath)
        print('created path: ', contourCoordsFolderFilePath)
    contactAngleListFilePath = os.path.join(analysisFolder, "ContactAngle_MedianListFile.txt")
    if os.path.exists(
            contourListFilePath):  # read in all contourline data from existing file (filenr ,+ i for obtaining contour location)
        f = open(contourListFilePath, 'r')
        lines = f.readlines()
        importedContourListData_n, importedContourListData_i, importedContourListData_thresh = extractContourNumbersFromFile(lines)
    else:
        f = open(contourListFilePath, 'w')
        f.write(f"file number (n); outputi; thresholdSensitivity a; thresholdSensitivity b\n")
        f.close()
        print("Created contour list file.txt")
        importedContourListData_n = []
        importedContourListData_i = []

    ndata = []
    if not os.path.exists(
            contactAngleListFilePath):  # Create a file for saving median contact angle, if not already existing
        f = open(contactAngleListFilePath, 'w')
        f.write(f"file number (n), delta time from 0 (s), median CA (deg), Horizontal component force (mN), middle X-coord, middle Y-coord\n")
        f.close()
        print("Created Median Contact Angle list file.txt")
    else:
        f = open(contactAngleListFilePath, 'r')
        lines = f.readlines()
        for line in lines[1:]:
            data = line.split(',')
            ndata.append(int(data[0]))

    angleDeg_afo_time = []  # for saving median CA's later
    totalForce_afo_time = []
    usedDeltaTs = [] # for saving delta t (IN SECONDS) for only the USED IMAGES
    FILTERED = False
    timestamps, deltatseconds, deltatFromZeroSeconds = getTimeStamps(
        imgList)  # get the timestamps of ALL images in folder, and the delta time of ALL images w/ respect to the 1st image
    deltat_formatted = timeFormat(
        deltatFromZeroSeconds)  # formatted delta t (seconds, minutes etc..) for ALL IMAGES in folder.
    # output array for the calculated median contact angle as a function of time (so for every used image)
    if unitZ != "nm" or unitXY != "mm":
        raise Exception("One of either units is not correct for good conversion. Fix manually or implement in code")
    for n, img in enumerate(imgList):
        if n in usedImages:
            if n in importedContourListData_n and MANUALPICKING > 0:
                try:
                    contouri = importedContourListData_i[importedContourListData_n.index(n)]
                    thresholdSensitivity = importedContourListData_thresh[importedContourListData_n.index(n)]
                except:
                    print(
                        f"Either contouri or thresholddata was not imported properly, even though n was in the importedContourListData")
            else:
                contouri = [-1]
                thresholdSensitivity = thresholdSensitivityStandard
            print(f"Determining contour for image n = {n}/{len(imgList)}, or nr {list(usedImages).index(n)+1} out of {len(usedImages)}")



            #Trying for automatic coordinate finding, using coordinates of a previous iteration.
            # TODO doesn't work as desired: now finds contour at location of previous one, but not the  CL one. Incorporate offset somehow, or a check for periodicity of intensitypeaks
            if MANUALPICKING == 2 and n != usedImages[0] and n - usedImages[list(usedImages).index(n) - 1] == everyHowManyImages:
                useablexlist, useableylist, usableContour, resizedimg, greyresizedimg = \
                    getContourCoordsV4(img, contourListFilePath, n, contouri, thresholdSensitivity, MANUALPICKING, usablecontour=usableContour, fitgapspolynomial=FITGAPS_POLYOMIAL)
            #in any other case
            else:
                coordinatesListFilePath = os.path.join(contourCoordsFolderFilePath, f"coordinatesListFilePath_{n}.txt")         #file for all contour coordinates
                filtered_coordinatesListFilePath = os.path.join(contourCoordsFolderFilePath, f"filtered_coordinatesListFilePath_{n}.txt")   #file for manually filtered contour coordinates
                #If allowing importing known coords:
                #-if filtered coordinates etc. already exist, import those
                if (MANUALPICKING in [1, 3]) and os.path.exists(filtered_coordinatesListFilePath):
                    useablexlist, useableylist, usableContour, resizedimg, greyresizedimg, vectorsFinal, angleDegArr = getfilteredContourCoordsFromDatafile(img, filtered_coordinatesListFilePath)
                    xArrFinal = useablexlist
                    yArrFinal = useableylist
                    IMPORTEDCOORDS = True
                    FILTERED = True #Bool for not doing any filtering operations anymore later

                    # For determining the middle coord by mean of surface area - must be performed on unfiltered CL to not bias
                    unfilteredCoordsx, unfilteredCoordsy, _, _, _ = getContourCoordsFromDatafile(img, coordinatesListFilePath)
                    middleCoord = determineMiddleCoord(unfilteredCoordsx, unfilteredCoordsy) #determine middle coord by making use of "mean surface" area coordinate
                    del unfilteredCoordsx, unfilteredCoordsy

                #-if coordinates were already written out, but not filtered
                elif (MANUALPICKING in [1, 3]) and os.path.exists(coordinatesListFilePath):
                    useablexlist, useableylist, usableContour, resizedimg, greyresizedimg = getContourCoordsFromDatafile(img, coordinatesListFilePath)
                    IMPORTEDCOORDS = True
                    FILTERED = False
                    #TODO ^ where to do filtering? -> check where filtering in code
                    # For determining the middle coord by mean of surface area - must be performed on unfiltered CL to not bias
                    middleCoord = determineMiddleCoord(useablexlist, useableylist)  # determine middle coord by making use of "mean surface" area coordinate

                #-if not allowing, or coords not known yet:
                else:
                    useablexlist, useableylist, usableContour, resizedimg, greyresizedimg = \
                        getContourCoordsV4(img, contourListFilePath, n, contouri, thresholdSensitivity, MANUALPICKING, fitgapspolynomial=FITGAPS_POLYOMIAL, saveCoordinates=saveCoordinates, contourCoordsFolderFilePath=coordinatesListFilePath)
                    IMPORTEDCOORDS = False
                    FILTERED = False


            imgshape = resizedimg.shape #tuple (height, width, channel)
            angleDegArr = []
            xArrFinal = []
            yArrFinal = []
            vectorsFinal = []
            omittedVectorCounter = 0

            fig_heightsCombined, ax_heightsCombined = plt.subplots()
            heightPlottedCounter = 0

            print(f"Contour succesfully obtained. Next: obtaining the normals of contour.")
            try:
                resizedimg = cv2.polylines(resizedimg, np.array([usableContour]), False, (0, 0, 255),
                                           2)  # draws 1 red good contour around the outer halo fringe
                #TODO temporary solution to import already filtered coordinates. Completely skip the obtaining coords & vectors part.
                if IMPORTEDCOORDS:
                    logging.info("USING IMPORTED coordinates to determine normal vectors")
                    # If coordinates have been imported already
                    #Get all normal vectors. We'll use only some of them later for plotting purposes :

                    x0arr, dxarr, y0arr, dyarr, vectors, dxnegarr, dynegarr, dxExtraOutarr, dyExtraOutarr = get_normalsV4(useablexlist, useableylist,
                                                                                            lengthVector,
                                                                                            outwardsLengthVector, smallExtraOutwardsVector)

                    logging.info("Starting to extract information from IMPORTED COORDS.\n"
                                 f"Plotting for vector nrs: {plotHeightCondition(useablexlist)} & {round(len(useablexlist)/2)}")
                    ax1, fig1, omittedVectorCounter, resizedimg, xOutwards, x_ax_heightsCombined, x_ks, y_ax_heightsCombined, y_ks = coordsToIntensity_CAv2(
                        FLIPDATA, analysisFolder, angleDegArr, ax_heightsCombined, conversionXY, conversionZ,
                        deltatFromZeroSeconds, dxarr, dxnegarr, dyarr, dynegarr, greyresizedimg,
                        heightPlottedCounter, lengthVector, n, omittedVectorCounter, outwardsLengthVector, path,
                        plotHeightCondition(useablexlist), resizedimg, sensitivityR2, vectors, vectorsFinal, x0arr, xArrFinal, y0arr,
                        yArrFinal, IMPORTEDCOORDS, SHOWPLOTS_SHORT, dxExtraOutarr, dyExtraOutarr, smallExtraOutwardsVector)

                else:
                    #If the CL coordinates have not been imported (e.g. for new img file)
                    # One of the main functions:
                    # Should yield the normal for every point: output is original x&y coords (x0,y0)
                    # corresponding normal coordinate inwards to droplet x,y (defined as dx and dy)
                    # and normal x,y coordinate outwards of droplet (dxneg & dyneg)
                    logging.info("USING CHOSEN CONTACT LINE to determine normal vectors")
                    x0arr, dxarr, y0arr, dyarr, vectors, dxnegarr, dynegarr, dxExtraOutarr, dyExtraOutarr = get_normalsV4(useablexlist, useableylist, lengthVector, outwardsLengthVector, smallExtraOutwardsVector)
                    print(f"Normals sucessfully obtained. Next: plot normals in image & obtain intensities over normals")
                    tempcoords = [[x0arr[k], y0arr[k]] for k in range(0, len(x0arr))]

                    if any(t < 0 for t in x0arr) or any(p < 0 for p in y0arr):#Check for weird x or y values, THEY NEVER SHOULD BE NEGATIVE
                        logging.critical("Either an x or y coordinate was found to be negative!\n This should not be possible.")

                    #TODO attempting to determine middle coord by making use of "mean surface" area coordinate
                    middleCoord = determineMiddleCoord(x0arr, y0arr)

                    tempimg = []
                    tempimg = cv2.polylines(resizedimg, np.array([tempcoords]), False, (0, 255, 0),
                                            20)  # draws 1 blue contour with the x0&y0arrs obtained from get_normals
                    print(f"Middle Coordinates from surface area [x,y] :\n"
                          f"[{middleCoord[0]}", f" {middleCoord[1]}]")
                    resizedimg = cv2.circle(tempimg, (middleCoord[0], middleCoord[1]), 63, (255, 255, 255), -1)  # draw median middle. abs(np.subtract(imgshape[0], medianmiddleY))
                    tempimg = cv2.resize(tempimg, [round(tempimg.shape[1] / 5), round(tempimg.shape[0] / 5)],
                                         interpolation=cv2.INTER_AREA)  # resize image

                    if SHOWPLOTS_SHORT in ['timed', 'manual']:
                        cv2.imshow( f"Contour of img {np.where(np.array(usedImages) == n)[0][0]} out of {len(usedImages)} with coordinates being used by get_normals", tempimg)
                        cv2.waitKey(3000)
                        cv2.destroyAllWindows()
                    #cv2.imwrite(os.path.join(analysisFolder, f"rawImage_x0y0Arr_blue{n}.png"), tempimg)

                    #TODO Important function!
                    ax1, fig1, omittedVectorCounter, resizedimg, xOutwards, x_ax_heightsCombined, x_ks, y_ax_heightsCombined, y_ks = coordsToIntensity_CAv2(
                        FLIPDATA, analysisFolder, angleDegArr, ax_heightsCombined, conversionXY, conversionZ,
                        deltatFromZeroSeconds, dxarr, dxnegarr, dyarr, dynegarr, greyresizedimg,
                        heightPlottedCounter, lengthVector, n, omittedVectorCounter, outwardsLengthVector, path,
                        plotHeightCondition(useablexlist), resizedimg, sensitivityR2, vectors, vectorsFinal, x0arr,
                        xArrFinal, y0arr,
                        yArrFinal, IMPORTEDCOORDS, SHOWPLOTS_SHORT, dxExtraOutarr, dyExtraOutarr,
                        smallExtraOutwardsVector)

                    print(f"Normals, intensities & Contact Angles Succesffuly obtained. Next: plotting overview of all data for 1 timestep")
                    logging.warning(f"Out of {len(x0arr)}, {omittedVectorCounter} number of vectors were omitted because the R^2 was too low.")


                #coordsBottom, coordsTop = determineTopAndBottomOfDropletCoords(x0arr, y0arr, dxarr, dyarr)
                #TODO testing the 'easy way' of determining top&bottom with only min/max because other method fails sometimes?
                coordsBottom, coordsTop = determineTopAndBottomOfDropletCoords_SIMPLEMINMAX(vectorsFinal, xArrFinal, yArrFinal)
                print(f"Calculated top and bottom coordinates of the droplet to be:\n"
                      f"Top: x={coordsTop[0]}, y={coordsTop[1]}\n"
                      f"Bottom: x={coordsBottom[0]}, y={coordsBottom[1]}")


                resizedimg = cv2.circle(resizedimg, (coordsBottom), 30, (255, 0, 0), -1)    #draw blue circle at calculated bottom/inflection point of droplet
                resizedimg = cv2.circle(resizedimg, (coordsTop), 30, (0, 255, 0), -1)       #green

                # determine middle of droplet & plot
                middleX, middleY, meanmiddleX, meanmiddleY, medianmiddleX, medianmiddleY = approxMiddlePointDroplet(list(zip(xArrFinal, yArrFinal)), vectorsFinal)

                DONEFILTERTIING = False
                temp_xArrFinal = xArrFinal
                temp_yArrFinal = yArrFinal
                temp_angleDegArr = angleDegArr
                temp_vectorsFinal = vectorsFinal

                if FILTERED:    #already filtered, only plot the contact angle scatterplot
                    pass
                    #TODO implement a way to still extract height profiles from desired locations.
                    #TODO Both inside droplet, and outside
                    # fig1, ax1 = plt.subplots(2, 2)
                    # ax1[0, 0].plot(profileOutwards + profile, 'k');
                    # if xOutwards[-1] != 0:
                    #     ax1[0, 0].plot(len(profileOutwards), profileOutwards[-1], 'r.',
                    #                    label='transition brush-droplet')
                    #     ax1[0, 0].axvspan(0, len(profileOutwards), facecolor='orange', alpha=0.5, label='brush profile')
                    # ax1[0, 0].axvspan(len(profileOutwards), len(profileOutwards + profile), facecolor='blue', alpha=0.5,
                    #                   label='droplet')
                    # ax1[0, 0].legend(loc='best')
                    # ax1[0, 0].set_title(f"Intensity profile");
                    #
                    # ax1[1, 0].plot(wrapped);
                    # ax1[1, 0].plot(peaks, wrapped[peaks], '.')
                    # ax1[1, 0].set_title("Wrapped profile (drop only)")
                    #
                    # # TODO unit unwrapped was in um, *1000 -> back in nm. unit x in um
                    # if xOutwards[-1] != 0:
                    #     ax1[0, 1].plot(xOutwards, heightNearCL[:len(profileOutwards)],
                    #                    label="Swelling fringe calculation",
                    #                    color='C0');  # plot the swelling ratio outside droplet
                    # ax1[0, 1].plot(x, unwrapped * 1000, label="Interference fringe calculation", color='C1');
                    # ax1[0, 1].plot(x[startIndex], unwrapped[startIndex] * 1000, 'r.',
                    #                label='Start linear regime droplet');
                    # # '\nCA={angleDeg:.2f} deg. ' Initially had this in label below, but because of code order change angledeg is not defined yet
                    # ax1[0, 1].plot(x, (np.poly1d(coef1)(x) + offsetDropHeight) * 1000, '--', linewidth=1,
                    #                label=f'Linear fit, R$^2$={r2:.3f}');
                    # ax1[0, 1].legend(loc='best')
                    # ax1[0, 1].set_title("Brush & drop height vs distance")
                    #
                    # ax1[0, 0].set_xlabel("Distance (nr.of datapoints)");
                    # ax1[0, 0].set_ylabel("Intensity (a.u.)")
                    # ax1[1, 0].set_xlabel("Distance (nr.of datapoints)");
                    # ax1[1, 0].set_ylabel("Amplitude (a.u.)")
                    # ax1[0, 1].set_xlabel("Distance (um)");
                    # ax1[0, 1].set_ylabel("Height profile (nm)")
                    # fig1.set_size_inches(12.8, 9.6)
                    #
                    # #plotting for fig3
                    # fig3, ax3 = plt.subplots()
                    # im3 = ax3.scatter(xArrFinal, abs(np.subtract(imgshape[0], yArrFinal)), c=angleDegArr, cmap='jet', vmin=min(angleDegArr), vmax=max(angleDegArr))
                    # ax3.set_xlabel("X-coord"); ax3.set_ylabel("Y-Coord"); ax3.set_title(f"Spatial Contact Angles Colormap n = {n}, or t = {deltat_formatted[n]}")
                    # ax3.legend([f"Median CA (deg): {(statistics.median(angleDegArr)):.2f}"], loc='center left')
                    # fig3.colorbar(im3)
                    # plt.show()
                    # fig3.savefig(os.path.join(analysisFolder, f'Colorplot XYcoord-CA {n:04}-filtered.png'), dpi=600)
                else:  #else, allow to manually filter the CA scatterplot & save filtered coords afterwards
                    while not DONEFILTERTIING:
                        #TODO trying to deselect regions manually, where e.g. a pinning point is
                        fig3, ax3 = plt.subplots()
                        im3 = ax3.scatter(temp_xArrFinal, abs(np.subtract(imgshape[0], temp_yArrFinal)), c=temp_angleDegArr, cmap='jet',
                                          vmin=min(temp_angleDegArr), vmax=max(temp_angleDegArr))
                        if MANUAL_FILTERING:
                            highlighter = Highlighter(ax3, np.array(temp_xArrFinal), np.array(abs(np.subtract(imgshape[0], temp_yArrFinal))))
                        ax3.set_xlabel("X-coord"); ax3.set_ylabel("Y-Coord"); ax3.set_title(f"Spatial Contact Angles Colormap n = {n}, or t = {deltat_formatted[n]}")
                        ax3.legend([f"Median CA (deg): {(statistics.median(temp_angleDegArr)):.2f}"], loc='center left')
                        fig3.colorbar(im3)
                        plt.show()
                        if MANUAL_FILTERING:
                            selected_regions = highlighter.mask
                            inverted_selected_regions = [not elem for elem in selected_regions] #invert booleans to 'deselect' the selected regions
                            xrange1, yrange1 = np.array(temp_xArrFinal)[inverted_selected_regions], np.array(temp_yArrFinal)[inverted_selected_regions]
                        fig3.savefig(os.path.join(analysisFolder, f'Colorplot XYcoord-CA {n:04}.png'), dpi=600)
                        plt.close()

                        if MANUAL_FILTERING:
                            filtered_angleDegArr = np.array(temp_angleDegArr)[inverted_selected_regions]
                            fig3, ax3 = plt.subplots()
                            im3 = ax3.scatter(xrange1, abs(np.subtract(imgshape[0], yrange1)), c=filtered_angleDegArr, cmap='jet',
                                              vmin=min(filtered_angleDegArr), vmax=max(filtered_angleDegArr))
                            ax3.set_xlabel("X-coord"); ax3.set_ylabel("Y-Coord"); ax3.set_title(f"Spatial Contact Angles Colormap n = {n}, or t = {deltat_formatted[n]}")
                            ax3.legend([f"Median CA (deg): {(statistics.median(filtered_angleDegArr)):.2f}"], loc='center left')
                            fig3.colorbar(im3)
                            plt.show()
                            fig3.savefig(os.path.join(analysisFolder, f'Colorplot XYcoord-CA {n:04}-filtered.png'), dpi=600)
                            plt.close()

                            choices = ["Good filtering: use leftover coordinates", "Bad filtering: filter more in current coordinates", "Bad filtering: redo entire process", "Bad filtering: don't filter"]
                            myvar = easygui.choicebox("What to do next?", choices=choices)
                            temp_vectorsFinal = np.array(temp_vectorsFinal)[inverted_selected_regions]
                            if myvar == choices[0]:
                                xArrFinal = xrange1
                                yArrFinal = yrange1
                                angleDegArr = filtered_angleDegArr
                                vectorsFinal = temp_vectorsFinal
                                DONEFILTERTIING = True
                            elif myvar == choices[1]:
                                temp_xArrFinal = xrange1
                                temp_yArrFinal = yrange1
                                temp_angleDegArr = filtered_angleDegArr
                            elif myvar == choices[2]:
                                temp_xArrFinal = xArrFinal
                                temp_yArrFinal = yArrFinal
                                temp_angleDegArr = angleDegArr
                                temp_vectorsFinal  = vectorsFinal
                            elif myvar == choices[3]:
                                DONEFILTERTIING = True
                    #TODO save filtered coordinates, contact angles and vectors to a .txt file for even faster analysis
                    if saveCoordinates == True:
                        if os.path.exists(os.path.dirname(filtered_coordinatesListFilePath)):
                            with open(filtered_coordinatesListFilePath, 'wb') as internal_filename:
                                pickle.dump([list(zip(xArrFinal, yArrFinal)), vectorsFinal, angleDegArr], internal_filename)
                        else:
                            logging.critical(
                                "Path to folder in which the contour coordinates file is to be saved DOES NOT exist.\n"
                                "When parsing 'saveCoordinates' = True, make sure 'coordinatesListFilePath' is parsed (correctly) as well")

                #calculate the nett force over given CA en droplet range
                tangent_forces = CA_And_Coords_To_Force(xArrFinal, abs(np.subtract(imgshape[0], yArrFinal)), vectorsFinal, angleDegArr, analysisFolder, lg_surfaceTension)


                fig2, ax2 = plt.subplots()
                ax2.plot(middleX, abs(np.subtract(imgshape[0], middleY)), 'b.', label='intersects of normal vectors')
                ax2.plot(xArrFinal, abs(np.subtract(imgshape[0], yArrFinal)), 'r', label='contour of droplet')
                ax2.plot(medianmiddleX, abs(np.subtract(imgshape[0], medianmiddleY)), 'k.', markersize=20, label='median middle coordinate')
                ax2.set_xlabel('X-coords'); ax2.set_ylabel('Y-coords')
                ax2.legend(loc='best')
                print(f"meanX = {meanmiddleX}, meanY:{meanmiddleY}, medianX = {medianmiddleX}, medianY = {medianmiddleY}")
                fig2.savefig(os.path.join(analysisFolder, f'Middle of droplet {n:04}.png'), dpi=600)
                showPlot(SHOWPLOTS_SHORT, [fig2])

                #PLOTTING various previously calculated OUTSIDE height profiles
                if not FILTERED and xOutwards[-1] != 0:
                    phi_k, _ = coordsToPhi(x_ks, abs(np.subtract(imgshape[0], y_ks)), medianmiddleX, abs(np.subtract(imgshape[0], medianmiddleY)))
                    for i in range(0, len(x_ax_heightsCombined)):
                        ax_heightsCombined.plot(x_ax_heightsCombined[i], y_ax_heightsCombined[i], label=f'$\phi$={convertPhiToazimuthal(phi_k[i])[1]/np.pi:.2f}$\pi$ rad')
                    ax_heightsCombined.legend(loc='best')
                    ax_heightsCombined.set(xlabel='Distance (um)', ylabel='Film height (nm)', title='Halo height profiles at\nvarious positions on droplet contour\nSavgol smoothened!')
                    fig_heightsCombined.savefig(os.path.join(analysisFolder, f'Combined height profiles imgNr {n}.png'), dpi=600)

                #TODO: middle point is not working too well yet, so left& right side are a bit skewed
                phi, rFromMiddleArray_pixel = coordsToPhi(xArrFinal, abs(np.subtract(imgshape[0], yArrFinal)), medianmiddleX, abs(np.subtract(imgshape[0], medianmiddleY)))
                phiTop, rTop_pixel = coordsToPhi(coordsTop[0], abs(np.subtract(imgshape[0], coordsTop[1])), medianmiddleX, abs(np.subtract(imgshape[0], medianmiddleY)))  #the phi at which the 'top' of the droplet is located
                phiBottom, rBot_pixel = coordsToPhi(coordsBottom[0], abs(np.subtract(imgshape[0], coordsBottom[1])), medianmiddleX, abs(np.subtract(imgshape[0], medianmiddleY))) #the phi at which the 'bottom' of the droplet is located

                rFromMiddleArray_m = rFromMiddleArray_pixel * conversionXY / 1000   #pixel* conversionXY is in mm, so divide by 1000 to yield in meters
                rTop_m = rTop_pixel * conversionXY / 1000  # pixel* conversionXY is in mm, so divide by 1000 to yield in meters
                rBot_m = rBot_pixel * conversionXY / 1000  # pixel* conversionXY is in mm, so divide by 1000 to yield in meters

                azimuthalX, phi_normalRadians = convertPhiToazimuthal(phi)

                fig4, ax4 = plt.subplots()
                #condition = [(elem < (np.pi * (1/2)) or elem > (np.pi * (3/2))) for elem in phi]    #condition for top half of sphere
                condition = [elem > 0 for elem in phi]  # condition for top half of sphere
                rightFromMiddle = azimuthalX[condition]
                leftFromMiddle = azimuthalX[np.invert(condition)]
                ax4.plot(rightFromMiddle, np.array(angleDegArr)[condition], '.', label='top side')
                ax4.plot(leftFromMiddle, np.array(angleDegArr)[np.invert(condition)], '.', label='bottom side')

                #halfDropletCondition = np.invert((phi>phiBottom) & (phi<phiTop))
                #phi_leftside = phi[halfDropletCondition]
                #ax4.plot(convertPhiToazimuthal(phi_leftside)[0], np.array(angleDegArr)[halfDropletCondition], '-', label='left side')

                #still playing around with the windowsize. (e.g. round(len(angleDegArr)/10))
                sovgol_windowSize = round(len(angleDegArr)/40); savgol_order = 3
                sovgol_windowSize = int(sovgol_windowSize + (np.mod(sovgol_windowSize, 2) == 0))    #ensure window size is uneven
                #azimuthal_savgol = scipy.signal.savgol_filter(angleDegArr, sovgol_windowSize, savgol_order, mode='wrap')
                #ax4.plot(azimuthalX, azimuthal_savgol, '--', label=f'savitsky golay filtered. Wsize = {sovgol_windowSize}, order = {savgol_order}')
                aziCA_savgol_nonuniformed = non_uniform_savgol(azimuthalX, angleDegArr, sovgol_windowSize, savgol_order, mode='periodic')
                phi_CA_savgol_nonuniformed = non_uniform_savgol(phi, angleDegArr, sovgol_windowSize, savgol_order, mode='periodic')
                phi_tangentF_savgol_nonuniformed = non_uniform_savgol(phi, tangent_forces, sovgol_windowSize, savgol_order, mode='periodic')
                ax4.plot(azimuthalX, aziCA_savgol_nonuniformed, '.', markersize=3, label=f'azi savgol filter, nonuniform')

                # TODO plotting a function of r against phi, so I can integrate properly later on. Trying the savgol& cubicSpline
                rFromMiddle_savgol = non_uniform_savgol(phi, rFromMiddleArray_m, sovgol_windowSize, savgol_order, mode='periodic')
                phi_sorted, aziCA_savgol_sorted, rFromMiddle_savgol_sorted, phiCA_savgol_sorted, phi_tangentF_savgol_sorted = [list(a) for a in zip(*sorted(zip(phi, aziCA_savgol_nonuniformed, rFromMiddle_savgol, phi_CA_savgol_nonuniformed, phi_tangentF_savgol_nonuniformed)))]  #TODO, check dit goed; snelle fix voor altijd increasing x, maar is misschien heel fout
                for i in range(1, len(phi_sorted)):
                    if phi_sorted[i] <= phi_sorted[i - 1]:
                        phi_sorted[i] = phi_sorted[i - 1] + 1e-5

                # #cubespline. +[x[0]] and +[y[0]] for required periodic boundary condition
                # phi_CA_savgol_cs = scipy.interpolate.CubicSpline(phi_sorted + [phi_sorted[-1] + 1e-5], phiCA_savgol_sorted + [phiCA_savgol_sorted[0]], bc_type='periodic')
                # phi_tangentF_savgol_cs = scipy.interpolate.CubicSpline(phi_sorted + [phi_sorted[-1] + 1e-5], phi_tangentF_savgol_sorted + [phi_tangentF_savgol_sorted[0]], bc_type='periodic')

                phiCA_fourierFit, phiCA_fourierFit_single, phiCA_N, _, _ = manualFitting(phi_sorted, phiCA_savgol_sorted, analysisFolder, ["Contact angle ", "[deg]"], N_for_fitting, SHOWPLOTS_SHORT)
                tangentF_fourierFit, tangentF_fourierFit_single, tangentF_N, _, _ = manualFitting(phi_sorted, phi_tangentF_savgol_sorted, analysisFolder, ["Horizontal Component Force ", "[mN/m]"], N_for_fitting, SHOWPLOTS_SHORT)
                rFromMiddle_fourierFit, rFromMiddle_fourierFit_single, rFromMiddle_N, _, _ = manualFitting(phi_sorted, rFromMiddle_savgol_sorted, analysisFolder, ["Radius", "[m]"], N_for_fitting, SHOWPLOTS_SHORT)

                phi_range = np.arange(min(phi), max(phi), 0.01) #TODO this step must be quite big, otherwise for whatever reason the cubicSplineFit introduces a lot of noise at positions where before the data interval was relatively large = bad interpolation
                # phiCA_cubesplined = phi_CA_savgol_cs(phi_range[:-1])      #if using a cubicSpline Fit
                ax4.plot(convertPhiToazimuthal(phi_range)[0], phiCA_fourierFit(phi_range), '.', label=f'Fourier Fit order: {phiCA_N[-1]}')
                ax4.set(title=f"Azimuthal contact angle.\nWsize = {sovgol_windowSize}, order = {savgol_order}", xlabel=f'sin($\phi$)', ylabel='contact angle (deg)')
                ax4.legend(loc='best')
                fig4.savefig(os.path.join(analysisFolder, f'Azimuthal contact angle {n:04}.png'), dpi=600)

                fig6, ax6 = plt.subplots()
                ax6.plot(phi[condition], np.array(angleDegArr)[condition], '.', label='raw data: top side')
                ax6.plot(phi[np.invert(condition)], np.array(angleDegArr)[np.invert(condition)], '.', label='raw data: bottom side')
                ax6.plot(phi, aziCA_savgol_nonuniformed, '.', markersize=3, label=f'savgol filter, nonuniform')
                ax6.plot(phi_range, phiCA_fourierFit(phi_range), '.', label=f'Fourier Fit order: {phiCA_N[-1]}')
                ax6.set(title=f"Radial contact angle.\nWsize = {sovgol_windowSize}, order = {savgol_order}", xlabel=f'$phi$ (rad))', ylabel='contact angle (deg)')
                ax6.legend(loc='best')
                fig6.savefig(os.path.join(analysisFolder, f'Radial contact angle {n:04}.png'), dpi=600)
                showPlot(SHOWPLOTS_SHORT, [fig4, fig6])

                # TODO plotting a function of r against phi, so I can integrate properly later on. Trying the savgol& cubicSpline
                fig5, ax5 = plt.subplots()
                ax5_2 = ax5.twinx()
                # phi_r_savgol_cs = scipy.interpolate.CubicSpline(phi_sorted + [phi_sorted[-1] + 1e-5], rFromMiddle_savgol_sorted + [rFromMiddle_savgol_sorted[0]], bc_type='periodic')

                #for plotting r vs phi
                #ax5.plot(phi_sorted, np.array(rFromMiddle_savgol_sorted) * 1000, 'r.', label='r vs phi data')
                #ax5.plot(phi_range, np.array(phi_r_savgol_cs(phi_range)) * 1000, 'k--', label='cubic spline fit')
                #ax5.set_ylabel('Radius length (millimeter)', color='r')
                ax5.plot(np.divide(phi_sorted, np.pi), np.array(phi_tangentF_savgol_sorted), 'r.', label='Azimuthal $F_{hor}$ data')
                #ax5.plot(phi_range, np.array(tangentF_fourierFit(phi_range)), 'k--', label=f'Fourier fit order: {tangentF_N[-1]}')
                ax5.plot(np.divide(phi_range, np.pi), np.array(tangentF_fourierFit(phi_range)), 'k--', label=f'Fourier fit')
                ax5.set_ylabel('Horizontal force (mN/m)', color='r')
                ax5_2.plot(np.divide(phi_sorted, np.pi), phiCA_savgol_sorted, 'b', label='Azimuthal CA data')
                #ax5_2.plot(phi_range, phiCA_fourierFit(phi_range), 'y--', label=f'Fourier fit order: {phiCA_N[-1]}')
                ax5_2.plot(np.divide(phi_range, np.pi), phiCA_fourierFit(phi_range), 'y--', label=f'Fourier fit')
                ax5.set_xlabel('Azimuthal angle $\phi$ (rad)')

                ax5_2.set_ylabel('Contact Angle (deg)', color='b')
                ax5.legend(loc='upper left');
                ax5_2.legend(loc='upper right')
                ax5.xaxis.set_major_formatter(FormatStrFormatter('%g $\pi$'))
                ax5.xaxis.set_major_locator(MultipleLocator(base=1.0))
                fig5.tight_layout()
                fig5.savefig(os.path.join(analysisFolder, f'Radius vs Phi {n:04}.png'), dpi=600)
                showPlot(SHOWPLOTS_SHORT, [fig5])

                #TODO calculate nett horizonal force in for each phi, and fit it with a cubic spline
                #total_force_quad, error_quad, trapz_intForce_function, trapz_intForce_data = calculateForceOnDroplet(phi_tangentF_savgol_cs, phi_r_savgol_cs, phiTop, phiBottom, analysisFolder, phi_sorted, rFromMiddle_savgol_sorted, phi_tangentF_savgol_sorted)

                total_force_quad, error_quad, trapz_intForce_function, trapz_intForce_data = calculateForceOnDroplet(
                    tangentF_fourierFit_single, rFromMiddle_fourierFit_single, phiTop, phiBottom, analysisFolder, phi_sorted,
                    rFromMiddle_savgol_sorted, phi_tangentF_savgol_sorted)

                # #TODO trying to fit the CA contour in 3D, to integrate etc. for force calculation
                # Z, totalZ = givemeZ(np.array(xArrFinal), np.array(yArrFinal), tangent_forces, np.array(x0arr), np.array(y0arr), conversionXY, analysisFolder, n)

                totalForce_afo_time.append(total_force_quad)

                angleDeg_afo_time.append(statistics.median(angleDegArr))
                usedDeltaTs.append(deltatFromZeroSeconds[n])    #list with delta t (IN SECONDS) for only the USED IMAGES


                # #TODO trying to deselect regions manually, where e.g. a pinning point is
                # fig3, ax3 = plt.subplots()
                # im3 = ax3.scatter(xArrFinal, abs(np.subtract(imgshape[0], yArrFinal)), c=angleDegArr, cmap='jet',
                #                   vmin=min(angleDegArr), vmax=max(angleDegArr))
                # highlighter = Highlighter(ax3, np.array(xArrFinal), np.array(abs(np.subtract(imgshape[0], yArrFinal))))
                # ax3.set_xlabel("X-coord"); ax3.set_ylabel("Y-Coord"); ax3.set_title(f"Spatial Contact Angles Colormap n = {n}, or t = {deltat_formatted[n]}")
                # ax3.legend([f"Median CA (deg): {(statistics.median(angleDegArr)):.2f}"], loc='center left')
                # fig3.colorbar(im3)
                # plt.show()
                # selected_regions = highlighter.mask
                # inverted_selected_regions = [not elem for elem in selected_regions] #invert booleans to 'deselect' the selected regions
                # xrange1, yrange1 = np.array(xArrFinal)[inverted_selected_regions], np.array(yArrFinal)[inverted_selected_regions]
                # fig3.savefig(os.path.join(analysisFolder, f'Colorplot XYcoord-CA {n:04}.png'), dpi=600)
                # plt.close()
                #
                #
                # filtered_angleDegArr = np.array(angleDegArr)[inverted_selected_regions]
                # fig3, ax3 = plt.subplots()
                # im3 = ax3.scatter(xrange1, abs(np.subtract(imgshape[0], yrange1)), c=filtered_angleDegArr, cmap='jet',
                #                   vmin=min(filtered_angleDegArr), vmax=max(filtered_angleDegArr))
                # ax3.set_xlabel("X-coord"); ax3.set_ylabel("Y-Coord"); ax3.set_title(f"Spatial Contact Angles Colormap n = {n}, or t = {deltat_formatted[n]}")
                # ax3.legend([f"Median CA (deg): {(statistics.median(angleDegArr)):.2f}"], loc='center left')
                # fig3.colorbar(im3)
                # fig3.savefig(os.path.join(analysisFolder, f'Colorplot XYcoord-CA {n:04}-filtered.png'), dpi=600)
                # plt.close()


                #TODO uncomment if above doesnt work: for plotting of CA in scatterplot w/ colorbar
                # fig3, ax3 = plt.subplots()
                # im3 = ax3.scatter(xArrFinal, abs(np.subtract(imgshape[0], yArrFinal)), c=angleDegArr, cmap='jet',
                #                   vmin=min(angleDegArr), vmax=max(angleDegArr))
                #
                # ax3.set_xlabel("X-coord"); ax3.set_ylabel("Y-Coord"); ax3.set_title(f"Spatial Contact Angles Colormap n = {n}, or t = {deltat_formatted[n]}")
                # ax3.legend([f"Median CA (deg): {(statistics.median(angleDegArr)):.2f}"], loc='center left')
                # fig3.colorbar(im3)
                # fig3.savefig(os.path.join(analysisFolder, f'Colorplot XYcoord-CA {n:04}.png'), dpi=600)
                # plt.close()

                im1 = ax1[1, 1].scatter(xArrFinal, abs(np.subtract(imgshape[0], yArrFinal)), c=angleDegArr, cmap='jet',
                                        vmin=min(angleDegArr), vmax=max(angleDegArr))
                ax1[1, 1].set_xlabel("X-coord"); ax1[1, 1].set_ylabel("Y-Coord"); ax1[1, 1].set_title(f"Spatial Contact Angles Colormap n = {n}, or t = {deltat_formatted[n]}")
                ax1[1, 1].legend([f"Median CA (deg): {(statistics.median(angleDegArr)):.2f}"], loc='center left')
                fig1.colorbar(im1)
                fig1.savefig(os.path.join(analysisFolder, f'Complete overview {n:04}.png'), dpi=600)

                showPlot(SHOWPLOTS_SHORT, [fig1])

                # Export Contact Angles to a csv file & add median CA to txt file
                wrappedPath = os.path.join(analysisFolder, f"ContactAngleData {n}.csv")
                d = dict({'x-coords': xArrFinal, 'y-coords': yArrFinal, 'contactAngle': angleDegArr})
                df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))  # pad shorter colums with NaN's
                df.to_csv(wrappedPath, index=False)

                if n not in ndata:  # if not already saved median CA, save to txt file.
                    CAfile = open(contactAngleListFilePath, 'a')
                    CAfile.write(f"{n}, {usedDeltaTs[-1]}, {angleDeg_afo_time[-1]}, {totalForce_afo_time[-1]}, {middleCoord[0]}, {middleCoord[1]}\n")
                    CAfile.close()

                print("------------------------------------Succesfully finished--------------------------------------------\n"
                      "------------------------------------   previous image  --------------------------------------------")

            except Exception:
                logging.critical(f"Some error occured. Still plotting obtained contour")
                print(traceback.format_exc())
            tstring = str(datetime.now().strftime("%Y_%m_%d"))  # day&timestring, to put into a filename    "%Y_%m_%d_%H_%M_%S"
            resizedimg = cv2.circle(resizedimg, (round(medianmiddleX), round(medianmiddleY)), 30, (0, 255, 0), -1)  # draw median middle. abs(np.subtract(imgshape[0], medianmiddleY))
            cv2.imwrite(os.path.join(analysisFolder, f"rawImage_contourLine_{tstring}_{n}.png"), resizedimg)

            plt.close() #close all existing figures

    #once all images are analysed, plot obtained data together. Can also be done separately afterwards with the "CA_analysisRoutine()" in this file
    fig2, ax2 = plt.subplots()
    ax2.plot(np.divide(usedDeltaTs, 60), totalForce_afo_time)
    ax2.set_xlabel("Time (minutes)"); ax2.set_ylabel("Horizontal component force (mN)"); ax2.set_title("Horizontal component force over Time")
    fig2.savefig(os.path.join(analysisFolder, f'Horizontal component force vs Time.png'), dpi=600)
    showPlot(SHOWPLOTS_SHORT, [fig2])
    #plt.show()


def plotPanelFig_I_h_wrapped_CAmap(coef1, heightNearCL, offsetDropHeight, peaks, profile, profileOutwards,
                                   r2, startIndex, unwrapped, wrapped, x, xOutwards):
    """"
    #for 4-panel plot:  Intensity vs datapoint,
                        height vs distance,
                        wrapped profile vs datapoint,
                        CA colormap x,y-coord
    """
    fig1, ax1 = plt.subplots(2, 2)
    ax1[0, 0].plot(profileOutwards + profile, 'k');
    if xOutwards[-1] != 0:
        ax1[0, 0].plot(len(profileOutwards), profileOutwards[-1], 'g.',
                       label='Transition brush-droplet (man. contour)')
        ax1[0, 0].plot(startIndex+len(profileOutwards), profile[startIndex], 'r.',
                       label='Start linear regime droplet')
        ax1[0, 0].axvspan(0, len(profileOutwards), facecolor='blue', alpha=0.5,
                          label='(Swollen) brush')
    ax1[0, 0].axvspan(len(profileOutwards), len(profileOutwards + profile),
                      facecolor='orange', alpha=0.5,
                      label='droplet')
    ax1[0, 0].legend(loc='best')
    ax1[0, 0].set_title(f"Intensity profile");
    ax1[1, 0].plot(wrapped);
    ax1[1, 0].plot(peaks, wrapped[peaks], '.')
    ax1[1, 0].set_title("Wrapped profile (drop only)")
    # TODO unit unwrapped was in um, *1000 -> back in nm. unit x in um
    if xOutwards[-1] != 0:
        ax1[0, 1].plot(xOutwards, heightNearCL[:len(profileOutwards)],
                       label="Swelling fringe calculation",
                       color='C0');  # plot the swelling ratio outside droplet
    ax1[0, 1].plot(x, unwrapped * 1000, label="Interference fringe calculation",
                   color='C1');
    ax1[0, 1].plot(x[startIndex], unwrapped[startIndex] * 1000, 'r.',
                   label='Start linear regime droplet');
    # '\nCA={angleDeg:.2f} deg. ' Initially had this in label below, but because of code order change angledeg is not defined yet
    ax1[0, 1].plot(x, (np.poly1d(coef1)(x) + offsetDropHeight) * 1000, '--', linewidth=1,
                   label=f'Linear fit, R$^2$={r2:.3f}');
    ax1[0, 1].legend(loc='best')
    ax1[0, 1].set_title("Brush & drop height vs distance")
    ax1[0, 0].set_xlabel("Distance (nr.of datapoints)");
    ax1[0, 0].set_ylabel("Intensity (a.u.)")
    ax1[1, 0].set_xlabel("Distance (nr.of datapoints)");
    ax1[1, 0].set_ylabel("Amplitude (a.u.)")
    ax1[0, 1].set_xlabel("Distance (um)");
    ax1[0, 1].set_ylabel("Height profile (nm)")
    fig1.set_size_inches(12.8, 9.6)
    return ax1, fig1


def CA_analysisRoutine(path, wavelength_laser=520):
    imgFolderPath, conversionZ, conversionXY, unitZ, unitXY = filePathsFunction(path, wavelength_laser)
    analysisFolder = os.path.join(imgFolderPath, "Analysis CA Spatial")
    contactAngleListFilePath = os.path.join(analysisFolder, "ContactAngle_MedianListFile.txt")

    ndata= []; tdata=[]; CAdata= []; forcedata = []
    f = open(contactAngleListFilePath, 'r')
    lines = f.readlines()
    for line in lines[1:]:
        data = line.split(',')
        ndata.append(int(data[0]))
        tdata.append(float(data[1])/60)
        CAdata.append(float(data[2]))
        forcedata.append(float(data[3]))

    fig1, ax1 = plt.subplots()
    ax1.plot(tdata, CAdata, '.')
    ax1.set_ylabel('Median Contact Angle (deg)')
    ax1.set_xlabel('Time passed (min)')
    ax1.set_title('Evolution of median contact angle entire droplet')
    fig1.savefig(os.path.join(analysisFolder, f'Median contact angle vs Time.png'), dpi=600)
    plt.show()
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(tdata, forcedata, '.')
    ax2.set_ylabel('Total Horizontal component force (mN)')
    ax2.set_xlabel('Time passed (min)')
    ax2.set_title('Evolution of Total Horizontal component force entire droplet')
    fig2.savefig(os.path.join(analysisFolder, f'Horizontal component force vs Time.png'), dpi=600)
    plt.show()
    plt.close(fig2)



# light wavelength in nm, INT.
# ZEISS: 520nm, Chroma ET520/10X dia18mm 10FWHM
# NIKON: 532nm, Thorlabs FLH532-10 dia25mm 10FWHM

def main():
    # procStatsJsonPath = os.path.join("D:\\2023_08_07_PLMA_Basler5x_dodecane_1_28_S5_WEDGE_1coverslip spacer_COVERED_SIDE\Analysis_1\PROC_20230809115938\PROC_20230809115938_statistics.json")
    # procStatsJsonPath = os.path.join("D:\\2023_09_22_PLMA_Basler2x_hexadecane_1_29S2_split\\B_Analysis\\PROC_20230927135916_imbed", "PROC_20230927135916_statistics.json")
    # imgFolderPath = os.path.dirname(os.path.dirname(os.path.dirname(procStatsJsonPath)))
    # path = os.path.join("G:\\2023_08_07_PLMA_Basler5x_dodecane_1_28_S5_WEDGE_1coverslip spacer_COVERED_SIDE\Analysis_1\PROC_20230809115938\PROC_20230809115938_statistics.json")

    path = "D:\\2023_11_13_PLMA_Dodecane_Basler5x_Xp_1_24S11los_misschien_WEDGE_v2" #outwardsLengthVector=[590]

    #path = "D:\\2023_07_21_PLMA_Basler2x_dodecane_1_29_S1_WEDGE_1coverslip spacer_____MOVEMENT"
    #path = "D:\\2023_11_27_PLMA_Basler10x_and5x_dodecane_1_28_S2_WEDGE\\10x"
    #path = "D:\\2023_12_08_PLMA_Basler5x_dodecane_1_28_S2_FULLCOVER"
    #path = "H:\\2023_12_12_PLMA_Dodecane_Basler5x_Xp_1_28_S2_FULLCOVER"
    #path = "H:\\2023_12_15_PLMA_Basler5x_dodecane_1_28_S2_WEDGE_Tilted"

    # path = "D:\\2023_08_07_PLMA_Basler5x_dodecane_1_28_S5_WEDGE_1coverslip spacer_AIR_SIDE"
    # path = "E:\\2023_10_31_PLMA_Dodecane_Basler5x_Xp_1_28_S5_WEDGE"
    # path = "F:\\2023_10_31_PLMA_Dodecane_Basler5x_Xp_1_29_S1_FullDropletInFocus"
    # path = "D:\\2023_11_27_PLMA_Basler10x_and5x_dodecane_1_28_S2_WEDGE"

    #path = "D:\\2024_02_05_PLMA 160nm_Basler17uc_Zeiss5x_dodecane_FULLCOVER_v2____GOOD"
    #path = "D:\\2024_02_05_PLMA 160nm_Basler17uc_Zeiss5x_dodecane_WEDGE_v2"

    #New P12MA dataset from 2024/05/07
    #path = "H:\\2024_05_07_PLMA_Basler15uc_Zeiss5x_dodecane_Xp1_31_S1_WEDGE_2coverslip_spacer_V4"
    #path = "H:\\2024_05_07_PLMA_Basler15uc_Zeiss5x_dodecane_Xp1_31_S1_WEDGE_Si_spacer"      #Si spacer, so doesn't move far. But for sure img 29 is pinning free
    #path = "H:\\2024_05_07_PLMA_Basler15uc_Zeiss5x_dodecane_Xp1_31_S2_WEDGE_2coverslip_spacer_V3"
    #path = "D:\\2024_05_17_PLMA_180nm_hexadecane_Basler15uc_Zeiss5x_Xp1_31_S3_v3FLAT_COVERED"
    #path = "D:\\2024_05_17_PLMA_180nm_dodecane_Basler15uc_Zeiss5x_Xp1_31_S3_v1FLAT_COVERED"

    #path = "D:\\2024_05_17_PLMA_180nm_dodecane_Basler15uc_Zeiss5x_Xp1_31_S3_v1FLAT_COVERED"
    #path = "D:\\2023_12_12_PLMA_Dodecane_Basler5x_Xp_1_28_S2_FULLCOVER"

    #P12MA dodecane - tilted stage
    path = "D:\\2024-09-04 PLMA dodecane Xp1_31_2 ZeissBasler15uc 5x M3 tilted drop"
    path = "D:\\2024-09-04 PLMA dodecane Xp1_31_2 ZeissBasler15uc 5x M2 tilted drop"

    #PODMA on heating stage:
    #path = "E:\\2023_12_21_PODMA_hexadecane_BaslerInNikon10x_Xp2_3_S3_HaloTemp_29_5C_AndBeyond\\40C"
    #path = "E:\\2023_07_31_PODMA_Basler2x_dodecane_2_2_3_WEDGE_1coverslip spacer____MOVEMENT"

    #Zeiss = 520nm, Nikon=533nm
    primaryObtainCARoutine(path, wavelength_laser=520)
    #CA_analysisRoutine(path, wavelength_laser = 533)


if __name__ == "__main__":
    main()
    exit()