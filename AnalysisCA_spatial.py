import itertools
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
import statistics

from matplotlib.widgets import RectangleSelector
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interpolate
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

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
def get_normalsV4(x, y, L):
    # For each coordinate, fit with nearby points to a polynomial to better estimate the dx dy -> tangent
    # Take derivative of the polynomial to obtain tangent and use that one.
    x0arr = []; dyarr = []; y0arr = []; dxarr = []
    window_size = 25            #!!! window size to use. 17 seems to be good for most normal vectors
    k = round((window_size+1)/2)

    middleCoordinate = [(max(x) + min(x))/2, (max(y) + min(y))/2]   #estimate for "middle coordinate" of contour. Will be used to direct normal vector to inside contour

    connectingCoordinatesDyDx = 30  #if coordinates are within 30 pixels of each other, probably they were connecting
    if abs(y[0]-y[-1]) < connectingCoordinatesDyDx and abs(x[0]-x[-1]) < connectingCoordinatesDyDx:   #if ends of contours are connecting, allow all the points also from the other end to be used for fitting
        x = x[-25:] + x + x[:25]
        y = np.hstack((y[-25:], y, y[:25]))

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

    vector = [[dxarr[i] - x0arr[i], dyarr[i] - y0arr[i]] for i in range(0, len(x0arr))]   #vector [dx, dy] for each coordinate
    return x0arr, dxarr, y0arr, dyarr, vector  # return the original data points x0&y0, and the coords of the normals dx&dy


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

def selectAreaAndFindContour(resizedimg, thresholdSensitivity):
    tempimg = []
    copyImg = resizedimg.copy()
    #tempimg = cv2.polylines(copyImg, np.array([contourList[i]]), False, (255, 0, 0), 8)  # draws 1 blue contour with the x0&y0arrs obtained from get_normals
    #if combineMultipleContours:
    #    tempimg = cv2.polylines(tempimg, np.array([contour]), False, (255, 0, 0), 8)  # draws 1 blue contour with the x0&y0arrs obtained from get_normals
    resizeFactor = [round(5328 / 5), round(4608 / 5)]
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
    adjustedContourList = []
    for contour in contourList:
        adjustedContourList.append([np.array([[elem[0][0] + P1[0], elem[0][1]+P1[1]]]) for elem in contour])

    return adjustedContourList


# Attempting to get a contour from the full-sized HQ image, and using resizefactor only for showing a copmressed image so it fits in the screen
# Parses all 'outer' coordinates, not only on right side of droplet
#With working popup box for checking and changing contour
def getContourCoordsV4(imgPath, contourListFilePath, n, contouri, thresholdSensitivity, MANUALPICKING, contourCoords = 0):
    minimalDifferenceValue = 100    #Value of minimal difference in x1 and x2 at same y-coord to check for when differentiating between entire droplet & partial contour & fish-hook-like contour
    img = cv2.imread(imgPath)  # read in image
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to greyscale
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
    #show img w/ contour to check if it is the correct one
    #make popup box to show next contour (or previous) if desired
    while goodContour == False:
        if len(contourList) > 0:
            tempimg = []
            copyImg = resizedimg.copy()
            tempimg = cv2.polylines(copyImg, np.array([contourList[i]]), False, (255, 0, 0), 8)  # draws 1 blue contour with the x0&y0arrs obtained from get_normals
            if combineMultipleContours:
                tempimg = cv2.polylines(tempimg, np.array([contour]), False, (255, 0, 0), 8)  # draws 1 blue contour with the x0&y0arrs obtained from get_normals
            tempimg = cv2.resize(tempimg, [round(5328 / 5), round(4608 / 5)], interpolation=cv2.INTER_AREA)  # resize image
            cv2.imshow(f"Contour img with current selection of contour {i+1} out of {nrOfContoursToShow}", tempimg)
            choices = ["One contour outwards (-i)", "Current contour is fine", "One contour inwards (+1)",
                       "Stitch multiple contours together: first selection",
                       "No good contours: Ajdust threshold sensitivities", "No good contours: quit programn",
                       "EXPERIMENTAL: Drawing box in which contour MUST be found (in case it never finds it there)"]
            myvar = easygui.choicebox("Is this a desired contour?", choices=choices)
        else:
            choices = ["One contour outwards (-i)", "Current contour is fine", "One contour inwards (+1)",
                       "Stitch multiple contours together: first selection",
                       "No good contours: Ajdust threshold sensitivities", "No good contours: quit programn",
                       "EXPERIMENTAL: Drawing box in which contour MUST be found (in case it never finds it there)"]
            myvar = easygui.choicebox("From the get-go, no contours were obtained with this threshold sensitivity. Choose option 5 to change this.", choices=choices)
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
            contourList = selectAreaAndFindContour(grayImg, thresholdSensitivity)
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
                usableContour = usableContourMax + usableContourMin[::-1]
                # TODO not sure if this works properly: meant to concate the coords of a partial contour such that the coords are on a 'smooth partial ellips' without a gap
                for ii in range(0, len(usableContour) - 1):
                        if abs(usableContour[ii][1] - usableContour[ii + 1][1]) > 20:  # if difference in y between 2 appending coords is large, a gap in the contour is formed
                            usableContour = usableContour[ii:] + usableContour[0:ii]
                            print("The order of coords has been changed")

                useableylist = np.array([elem[1] for elem in usableContour])
                useablexlist = [elem[0] for elem in usableContour]

            else:   #far spaced x-values: probably contour of an entire droplet: take the min & max at every y-coordinate
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
                        usableContour = usableContour[ii:] + usableContour[0:ii]        #shift coordinates in list such that the coordinates are sequential neighbouring
                useableylist = np.array([elem[1] for elem in usableContour])
                useablexlist = [elem[0] for elem in usableContour]



        else:   #if only 1 value
            print(f"For now something seems to be weird. Either bad contour, or accidentally only 1 real Y at that level."
                  f"Might therefore be a min/max in image, or the contour at the other side somehow skipped 1 Y-value (if this happens, implement new code)")
            exit()

        if contouri[0] < 0:  # Save which contour & thresholdSensitivity is used to a txt file for this n, for ease of further iterations
            file = open(contourListFilePath, 'a')
            if len(iout) > 1:
                file.write(f"{n}; {','.join(map(str, iout))}; {thresholdSensitivity[0]}; {thresholdSensitivity[1]}\n")
            else:
                file.write(f"{n}; {iout[0]}; {thresholdSensitivity[0]}; {thresholdSensitivity[1]}\n")
            file.close()

    return useablexlist, useableylist, usableContour, resizedimg, greyresizedimg


def importConversionFactors(procStatsJsonPath):
    with open(procStatsJsonPath, 'r') as f:
        procStats = json.load(f)
    conversionZ = procStats["conversionFactorZ"]
    conversionXY = procStats["conversionFactorXY"]
    unitZ = procStats["unitZ"]
    unitXY = procStats["unitXY"]
    #lensUsed = procStats['UsedLens']
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
    :param filenames_fullpath: list with full path the filnames of which the times are to be obtained.
    :return: absolute timestamps
    :return: difference in time (seconds) between two sequential images. First image = 0
    :return: difference in time (seconds) between first image and image(n).
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
    #TODO: make it so that if error is too large for linear fit, a NaN is return instead of a completely bogus CA
    minimalnNrOfDatapoints = round(len(yarr) * (2/4))
    residualLastTime = 10000        #some arbitrary high value to have the first iteration work
    for i in range(0, len(yarr)-minimalnNrOfDatapoints):    #iterate over the first 2/4 of the datapoints as a starting value
        coef1, residuals, _, _, _ = np.polyfit(xarr[i:], yarr[i:], 1, full=True)
        startLinRegimeIndex = i
        if not (residualLastTime/(len(yarr)-i+1) - residuals[0]/(len(yarr)-i)) / (residuals[0]/(len(yarr)-i)) > 0.05:    #if difference between
            break
        residualLastTime = residuals
    if i == (len(yarr)-minimalnNrOfDatapoints-1):
        print(f"Apparently no the difference in squared sum differs a lot for all 2/4th first datapoints. "
              f"\nTherefore the data is fitted from 2/4th of the data onwards.")
    r2 = r2_score(yarr, np.poly1d(coef1)(xarr))
    #if r2 < sensitivityR2:
    #    print(f"{k}: R^2 of fit is worse than {sensitivityR2}, namely: {r2:.4f}. This fit is not to be trusted")
    return startLinRegimeIndex, coef1, r2

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
    ax10.set_xlabel("X-coord"); ax10.set_ylabel("Y-Coord"); ax10.set_title(f"Spatial Nett Forces ({unitST}) Colormap")
    ax10.legend([f"Median Nett Force: {(statistics.median(nettforces)):.2f} {unitST}"], loc='center left')
    fig10.colorbar(im10, format="%.5f")
    #plt.show()
    fig10.savefig(os.path.join(analysisFolderPath, 'Spatial Nett Force.png'), dpi=600)
    plt.close(fig10)

    fig11, ax11 = plt.subplots()
    im11 = ax11.scatter(xcoords, ycoords, c=tangentforces, cmap='jet', vmin=min(tangentforces), vmax=max(tangentforces), label=f'Horizontal Line Force ({unitST})')
    ax11.set_xlabel("X-coord"); ax11.set_ylabel("Y-Coord"); ax11.set_title(f"Spatial Tangential Forces ({unitST}) Colormap")
    ax11.legend([f"Median Horizontal Component Force: {(statistics.median(tangentforces)):.2f} {unitST}"], loc='center left')
    fig11.colorbar(im11)
    #plt.show()
    fig11.savefig(os.path.join(analysisFolderPath, 'Spatial Tangential Force.png'), dpi=600)
    plt.close(fig11)

    print(f"Sum of Horizontal Components forces = {sum(tangentforces)} (compare with total (abs horizontal) = {sum(abs(np.array(tangentforces)))}")
    return tangentforces

#for 2 linear lines; y = ax + c & y= bx + d, their intersect is at (x,y) = {(d-c)/(a-b), (a*(d-c)/(a-b))+c}
def approxMiddlePointDroplet(coords, vectors):
    intersectCoordsX = []
    intersectCoordsY = []
    for i in itertools.chain(range(0, round(len(vectors)/4)), range(round(len(vectors)/2), round(len(vectors)*3/4))):
        a = vectors[i][1] / vectors[i][0]           #a=dy/dx
        b = vectors[i+round(len(vectors)/4)][1] / vectors[i+round(len(vectors)/4)][0]  # c=dy/dx    (of vector 1 quarter away from current one)
        c = coords[i][1] - (coords[i][0] * a)
        d = coords[i+round(len(vectors)/4)][1] - (coords[i+round(len(vectors)/4)][0] * b)
        intersectCoordsX.append(round((d - c) / (a - b)))
        intersectCoordsY.append(round((a * (d - c) / (a - b)) + c))
    meanmiddleX = np.mean(intersectCoordsX)
    meanmiddleY = np.mean(intersectCoordsY)
    return intersectCoordsX, intersectCoordsY, meanmiddleX, meanmiddleY

#TODO get this to work? : fitting CA = (x,y) to interpolate for missing datapoints & get total contour length for good force calculation
#part of OG code: https://www.geeksforgeeks.org/3d-curve-fitting-with-python/
def givemeZ(xin, yin, zin, xout, yout, conversionXY, analysisFolder, n):
    #tck = interpolate.bisplrep(xin, yin, zin, s=0)
    #f = scipy.interpolate.interp2d(xin, yin, zin, kind="cubic")
    # Define mathematical function for curve fitting
    yin = abs(np.subtract(4608, yin))       #flip y's for good plotting of data
    yout = abs(np.subtract(4608, yout))
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

def primaryObtainCARoutine(path):
    #blockSize	Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
    #C Constant subtracted from the mean or weighted mean.
    thresholdSensitivityStandard = [11 * 3, 3 * 5]  # [blocksize, C].   OG: 11 * 5, 2 * 5;     working better now = 11 * 3, 2 * 5

    imgFolderPath, conversionZ, conversionXY, unitZ, unitXY = filePathsFunction(path)

    imgList = [f for f in glob.glob(os.path.join(imgFolderPath, f"*tiff"))]
    everyHowManyImages = 10
    #usedImages = np.arange(10, len(imgList), everyHowManyImages)  # 200 is the working one
    usedImages = [120]
    analysisFolder = os.path.join(imgFolderPath, "Analysis CA Spatial")
    lengthVector = 200  # 200 length of normal vector over which intensity profile data is taken
    FLIPDATA = True
    SHOWPLOTS_SHORT = 1  # 0 Don't show plots&images at all; 1 = show images for only 2 seconds; 2 = remain open untill clicked away manually
    sensitivityR2 = 0.997    #sensitivity for the R^2 linear fit for calculating the CA. Generally, it is very good fitting (R^2>0.99)
    # MANUALPICKING:Manual (0/1):  0 = always pick manually. 1 = only manual picking if 'correct' contour has not been picked & saved manually before.
    # All Automatical(2/3): 2 = let programn pick contour after 1st manual pick (TODO: not advised, doesn't work properly yet). 3 = use known contour IF available, else automatically use the second most outer contour
    MANUALPICKING = 1
    lg_surfaceTension = 27     #surface tension hexadecane liquid-gas (N/m)
    if not os.path.exists(analysisFolder):
        os.mkdir(analysisFolder)
        print('created path: ', analysisFolder)
    contourListFilePath = os.path.join(analysisFolder, "ContourListFile.txt")
    contactAngleListFilePath = os.path.join(analysisFolder, "ContactAngle_MedianListFile.txt")
    if os.path.exists(
            contourListFilePath):  # read in all contourline data from existing file (filenr ,+ i for obtaining contour location)
        f = open(contourListFilePath, 'r')
        lines = f.readlines()
        importedContourListData_n, importedContourListData_i, importedContourListData_thresh = extractContourNumbersFromFile(lines)
        # importedContourListData = np.loadtxt(contourListFilePath, dtype='int', delimiter=',',  usecols=(0,1))
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
        f.write(f"file number (n), delta time from 0 (s), median CA (deg), Horizontal component force (mN)\n")
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
            # One of the main functions: outputs the coordinates of the desired contour of current image
            if n == usedImages[0] and MANUALPICKING != 0:  # on first iteration, don't parse previous coords (because there are none)
                useablexlist, useableylist, usableContour, resizedimg, greyresizedimg = getContourCoordsV4(img,
                                                                                                           contourListFilePath,
                                                                                                           n, contouri,
                                                                                                           thresholdSensitivity,
                                                                                                           MANUALPICKING)
            # TODO doesn't work as desired: now finds contour at location of previous one, but not the aout CL one. Incorporate offset somehow, or a check for periodicity of intensitypeaks
            elif n - usedImages[list(usedImages).index(
                    n) - 1] == everyHowManyImages and MANUALPICKING == 2:  # if using sequential images, use coordinates of previous image
                useablexlist, useableylist, usableContour, resizedimg, greyresizedimg = getContourCoordsV4(img,
                                                                                                           contourListFilePath,
                                                                                                           n, contouri,
                                                                                                           thresholdSensitivity,
                                                                                                           MANUALPICKING,
                                                                                                           usableContour)
            else:  # else, don't parse coordinates (let user define them themselves)
                useablexlist, useableylist, usableContour, resizedimg, greyresizedimg = getContourCoordsV4(img,
                                                                                                           contourListFilePath,
                                                                                                           n, contouri,
                                                                                                           thresholdSensitivity,
                                                                                                           MANUALPICKING)
            print(f"Contour succesfully obtained. Next: obtaining the normals of contour.")
            try:
                resizedimg = cv2.polylines(resizedimg, np.array([usableContour]), False, (0, 0, 255),
                                           2)  # draws 1 red good contour around the outer halo fringe

                # One of the main functions:
                # Should yield the normal for every point: output is original x&y, and corresponding normal x,y (defined as dx and dy)
                x0arr, dxarr, y0arr, dyarr, vectors = get_normalsV4(useablexlist, useableylist, lengthVector)
                print(f"Normals sucessfully obtained. Next: plot normals in image & obtain intensities over normals")
                tempcoords = [[x0arr[k], y0arr[k]] for k in range(0, len(x0arr))]
                for k in x0arr:     #Check for weird x or y values, THEY NEVER SHOULD BE NEGATIVE
                    if k < 0:
                        print(f"xval: {k}, index: {x0arr.index(k)}")
                for k in y0arr:
                    if k < 0:
                        print(f"yval {k}, index: {y0arr.index(k)}")

                tempimg = []
                tempimg = cv2.polylines(resizedimg, np.array([tempcoords]), False, (0, 255, 0),
                                        20)  # draws 1 blue contour with the x0&y0arrs obtained from get_normals
                tempimg = cv2.resize(tempimg, [round(5328 / 5), round(4608 / 5)],
                                     interpolation=cv2.INTER_AREA)  # resize image
                if SHOWPLOTS_SHORT > 0:
                    cv2.imshow( f"Contour of img {np.where(np.array(usedImages) == n)[0][0]} out of {len(usedImages)} with coordinates being used by get_normals", tempimg)
                    cv2.waitKey(2000)
                    cv2.destroyAllWindows()
                cv2.imwrite(os.path.join(analysisFolder, f"rawImage_x0y0Arr_blue{n}.png"), tempimg)

                angleDegArr = []
                xArrFinal = []
                yArrFinal = []
                vectorsFinal = []
                counter = 0

                for k in range(0, len(x0arr)):  # for every contour-coordinate value; plot the normal, determine intensity profile & calculate CA from the height profile
                    try:
                        # if k == 101:
                        #     print(f"at this k we break={k}")
                        # TODO trying to get this to work: plotting normals obtained with above function get_normals
                        # resizedimg = cv2.polylines(resizedimg, np.array([[x0arr[k], y0arr[k]], [dxarr[k], dyarr[k]]]), False, (0, 255, 0), 2)  # draws 1 good contour around the outer halo fringe#
                        if k % 25 == 0:  # only plot 1/25th of the vectors to not overcrowd the image
                            resizedimg = cv2.line(resizedimg, ([x0arr[k], y0arr[k]]), ([dxarr[k], dyarr[k]]), (0, 255, 0),
                                                  2)  # draws 1 good contour around the outer halo fringe

                        if dxarr[k] - x0arr[k] == 0:  # constant x, variable y
                            xarr = np.ones(lengthVector) * x0arr[k]
                            if y0arr[k] > dyarr[k]:
                                yarr = np.arange(dyarr[k], y0arr[k] + 1)
                            else:
                                yarr = np.arange(y0arr[k], dyarr[k] + 1)
                            coords = list(zip(xarr.astype(int), yarr.astype(int)))
                        else:
                            a = (dyarr[k] - y0arr[k]) / (dxarr[k] - x0arr[k])
                            b = y0arr[k] - a * x0arr[k]
                            coords, lineLengthPixels = coordinates_on_line(a, b,
                                                                           [x0arr[k], dxarr[k], y0arr[k], dyarr[k]])  #
                        profile = [np.transpose(greyresizedimg)[pnt] for pnt in coords]

                        profile_fft = np.fft.fft(profile)  # transform to fourier space
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
                        x = np.linspace(0, lineLengthPixels,
                                        len(unwrapped)) * conversionXY * 1000  # converts pixels to desired unit (prob. um)

                        # finds linear fit over most linear regime (read:excludes halo if contour was not picked ideally).
                        startIndex, coef1, r2 = linearFitLinearRegimeOnly(x, unwrapped, sensitivityR2, k)

                        if r2 > sensitivityR2:      #R^2 should be very high, otherwise probably e.g. near pinning point
                            a_horizontal = 0
                            angleRad = math.atan((coef1[0] - a_horizontal) / (1 + coef1[0] * a_horizontal))
                            angleDeg = math.degrees(angleRad)
                            if angleDeg > 45:  # Flip measured CA degree if higher than 45.
                                angleDeg = 90 - angleDeg
                            xArrFinal.append(x0arr[k])
                            yArrFinal.append(y0arr[k])
                            vectorsFinal.append(vectors[k])
                            angleDegArr.append(angleDeg)
                        else:
                            counter += 1    #TEMP: to check how many vectors should not be taken into account because the r2 value is too low

                        if k == round(
                                len(x0arr) / 2):  # plot 1 profile of each image with intensity, wrapped, height & resulting CA
                            fig1, ax1 = plt.subplots(2, 2)
                            ax1[0, 0].plot(profile);
                            ax1[1, 0].plot(wrapped);
                            ax1[1, 0].plot(peaks, wrapped[peaks], '.')
                            ax1[0, 0].set_title(f"Intensity profile");
                            ax1[1, 0].set_title("Wrapped profile")
                            # TODO unit unwrapped was in um, *1000 -> back in nm. unit x in um
                            ax1[0, 1].plot(x, unwrapped * 1000);
                            ax1[0, 1].plot(x[startIndex], unwrapped[startIndex] * 1000, 'r.');
                            ax1[0, 1].set_title("Drop height vs distance (unwrapped profile)")
                            ax1[0, 1].plot(x, np.poly1d(coef1)(x) * 1000, linewidth=0.5, label=f'R2={r2:.3f}\nCA={angleDeg:.2f} deg');
                            ax1[0, 1].legend(loc='best')
                            ax1[0, 0].set_xlabel("Distance (nr.of datapoints)");
                            ax1[0, 0].set_ylabel("Intensity (a.u.)")
                            ax1[1, 0].set_xlabel("Distance (nr.of datapoints)");
                            ax1[1, 0].set_ylabel("Amplitude (a.u.)")
                            ax1[0, 1].set_xlabel("Distance (um)");
                            ax1[0, 1].set_ylabel("Height profile (nm)")
                            fig1.set_size_inches(12.8, 9.6)

                        # if angleDeg < 1.8:
                        #     print("we pausin'")
                        #     plt.plot(x, unwrapped * 1000);
                        #     plt.title("For deg< 1.8: Drop height vs distance (unwrapped profile)")
                        #     plt.plot(x, np.poly1d(coef1)(x) * 1000, linewidth=0.5)
                        #     plt.legend([f'R2={r2}'])
                        #     plt.show()
                    except:
                        logging.info(f"!{k}: Analysing each coordinate & normal vector broke!")
                print(f"Normals, intensities & Contact Angles Succesffuly obtained. Next: plotting overview of all data for 1 timestep")
                print(f"Out of {len(x0arr)}, {counter} number of vectors were omitted because the R^2 was too low.")

                #calculate the nett force over given CA en droplet range
                forces = CA_And_Coords_To_Force(xArrFinal, abs(np.subtract(4608, yArrFinal)), vectorsFinal, angleDegArr, analysisFolder, lg_surfaceTension)

                #determine middle of droplet & plot
                middleX, middleY,meanmiddleX, meanmiddleY = approxMiddlePointDroplet(list(zip(xArrFinal, yArrFinal)), vectorsFinal)
                fig2, ax2 = plt.subplots()
                ax2.plot(middleX, middleY, 'b.', label='intersects of normal vectors')
                ax2.plot(xArrFinal, yArrFinal, 'r', label='contour of droplet')
                ax2.plot(meanmiddleX, meanmiddleY, 'k.', markersize=20, label='average middle coordinate')
                ax2.set_xlabel('X-coords'); ax2.set_ylabel('Y-coords')
                ax2.legend(loc='best')
                print(f"meanX = {meanmiddleX}, meanY:{meanmiddleY}")
                fig2.savefig(os.path.join(analysisFolder, f'Middle of droplet {n:04}.png'), dpi=600)
                #plt.show()
                plt.close(fig2)

                ##TODO Plot CA vs. radial angle
                dx = np.subtract(xArrFinal, meanmiddleX)
                dy = np.subtract(yArrFinal, meanmiddleY)
                phi = np.arctan2(dy, dx)      #angle of contour coordinate w/ respect to 12o'clock (radians)
                phi = 0.5 * np.pi - phi
                phi = np.mod(phi, (2.0 * np.pi))    #converting atan2 to normal radians: https://stackoverflow.com/questions/17574424/how-to-use-atan2-in-combination-with-other-radian-angle-systems
                fig4, ax4 = plt.subplots()

                azimuthalX = np.sin(phi)
                rightFromMiddle = azimuthalX[np.where(phi < np.pi)]
                leftFromMiddle = azimuthalX[np.invert(np.where(phi < np.pi)[0])]
                ax4.plot(rightFromMiddle, np.array(angleDegArr)[np.where(phi < np.pi)], '.', label='right side')
                ax4.plot(leftFromMiddle, np.array(angleDegArr)[np.invert(np.where(phi < np.pi)[0])], '.', label='left side')

                ax4.set_xlabel('sin(\phi)'); ax4.set_ylabel('contact angle (deg)')
                ax4.legend(loc='best')
                fig4.savefig(os.path.join(analysisFolder, f'Azimuthal contact angle {n:04}.png'), dpi=600)
                plt.close(fig4)

                #TODO trying to fit the CA contour in 3D, to integrate etc. for force calculation
                Z, totalZ = givemeZ(np.array(xArrFinal), np.array(yArrFinal), forces, np.array(x0arr), np.array(y0arr), conversionXY, analysisFolder, n)
                totalForce_afo_time.append(totalZ)
                #fig5 = plt.figure()
                #ax5 = fig5.add_subplot(1, 1, 1, projection='3d')
                #ax5.plot_surface(x0arr, y0arr, fittedCA, color='r')
                #ax5.set_xlabel("X-coords"); ax5.set_ylabel("Y_Coords"); ax5.set_zlabel("Contact Angle")
                #plt.show()

                angleDeg_afo_time.append(statistics.median(angleDegArr))
                usedDeltaTs.append(deltatFromZeroSeconds[n])    #list with delta t (IN SECONDS) for only the USED IMAGES
                # Fit an ellipse around the contour
                # ellipse = cv2.fitEllipse(contour)
                # Draw the ellipse on the original frame
                # resizedimg = cv2.ellipse(resizedimg, ellipse, (0, 255, 0), 2)                #draws an ellips, which fits poorly
                # get the middlepoint of the contour and draw a circle in it
                # M = cv2.moments(contour)
                # cx = int(M["m10"] / M["m00"])
                # cy = int(M["m01"] / M["m00"])
                # resizedimg = cv2.circle(resizedimg, (cx, cy), 13, (255, 0, 0), -1)           #draws a white circle
                # cx_save.append(cx)
                # cy_save.append(cy)

                fig3, ax3 = plt.subplots()
                im3 = ax3.scatter(xArrFinal, abs(np.subtract(4608, yArrFinal)), c=angleDegArr, cmap='jet',
                                  vmin=min(angleDegArr), vmax=max(angleDegArr))
                ax3.set_xlabel("X-coord"); ax3.set_ylabel("Y-Coord"); ax3.set_title(f"Spatial Contact Angles Colormap n = {n}, or t = {deltat_formatted[n]}")
                ax3.legend([f"Median CA (deg): {(statistics.median(angleDegArr)):.2f}"], loc='center left')
                fig3.colorbar(im3)
                fig3.savefig(os.path.join(analysisFolder, f'Colorplot XYcoord-CA {n:04}.png'), dpi=600)
                plt.close()

                im1 = ax1[1, 1].scatter(xArrFinal, abs(np.subtract(4608, yArrFinal)), c=angleDegArr, cmap='jet',
                                        vmin=min(angleDegArr), vmax=max(angleDegArr))
                ax1[1, 1].set_xlabel("X-coord"); ax1[1, 1].set_ylabel("Y-Coord"); ax1[1, 1].set_title(f"Spatial Contact Angles Colormap n = {n}, or t = {deltat_formatted[n]}")
                ax1[1, 1].legend([f"Median CA (deg): {(statistics.median(angleDegArr)):.2f}"], loc='center left')
                fig1.colorbar(im1)
                fig1.savefig(os.path.join(analysisFolder, f'Complete overview {n:04}.png'), dpi=600)
                if SHOWPLOTS_SHORT == 1:
                    plt.show(block=False)
                    plt.pause(2)
                    plt.close()
                elif SHOWPLOTS_SHORT == 2:
                    fig1.show()
                    plt.close(fig1)
                    fig3.show()
                    plt.close(fig3)


                # Export Contact Angles to a csv file & add median CA to txt file
                wrappedPath = os.path.join(analysisFolder, f"ContactAngleData {n}.csv")
                d = dict({'x-coords': xArrFinal, 'y-coords': yArrFinal, 'contactAngle': angleDegArr})
                df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))  # pad shorter colums with NaN's
                df.to_csv(wrappedPath, index=False)

                if n not in ndata:  # if not already saved median CA, save to txt file.
                    CAfile = open(contactAngleListFilePath, 'a')
                    CAfile.write(f"{n}, {usedDeltaTs[-1]}, {angleDeg_afo_time[-1]}, {totalForce_afo_time[-1]}\n")
                    CAfile.close()

            except:
                print(f"Some error occured. Still plotting obtained contour")
            tstring = str(
                datetime.now().strftime("%Y_%m_%d"))  # day&timestring, to put into a filename    "%Y_%m_%d_%H_%M_%S"
            cv2.imwrite(os.path.join(analysisFolder, f"rawImage_contourLine_{tstring}_{n}.png"), resizedimg)
    fig2, ax2 = plt.subplots()
    ax2.plot(np.divide(usedDeltaTs, 60), totalForce_afo_time)
    ax2.set_xlabel("Time (minutes)"); ax2.set_ylabel("Horizontal component force (mN)"); ax2.set_title("Horizontal component force over Time")
    fig2.savefig(os.path.join(analysisFolder, f'Horizontal component force vs Time.png'), dpi=600)
    plt.close()


def CA_analysisRoutine(path):
    imgFolderPath, conversionZ, conversionXY, unitZ, unitXY = filePathsFunction(path)
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
def main():
    # procStatsJsonPath = os.path.join("D:\\2023_08_07_PLMA_Basler5x_dodecane_1_28_S5_WEDGE_1coverslip spacer_COVERED_SIDE\Analysis_1\PROC_20230809115938\PROC_20230809115938_statistics.json")
    # procStatsJsonPath = os.path.join("D:\\2023_09_22_PLMA_Basler2x_hexadecane_1_29S2_split\\B_Analysis\\PROC_20230927135916_imbed", "PROC_20230927135916_statistics.json")
    # imgFolderPath = os.path.dirname(os.path.dirname(os.path.dirname(procStatsJsonPath)))
    # path = os.path.join("G:\\2023_08_07_PLMA_Basler5x_dodecane_1_28_S5_WEDGE_1coverslip spacer_COVERED_SIDE\Analysis_1\PROC_20230809115938\PROC_20230809115938_statistics.json")
    #path = "E:\\2023_11_13_PLMA_Dodecane_Basler5x_Xp_1_24S11los_misschien_WEDGE_v2"
    #path = "D:\\2023_07_21_PLMA_Basler2x_dodecane_1_29_S1_WEDGE_1coverslip spacer_____MOVEMENT"
    #path = "D:\\2023_11_27_PLMA_Basler10x_and5x_dodecane_1_28_S2_WEDGE\\10x"
    #path = "D:\\2023_12_08_PLMA_Basler5x_dodecane_1_28_S2_FULLCOVER"
    #path = "E:\\2023_12_12_PLMA_Dodecane_Basler5x_Xp_1_28_S2_FULLCOVER"
    path = "G:\\2023_12_15_PLMA_Basler5x_dodecane_1_28_S2_WEDGE_Tilted"

    # path = "D:\\2023_08_07_PLMA_Basler5x_dodecane_1_28_S5_WEDGE_1coverslip spacer_AIR_SIDE"
    # path = "E:\\2023_10_31_PLMA_Dodecane_Basler5x_Xp_1_28_S5_WEDGE"
    # path = "F:\\2023_10_31_PLMA_Dodecane_Basler5x_Xp_1_29_S1_FullDropletInFocus"
    # path = "D:\\2023_11_27_PLMA_Basler10x_and5x_dodecane_1_28_S2_WEDGE"


    primaryObtainCARoutine(path)
    #CA_analysisRoutine(path)


if __name__ == "__main__":
    main()
    exit()