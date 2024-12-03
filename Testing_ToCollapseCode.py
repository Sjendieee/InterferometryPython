import itertools
import os.path
import pickle
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
import traceback
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import matplotlib
import time
from sklearn.linear_model import LinearRegression

matplotlib.use('TkAgg')         #to view plots in debugger mode. Might differ per device to work 'QtAgg'  'TkAgg'
import matplotlib.animation as animation
from matplotlib.widgets import RectangleSelector
#from matplotlib.animation import PillowWriter
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits import mplot3d

from scipy import integrate
from scipy.interpolate import interpolate
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from SwellingFromAbsoluteIntensity import heightFromIntensityProfileV2
from IntensityToSwellingKnownPeakLocation import heightFromIntensityProfileV3

__author__ = "Sander Reuvekamp"
__version__ = "2.0"

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

    Note: line MUST intersect the image edges, otherwise it throws an error.
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


def get_normalsV4(x, y, L, L2 = 0, L3 = 0):
    """
    For a dataset of x & y coordinates, in which the x&y are already ordened, determine the x&y coordinates at a given distance L, L2 and L3 normal to the given coords
    by fitting a polynomial through neigbouring coordinates.
    Return lists of corresponding coordinates
    :param x: xcoords of contour
    :param y: ycoords of contour
    :param L: desired length of normal vector in pixels (determines how many fringes will be accounted for later on)
    :param L2: normal vector in opposite direction of L (for positive L2 values). If 0, not calculated
    :param L3: normal vector in opposite direction of L (for positive L3 values). If 0, not calculated
    :return all: lists of the original data points x0&y0, and the coords of the inwards normals dx&dy, and the outwards normals dxneg&dyneg
    """
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


def move_insert_k_half_data(data_k_half, k_half_unfiltered, x0arr, dxarr, y0arr, dyarr, vectors, dxnegarr, dynegarr, dxExtraOutarr, dyExtraOutarr):
    """
    Shift the (in a previous run obtained) saved data of index k_half_unfiltered, extracted from 'data_k_half', to index k_half_unfiltered in new lists with all the other data
    The saved data consists of:
    -coord (x,y) contact line = x0arr, y0arr
    -coord (x,y) end-of-line inside drop (drop profile->contact angle) = dxarr, dyarr
    -coord (x,y) end-of-line outside drop (swelling profile) = dxnegarr, dynegarr,
    -coords (x,y) end-of-line extra bit outside droplet (for extra wrapped calculation) = dxExtraOutarr, dyExtraOutarr
    """
    x0arr.insert(k_half_unfiltered, data_k_half[0])
    dxarr.insert(k_half_unfiltered, data_k_half[1])
    y0arr.insert(k_half_unfiltered, data_k_half[2])
    dyarr.insert(k_half_unfiltered, data_k_half[3])
    vectors.insert(k_half_unfiltered, data_k_half[4])
    dxnegarr.insert(k_half_unfiltered, data_k_half[5])
    dynegarr.insert(k_half_unfiltered, data_k_half[6])
    dxExtraOutarr.insert(k_half_unfiltered, data_k_half[7])
    dyExtraOutarr.insert(k_half_unfiltered, data_k_half[8])

    logging.info(f"Shifted k_half data of OG run to index k_half, to ensure the programn finds the correct 'MinAndMaximaHandpicked ... .txt' file later.")
    return x0arr, dxarr, y0arr, dyarr, vectors, dxnegarr, dynegarr, dxExtraOutarr, dyExtraOutarr


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
            msg = (f"Inputted threshold sensitivity didn't work! Input new.\nCurrent threshold sensitivity is: {thresholdSensitivity}. Change to (comma seperated):\n"
                   f"1 BlockSize: Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on. "
                   f"2 Constant subtracted from the mean or weighted mean.")
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


# Attempting to get a contour from the full-sized HQ image, and using resizefactor only for showing a compressed image so it fits in the screen
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
            choices = ["One contour outwards (-i)",
                       "Current contour is fine",
                       "One contour inwards (+1)",
                       "Stitch multiple contours together: first selection",
                       "No good contours: Ajdust threshold sensitivities",
                       "No good contours: quit programn",
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
                # concate the coords of a partial contour such that the coords are on a 'smooth partial ellips' without a gap
                for ii in range(windowSizePolynomialCheck, len(usableContourCopy)-(windowSizePolynomialCheck)):
                    FITXGAP = False
                    FITYGAP = False
                    #Check for gaps, and determine if it's bigger in x or y direction. Fit for the biggest
                    XGAP = abs(usableContourCopy[ii][0] - usableContourCopy[ii + 1][0])     #determine x-distance between adjacent indices
                    YGAP = abs(usableContourCopy[ii][1] - usableContourCopy[ii + 1][1])     #determine y-distance between adjacent indices
                    if XGAP > 20 or YGAP > 20:
                        if YGAP > XGAP:
                            FITYGAP = True
                        else:
                            FITXGAP = True

                    # TODO fit for gaps in x axis
                    if FITXGAP:      #if difference in x between 2 appending coords is large, a horizontal gap in the contour is formed
                        xrange_for_fitting = [i[0] for i in usableContourCopy[(ii-windowSizePolynomialCheck):(ii+windowSizePolynomialCheck)]]
                        yrange_for_fitting = [i[1] for i in usableContourCopy[(ii - windowSizePolynomialCheck):(ii + windowSizePolynomialCheck)]]
                        # xrange_for_fitting = usableContour[(ii-windowSizePolynomialCheck):(ii+windowSizePolynomialCheck)][0] #to fit polynomial with 30 points on both sides of the gap   #todo gaat fout als ii<30 of > (len()-30)
                        # yrange_for_fitting = usableContour[ii-windowSizePolynomialCheck:ii+windowSizePolynomialCheck][1]

                        #TODO commented below: trying other step
                        # if usableContourCopy[ii][0] < usableContourCopy[ii + 1][0]:    #find if x is increasing
                        #     x_values_to_be_fitted = np.arange(usableContourCopy[ii][0]+1, usableContourCopy[ii + 1][0]-1, 1)
                        # else:
                        #     x_values_to_be_fitted = np.arange(usableContourCopy[ii + 1][0]+1, usableContourCopy[ii][0]-1, 1)
                        if usableContourCopy[ii][0] < usableContourCopy[ii + 1][0]:    #find if x is increasing
                            x_values_to_be_fitted = np.arange(usableContourCopy[ii][0]+1, usableContourCopy[ii + 1][0]-1, 1)
                        else:
                            x_values_to_be_fitted = np.arange(usableContourCopy[ii][0]-1, usableContourCopy[ii + 1][0]+1, -1)

                        localfit = np.polyfit(xrange_for_fitting, yrange_for_fitting, 2)    #horizontal gap: fit y(x)
                        y_fitted = np.poly1d(localfit)(x_values_to_be_fitted).astype(int)
                        usableContourCopy_instertion = np.insert(usableContourCopy_instertion, ii+ii_inserted+1, list(map(list, zip(x_values_to_be_fitted, y_fitted))), axis=0)
                        ii_inserted+=len(x_values_to_be_fitted) #offset index of insertion by length of previous arrays which were inserted
                        plt.plot(xrange_for_fitting, yrange_for_fitting, '.', label='x-gap data')
                        plt.plot(x_values_to_be_fitted, y_fitted, label='x-gap fit')
                        plt.legend(loc='best')
                        #plt.show()

                    #TODO fit for gaps in y-direction
                    #TODO THIS IS STILL WRONG !!!
                    elif FITYGAP:      #if difference in y between 2 appending coords is large, a vertical gap in the contour is formed
                        # xrange_for_fitting = usableContour[ii-windowSizePolynomialCheck:ii+windowSizePolynomialCheck][0] #to fit polynomial with 30 points on both sides of the gap   #todo gaat fout als ii<30 of > (len()-30)
                        # yrange_for_fitting = usableContour[ii-windowSizePolynomialCheck:ii+windowSizePolynomialCheck][1]
                        xrange_for_fitting = [i[0] for i in usableContourCopy[(ii-windowSizePolynomialCheck):(ii+windowSizePolynomialCheck)]]
                        yrange_for_fitting = [i[1] for i in usableContourCopy[(ii - windowSizePolynomialCheck):(ii + windowSizePolynomialCheck)]]
                        # TODO commented below: trying other step
                        # if usableContourCopy[ii][1] < usableContourCopy[ii + 1][1]:    #find if y is increasing
                        #     y_values_to_be_fitted = np.arange(usableContourCopy[ii][1]+1, usableContourCopy[ii+1][1]-1, 1)
                        # else:
                        #     y_values_to_be_fitted = np.arange(usableContourCopy[ii+1][1]+1, usableContourCopy[ii][1]-1, 1)
                        if usableContourCopy[ii][1] < usableContourCopy[ii + 1][1]:    #find if y is increasing
                            y_values_to_be_fitted = np.arange(usableContourCopy[ii][1]+1, usableContourCopy[ii+1][1]-1, 1)
                        else:   #if decreasing, step = -1
                            y_values_to_be_fitted = np.arange(usableContourCopy[ii][1]-1, usableContourCopy[ii+1][1]+1, -1)

                        localfit = np.polyfit(yrange_for_fitting, xrange_for_fitting, 2)    #horizontal gap: fit x(y)
                        x_fitted = np.poly1d(localfit)(y_values_to_be_fitted).astype(int)
                        usableContourCopy_instertion = np.insert(usableContourCopy_instertion, ii+ii_inserted+1, list(map(list, zip(x_fitted, y_values_to_be_fitted))), axis=0)
                        ii_inserted+=len(y_values_to_be_fitted) #offset index of insertion by length of array which was just inserted
                        plt.plot(xrange_for_fitting, yrange_for_fitting, '.', label='y-gap data')
                        plt.plot(x_fitted, y_values_to_be_fitted, label='y-gap fit')
                        plt.legend(loc='best')
                        #plt.show()
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
                        logging.info(f"Polynomial fits inserted. NEW length of coords_arr = {len(usableContourCopy_instertion)} vs OLD length = {len(usableContour)}")
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
            logging.info(f"SAVED threshold info to {os.path.split(contourListFilePath)[1]}")

        if saveCoordinates == True:
            if os.path.exists(os.path.dirname(coordinatesListFilePath)):
                with open(coordinatesListFilePath, 'wb') as internal_filename:
                    pickle.dump(usableContour, internal_filename)
                logging.info(f"DUMPED contour X-Y (len={len(usableContour)}) coordinates to pickle file: {os.path.split(coordinatesListFilePath)[0]}")
            else:
                logging.error("Path to folder in which the contour coordinates file is to be saved DOES NOT exist.\n"
                              "When parsing 'saveCoordinates' = True, make sure 'coordinatesListFilePath' is parsed (correctly) as well")

    return useablexlist, useableylist, usableContour, resizedimg, greyresizedimg, thresholdSensitivity

def getContourCoordsFromDatafile(imgPath, coordinatesListFilePath, FITGAPS_POLYOMIAL = False):
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
                plt.plot(x_values_to_be_fitted, y_fitted, '*',  label='x-gap fit')
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
                plt.plot(x_fitted, y_values_to_be_fitted, '*', label='y-gap fit')
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
    """

    :param path:
    :param wavelength_laser:
    :param refr_index:
    :return:
    """
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

    choices = ["ZEISS_OLYMPUSX2", "ZEISS_ZEISSX5", "ZEISS_ZEISSX10", "SR_NIKON_NIKONX10_PIEZO", "EnqL_ZEISSX20"]
    answer = easygui.choicebox("What lens preset was used?", choices=choices)
    if answer == choices[0]:
        preset = 672
    elif answer == choices[1]:
        preset = 1836
    elif answer == choices[2]:
        preset = 3695
    elif answer == choices[3]:
        preset = 3662
    elif answer == choices[4]:
        preset = 7360
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


#TODO fix dit (NOT WORKING IDEALLY/EVEN PROPERLY NOW): try to get linear fitting to work from the inside backing up for the startindex.
def linearFitLinearRegimeOnly_wPointsOutsideDrop_v3(xarr, yarr, sensitivityR2, k, lenArrOutsideDrop):
    """
    Version 3 of 'linearFitLinearRegimeOnly()'
    Try fitting in linear regime of droplet shape for contact angle analysis.
    Fit in range from end backwards (iun case of artifacts there),
    and for beginning part: iterate backwards from +- 1/4 of droplet regime. Stop when deviating from desired R^2.
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
    # with open(os.path.join(os.getcwd(), "tempForlInearFit.pickle"), 'wb') as internal_filename:
    #     print(internal_filename)
    #     pickle.dump([xarr, yarr, sensitivityR2, k, lenArrOutsideDrop], internal_filename)

    minimalnNrOfDatapointsi = round((len(yarr)-lenArrOutsideDrop) * (2 / 4))  #minim nr of datapoints over which should be fitted = half the array inside droplet

    residualLastTime = []  # some arbitrary high value to have the first iteration work

    rangeEndi = round(lenArrOutsideDrop/2) # start somewhere outside the CL, but not super far (So here, at half the length outside drop)
    rangeStarti = round((len(yarr)-lenArrOutsideDrop) / 7) + lenArrOutsideDrop #only try to fit up untill from the first 1/8th of datapoints inside the drop onwards #len(yarr) - minimalnNrOfDatapointsi

    rangeEndk = -round((len(yarr)-lenArrOutsideDrop) * (2 / 4)) # iterate till at most 2/4th back from the end of the array.
    CurrentFitGood = True
    GoodFitVector = False
    #Vary linear fit range of dataset. Always start by changing the 'first' index in array and fit till end of array.
    #However since sometimes, due to artifacts, the end of array gives poor height values, also allow to omit datapoints from there.
    stepj = round(minimalnNrOfDatapointsi/10)
    startLinRegimeIndex = 0
    # TODO TEMP for getting some quick datanalysis can be removed
    # if k in [1690, 6338, 4225]:       #which vector to plot for    # k == round(len(xarr) / 2)
    #     ###plot linear fits for variable i ranges
    #     for i in [12, 30, 50, 70, 100]:     #which starting index to plot for
    #         fig, ax = plt.subplots()
    #         ax.plot(yarr, '.', label='data')
    #         j = -4
    #         coef1, residuals, _, _, _ = np.polyfit(xarr[i:j], yarr[i:j], 1, full=True)
    #         r2 = r2_score(yarr[i:i + 30], np.poly1d(coef1)(xarr[i:i + 30]))
    #         ax.plot(np.arange(i, len(xarr) + j, 1), np.poly1d(coef1)(xarr[i:j]), '.', markersize = 3,
    #                 label=f"r2 first 30 points: {r2:.3f}")
    #         ax.set_title(f'k={k}. Fit range index = {i} - {len(xarr) + j}')
    #         ax.set(xlabel='index (-)', ylabel='Some height (um?)')
    #         ax.legend(loc='best')
    #         #fig.savefig(f"C:\\Downloads\\linfit - vec{k} i {i}.png", dpi=900)
    #         plt.close(fig)
    #
    #     ###plot r2's
    #     fig, ax = plt.subplots()
    #     r2 = [];
    #     j = -4
    #     i_range = np.arange(10, 100, 1)
    #     for i in i_range:
    #         coef1, residuals, _, _, _ = np.polyfit(xarr[i:j], yarr[i:j], 1, full=True)
    #         r2.append(r2_score(yarr[i:i + 30], np.poly1d(coef1)(xarr[i:i + 30])))
    #     ax.plot(i_range, r2)
    #     ax.set(title= 'R^2 plot: vary starting index of linear fit-end of dataset', xlabel = 'starting index i of linear fit', ylabel = 'Calculated R^2 value')
    #     #fig.savefig(f"C:\\Downloads\\r2 plot - vec{k}.png", dpi = 600)
    #     plt.close(fig)

    sensitivityR2 = 0.997
    for j in range(-4, rangeEndk, -stepj): # omit last 5 datapoints (since these are generally not great due to wrapped function). Make steps of 10 lengths
        if not CurrentFitGood: #if fit is not good, break out of loop
            break
        if startLinRegimeIndex == rangeEndi+1:  #if we got here, that means over the entire range fit was very good. Stop trying to fit over smaller range and just parse this
            print(f"A good fit (R2 = {r2:.4f}) was found over the entire range at vector {k}. That might be suspicious..")
            CurrentFitGood = False
            GoodFitVector = True
            break
        for i in range(rangeStarti, rangeEndi, -1):  # iterate backwards from +- 1/8 of droplet regime. Stop when deviating from desired R^2.
            coef1, residuals, _, _, _ = np.polyfit(xarr[i:j], yarr[i:j], 1, full=True)
            startLinRegimeIndex = i
            r2 = r2_score(yarr[i:i+30], np.poly1d(coef1)(xarr[i:i+30]))           #r2 of 'first' 10 datapoints to
            #r2 = r2_score(yarr[i:j], np.poly1d(coef1)(xarr[i:j]))
            if r2 < sensitivityR2 and k != round(len(xarr) / 2):  # stop iterating when R2 deviates too much
                CurrentFitGood = False
                GoodFitVector = True
                break
            residualLastTime.append(residuals**0.5 / len(xarr[i:j]))
    if not GoodFitVector:
        print(f"No good linear fit was found inside the droplet. Skipping this vector {k}")
              #f"\nTherefore the data is fitted from 2/4th of the data onwards.")

    return startLinRegimeIndex+rangeStarti, coef1, r2, GoodFitVector

def linearFitLinearRegimeOnly_wPointsOutsideDrop_v4(x, y, sensitivityR2=0.999, lenOutside=0):
    """
    Version 4 of 'linearFitLinearRegimeOnly()'
    Finds the largest regime in the dataset that can be fitted with a good linear fit.

    Parameters:
    - x (array-like): Independent variable.
    - y (array-like): Dependent variable.
    - threshold (float): Minimum R² value to consider the fit as "good".

    Returns:
    - best_start (int): Starting index of the best linear regime.
    - best_end (int): Ending index of the best linear regime.
    - best_r2 (float): R² value of the best linear fit.
    """
    n = len(x)
    best_start, best_end, best_r2 = 0, 0, 0
    bestcoef1 = [0,0]
    GoodFitVector = False
    omitted_finalIndices = 5
    iteration_step_start = 5
    iteration_step_end = 20
    if lenOutside > 10:
        startIndex = lenOutside - 10
    else:
        startIndex = 0
    #for start in range(startIndex, n - 1 - omitted_finalIndices, iteration_step_start):  # Iterate over possible start indices
    for start in range(startIndex, startIndex + (n-lenOutside)//5, iteration_step_start):  # Iterate over possible start indices: only some startindices near CL: bit outside untill len(inside)/5
        for end in range(start + 30, n - omitted_finalIndices, iteration_step_end):  # Iterate over possible end indices (at least 30 points)
            # Subset data
            x_subset = x[start:end]#.reshape(-1, 1)
            y_subset = y[start:end]

            # Fit linear model
            #model = LinearRegression()
            #model.fit(x_subset, y_subset)
            coef1, residuals, _, _, _ = np.polyfit(x_subset, y_subset, 1, full=True)

            # Compute R² score
            #y_pred = model.predict(x_subset)
            y_pred = np.poly1d(coef1)(x_subset)
            r2 = r2_score(y_subset, y_pred)

            # Check if this regime is the best so far
            if r2 >= sensitivityR2 and (end - start) > (best_end - best_start):
                best_start, best_end, best_r2 = start, end, r2
                #coef1 = model.coef_
                bestcoef1 = coef1
                GoodFitVector = True

    return best_start, bestcoef1, best_r2, GoodFitVector, best_end


def linearFitLinearRegimeOnly_wPointsOutsideDrop_v5(x, y, sensitivityR2=0.99, lenOutside=0, k = 0):
    """
    Version 5 of 'linearFitLinearRegimeOnly()'
    Finds the largest regime in the dataset that can be fitted with a good linear fit.

    Parameters:
    - x (array-like): Independent variable.
    - y (array-like): Dependent variable.
    - threshold (float): Minimum R² value to consider the fit as "good".

    Returns:
    - best_start (int): Starting index of the best linear regime.
    - best_end (int): Ending index of the best linear regime.
    - best_r2 (float): R² value of the best linear fit.
    """
    # if k == 760:
    #     with open(os.path.join(os.getcwd(), "tempForlInearFit2.pickle"), 'wb') as internal_filename:
    #         print(internal_filename)
    #         pickle.dump([x, y, sensitivityR2, k, lenOutside], internal_filename)

    n = len(x)
    best_start, best_end, best_r2 = 0, 0, 0
    bestcoef1 = [0, 0]
    end_final = []
    GoodFitVector = False
    omitted_finalIndices = 5
    iteration_step_start = 5
    iteration_step_end = 20
    if lenOutside > 10:
        startIndex = lenOutside - 10
    else:
        startIndex = 0
    # for start in range(startIndex, n - 1 - omitted_finalIndices, iteration_step_start):  # Iterate over possible start indices
    for i, start in enumerate(range(startIndex, startIndex + (n - lenOutside) // 5,
                                    iteration_step_start)):  # Iterate over possible start indices: only some startindices near CL: bit outside untill len(inside)/5
        currentRangeGoodFit = False

        end_start = start + 50
        if i == 0 or not GoodFitVector:
            end_end = n - omitted_finalIndices
        else:
            end_start = max([start + 50, round(np.mean(end_final)) - iteration_step_end*3])     #use the end_final value (for efficiency) if it is larger than the OG starting end_start (this way the subsets will always have some range)
            end_end = round(np.mean(end_final)) + iteration_step_end * 3
        if i > 3 and GoodFitVector:  # from the 4th starting index forward, check if end-index is changing a lot. If no, prob. won't change later = quit function
            end_final_diff = sum(abs(np.array(end_final[-3:]) - np.mean(end_final[-3:])))

            if end_final_diff / 3 < iteration_step_end:
                #print('stop', i, end_final_diff)
                break
            else:
                pass
                #print(i, end_final_diff)
        end_range = np.arange(end_start, end_end, iteration_step_end)
        for end in end_range:  # Iterate over possible end indices (at least 30 points)
            # Subset data
            x_subset = x[start:end]
            y_subset = y[start:end]

            # Fit linear model
            coef1, residuals, _, _, _ = np.polyfit(x_subset, y_subset, 1, full=True)

            # Compute R² score
            y_pred = np.poly1d(coef1)(x_subset)
            r2 = r2_score(y_subset, y_pred)

            # Check if this regime is the best so far
            if (r2 >= sensitivityR2):
                currentRangeGoodFit = True
                if (end - start) > (best_end - best_start):
                    best_start, best_end, best_r2 = start, end, r2
                    bestcoef1 = coef1
                    GoodFitVector = True
            elif currentRangeGoodFit:  # if r2 score is bad, but in current startIndex range a good fit was already found, then the endIndex is messing up the fit = stop trying for higher order, which will likely just worsen the fit
                end_final.append(end)
                #print(f"Breaking out of loop {i, start, end, best_start, best_end}")
                break
            if end == end_range[-1]:
                end_final.append(end)

    return best_start, bestcoef1, best_r2, GoodFitVector, best_end

def extractContourNumbersFromFile(lines):
    """
    Check if in a previous iteration the contour was already determined:
    if yes, import the img nr (n) and the corresponding (i) & threshold sensitivities.
    :param lines:
    :return:
    """
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
def swellingRatioNearCL(xdata, ydata, elapsedtime, path, imgNumber, vectorNumber, outwardsLengthVector, extraPartIndroplet):
    """
    :param xdata: np.aray of x-position data (unit=pixels)
    :param ydata: np.array of y-Intensity data
    :param elapsedtime: elapsed time with respect to t=0 (unit=seconds, format = int or float)
    :param imgNumber: image number of analysed tiff file in folder. Required for saving & importing peak-picking data
    :param vectorNumber: vector number k analysed. Required for saving & importing peak-picking data
    :return height: np.array with absolute heights (nm).
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
    idxx = 0    #in the OG code, idxx is used for the knownheight array and only incremented when a desired idx is investigated. Such that for e.g. 4 idx's to be investigated, knownHeigthArr[idxx] is used
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
                                 range1, range2, source, xshifted, vectorNumber, outwardsLengthVector, unitXY="pixels", extraPartIndroplet=extraPartIndroplet)
    return height, h_ratio

def swellingRatioNearCL_knownpeaks(xdata, ydata, elapsedtime, path, imgNumber, vectorNumber, outwardsLengthVector, extraPartIndroplet, peaks, minima):
    """
    :param xdata: np.aray of x-position data (unit=pixels)
    :param ydata: np.array of y-Intensity data
    :param elapsedtime: elapsed time with respect to t=0 (unit=seconds, format = int or float)
    :param imgNumber: image number of analysed tiff file in folder. Required for saving & importing peak-picking data
    :param vectorNumber: vector number k analysed. Required for saving & importing peak-picking data
    :return height: np.array with absolute heights (nm).
    :return h_ratio: np.array with swelling ratios.
    """
    EVALUATERIGHTTOLEFT = True         #evaluate from left to right, or the other way around    (required for correct conversion of intensity to height profile)
    if EVALUATERIGHTTOLEFT:
        list(xdata).reverse()
        list(ydata).reverse()
        #peaks = len(ydata) - peaks - 1
        #minima = len(ydata) - minima - 1

    FLIP = False                 #True=flip data after h analysis to have the height increase at the left
    MANUALPEAKSELECTION = False
    PLOTSWELLINGRATIO = False
    SAVEFIG = False
    SEPERATEPLOTTING = True
    USESAVEDPEAKS = True
    figswel0, axswel0 = plt.subplots()
    figswel1, axswel1, = plt.subplots()
    knownHeightArr = [165]  #TODO: for now, set the known height of all profiles to 165. Hopefully outwardsLengthVector is long enough that it is hardly swelling at the end
    colorscheme = 'plasma'; cmap = plt.get_cmap(colorscheme); colorGradient = np.linspace(0, 1, len(knownHeightArr))
    dryBrushThickness = 160 #TODO: for now, set the known dry height of all profiles to 160. Hopefully outwardsLengthVector is long enough that it is hardly swelling at the end
    idx = imgNumber             #idx is the image number being investigated
    idxx = 0    #in the OG code, idxx is used for the knownheight array and only incremented when a desired idx is investigated. Such that for e.g. 4 idx's to be investigated, knownHeigthArr[idxx] is used
    intensityProfileZoomConverted = ydata
    knownPixelPosition = 30     #TODO random pixel near end of outwardsLengthVector for now, just to test if entire function works
    normalizeFactor = 1
    range1 = 0
    range2 = len(ydata)
    source = path
    xshifted = xdata

    height, h_ratio = heightFromIntensityProfileV3(FLIP, MANUALPEAKSELECTION, PLOTSWELLINGRATIO, SAVEFIG, SEPERATEPLOTTING, USESAVEDPEAKS,
                                 axswel0, axswel1, cmap, colorGradient, dryBrushThickness, elapsedtime, figswel0, figswel1, idx, idxx,
                                 intensityProfileZoomConverted, knownHeightArr, knownPixelPosition, normalizeFactor,
                                 range1, range2, source, xshifted, vectorNumber, outwardsLengthVector, unitXY="pixels", extraPartIndroplet=extraPartIndroplet, knownPeaks=peaks, knownMinima=minima)
    return height, h_ratio

#TODO ik denk dat constant x, var y nog niet goed kan werken: Output geen lineLength pixel & lengthVector moet langer zijn dan aantal punten van np.arrange (vanwege eerdere normalisatie)?
def profileFromVectorCoords(x0arrcoord, y0arrcoord, dxarrcoord, dyarrcoord, lengthVector, greyresizedimg):
    """
    Returns the intensity profile between 2 coordinates a=[x0arrcoord, y0arrcoord] - b=[dxarrcoord, dyarrcoord].
    The length between the 2 coordinates should be very similar to the set length of the line (e.g. 200)
    Along the line between a-b, steps of 1 full pixel in horizontal (x) or vertical (y) (depending on a-b dx<>dy) w/ corresponding x or y are evaluated.
    Therefore, the len() of returned 'profile' DOES NOT have to be that length!!
    For horizontal&vertical lines len(profile) = lengtVector
    For tilted lines, len(profile) < lengtVector    because less 1 pixel hor/vertical steps have to be evaluated to have a line of the desired length

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
        if y0arrcoord > dyarrcoord: #line is pointing up in the image (x,y=0,0 is top left of image)
            yarr = np.arange(y0arrcoord, dyarrcoord,-1)
        else:   #line is pointing down
            yarr = np.arange(y0arrcoord, dyarrcoord)
        coords = list(zip(xarr.astype(int), yarr.astype(int))) #list of [(x,y), (x,y), ....]
        lineLengthPixels = lengthVector
    else:   #for any other vector orientation
        a = (dyarrcoord - y0arrcoord) / (dxarrcoord - x0arrcoord)
        b = y0arrcoord - a * x0arrcoord
        coords, lineLengthPixels = coordinates_on_line(a, b, [x0arrcoord, dxarrcoord, y0arrcoord, dyarrcoord])


    #Check if line exceeds image boundaries: if so, set bool to false. Else, obtain intensity profile from coordinates
    sy, sx = greyresizedimg.shape
    xlist = [coord[0] for coord in coords]
    ylist = [coord[1] for coord in coords]
    HARDPASS = False        #bool for completely skipping vectors that are outside image range
    if coords[0][0] < 0 or coords[0][1] < 0 or coords[-1][0] >= sx or coords[-1][1] >= sy:          #x1<0, y1<0, xn>=sx, yn>=sy
        n_endx = len(xlist)-1
        n_endy = len(ylist)-1
        for n, coordx in enumerate(xlist):  #iterate over x coords in list to find at which index it passes outside of image range
            if coordx < 0 or coordx >= sx:
                n_endx = n
                break               #break there, and save the index
        for n, coordy in enumerate(ylist):  #iterate over y coords in list to find at which index it passes outside of image range
            if coordy < 0 or coordy >= sy:
                n_endy = n
                break               #break there, and save the index
        if n_endx > n_endy:
            xlist = xlist[:n_endy]
            ylist = ylist[:n_endy]
        else:
            xlist = xlist[:n_endx]
            ylist = ylist[:n_endx]

        coords = list(zip(xlist, ylist))
        lineLengthPixels = ((xlist[0]-xlist[-1])**2 + (ylist[0]-ylist[-1])**2)**0.5
        profile = [np.transpose(greyresizedimg)[pnt] for pnt in coords]
        fitInside = True
        logging.warning(f"Trying to extract intensity data from outside the given image. New line lenght = {lineLengthPixels}.")
        #lineLengthPixels = 0
    elif HARDPASS:
        lineLengthPixels = 0
        fitInside = False
        logging.warning(f"Trying to extract intensity data from outside the given image. Skipping this vector.")
    else:
        profile = [np.transpose(greyresizedimg)[pnt] for pnt in coords]

    return profile, lineLengthPixels, fitInside, coords

def intensityToHeightProfile(profile, lineLengthPixels, conversionXY, conversionZ, FLIPDATA, nrOfLinesDifferenceInMaxMin_unwrappingHeight):
    """
    Convert an intensity profile to a relative height profile by using monochromatic interferometry.
    Best applied when many interference fringes are visible. Not suitable for e.g. less than 5 fringes.
    Intensity profile with fringes is converted to a 'wrapped' or 'sawtooth' profile after filtering in fourier space.
    This allows us to distinguish full phases present in the intensity profile, which can then be converted
    to a smooth height profile by 'unwrapping' (stacking the phases).

    Lowpass & highpass values for filtering in frequency domain are important w/ respect to the obtain results.
    So far, I found generally 'lowpass = 1/2 profile length' & 'highpass = 2' work fine for CA fringes.

    :param profile: intensity profile on line
    :param lineLengthPixels: length of the line (pixels)
    :param conversionXY: conversion factor for pixels -> um in xy-plane
    :param conversionZ: conversion factor for pixels -> nm in z-plane
    :param FLIPDATA: boolean to reverse data in xy-plane (only for plotting purposes)
    :return unwrapped: calculated height profile (units = what was specified by conversionZ)
    :return x: calculated x-values on corresponding with the unwrapped height profile (distance. units = what was specified by conversionXY)
    :return wrapped: wrapped profile
    :return peaks: indices of calculated maxima
    """

    # with open(os.path.join(os.getcwd(), "tempIntensityProfileForWrappedFixing.pickle"), 'wb') as internal_filename:
    #     print(internal_filename)
    #     pickle.dump([profile, lineLengthPixels, conversionXY, conversionZ, FLIPDATA], internal_filename)      #TODO TEMP

    # transform to fourier space
    profile_fft = np.fft.fft(profile)
    mask = np.ones_like(profile).astype(float)
    lowpass = round(len(profile) / 4);
    highpass = 2  # NOTE: lowpass seems most important for a good sawtooth profile. Filtering half of the data off seems fine
    mask[0:lowpass] = 0;
    mask[-highpass:] = 0
    profile_fft = profile_fft * mask
    profile_filtered = np.fft.ifft(profile_fft)

    # calculate the wrapped space
    wrapped = np.arctan2(profile_filtered.imag, profile_filtered.real)
    #obtain indeces to show maxima in plot: since wrapped reaches from approx.[pi, -pi], minimal peak height = 2; minimal peak prominence (w/ respect to basline = 1); min. distance between peaks = 3 datapoints
    #MIND YOU, THIS IS NOT USED IN THE UNWRAPPING, ONLY FOR VISUALISATION PLOTTING PURPOSES
    peaks_initialGuess, _ = scipy.signal.find_peaks(wrapped, height=2, prominence = 1)
    minDistance = round(len(wrapped) / (len(peaks_initialGuess)*5))        #Adaptive peak distance. nr of datapoints / (amount of peaks*3) to be safe
    peaks, _ = scipy.signal.find_peaks(wrapped, height=2, prominence=1, distance=minDistance)

    unwrapped = np.unwrap(wrapped)
    if FLIPDATA:
        unwrapped = -unwrapped + max(unwrapped)
    # TODO conversionZ generally in nm, so /1000 -> in um

    #CHECK IF CALCULATED UNWRAPPED 'HEIGHT' IS APPROX. EQUAL TO HEIGHT STRAIGHT FROM FOUND MAXIMA (to ensure unwrapping did not do something totally crazy)

    h_unwrapped = unwrapped[-1] * conversionZ       #nm
    # conversionZ * 2pi since conversionZ is divided by 1 phase = 2pi. So for 1 phase it's: wavelength / (2*refr_index) = height/phase
    # So (nr of peaks-1) * h/phase = height from peaks alone, which is a lower bound height     (1 phase = 2 peaks = e.g. 181nm, 2 phase = 3 peaks = 360nm, 4 peaks = 540nm etc..)
    h_peaks_lower = (len(peaks)-1) * (conversionZ * 2 * np.pi)
    h_peaks_upper = (len(peaks)+1) * (conversionZ * 2 * np.pi)  #upper bound, peaks+2 (since in principle on both sides it oculd be close to a maximum
    if h_unwrapped < h_peaks_lower or  h_unwrapped > h_peaks_upper:
        if nrOfLinesDifferenceInMaxMin_unwrappingHeight < 20:
            logging.critical(f"WATCH OUT: calculated height from unwrapping ({h_unwrapped:.2f}) is very different from calculated height straight from nr. of maximas (lower bound={h_peaks_lower:.2f}, upper bound = {h_peaks_upper:.2f})"
                             f"used minDistance = {minDistance:.2f}")
        elif nrOfLinesDifferenceInMaxMin_unwrappingHeight == 20:
            logging.critical(
                f"WATCH OUT: 20th line encoutnered where alculated height from unwrapping ({h_unwrapped:.2f}) is very different from calculated height straight from nr. of maximas (lower bound={h_peaks_lower:.2f}, upper bound = {h_peaks_upper:.2f})"
                f"\n NOT SHOWING THIS TEXT MESSAGE FOR FURTHER LINES")
        nrOfLinesDifferenceInMaxMin_unwrappingHeight += 1

        # fig1, ax1 = plt.subplots(2, 2)
        # ax1[0, 0].plot(profile);
        # ax1[0, 0].plot(peaks, np.array(profile)[peaks], '.')
        # ax1[1, 0].plot(wrapped);
        # ax1[1, 0].plot(peaks, wrapped[peaks], '.')
        # ax1[0, 0].set_title(f"Intensity profile with, meaning LOWPASS = {lowpass}");
        # ax1[1, 0].set_title("Wrapped profile")
        # # ax1[0, 1].plot(x, unwrapped * 1000);  # TODO unit unwrapped was in um, *1000 -> back in nm. unit x in um
        # # ax1[0, 1].set_title("Drop height vs distance (unwrapped profile)")
        # # ax1[0, 1].legend(loc='best')
        # ax1[0, 0].set_xlabel("Distance (nr.of datapoints)");
        # ax1[0, 0].set_ylabel("Intensity (a.u.)")
        # ax1[1, 0].set_xlabel("Distance (nr.of datapoints)");
        # ax1[1, 0].set_ylabel("Amplitude (a.u.)")
        # # ax1[0, 1].set_xlabel("Distance (um)");
        # # ax1[0, 1].set_ylabel("Height profile (nm)")
        # fig1.set_size_inches(12.8, 9.6)
        # fig1.tight_layout()
        # plt.show()
        # exit()

    unwrapped *= conversionZ / 1000  # if unwapped is in um: TODO fix so this can be used for different kinds of Z-unit

    # x = np.arange(0, len(unwrapped)) * conversionXY * 1000 #TODO same ^
    # TODO conversionXY generally in mm, so *1000 -> unit in um.
    x = np.linspace(0, lineLengthPixels, len(unwrapped)) * conversionXY * 1000  # converts pixels to desired unit (prob. um)

    # fig1, ax1 = plt.subplots(2, 2)
    # ax1[0, 0].plot(profile);
    # ax1[0, 0].plot(peaks, np.array(profile)[peaks], '.')
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

    return unwrapped, x, wrapped, peaks, nrOfLinesDifferenceInMaxMin_unwrappingHeight


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


def determineTopBottomLeftRight_DropletCoords(x0arr, y0arr):
    """"
    Determine Top, bottom, left & right coordinate simply by checking for the min & max arguments in x & y arrays.

    :param x0arr: input array with x coordinate values
    :param y0arr: input array with y coordinate values
    :param dxarr: input array with x coordinate values, at end of vector
    :param dyarr: input array with y coordinate values, at end of vector
    :return coords: [x, y] values of coordinates at minimum & maximum y-value
    """

    miny_index = np.argmin(y0arr)
    maxy_index = np.argmax(y0arr)
    top_coord = [x0arr[maxy_index], y0arr[maxy_index]]
    bottom_coord = [x0arr[miny_index], y0arr[miny_index]]

    minx_index = np.argmin(x0arr)
    maxx_index = np.argmax(x0arr)
    left_coord = [x0arr[minx_index], y0arr[minx_index]]
    right_coord = [x0arr[maxx_index], y0arr[maxx_index]]
    return top_coord, bottom_coord, left_coord, right_coord




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
    """
    Calculate the total horizontal force on a droplet by integrating along the contact line in various ways:
    -Quad integration  : input function f(phi), from a-b   - (compute definite integral using a technique from Fortran library QUADPACK)
    -Trapz integration : input = datapoints (x, y)  - (trapezoidal rule)

    input: cubicSplineFunction
    :param boundaryPhi1: positive phi value
    :param boundaryPhi2: negative phi value
    such that boundaryPhi2:boundaryPhi1 is the right side of droplet
    :return total_force_quad:           Total horizontal force, integral on function f(phi)
    :return error_quad:                 Error in ^
    :return trapz_intForce_function:    Total horizontal force, integral on 'datapoints' from function f(phi) describing raw data  (so smoothened a bit, but also more datapoints available than raw data)
    :return trapz_intForce_data:        Total horizontal force, integral on raw datapoints
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

        #Simpson integration
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

    # with open(os.path.join(os.getcwd(), "tempForManualFitting.pickle"), 'wb') as internal_filename:
    #     print(internal_filename)
    #     pickle.dump([inputX, inputY, path, Ylabel, N], internal_filename)      #TODO TEMP

    if abs(abs(inputX[0]) - np.pi) > 0.02:
        logging.critical(f"Gap in contour coords (={abs(abs(inputX[0]) - np.pi):.3f}rad) at -pi&pi -> Check if correctly fitted w/ implemented function in 'manualFitting(..)' ")
    inputX = inputX + [np.pi + (np.pi-abs(inputX[0]))]   #add 1 value of x to the end of array at the 'positive x position' of the first x-value, for correct trapz integration
    inputY = inputY + [inputY[0]]

    #TODO: this will not work well if given range is filtered at -pi / pi (fit will oscillate wildly near -pi/pi).
    # Solution to try: since its periodic, shift entire set such that the 'gap' is not at -pi/pi anymore, but somewhere else.

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

    for k in range(1, N[-1]+1): #for all orders in range 1 to N, determine the sigma's sin & cos.
        sigma_k_s.append(f_k__s(I_k__s_j, k, inputX, inputY))
        sigma_k_c.append(f_k__c(I_k__c_j, k, inputX, inputY))
    N = np.array([0] + N)
    X_range = np.linspace(-np.pi, np.pi, 1000)

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
    #showPlot('manual', [fig1])

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



def matchingCAIntensityPeak(x_units, y_intensity, minIndex_maxima, minIndex_minima, I_peaks, I_minima):
    """
    Return the index of the 4th maximum peak from the dry brush -> droplet side.
    The function finds the maxima automatically, removes 'fake' detected peaks, and then returns the 4th peak index.
    :param x_units:
    :param y_intensity:
    :param h_profile:
    :return:
    """
    peaks, minima, _ ,_ = FindMinimaAndMaxima(x_units, y_intensity, minIndex_maxima, minIndex_minima, Ipeaks = I_peaks, Iminima = I_minima,  nomsgbox=True)
    return peaks[3]

def FindMinimaAndMaxima(x_units, y_intensity, minIndex_maxima, minIndex_minima, vectornr=-1, **kwargs):
    """
    Return the indices of all maxima and minima of 'y_intensity'.
    The function finds the extrema automatically, removes 'fake' detected peaks, and then returns the indices.
    :param x_units:
    :param y_intensity: array (or list) with intensity values
    :param minIndex_minima: index below which NO minima are to be found.    Usefull when the intensity profile is very similar across all lines TO FILTER NON MINIMA below a certain index
    :param minIndex_maxima: index below which NO maxima are to be found.    Usefull when the intensity profile is very similar across all lines TO FILTER NON MAXIMA below a certain index
    :return:
    """

    I_peaks_standard = 135   #intensity above which peaks must be found.
    I_minima_standard = 135  #intensity below which minima must be found
    I_peaks = I_peaks_standard  # intensity above which peaks must be found.
    I_minima = I_minima_standard  # intensity below which minima must be found
    validAnswer = False

    for keyword, value in kwargs.items():
        if keyword == "Ipeaks":
            I_peaks = value
        elif keyword == "Iminima":
            I_minima = value
        elif keyword == 'nomsgbox':
            validAnswer = value
        else:
            logging.error(f"Incorrect keyword inputted: {keyword} is not known")

    temp_indexToShowPlot = 10000


    #input values for I_peaks & I_minima
    figtemp, axtemp = plt.subplots()
    axtemp.plot(y_intensity, 'k')
    axtemp.axhspan(min(y_intensity)-5, I_minima, facecolor='orange', alpha=0.3)     #color below certain intensity orange
    axtemp.axhspan(I_peaks, max(y_intensity)+5, facecolor='blue', alpha=0.3)  # color below certain intensity orange

    axtemp.set(title='FindMinimaAndMaxima: to check ', xlabel='Index (-)', ylabel='Intensity (-)')
    figtemp.show();

    msg = f"Input intensity integer values |above, below| which the maxima and minima are found \n(comma seperated. If nothing is inputted, standard = {I_peaks_standard},{I_minima_standard} ):"
    while not validAnswer:
        title = "Find maxima and minima above and below which intensity value?"
        out = easygui.enterbox(msg, title)
        if len(out) > 0:
            try:
                I_peaks, I_minima = list(map(int, out.split(',')))
                if I_peaks and I_minima:    #if not empty
                    validAnswer = True
            except:
                msg = (f"Inputted intensity values were incorrect: possibly not an integer or not comma seperated. Try again: "
                       f"\n(comma seperated. Typically e.g. 130,130):")
        else:  # if empty, use standard
            I_peaks = I_peaks_standard  # intensity above which peaks must be found.
            I_minima = I_minima_standard  # intensity below which minima must be found
            validAnswer = True
    plt.close(figtemp)



    y_intensity = np.array(y_intensity)
    spacing_peaks = 5 #at least 5 pixels between peaks
    prominence_peaks = I_peaks - I_minima#minimum difference between an extremum & the baseline (which for here is always near a minimum). So I input the difference between expected  expected
    if prominence_peaks < 15:
        prominence_peaks = 15
    #Find peaks based on expected intensity values, minimum spacing of peaks, and prominence
    peaks, _ = scipy.signal.find_peaks(y_intensity, height=I_peaks, distance = spacing_peaks, prominence = prominence_peaks)  # obtain indeces of maxima
    minima, _ = scipy.signal.find_peaks(-y_intensity, height=-I_minima, distance = spacing_peaks,  prominence = prominence_peaks)  # obtain indeces of minima

    if vectornr > temp_indexToShowPlot:
        figtemp, axtemp = plt.subplots()
        axtemp.plot(y_intensity)
        axtemp.plot(peaks, y_intensity[peaks], '.', markersize=8)
        axtemp.plot(minima, y_intensity[minima], '.', markersize=8)
        axtemp.set(title='Unfiltered FindMinimaAndMaxima: intensities, min- & maxima', xlabel='Index (-)', ylabel='Intensity (-)')
        plt.show(); plt.close()

    #Filter min & maxima below a given corresponding index
    peaks, minima = filterExtremaBelowIndex(peaks, minima, minIndex_maxima, minIndex_minima)

    if vectornr > temp_indexToShowPlot:
        figtemp, axtemp = plt.subplots()
        axtemp.plot(y_intensity)
        axtemp.plot(peaks, y_intensity[peaks], '.', markersize=8)
        axtemp.plot(minima, y_intensity[minima], '.', markersize=8)
        axtemp.set(title=f'Filtered below index max: {minIndex_maxima}, min: {minIndex_minima}.\n FindMinimaAndMaxima: intensities, min- & maxima', xlabel='Index (-)',
                   ylabel='Intensity (-)')
        plt.show(); plt.close()

    peaks, minima = removeLeftLocalExtrama(peaks, minima, y_intensity)

    # figtemp, axtemp = plt.subplots()
    # axtemp.plot(y_intensity)
    # axtemp.plot(peaks, y_intensity[peaks], '.', markersize=8)
    # axtemp.plot(minima, y_intensity[minima], '.', markersize=8)
    # axtemp.set(title='Filtered local extrema left side. \nFindMinimaAndMaxima: intensities, min- & maxima', xlabel='Index (-)',
    #            ylabel='Intensity (-)')
    # plt.show(); plt.close()

    peaks = removeLeftNonPeak(peaks, y_intensity)
    minima = removeLeftNonPeak(minima, -y_intensity)

    # figtemp, axtemp = plt.subplots()
    # axtemp.plot(y_intensity)
    # axtemp.plot(peaks, y_intensity[peaks], '.', markersize = 8)
    # axtemp.plot(minima, y_intensity[minima], '.', markersize=8)
    # axtemp.set(title='Filtered "left peak" FindMinimaAndMaxima: intensities, min- & maxima', xlabel='Index (-)', ylabel='Intensity (-)')
    # plt.show()

    return peaks, minima, I_peaks, I_minima

def FindMinimaAndMaxima_v2(x_units, y_intensity, minIndex_maxima, minIndex_minima, vectornr=-1, lenIn = 0, lenOut = 0):
    """
    Return the indices of all maxima and minima of 'y_intensity'.
    Divide plot into 2 regimes: left side = before most absolute min or maximum (whichever is more to the right)
    right side: after the index above.
    As such, on the left we can use a higher prominence, peak spacing and intensity values.
    On the right (where the drop, closely spaced, fringes are, a lower prominence & spacing is used)

    The function finds the extrema automatically, removes 'fake' detected peaks, and then returns the indices.
    :param x_units:
    :param y_intensity: array (or lsit) with intensity values
    :param minIndex_minima: index below which NO minima are to be found.    Usefull when the intensity profile is very similar across all lines TO FILTER NON MINIMA below a certain index
    :param minIndex_maxima: index below which NO maxima are to be found.    Usefull when the intensity profile is very similar across all lines TO FILTER NON MAXIMA below a certain index
    :return:
    """
    temp_indexToShowPlot = 10000

    y_intensity = np.array(y_intensity, dtype='int32')

    #look for the abs min and max outside the droplet, but close to the CL:
    #From half of the swelling profile till the CL position
    lowerLimit_I = 30       #any intensity below 'value' must be an artifact
    if any(y_intensity[:lenOut//2] < lowerLimit_I): #Check from 0:half brush part
        peaks = []
        minima = []
        # fig, ax = plt.subplots()
        # ax.plot(y_intensity, '.')
        # plt.show()
        logging.error(f"Some Intensity lower than lowerLimit_I - Probably dirt, so NO Peaks and Minima are parsed"
                      f"CHECK if this is indeed dirt, or intensity is simply lower than {lowerLimit_I}. Then code needs change!")
    #np.where(y_intensity < LowerLimit_I)
    else:
        #smoothen y-data untill just outside CL (lenOut + e.g. -30 datapoints)=
        smoothened_y_partial = scipy.signal.savgol_filter(y_intensity[:(lenOut - 30)], len(y_intensity) // 10, 3)  # apply a savgol filter for data smoothing
        smoothened_y = np.concatenate((smoothened_y_partial, y_intensity[len(smoothened_y_partial):]))
        y_intensity = smoothened_y

        range1 = round(lenOut/2)
        abs_min_index = range1 + np.argmin(y_intensity[range1:lenOut])  #index of absolute minimum
        abs_max_index = range1 + np.argmax(y_intensity[range1:lenOut])  #index of absolute maximum
        #initial guess for intense peaks:
        peaks_guess, _ = scipy.signal.find_peaks(y_intensity, height=y_intensity[abs_max_index]-15)  # obtain indeces of maxima
        peak_outer_high = min(peaks_guess[peaks_guess>range1])
        minimum_guess, _ = scipy.signal.find_peaks(-y_intensity, height=-y_intensity[abs_min_index]-15)  # obtain indeces of maxima
        minimum_outer_low = min(minimum_guess[minimum_guess > range1])

        i_transition = max(minimum_outer_low, peak_outer_high)    #transition index is highest of those two.

        I_peaks_left = y_intensity[peak_outer_high]-20  # intensity above which peaks must be found: in range of 20 intensity points of abs max
        I_minima_left = y_intensity[minimum_outer_low] + 20  # intensity below which minima must be found: in range of 20 intensity points of abs min

        #T
        spacing_peaks_left = 40  # at least 5 pixels between peaks
        prominence_left = 30 #minimum difference between an extremum & the baseline (which for here is always near a minimum).

        I_peaks_right = 130   #intensity above which peaks must be found.
        I_minima_right = 130  #intensity below which minima must be found
        spacing_peaks_right = 6  # at least 5 pixels between peaks
        prominence_right = 15



        #TODO check hier weer peak finding wanneer de distance > 20 (zie onderaan)
        #Find peak indices based on expected intensity values, minimum spacing of peaks, and prominence
        #Left side of i_transition:
        #peaks_left, _ = scipy.signal.find_peaks(y_intensity[:i_transition], height=I_peaks_left, distance = spacing_peaks_left, prominence = prominence_left)  # obtain indeces of maxima
        peaks_left, _ = scipy.signal.find_peaks(y_intensity, height=I_peaks_left, distance = spacing_peaks_left, prominence = prominence_left)  # obtain indeces of maxima
        peaks_left = peaks_left[peaks_left <= i_transition]

        #minima_left, _ = scipy.signal.find_peaks(-y_intensity[:i_transition], height=-I_minima_left, distance = spacing_peaks_left,  prominence = prominence_left)  # obtain indeces of minima
        minima_left, _ = scipy.signal.find_peaks(-y_intensity, height=-I_minima_left, distance = spacing_peaks_left,  prominence = prominence_left)  # obtain indeces of minima
        minima_left = minima_left[minima_left <= i_transition]

        # Right side of i_transition:
        peaks_right, _ = scipy.signal.find_peaks(y_intensity, height=I_peaks_right, distance=spacing_peaks_right, prominence=prominence_right)  # obtain indeces of maxima
        peaks_right = peaks_right[peaks_right >= i_transition]
        minima_right, _ = scipy.signal.find_peaks(-y_intensity, height=-I_minima_right, distance=spacing_peaks_right, prominence=prominence_right)  # obtain indeces of minima
        minima_right = minima_right[minima_right >= i_transition]

        peaks = np.unique(np.concatenate((peaks_left, peaks_right)))    #Combine left & right peaks. remove duplicate peak if present
        minima = np.unique(np.concatenate((minima_left, minima_right)))

        if vectornr > temp_indexToShowPlot:
            figtemp, axtemp = plt.subplots(2,2)
            axtemp[0,0].plot(y_intensity)
            axtemp[0,0].plot(peaks_left, y_intensity[peaks_left], 'o', markersize=8, label = 'left')
            axtemp[0,0].plot(minima_left, y_intensity[minima_left], 'o', markersize=8, label = 'left')
            axtemp[0,0].plot(peaks_right, y_intensity[peaks_right], '.', markersize=8, label = 'right')
            axtemp[0,0].plot(minima_right, y_intensity[minima_right], '.', markersize=8, label = 'right')
            axtemp[0,0].set(title=f'Unfiltered FindMinimaAndMaxima {vectornr}: intensities, min- & maxima\n i_transition={i_transition}', xlabel='Index (-)', ylabel='Intensity (-)')
            axtemp[0,0].legend()
            # plt.show()
            # plt.close()

        #Filter min & maxima below a given corresponding index
        peaks, minima = filterExtremaBelowIndex(peaks, minima, minIndex_maxima, minIndex_minima)

        if vectornr > temp_indexToShowPlot:
            axtemp[0,1].plot(y_intensity)
            axtemp[0,1].plot(peaks, y_intensity[peaks], '.', markersize=8)
            axtemp[0,1].plot(minima, y_intensity[minima], '.', markersize=8)
            axtemp[0,1].set(title=f'Filtered below index max: {minIndex_maxima}, min: {minIndex_minima}.\n FindMinimaAndMaxima: intensities, min- & maxima', xlabel='Index (-)',
                       ylabel='Intensity (-)')
            # plt.show();
            # plt.close()

        peaks, minima = removeLeftLocalExtrama(peaks, minima, y_intensity)

        if vectornr > temp_indexToShowPlot:

            axtemp[1,0].plot(y_intensity)
            axtemp[1,0].plot(peaks, y_intensity[peaks], '.', markersize=8)
            axtemp[1,0].plot(minima, y_intensity[minima], '.', markersize=8)
            axtemp[1,0].set(title='Filtered local extrema left side. \nFindMinimaAndMaxima: intensities, min- & maxima', xlabel='Index (-)',
                       ylabel='Intensity (-)')
            # plt.show();
            # plt.close()

        peaks = removeLeftNonPeak(peaks, y_intensity)
        minima = removeLeftNonPeak(minima, -y_intensity)

        if vectornr > temp_indexToShowPlot:
            axtemp[1,1].plot(y_intensity)
            axtemp[1,1].plot(peaks, y_intensity[peaks], '.', markersize = 8)
            axtemp[1,1].plot(minima, y_intensity[minima], '.', markersize=8)
            axtemp[1,1].set(title='Filtered "left peak" FindMinimaAndMaxima: intensities, min- & maxima', xlabel='Index (-)', ylabel='Intensity (-)')
            figtemp.set_size_inches(12.8, 9.6)
            figtemp.tight_layout
            plt.show()
    return peaks, minima, y_intensity

def filterExtremaBelowIndex(peaks, minima, minIndex_maxima, minIndex_minima):
    """
    Filter min & maxima below a given corresponding index.
    Return only extrema above the given index.
    Usefull when the intensity profile is very similar across all lines TO FILTER NON MAXIMA & MINIMA below a certain index

    :param peaks: peak indices
    :param minima:  minima indices
    :param minIndex_maxima: index below which no maxima are to be found
    :param minIndex_minima: index below which no minima are to be found
    :return:
    """
    newpeaks = peaks[peaks > minIndex_maxima]
    newminima = minima[minima > minIndex_minima]
    return newpeaks, newminima

def removeLeftNonPeak(peaks, y_intensity):
    """
    Remove the most left peak if it's just a local maximum.
    Check is done by comparing the peak intensity to the intensity of the drop fringes.
    :param peaks:
    :param y_intensity:
    :return:
    """
    mean_y_peaks = np.mean(y_intensity[peaks[1:]])
    if (mean_y_peaks - y_intensity[peaks[0]]) > 5:  #if far left peak height is lower intensity (w/ error of e.g.5) than drop fringes intensity, it's not a real peak and will be removed from list
        newPeaks = peaks[1:]
    else:
        newPeaks = peaks
    return newPeaks

def removeLeftLocalExtrama(peaks, minima, y_intensity: np.array):
    """
    Remove, if present, multiple local maxima or minima on the left of an intensity profile
    :param peaks:
    :param minima:
    :param y_intensity:
    :return: array with adjusted minima or maxima
    """
    # final peak is known, might be multiple local minima that need removing
    if peaks[0] > minima[0]:
        #locate indices of possible local extrema
        i_extrema = np.where(np.array(minima) < peaks[0])
        #find index of minimum of those values
        i_loc_min = np.argmin(y_intensity[minima[i_extrema]])
        #return combined list of indices of local minimum, with all 'good' known minima after
        newminima = np.insert(minima[(i_extrema[0][-1]+1):], 0, minima[i_loc_min])
        newpeaks = peaks
    else:   #local maxima might need removing
        # locate indices of possible local extrema
        i_extrema = np.where(np.array(peaks) < minima[0])
        # find index of minimum of those values
        i_loc_max = np.argmax(y_intensity[peaks[i_extrema]])
        newpeaks = np.insert(peaks[(i_extrema[0][-1]+1):], 0, peaks[[i_loc_max]])
        newminima = minima
    return newpeaks, newminima

def coordsToIntensity_CAv2(FLIPDATA, analysisFolder, angleDegArr, ax_heightsCombined, conversionXY, conversionZ,
                         deltatFromZeroSeconds, dxarr, dxnegarr, dyarr, dynegarr, greyresizedimg, heightPlottedCounter,
                         lengthVector, n, omittedVectorCounter, outwardsLengthVector, path, plotHeightCondition,
                         resizedimg, sensitivityR2, vectors, vectorsFinal, x0arr, xArrFinal, y0arr, yArrFinal, IMPORTEDCOORDS,
                         SHOWPLOTS_SHORT, dxExtraOutarr, dyExtraOutarr, extraPartIndroplet, smallExtraOutwardsVector, minIndex_maxima, minIndex_minima, middleCoord, k_half_unfiltered, makeVideoOfData = False):
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
    DETERMINE_HEIGHT_NEAR_CL = False

    # Counter for in how many lines a difference was found between the total drop height from the wrapping/unwrapping function (used in CA calculation), and height purely from maxima in the wrapped profile.
    # Note here, that the nr of determined maxima is prone to 'errors' in the peak_finding: near the edges peaks are not well found, and e.g. dirt spots mess with the peaks.
    # Typically, this does NOT matter too much for the total height profile from the wrapping/unwrapping function (BUT if this number is very large, BE SCEPTICAL of the obtained CA profile & check it in more detail!)
    nrOfLinesDifferenceInMaxMin_unwrappingHeight = 0

    #Create folder in which pickle files will be dumped, if it doesn't exist already:
    output_pickleFolder = os.path.join(analysisFolder, f"pickle dumps")
    if not os.path.exists(output_pickleFolder):
        os.mkdir(output_pickleFolder)

    x_ax_heightsCombined = []
    y_ax_heightsCombined = []
    x_ks = []
    y_ks = []

    matchedPeakIndexArr = []

    x_totalProfileCombined = []
    y_totalIntensityProfileCombined = []
    y_totalHeightProfileCombined = []

    peakdistanceFromCL = []     #distance of 1st peak outside CL. (um)
    #Mp4 video of plots in which intensities & automatically chosen min&maxima will be displayed, of lines around CL.
    figvid, axvid = plt.subplots()
    metadata = dict(title='Intensity Movie', artist = 'Sjendieee')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    outputPath_movie1 = os.path.join(analysisFolder, f"{n}-Intensity along CL.mp4")

    paths_ffmpeg = ['C:\\Users\\ReuvekampSW\\Desktop\\ffmpeg-7.1-essentials_build\\bin\\ffmpeg.exe',            #UT pc & laptop
            'C:\\Users\\Sander PC\\Desktop\\ffmpeg-7.1-essentials_build\\bin\\ffmpeg.exe'                       #thuis pc
            ]
    for ffmpeg_path in paths_ffmpeg:
        if os.path.exists(ffmpeg_path):
            plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path        #set path to ffmpeg file.
            break
    if not os.path.exists(ffmpeg_path):
        logging.critical("No good path to ffmpeg.exe.\n Correct path, or install from e.g. https://www.gyan.dev/ffmpeg/builds/#git-master-builds")

    #Mp4 video of scatterplots in which the automatically found height profiles are plotted in 3D.
    fig3D = plt.figure()
    ax3D = fig3D.add_subplot(111, projection = '3d')
    metadata2 = dict(title='3D Height Movie', artist='Sjendieee')
    writer2 = FFMpegWriter(fps=15, metadata=metadata2)
    outputPath_movie2 = os.path.join(analysisFolder, f"{n}-Height profiles 3D.mp4")

    heightPlottedCounter_3dplot = 0
    x3d_matrix = []
    y3d_matrix = []
    z3d_matrix = []

    # [4000, round(len(x0arr) / 2)]:#
    with writer.saving(figvid, outputPath_movie1, 300) as a, writer2.saving(fig3D, outputPath_movie2, 300) as b:
        # The elements to combine
        middle = np.array([k_half_unfiltered])              #TODO changed: round(len(x0arr) / 2) to k_half_unfiltered
        left_part = np.arange(0, k_half_unfiltered)
        right_part = np.arange(k_half_unfiltered + 1, len(x0arr))
        # Concatenate into a single array
        k_range = np.concatenate((middle, left_part, right_part))
        #k_range = np.concatenate((middle, np.array([0,1,2])))       #TODO Temp, to have the code run faster
        logging.info(f"STARTING with k={k_range[0]}")
        for k in k_range:  # for every contour-coordinate value; plot the normal, determine intensity profile & calculate CA from the height profile
            try:
                xOutwards = [0]     #x length pointing outwards of droplet, for possible swelling analysis
                profileOutwards = []
                if outwardsLengthVector != 0:
                    #extracts intensity profile purely outside droplet.
                    profileOutwards, lineLengthPixelsOutwards, fitInside, coords_Outside = profileFromVectorCoords(x0arr[k], y0arr[k], dxnegarr[k],
                                                                                        dynegarr[k], outwardsLengthVector,
                                                                                        greyresizedimg)


                    # If intensities fit inside profile & are obtained as desired, fill an array with x-positions.
                    # If not keep list empty and act as if we don't want the outside vector
                    # xOutwards is the x-distance (units) purely of swelling profile outside drop
                    if fitInside:
                        xOutwards = np.linspace(0, lineLengthPixelsOutwards,
                                            len(profileOutwards)) * conversionXY * 1000  # converts pixels to desired unit (prob. um)
                    profileOutwards.reverse()  # correct stitching of in-&outwards profiles requires reversing of the outwards profile


                if k in plotHeightCondition or k == k_half_unfiltered: #color & show the vectors of the desried swelling profiles & always the 'middle' vector (of OG contour dataset)
                    colorInwards = (255, 0, 0)  # draw blue vectors for desired swelling profiles
                    colorOutwards = (255, 0, 0)
                    resizedimg = cv2.line(resizedimg, ([x0arr[k], y0arr[k]]), ([dxarr[k], dyarr[k]]), colorInwards,
                                          2)  # draws 1 good contour around the outer halo fringe
                    if outwardsLengthVector != 0:  # if a swelling profile is desired, also plot it in the image
                        resizedimg = cv2.line(resizedimg, ([x0arr[k], y0arr[k]]), ([dxnegarr[k], dynegarr[k]]),
                                              colorOutwards, 2)  # draws 1 good contour around the outer halo fringe
                elif k % 25 == 0:   #Then also plot only 1/25 vectors to not overcrowd the image
                    colorInwards = (0, 255, 0)  # color the others pointing inwards green
                    colorOutwards = (0, 0, 255)  # color the others pointing outwards red
                    resizedimg = cv2.line(resizedimg, ([x0arr[k], y0arr[k]]), ([dxarr[k], dyarr[k]]), colorInwards,
                                          2)  # draws 1 good contour around the outer halo fringe
                    if outwardsLengthVector != 0:  # if a swelling profile is desired, also plot it in the image
                        resizedimg = cv2.line(resizedimg, ([x0arr[k], y0arr[k]]), ([dxnegarr[k], dynegarr[k]]),
                                              colorOutwards, 2)  # draws 1 good contour around the outer halo fringe

                # intensity profile between x0,y0 & inwards vector coordinate (dx,dy)
                profile, lineLengthPixels, _, coords_Profile = profileFromVectorCoords(x0arr[k], y0arr[k], dxarr[k], dyarr[k], lengthVector,
                                                                    greyresizedimg)

                #TODO incoorp. functionality profile + bit outside drop to check for correctness of CA & finding the linear regime
                profileExtraOut = []
                lineLengthPixelsExtraOut = 0
                if smallExtraOutwardsVector != 0:   #extract intensity profile a bit outside the drop
                    profileExtraOut, lineLengthPixelsExtraOut, _, _ = profileFromVectorCoords(x0arr[k], y0arr[k], dxExtraOutarr[k], dyExtraOutarr[k],
                                                                                  smallExtraOutwardsVector, greyresizedimg)

                profileExtraOut.reverse()
                profileExtraOut = profileExtraOut[:-1]  #remove the last datapoint, as it's the same as the start of the CA profile
                # Converts intensity profile to height profile by unwrapping fourier transform wrapping & unwrapping of interferometry peaks
                unwrapped, x, wrapped, peaks, nrOfLinesDifferenceInMaxMin_unwrappingHeight = intensityToHeightProfile(profileExtraOut + profile, lineLengthPixelsExtraOut + lineLengthPixels, conversionXY,
                                                                        conversionZ, FLIPDATA, nrOfLinesDifferenceInMaxMin_unwrappingHeight)

                # (units) shift x (bitbrush+drop) to match with the end of xOutwards (brush).
                #If xOutwards = 0, no shift occurs. Otherwise, x[0] and xOutwards[-1] are overlapped.
                xshift = xOutwards[-1] - x[smallExtraOutwardsVector-1]
                x += xshift          #TODO check of dit goed geimplementeerd is -> check x,y plots & overlap drop&brush

                # finds linear fit over most linear regime (read:excludes halo if contour was not picked ideally).
                # startIndex, coef1, r2 = linearFitLinearRegimeOnly(x[len(profileOutwards):], unwrapped[len(profileOutwards):], sensitivityR2, k)
                #startIndex, coef1, r2, GoodFit = linearFitLinearRegimeOnly_wPointsOutsideDrop_v3(x, unwrapped, sensitivityR2, k, smallExtraOutwardsVector)
                startIndex, coef1, r2, GoodFit, endIndex = linearFitLinearRegimeOnly_wPointsOutsideDrop_v5(x, unwrapped, sensitivityR2, smallExtraOutwardsVector, k)



                if not GoodFit: #if the linear fit is not good, skip this vector and continue w/ next
                    omittedVectorCounter += 1  # TEMP: to check how many vectors should not be taken into account because the r2 value is too low
                    logging.warning(f"Fit inside drop was not good - skipping vector {k}")
                    if k == k_half_unfiltered:
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

                #TODO WIP: check of deze functie werkt naar behoren
                # Always plot 1 drop (and possibly swelling) profile with intensity, wrapped, height & resulting CA for k@half the datapoints
                # If plotting swelling as well, combine that with drop profile into the same figure
                # For the k's in plotHeightCondition, obtain the swelling ratios & plot them.
                if k in plotHeightCondition or k == k_half_unfiltered:
                    offsetDropHeight = 0
                    heightNearCL_smoothened = []  # empty list, which is filled when determining the swelling profile outside droplet.

                    #If plotting swelling, determine swellingprofile outside drop
                    if xOutwards[-1] != 0:
                        xBrushAndDroplet = np.arange(0,len(profileOutwards) + extraPartIndroplet - 1)  # distance of swelling outside drop + some datapoints within of the drop profile (nr of datapoints (NOT pixels!))

                        yBrushAndDroplet = profileOutwards + profile[1:extraPartIndroplet]  # intensity data of brush & some datapoints within droplet

                        heightNearCL_smoothened, xBrushAndDroplet, yBrushAndDroplet_smoothened, matchedPeakIndexArr = intensityToHeightOutside_bitInsideDrop(deltatFromZeroSeconds, k, matchedPeakIndexArr, n,
                                                                                                                                       lineLengthPixelsOutwards, path, profile,
                                                                                                                                       profileOutwards, extraPartIndroplet, minIndex_maxima, minIndex_minima,xBrushAndDroplet,yBrushAndDroplet)

                        x_ks.append(x0arr[k])       #x-coord of 'chosen CL' current line
                        y_ks.append(y0arr[k])       #y-coord of 'chosen CL' current line
                        #below is weird looking, but correct! y is obtained through correctly stitched intensity profiles
                        #x is just 'added' together from 2 profiles (x was shifted before). Since the 1st point was overlapped in that shift, we take 1:extraPartInDroplet .
                        #xBrushAndDroplet_units = np.concatenate([xOutwards, x[1:extraPartIndroplet]])           #x-distance swelling profile + bit inside drop. units= um
                        #Same as above, but more intuitively written
                        xBrushAndDroplet_units = np.linspace(0, lineLengthPixelsOutwards + extraPartIndroplet - 1, len(heightNearCL_smoothened)) * conversionXY * 1000# x-distance swelling profile + bit inside drop. units= um
                        x_ax_heightsCombined.append(xBrushAndDroplet_units)
                        y_totalIntensityProfileCombined.append(yBrushAndDroplet_smoothened)                                #intensity profile. units= (-)
                        y_ax_heightsCombined.append(heightNearCL_smoothened)                                               #height profile brush & bit drop.    units= nm

                        # TODO check dit: nu gewoon overgenomen van eerder. Check wat nog meer nodig is/ wat overbodig is in functie
                        # #remove overlapping datapoints to do proper plotting later:
                        # remove either the extra vectors inside of droplet from the extended swelling profile: ('xOutwards', 'profileOutwards')    (distance x, and intensity resp.)
                        #TODO deed dit: xOutwards = xOutwards[:-smallExtraOutwardsVector]; profileOutwards = profileOutwards[:-smallExtraOutwardsVector]  # TODO CHECK waarom dit niet ":-extraPartIndroplet" is??!

                        # or overlapping vectors from droplet profile ('x', 'unwrapped')
                        #x = x[extraPartIndroplet:]; profile = profile[extraPartIndroplet:]

                        #Matching table y with x distance:
                        #   y variable (units)          x variable (units=um)              what location           how many extra datapoints
                        #   unwrapped - (um)            x (shifted at this point)          bitbrush+droplet        smallExtraOutwardsVector
                        #   heightNearCL (nm)           xBrushAndDroplet_units             brush+bitdroplet        extraPartIndroplet
                        #

                        # Determine difference in h between brush+bitdroplet & bitbrush+droplet profile at 'profileExtraOut' distance from contour
                        offsetDropHeight = (unwrapped[smallExtraOutwardsVector] - (heightNearCL_smoothened[-extraPartIndroplet] / 1000))

                    #TODO: shift x-index of droplet profile to

                    # set equal height of swelling profile & droplet
                    unwrapped = unwrapped - offsetDropHeight
                    # Also, shift x-axis of 'x' to stitch with 'xOutwards properly'
                    #xshift = (x[len(profileExtraOut)] - x[0])
                    #x = np.array(x) - xshift

                    if k == k_half_unfiltered:
                        # big function for 4-panel plot: Intensity, height, wrapped profile, CA colormap
                        ax1, fig1 = plotPanelFig_I_h_wrapped_CAmap(coef1, heightNearCL_smoothened,
                                                                   offsetDropHeight, peaks, profile,
                                                                   profileOutwards, r2, startIndex, unwrapped,
                                                                   wrapped, x, xBrushAndDroplet_units, xshift, smallExtraOutwardsVector, endIndex)
                    else:       #for the profiles in plottingHeightCondition
                        # TODO WIP: swelling or height profile outside droplet
                        # TODO this part below sets the anchor at some index within the droplet regime
                        if heightPlottedCounter == 0:
                            distanceOfEqualHeight = 10  # can be changed: distance (units) at which the profiles must overlap. xOutwards[-1]
                            indexOfEqualHeight = np.argmin(abs(xOutwards - distanceOfEqualHeight))
                            equalHeight = heightNearCL_smoothened[indexOfEqualHeight]

                            ax_heightsCombined.plot(distanceOfEqualHeight, equalHeight, '.', markersize=15,
                                                    zorder=len(x0arr),
                                                    label=f'Anchor at = {distanceOfEqualHeight:.2f} um, {equalHeight:.2f} nm')
                            ax_heightsCombined.axvspan(0, xOutwards[-1], facecolor='orange', alpha=0.3)
                            ax_heightsCombined.axvspan(xOutwards[-1], x[extraPartIndroplet - 1], facecolor='blue',
                                                       alpha=0.3)
                        else:
                            indexOfEqualHeight = np.argmin(abs(xOutwards - distanceOfEqualHeight))
                            heightNearCL_smoothened = heightNearCL_smoothened - (heightNearCL_smoothened[indexOfEqualHeight] - equalHeight)  # to set all height profiles at some index to the same height
                    heightPlottedCounter += 1  # increment counter


                #TODO TEMP voor trying overlap 3d intensity peak & height profiles
                # if k == 6338:
                #     print(f"pausin")
                #     vectorsOfInterest = [1690, 4225, 6338]
                #     CA_s = [angleDegArr[i] for i in vectorsOfInterest]
                #     figtemp, axtemp = plt.subplots(1,2)
                #     overlapping_indices = np.array(matchedPeakIndexArr)
                #     refIndex = overlapping_indices[0]
                #     refX_at_index = x_ax_heightsCombined[0][refIndex]
                #     refH_at_index =  y_ax_heightsCombined[0][refIndex]
                #
                #     axtemp[0].plot(x_ax_heightsCombined[0], y_totalIntensityProfileCombined[0], label=f'Data 1 (reference set)')
                #     axtemp[0].plot(x_ax_heightsCombined[0][refIndex], y_totalIntensityProfileCombined[0][refIndex], '.', markersize = 8, label=f'Reference datapoint 1')
                #     axtemp[1].plot(x_ax_heightsCombined[0], y_ax_heightsCombined[0], label = f'Data 1 (reference set), CA: {CA_s[0]:.3f}')
                #     axtemp[1].plot(refX_at_index, refH_at_index, '.', label = 'Reference datapoint')
                #     for nr_dataset, overlapIndex in enumerate(overlapping_indices[1:]):
                #         nr_dataset = nr_dataset + 1
                #         offsetX = x_ax_heightsCombined[nr_dataset][overlapIndex] - refX_at_index
                #         offsetY = y_ax_heightsCombined[nr_dataset][overlapIndex] - refH_at_index
                #
                #         axtemp[0].plot(x_ax_heightsCombined[nr_dataset] - offsetX, y_totalIntensityProfileCombined[nr_dataset], label = f'Data {nr_dataset+1}')
                #         axtemp[0].plot(x_ax_heightsCombined[nr_dataset][overlapIndex] - offsetX, y_totalIntensityProfileCombined[nr_dataset][overlapIndex],'.', markersize=8, label=f'Reference datapoint {nr_dataset+1}')
                #
                #         axtemp[1].plot(x_ax_heightsCombined[nr_dataset] - offsetX, y_ax_heightsCombined[nr_dataset] - offsetY, label = f'Data {nr_dataset+1}, CA: {CA_s[nr_dataset]:.3f}')
                #     axtemp[0].legend(); axtemp[1].legend()
                #     axtemp[0].set(title = 'Intensity profiles, shifted', xlabel = 'distance (um)', ylabel = 'intensity (-)'); axtemp[1].set(title = "Height profiles, overlapped", xlabel = 'distance (um)', ylabel = 'hieght (nm)')
                #
                #     figtemp.set_size_inches(12.8, 4.8)
                #     figtemp.tight_layout()
                #     figtemp.savefig(os.path.join(analysisFolder, f"Combined Height profiles - imageNr {n}.png"), dpi=600)
                #     #TODO TEMP tot hier
                #
                #
                #     ax10, fig10 = plotPanelFig_I_h_wrapped_CAmap(coef1, heightNearCL_smoothened,
                #                                                offsetDropHeight, peaks, profile,
                #                                                profileOutwards, r2, startIndex, unwrapped,
                #                                                wrapped, x, xOutwards, xshift, smallExtraOutwardsVector)
                #
                #     fig10.suptitle(f"Data profiles: imageNr {n}, vectorNr {k}", size=14)
                #     fig10.tight_layout()
                #     fig10.subplots_adjust(top=0.88)
                #     fig10.savefig(os.path.join(analysisFolder, f"Height profiles - imageNr {n}, vectorNr {k}.png"), dpi=600)
                #     plt.close(fig10)

                #TODO :
                # 1) for every vector, determine peak positions near CL. Determine distance of 1st peak outside CL.
                # 2) Extend to determine swelling profile near CL for every vector
                if DETERMINE_HEIGHT_NEAR_CL:
                    #TODO removed: altered code before so this doesnt have to be done anymore
                    # if k in plotHeightCondition or k == round(len(x0arr) / 2):  # redo the profileOutwards to correctly determine it for automatic profiles (we adjusted it above somewhere)
                    #     profileOutwards, lineLengthPixelsOutwards, fitInside, _ = profileFromVectorCoords(x0arr[k], y0arr[k], dxnegarr[k], dynegarr[k], outwardsLengthVector, greyresizedimg)
                    #     profileOutwards.reverse() # correct stitching of in-&outwards profiles requires reversing of the outwards profile

                    """
                    Important variables here.
                    Inside & outside droplet:
                    - x & y coordinates of line     -> xCoordsProfile, yCoordsProfile
                    - x-distance (index)            -> xBrushAndDroplet
                    - x-distance (units)            -> xBrushAndDroplet_units
                    - y-intensity (-)               -> yBrushAndDroplet
                    - y-height (nm or um)           -> heightNearCL
                    """
                    #Set correct x & y profiles over total line
                    xBrushAndDroplet = np.arange(0, len(profileOutwards) + extraPartIndroplet - 1)  # distance of swelling outside drop + some datapoints within of the drop profile (nr of datapoints (NOT pixels!))
                    xBrushAndDroplet_units = np.linspace(0, outwardsLengthVector + extraPartIndroplet - 1, len(xBrushAndDroplet)) * conversionXY * 1000  # x-distance swelling profile + bit inside drop. units= um
                    yBrushAndDroplet = profileOutwards + profile[1:extraPartIndroplet]  # intensity data of brush & some datapoints within droplet

                    #Make arrays with all x & y coords of Outside part
                    xCoordsOutside = np.array([val[0] for val in coords_Outside])
                    yCoordsOutside = np.array([val[1] for val in coords_Outside])
                    # Make arrays with all x & y coords of Outside part
                    xCoordsInside = np.array([val[0] for val in coords_Profile[1:extraPartIndroplet]])
                    yCoordsInside = np.array([val[1] for val in coords_Profile[1:extraPartIndroplet]])
                    xCoordsProfile = np.concatenate((np.flip(xCoordsOutside), xCoordsInside))
                    yCoordsProfile = np.concatenate((np.flip(yCoordsOutside), yCoordsInside))

                    # TODO filter until the first peak from the left
                    try:
                        # Find peaks & minima automatically in [brush & part inside droplet], by automatic peakfinding.
                        peaks, minima, y_intensity_smoothened = FindMinimaAndMaxima_v2(xBrushAndDroplet, yBrushAndDroplet, minIndex_maxima, minIndex_minima, vectornr = k, lenIn = extraPartIndroplet, lenOut = len(profileOutwards))
                        if k == 0:
                            cmap_minval = 0     #set initial cmap values for h-spatial plot, and overwrite later to the actual min & max in entire dataset
                            cmap_maxval = 1
                            y_lim_min = min(y_intensity_smoothened) - 10    #set y-lim to minimum value w/ some extra room.
                            y_lim_max = max(y_intensity_smoothened) + 10
                        if len(peaks) == 0 or len(minima) == 0:     #if either list is empty, fill 0 for now
                            peakdistanceFromCL.append(0)
                            print(f'TEMPORARY {k}; NO Min or MAX found for 3D plotting. appending "peakdistanceFromCL" = 0 te fill in something')
                        else:
                            peakdistanceFromCL.append(xBrushAndDroplet_units[peaks[3]] - xBrushAndDroplet_units[peaks[2]])
                            if k % 25  == 0:  #TODO for movie plotting purposes only - can be removed
                                #TODO for intensity plots & videos
                                axvid.set(ylim=[y_lim_min, y_lim_max], xlabel='Distance (um)', ylabel='Intensity(-)', title = f'Intensity profile: {k}')
                                axvid.plot(xBrushAndDroplet_units, y_intensity_smoothened)
                                axvid.plot(xBrushAndDroplet_units[peaks], y_intensity_smoothened[peaks], 'o')
                                axvid.plot(xBrushAndDroplet_units[minima], y_intensity_smoothened[minima], 'o')
                                writer.grab_frame()
                                axvid.clear()

                                #Determine height profile in [brush & part inside droplet], by making use of peaks found before.
                                heightNearCL, heightRatioNearCL = swellingRatioNearCL_knownpeaks(xBrushAndDroplet_units, y_intensity_smoothened, deltatFromZeroSeconds[n], path, n, k, outwardsLengthVector, extraPartIndroplet, peaks, minima)

                                # TODO this part below sets the anchor at some index within the droplet regime
                                # TODO But it's not (really) valid if even far away on one side it's more swollen than the other. So removed for now
                                # if heightPlottedCounter_3dplot == 0:
                                #     distanceOfEqualHeight_3dplot = 10  # can be changed: distance (units) at which the profiles must overlap. xOutwards[-1]
                                #     indexOfEqualHeight_3dplot = np.argmin(abs(xBrushAndDroplet_units - distanceOfEqualHeight_3dplot))
                                #     equalHeight_3dplot = heightNearCL[indexOfEqualHeight_3dplot]
                                #
                                #     ax_heightsCombined.plot(distanceOfEqualHeight_3dplot, equalHeight_3dplot, '.', markersize=15, zorder=len(x0arr))
                                #     ax_heightsCombined.axvspan(0, xBrushAndDroplet_units[-1], facecolor='orange', alpha=0.3)
                                #     ax_heightsCombined.axvspan(xBrushAndDroplet_units[-1], x[extraPartIndroplet - 1],
                                #                                facecolor='blue',
                                #                                alpha=0.3)
                                # else:
                                #     indexOfEqualHeight_3dplot = np.argmin(abs(xBrushAndDroplet_units - distanceOfEqualHeight_3dplot))
                                #     heightNearCL = heightNearCL - (heightNearCL[indexOfEqualHeight_3dplot] - equalHeight_3dplot)  # to set all height profiles at some index to the same height
                                # heightPlottedCounter += 1  # increment counter


                                xCoordsProfile_reduced = [xCoordsProfile[i] for i in range(0, len(heightNearCL), 3)] #plot half the data
                                yCoordsProfile_reduced = [yCoordsProfile[i] for i in range(0, len(heightNearCL), 3)] #plot half the data
                                heightNearCL_reduced = [heightNearCL[i] for i in range(0, len(heightNearCL), 3)] #plot half the data

                                # axvid.set(ylim=[0, 1200], xlabel='Distance (um)', ylabel='Height(-)', title = f'Height profile: {k}')
                                # axvid.plot(xBrushAndDroplet, heightNearCL, 'b')
                                # writer.grab_frame()
                                # axvid.clear()

                                #x_3d_units = np.array(xCoordsProfile_reduced) * conversionXY       #in mm
                                #y_3d_units = np.array(resizedimg.shape[0]-np.array(yCoordsProfile_reduced)) * conversionXY       #in mm
                                ax3D.scatter3D(xCoordsProfile_reduced, resizedimg.shape[0]-np.array(yCoordsProfile_reduced), heightNearCL_reduced, c = heightNearCL_reduced, cmap='jet')
                                ax3D.set(xlabel='X-Coord', ylabel='Y-Coord', zlabel = 'Height (nm)', title=f'Spatial Height Profile Colormap ')
                                x3d_matrix.append(xCoordsProfile_reduced)
                                y3d_matrix.append(resizedimg.shape[0]-np.array(yCoordsProfile_reduced))
                                z3d_matrix.append(heightNearCL_reduced)

                                writer2.grab_frame()
                                if min(heightNearCL_reduced) < cmap_minval:
                                    cmap_minval = min(heightNearCL_reduced)
                                if max(heightNearCL_reduced) > cmap_maxval:
                                    cmap_maxval = max(heightNearCL_reduced)
                                print(f"{k} / {len(x0arr)} 3d plotted")

                    except: #TODO remove this at some point - when peakdistanceFromCL fully functional
                        print(f'TEMPORARY {k};  just so peakdistanceFromCL has a catch function: append "peakdistanceFromCL" = 0 te fill in something')
                        peakdistanceFromCL.append(0)

            except:
                logging.error(f"!{k}: Analysing each coordinate & normal vector broke!")
                print(traceback.format_exc())

    logging.info(f"FINISHED analysing all {len(k_range)} lines. "
                 f"\nNr of lines with a difference in calculated height between unwrapped function and from pure maxima/minima: {nrOfLinesDifferenceInMaxMin_unwrappingHeight}."
                 f"\nNow plotting various contact angle & swelling profile plots.")

    if DETERMINE_HEIGHT_NEAR_CL:
        ax3D.set(xlabel = 'X-Coord', ylabel = 'Y-Coord', zlabel = 'Height (nm)', title = f'Spatial Height Profile Colormap n = {n}, or t = ...')   #{deltat_formatted[n]}
        # Create the color bar
        #cax = fig3D.add_axes([0.94, 0.1, 0.05, 0.75])  # [left, bottom, width 5% of figure width, height 75% of figure height]
        #cax.set_title('H (nm)')
        cbar = fig3D.colorbar(matplotlib.cm.ScalarMappable(norm = plt.Normalize(cmap_minval, cmap_maxval), cmap = 'jet'), label='height (nm)', orientation='vertical')
        fig3D.set_size_inches(12.8/1.5, 9.6/1.5)

        try:
            pickle.dump([x3d_matrix, y3d_matrix, z3d_matrix], open(os.path.join(output_pickleFolder, f"{n}-plot3d_data.pickle"), "wb"))
        except:
            logging.critical(f"3D pickle dump did not work")
        #plt.show()
        showPlot(SHOWPLOTS_SHORT, [fig3D])

        fig2, ax2 = plt.subplots()
        #TODO coords to phi here:

        if len(peakdistanceFromCL) != len(x0arr):   #TODO temp solution. check why
            logging.critical((f"For some reason x0arr{len(x0arr)} is not as long as peakdistanceFromCL={len(peakdistanceFromCL)}. \nCheck WHY!"))
            vector_nrs = np.arange(0, len(peakdistanceFromCL))
            x0arr_3dplotting = x0arr[:-1]
            y0arr_3dplotting = abs(np.subtract(resizedimg.shape[0], y0arr))[:-1]

        else:
            vector_nrs = np.arange(0, len(x0arr))
            x0arr_3dplotting = x0arr
            y0arr_3dplotting = abs(np.subtract(resizedimg.shape[0], y0arr))
        phi, rArray = coordsToPhi(x0arr_3dplotting, y0arr_3dplotting, middleCoord[0], middleCoord[1])
        idk1, idk2 = convertPhiToazimuthal(phi)

        ax2.plot(vector_nrs, peakdistanceFromCL, '.')
        ax2.set(title = 'Distance of first fringe peak outside CL', xlabel = 'line nr. in clockwise direction', ylabel = 'distance (um)')
        fig2.savefig(os.path.join(analysisFolder, f'Distance of first fringe peak outside CL {n} lines.png'), dpi=600)

        fig3, ax3 = plt.subplots()
        ax3.plot(phi, peakdistanceFromCL, '.')
        ax3.set(title='Distance of first fringe peak outside CL', xlabel='Phi (rad)', ylabel='distance (um)')
        fig3.savefig(os.path.join(analysisFolder, f'Distance of first fringe peak outside CL {n} phi.png'), dpi=600)

        try:        #TODO temp: dump this plot for easier data retrieval
            pickle.dump([phi, peakdistanceFromCL], open(os.path.join(output_pickleFolder, f"{n}-phi_distance.pickle"), "wb"))
        except:
            logging.critical(f"ax2pickle dump did not work")


        fig4, ax4 = plt.subplots()
        ax4.plot(idk2, peakdistanceFromCL, '.')
        ax4.set(title='Distance of first fringe peak outside CL', xlabel='Azimuthal angle (rad)', ylabel='distance (um)')
        fig4.savefig(os.path.join(analysisFolder, f'Distance of first fringe peak outside CL {n} azi.png'), dpi=600)


        fig5, ax5 = plt.subplots()
        im5 = ax5.scatter(x0arr_3dplotting,  y0arr_3dplotting, c=peakdistanceFromCL, cmap='jet')#, vmin=5, vmax=16)
        ax5.set_xlabel("X-coord");
        ax5.set_ylabel("Y-Coord");
        ax5.set_title(f"Spatial Distance from First Drop Fringe Peak Outside CL ")
        fig5.colorbar(im5)
        fig5.savefig(os.path.join(analysisFolder, f'Distance of first fringe peak outside CL {n} colormap.png'), dpi=600)
        try:        #TODO temp: dump this plot for easier data retrieval
            pickle.dump([x0arr_3dplotting, y0arr_3dplotting, peakdistanceFromCL], open(os.path.join(output_pickleFolder, f"{n}-plot_spatial_distance-XY_data.pickle"), "wb"))
        except:
            logging.critical(f"ax5pickle dump did not work")

        showPlot(SHOWPLOTS_SHORT, [fig2, fig3, fig4, fig5])

    return ax1, fig1, omittedVectorCounter, resizedimg, xOutwards, x_ax_heightsCombined, x_ks, y_ax_heightsCombined, y_ks


def intensityToHeightOutside_bitInsideDrop(deltatFromZeroSeconds, k, matchedPeakIndexArr, n, outwardsLengthVector,
                                           path, profile, profileOutwards, extraPartIndroplet, minIndex_maxima, minIndex_minima, xBrushAndDroplet,yBrushAndDroplet):
    """
    Convert (or import) the intensity data on a line to a height profile.
    Combines the brush profile (from profileOutwards) with some datapoints inside the droplet (extraPartIndroplet) for more fringes.
    :returns: an array of height: swollen brush & bit of droplet
    :return heightNearCL: array of (smoothened) height data, corresponding to xBrushAndDroplet
    :return xBrushAndDroplet: array of distance of [Outside drop + extraPartIndroplet] (nr of datapoints (NOT pixels!))
    :return yBrushAndDroplet: array of intensity values correcponding to ^
    """
    fig, ax = plt.subplots()


    ax.plot(xBrushAndDroplet, yBrushAndDroplet, 'ob', label = 'raw data')
    #TODO filter until the first peak from the left
    peaks, minima, I_peaks, I_minima = FindMinimaAndMaxima(xBrushAndDroplet, yBrushAndDroplet, minIndex_maxima, minIndex_minima)
    yBrushAndDroplet_smoothened = list(scipy.signal.savgol_filter(yBrushAndDroplet[0:peaks[0]], len(yBrushAndDroplet)//10, 3)) + yBrushAndDroplet[peaks[0]:] # apply a savgol filter for data smoothing
    # TODO check if I really want savgol filtering on input data: peaks of
    ax.plot(xBrushAndDroplet, yBrushAndDroplet_smoothened, 'r.', label= 'smoothened before 1st peak')
    ax.set(title = 'intensityToHeightOutside_bitInsideDrop function', xlabel = 'index (-)', ylabel = 'intensity(-)')
    ax.legend()
    #plt.show()

    if extraPartIndroplet >= outwardsLengthVector:
        logging.critical(f'This will break. OutwardsLengthVector ({outwardsLengthVector}) must be longer than extraPartInDroplet ({extraPartIndroplet}).')

    # Function below determines swelling ratio outside droplet by manual fringe finding followed by inter&extrapolation.
    # This height is then only relative to 'itself', su must be corrected & stitched to droplet contact angle profile.
    heightNearCL, heightRatioNearCL = swellingRatioNearCL(xBrushAndDroplet, yBrushAndDroplet_smoothened, deltatFromZeroSeconds[n], path, n, k,outwardsLengthVector, extraPartIndroplet)

    # TODO check if I really want savgol filtering
    heightNearCL_smoothened = scipy.signal.savgol_filter(heightNearCL, len(heightNearCL) // 10, 3)  # apply a savgol filter for data smoothing

    # For matching the 4th (or something) peak of droplet profile in combined height profiles later
    matchedPeakIndex = matchingCAIntensityPeak(xBrushAndDroplet, yBrushAndDroplet_smoothened, minIndex_maxima, minIndex_minima, I_peaks, I_minima)
    matchedPeakIndexArr.append(matchedPeakIndex)
    return heightNearCL_smoothened, xBrushAndDroplet, yBrushAndDroplet_smoothened, matchedPeakIndexArr


def showPlot(display_mode: str, figures: list):
    """
    Display one or more plots with the specified display mode.
    Parameters:
    - display_mode: A string that specifies the display mode. It can be:
    :param display_mode: A string that specifies the display mode. It can be:
        - 'none': Do not display the plots.
        - 'timed': Display the plots for 3 seconds (or when clicked on the plot)
        - 'manual': Display the plots until manually closed. Code continues to execute while plots are open.
    - figures: A list of matplotlib figure objects to be displayed.
    :param figures: A list of matplotlib figure objects to be displayed.
    """

    if display_mode == 'none':
        return

    figs_min = []
    figs_interest = []
    for i in plt.get_fignums():
        fig = plt.figure(i)
        if not fig in figures:
            figs_min.append(fig)
        else:
            figs_interest.append(fig)

    if display_mode == 'timed':
        for fig in figs_interest:
            fig.show()
            fig.waitforbuttonpress(3)   #shows figure for 3 seconds by stopping loop (or click on figure)
            plt.close(fig)

    elif display_mode == 'manual':
        for fig in figs_interest:
            fig.show()      #show figure, without managing event loop : code will continue to execute

    else:
        raise ValueError("Invalid display_mode. Use 'none', 'timed', or 'manual'.")
    return


def find_k_half_filtered(Xfiltered, Yfiltered, Xunfiltered, Yunfiltered):
    """
    Find &return the index in the new dataset to which the OG k_half x&y-coords correspond, if it still exists in the filtered set.
    Allows for importing peaks & minima from previous analysis.
    If it is filtered out, return closest k-half value. This means later peaks & minima must be chosen again.

    :param Xfiltered: list of xcoord of filtered set at k_half
    :param Yfiltered: list ycoord of filtered set at k_half
    :param Xunfiltered: list of unfiltered x-coords
    :param Yunfiltered: list of unfiltered y-coords
    :return k_half_unfiltered: index of x&y-coord corresponding to x&y-coord of k_half_unfiltered dataset
    """
    Xfiltered_khalf = Xfiltered[0]          #k_half was placed at index 0 during first run
    Yfiltered_khalf = Yfiltered[0]
    for i in range(0, len(Xunfiltered)):
        if Xunfiltered[i] == Xfiltered_khalf and Yunfiltered[i] == Yfiltered_khalf:
            k_half_unfiltered = i
            print(f"khalf = {k_half_unfiltered}")
            break

    if k_half_unfiltered < 0:
        logging.critical(f"NO corresponding k_half value found between OG and filtered dataset!\n"
                         f"It might have been filtered out - k_half will be set to half of the filtered dataset, thus new peaks must be selected manually")
        k_half_unfiltered = round(len(Xunfiltered) / 4)
    elif k_half_unfiltered > len(Xfiltered):
        logging.critical(f"k_half of OG set > len(filtered set), which would give errors later."
                         f"k_half will be set to half of the filtered dataset, thus new peaks must be selected manually")
        k_half_unfiltered = round(len(Xunfiltered) / 4)
    return k_half_unfiltered

def reposition_k_half_point(x_listOG, y_listOG, k_half_unfiltered):
    useablexlist = []
    useableylist = []
    useablexlist += x_listOG[1:k_half_unfiltered]
    useablexlist += [x_listOG[0]]
    useablexlist += x_listOG[k_half_unfiltered:]

    useableylist += y_listOG[1:k_half_unfiltered]
    useableylist += [y_listOG[0]]
    useableylist += y_listOG[k_half_unfiltered:]

    return useablexlist, useableylist

def set_k_half_Factor(analysedData_folder):
    """
    Return the k_half factor to use in analysis:
    older data used a factor of 2 (so line at half of the data was taken). Later, this was changed to 4 (so at 3'o clock).
    To analyse old data correctly, set that factor to 2. Else, 4.
    For the older data to work, manually a txt file was made with the name 'k_half2.txt'. If this exists, set factor to 2.
    :param analysedData_folder:
    :return: k_half_factor
    """
    if os.path.exists(os.path.join(analysedData_folder, 'k_half2.txt')):
        k_half_factor = 2
        logging.info(f"USING k_half_factor of 2 = OLDER DATASET! The 4-panel plot will show the data at half of the length of the dataset")
    else:
        k_half_factor = 4
        logging.info(f"USING k_half_factor of 4! The 4-panel plot will show the data at a quarter of the length of the dataset")
    return k_half_factor

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
    everyHowManyImages = 4  # when a range of image analysis is specified, analyse each n-th image
    #usedImages = np.arange(4, 161, everyHowManyImages)  # len(imgList)
    #usedImages = list(np.arange(34, 39, everyHowManyImages))
    usedImages = [53]       #36, 57

    #usedImages = [32]       #36, 57
    thresholdSensitivityStandard = [11, 5]      #typical [13, 5]     [5,3] for higher CA's or closed contours

    imgFolderPath, conversionZ, conversionXY, unitZ, unitXY = filePathsFunction(path, wavelength_laser)

    imgList = [f for f in glob.glob(os.path.join(imgFolderPath, f"*tiff"))]
    analysisFolder = os.path.join(imgFolderPath, "Analysis CA Spatial") #name of output folder of Spatial Contact Analysis
    lengthVector = 200  # typically:200 .length of normal vector over which intensity profile data is taken    (pointing into droplet, so for CA analysis)
    outwardsLengthVector = 590      #0 if no swelling profile to be measured., 590

    extraPartIndroplet = 50  # extra datapoints from interference fringes inside droplet for manual calculating swelling profile outside droplet. Fine as is - don't change unless really desired
    smallExtraOutwardsVector = 50    #small vector e.g. '25', pointing outwards from CL (for wrapped calculation). Goal: overlap some height fitting from CA analysis inside w/ swelling profile outside. #TODO working code, but profile  outside CL has lower frequency than fringes inside, and this seems to mess with the phase wrapping & unwrapping. So end of height profile is flat-ish..

    minIndex_maxima =  400; minIndex_minima = 0; #index below which no minima are to be found (for filtering of extrema when investigating swelling profiles or fringe locations outside drop). Default = 0.

    FLIPDATA = True
    SHOWPLOTS_SHORT = 'timed'  # 'none' Don't show plots&images at all; 'timed' = show images for only 3 seconds; 'manual' = remain open untill clicked away manually
    sensitivityR2 = 0.999    #OG: 0.997  sensitivity for the R^2 linear fit for calculating the CA. Generally, it is very good fitting (R^2>0.99)
    FITGAPS_POLYOMIAL = True    #between gaps in CL coordinates, especially when manually picked multiple, fit with a 2d order polynomial to obtain coordinates in between
    saveCoordinates = True  #for saving the actual pixel coordinates for each file analyzed.
    MANUAL_FILTERING = True     #Manually remove certain coordinates from the contour e.g. at pinning sites

    # MANUALPICKING:Manual (0/1):  0 = always pick manually. 1 = only manual picking if 'correct' contour has not been picked & saved manually before.
    # All Automatical(2/3): 2 = let programn pick contour after 1st manual pick (TODO: not advised, doesn't work properly yet). 3 = use known contour IF available, else automatically use the second most outer contour
    MANUALPICKING = 1
    lg_surfaceTension = 27     #surface tension hexadecane liquid-gas (N/m)

    # A list of vector numbers, for which an outwardsVector (if desired) will be shown & heights can be plotted
    #plotHeightCondition = lambda xlist: [round(len(xlist) / 4), round(len(xlist) * 3 / 2)]                  #[300, 581, 4067, 4300]
    #plotHeightCondition = lambda xlist: [round(8450/5), round(8450*0.75)]        #don't use 'round(len(xlist)/2)', as this one always used automatically

    #plotHeightCondition = lambda xlist: [900, 4000]        #misschienV2 dataset. don't use 'round(len(xlist)/2)', as this one always used automatically

    plotHeightCondition = lambda xlist: []

    # Order of Fourier fitting: e.g. 8 is fine for little noise/movement. 20 for more noise (can be multiple values: all are shown in plot - highest is used for analysis)
    N_for_fitting = [5, 20]  #TODO fix dit zodat het niet manually moet // order of fitting data with fourier. Higher = describes data more accurately. Useful for noisy data.


    """"End primary changeables"""

    if not os.path.exists(analysisFolder):
        os.mkdir(analysisFolder)
        print('created path: ', analysisFolder)
    contourCoordsFolderFilePath = os.path.join(analysisFolder, "ContourCoords")     #folder for saving individual .txt files containing contour coordinates
    contourListFilePath = os.path.join(contourCoordsFolderFilePath, "ContourListFile.txt")       #for saving the settings how the contour was obtained (but fails when the experimental box is drawn manually for getting contour)

    if not os.path.exists(contourCoordsFolderFilePath):
        os.mkdir(contourCoordsFolderFilePath)
        print('created path: ', contourCoordsFolderFilePath)
    contactAngleListFilePath = os.path.join(analysisFolder, "ContactAngle_MedianListFile.txt")

    #Import known img nr - sensitivity threshold combinations to extract coordinates from contour more easily
    if os.path.exists(contourListFilePath):  # read in all contourline data from existing file (filenr ,+ i for obtaining contour location)
        f = open(contourListFilePath, 'r')
        lines = f.readlines()
        importedContourListData_n, importedContourListData_i, importedContourListData_thresh = extractContourNumbersFromFile(lines)
        logging.info(f"Imported threshold data for img nr's {importedContourListData_n}")
    else:
        f = open(contourListFilePath, 'w')
        f.write(f"file number (n); outputi; thresholdSensitivity a; thresholdSensitivity b\n")
        f.close()
        print("Created contour list file.txt")
        importedContourListData_n = []
        importedContourListData_i = []

    ndata = []
    if not os.path.exists(contactAngleListFilePath):  # Create a file for saving median contact angle, if not already existing
        f = open(contactAngleListFilePath, 'w')
        f.write(f"file number (n), delta time from 0 (s), median CA (deg), Horizontal component force (mN), middle X-coord, middle Y-coord\n")
        f.close()
        print("Created Median Contact Angle list file.txt")
    else:
        f = open(contactAngleListFilePath, 'r')
        lines = f.readlines()
        for line in lines[1:]:
            data = line.split(',')
            ndata.append(int(data[0]))      #add already analyzed img nr's into a list, so later we can check if this analysis already exists

    #folder for saving analyzed data: txt files with forces, middle of drop positions, etc.
    analysedData_folder = os.path.join(analysisFolder, 'Analyzed Data')
    if not os.path.exists(analysedData_folder):
        os.mkdir(analysedData_folder)
        print(f"created path: {analysedData_folder}")

    k_half_factor = set_k_half_Factor(analysedData_folder)  #Will be set to 4 for all next purposes, unless for older data analysis (2 was used, so to have that working correctly)

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
    if len(imgList) == 0:
        raise Exception("Image list is empty: make sure correct path is set, and correct img format is put in (e.g. tif, tiff, png, etc)")

    thresholdSensitivity = thresholdSensitivityStandard
    for n, img in enumerate(imgList):
        if n in usedImages:
            #initialize all json-data relevant so it will be saved later on, even if function crashes
            stats = {}  # for saving relevant (analysed) data into a json data
            stats['About'] = {}
            stats['About']['__author__'] = __author__
            stats['About']['__version__'] = __version__
            stats['startDateTime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f %z')
            stats['imgnr'] = n
            coordLeft = []; coordRight = []; coordsBottom = []; coordsTop = []; error_quad = [] ; meanmiddleX = [];
            meanmiddleY = []; medianmiddleX = []; medianmiddleY = []; middleCoord = []
            total_force_quad = []; trapz_intForce_data = []; trapz_intForce_function = []

            if n in importedContourListData_n and MANUALPICKING > 0:
                logging.info(f"img {n} found in thresholdList (ContourListFile.txt). Importing contour i & thresholds sensitivities")
                try:
                    contouri = importedContourListData_i[importedContourListData_n.index(n)]
                    thresholdSensitivity = importedContourListData_thresh[importedContourListData_n.index(n)]
                except:
                    print(
                        f"Either contouri or thresholddata was not imported properly, even though n was in the importedContourListData")
            else:
                contouri = [-1]
                #moved "thresholdSensitivity = thresholdSensitivityStandard" To before for loop, so now used thresholdsensitivy of last image is inputted as suggestion for next analyzed image
            logging.info(f"START determining contour for image n = {n}/{len(imgList)}, or nr {list(usedImages).index(n)+1} out of {len(usedImages)}")



            #Trying for automatic coordinate finding, using coordinates of a previous iteration.
            # TODO doesn't work as desired: now finds contour at location of previous one, but not the  CL one. Incorporate offset somehow, or a check for periodicity of intensitypeaks
            if MANUALPICKING == 2 and n != usedImages[0] and n - usedImages[list(usedImages).index(n) - 1] == everyHowManyImages:
                useablexlist, useableylist, usableContour, resizedimg, greyresizedimg, thresholdSensitivity = \
                    getContourCoordsV4(img, contourListFilePath, n, contouri, thresholdSensitivity, MANUALPICKING, usablecontour=usableContour, fitgapspolynomial=FITGAPS_POLYOMIAL)
                # For determining the middle coord by mean of surface area - must be performed on unfiltered CL to not bias
                middleCoord = determineMiddleCoord(useablexlist, useableylist)  # determine middle coord by making use of "mean surface" area coordinate
            #in any other case
            else:
                coordinatesListFilePath = os.path.join(contourCoordsFolderFilePath, f"coordinatesListFilePath_{n}.txt")         #file for all contour coordinates
                filtered_coordinatesListFilePath = os.path.join(contourCoordsFolderFilePath, f"filtered_coordinatesListFilePath_{n}.txt")   #file for manually filtered contour coordinates
                #If allowing importing known coords:
                #-if filtered coordinates etc. already exist, import those
                if (MANUALPICKING in [1, 3]) and os.path.exists(filtered_coordinatesListFilePath):
                    logging.info(f"IMPORTING FILTERED contact line coordinates")
                    useablexlist, useableylist, usableContour, resizedimg, greyresizedimg, vectorsFinal, angleDegArr = getfilteredContourCoordsFromDatafile(img, filtered_coordinatesListFilePath)
                    logging.info(f"SUCCESSFULLY IMPORTED FILTERED contact line coordinates")

                    try:
                        # #JSON file w/ data from first coordinates determination should exist already. Open and read the JSON file
                        with open(os.path.join(analysedData_folder, f"{n}_analysed_data.json"), 'r') as file:
                            json_data = json.load(file)
                        data_k_half = json_data['khalf_dxdy']
                        logging.info(f"SUCCESSFULLY IMPORTED Json data")
                    except:
                        logging.critical(
                            "No previously saved k_half stats. Could be because of an older data-analysis version json file:\n"
                            "Remove filtered-coordinates file, or redo entire analysis to fix.")
                        break

                    xArrFinal = useablexlist
                    yArrFinal = useableylist
                    IMPORTEDCOORDS = True
                    FILTERED = True #Bool for not doing any filtering operations anymore later

                    # For determining the middle coord by mean of surface area - must be performed on unfiltered CL to not bias
                    #The first value is k_half of OG dataset
                    unfilteredCoordsx, unfilteredCoordsy, _, _, _ = getContourCoordsFromDatafile(img, coordinatesListFilePath)
                    unfilteredCoordsx = [i.astype(int).tolist()for i in unfilteredCoordsx]          #set to type int to make json serializable later
                    unfilteredCoordsy = [i.astype(int).tolist() for i in unfilteredCoordsy]

                    # at index 0 of xArrFinal, yArrFinal, the k_half_OG coordinate is placed, which will mess with the polynomial fitting for getting the vector orientations
                    # so we have to put that x&y coord back in between the closest coord-values for determining dx&dy vectors, and
                    # then later to index=k_half_unfiltered to import the proper peak&minima's

                    k_half_unfiltered = round(len(unfilteredCoordsx) / k_half_factor)       #k_half of the unfiltered dataset

                    #find value where OG-x&y(k_half) = filtered-x&y(k)
                    #adjusted k_half for the fact that some lines were filtered
                    #k_half_filtered = find_k_half_filtered(useablexlist, useableylist, unfilteredCoordsx, unfilteredCoordsy)

                    #reposition k_half to from x&y[0] to middle of x&y list for proper peak retrieval later.
                    #useablexlist, useableylist = reposition_k_half_point(useablexlist, useableylist, k_half_unfiltered)

                    middleCoord = determineMiddleCoord(unfilteredCoordsx, unfilteredCoordsy) #determine middle coord by making use of "mean surface" area coordinate

                #-if coordinates were already written out, but not filtered
                elif (MANUALPICKING in [1, 3]) and os.path.exists(coordinatesListFilePath):
                    logging.info(f"IMPORTING UNFILTERED contact line coordinates")

                    #Obtain exact same (unfiltered) coordinates as in previous contour selection (+optional polynomial fitting).
                    useablexlist, useableylist, usableContour, resizedimg, greyresizedimg = getContourCoordsFromDatafile(img, coordinatesListFilePath)
                    #TODO ^ where to do filtering? -> check where filtering in code
                    logging.info(f"SUCCESFULLY IMPORTED UNFILTERED contact line coordinates")

                    IMPORTEDCOORDS = True
                    FILTERED = False

                    k_half_unfiltered = round(len(useablexlist)/k_half_factor)          #not at i_1/2, but i_1/4th now (=usually advancing coordinate point)
                    #TODO fix khalf
                    stats['XYcoord_k_half'] = [useablexlist[k_half_unfiltered].astype(int).tolist(), list(useableylist)[k_half_unfiltered].astype(int).tolist()]

                    # For determining the middle coord by mean of surface area - must be performed on unfiltered CL to not bias
                    middleCoord = determineMiddleCoord(useablexlist, useableylist)  # determine middle coord by making use of "mean surface" area coordinate

                #-if not allowing, or coords not known yet:
                else:
                    logging.info(f"MANUALLY DETERMINING contact line coordinates ")
                    useablexlist, useableylist, usableContour, resizedimg, greyresizedimg, thresholdSensitivity =  getContourCoordsV4(img, contourListFilePath, n, contouri, thresholdSensitivity, MANUALPICKING, fitgapspolynomial=FITGAPS_POLYOMIAL, saveCoordinates=saveCoordinates, contourCoordsFolderFilePath=coordinatesListFilePath)
                    logging.info(f"SUCCESFULLY DETERMINED contact line coordinates manually")

                    stats['len-x0arr-OG'] = len(useablexlist)
                    k_half_unfiltered = round(stats['len-x0arr-OG'] / k_half_factor)
                    stats['XYcoord_k_half'] = [useablexlist[k_half_unfiltered].astype(int).tolist(), list(useableylist)[k_half_unfiltered].astype(int).tolist()]

                    IMPORTEDCOORDS = False
                    FILTERED = False
                    # For determining the middle coord by mean of surface area - must be performed on unfiltered CL to not bias
                    middleCoord = determineMiddleCoord(useablexlist, useableylist)  # determine middle coord by making use of "mean surface" area coordinate


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
                    if not FILTERED: #parsed coordinates are in "good order"
                        x0arr, dxarr, y0arr, dyarr, vectors, dxnegarr, dynegarr, dxExtraOutarr, dyExtraOutarr = get_normalsV4(
                            useablexlist, useableylist, lengthVector, outwardsLengthVector, smallExtraOutwardsVector)

                        #Save k_half stats to json file, to retrieve when analyzing a next time using the FILTERED data
                        stats['khalf_dxdy'] = [x0arr[k_half_unfiltered], dxarr[k_half_unfiltered], y0arr[k_half_unfiltered],
                                               dyarr[k_half_unfiltered], vectors[k_half_unfiltered],
                                               dxnegarr[k_half_unfiltered], dynegarr[k_half_unfiltered],
                                               dxExtraOutarr[k_half_unfiltered], dyExtraOutarr[k_half_unfiltered]]
                    elif FILTERED: #at position index 0, the k_half_OG coordinate is placed, which will mess with the polynomial fitting for getting the vector orientations
                        x0arr, dxarr, y0arr, dyarr, vectors, dxnegarr, dynegarr, dxExtraOutarr, dyExtraOutarr = get_normalsV4(useablexlist[1:], useableylist[1:], lengthVector, outwardsLengthVector, smallExtraOutwardsVector)

                        #insert k_half OG data in index k_half_unfiltered
                        x0arr, dxarr, y0arr, dyarr, vectors, dxnegarr, dynegarr, dxExtraOutarr, dyExtraOutarr = move_insert_k_half_data(data_k_half, k_half_unfiltered, x0arr, dxarr, y0arr, dyarr, vectors, dxnegarr, dynegarr, dxExtraOutarr, dyExtraOutarr)

                        #del x0arr_temp, dxarr_temp, y0arr_temp, dyarr_temp, vectors_temp, dxnegarr_temp, dynegarr_temp, dxExtraOutarr_temp, dyExtraOutarr_temp
                        stats['khalf_dxdy'] = [x0arr[k_half_unfiltered], dxarr[k_half_unfiltered], y0arr[k_half_unfiltered],
                                               dyarr[k_half_unfiltered], vectors[k_half_unfiltered],
                                               dxnegarr[k_half_unfiltered], dynegarr[k_half_unfiltered],
                                               dxExtraOutarr[k_half_unfiltered], dyExtraOutarr[k_half_unfiltered]]


                    logging.info("Starting to extract information from IMPORTED COORDS.\n"
                                 f"Plotting for vector nrs: {plotHeightCondition(useablexlist)} & {k_half_unfiltered} from the total {len(useablexlist)} vectors possible")
                    start_time = time.time()
                    ax1, fig1, omittedVectorCounter, resizedimg, xOutwards, x_ax_heightsCombined, x_ks, y_ax_heightsCombined, y_ks = coordsToIntensity_CAv2(
                        FLIPDATA, analysisFolder, angleDegArr, ax_heightsCombined, conversionXY, conversionZ,
                        deltatFromZeroSeconds, dxarr, dxnegarr, dyarr, dynegarr, greyresizedimg,
                        heightPlottedCounter, lengthVector, n, omittedVectorCounter, outwardsLengthVector, path,
                        plotHeightCondition(useablexlist), resizedimg, sensitivityR2, vectors, vectorsFinal, x0arr, xArrFinal, y0arr,
                        yArrFinal, IMPORTEDCOORDS, SHOWPLOTS_SHORT, dxExtraOutarr, dyExtraOutarr, extraPartIndroplet, smallExtraOutwardsVector, minIndex_maxima, minIndex_minima, middleCoord, k_half_unfiltered)
                    elapsed_time = time.time() - start_time
                    logging.info(f"Finished coordsToIntensity_CAv2: Extracted intensity profiles, contact angles, and possibly height profiles.\n"
                                 f"This took {elapsed_time / 60 if elapsed_time > 90 else elapsed_time:.2f} {'min' if elapsed_time > 90 else 'sec'}")
                else:
                    #If the CL coordinates have not been imported (e.g. for new img file)
                    # One of the main functions:
                    # Should yield the normal for every point: output is original x&y coords (x0,y0)
                    # corresponding normal coordinate inwards to droplet x,y (defined as dx and dy)
                    # and normal x,y coordinate outwards of droplet (dxneg & dyneg)
                    logging.info("USING CHOSEN CONTACT LINE to determine normal vectors")
                    x0arr, dxarr, y0arr, dyarr, vectors, dxnegarr, dynegarr, dxExtraOutarr, dyExtraOutarr = get_normalsV4(useablexlist, useableylist, lengthVector, outwardsLengthVector, smallExtraOutwardsVector)

                    #Save the stats of k_half_unfiltered - these might be found at a different k value later because of filtering, making it very hard to find back properly.
                    #Saving and importing is much easier
                    stats['khalf_dxdy'] = [x0arr[k_half_unfiltered], dxarr[k_half_unfiltered], y0arr[k_half_unfiltered], dyarr[k_half_unfiltered], vectors[k_half_unfiltered],
                                           dxnegarr[k_half_unfiltered], dynegarr[k_half_unfiltered], dxExtraOutarr[k_half_unfiltered], dyExtraOutarr[k_half_unfiltered]]

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
                        yArrFinal, IMPORTEDCOORDS, SHOWPLOTS_SHORT, dxExtraOutarr, dyExtraOutarr, extraPartIndroplet,
                        smallExtraOutwardsVector, minIndex_maxima, minIndex_minima, middleCoord, k_half_unfiltered)

                    print(f"Normals, intensities & Contact Angles Succesffuly obtained. Next: plotting overview of all data for 1 timestep")
                    logging.warning(f"Out of {len(x0arr)}, {omittedVectorCounter} number of vectors were omitted because the R^2 was too low.")


                #coordsBottom, coordsTop = determineTopAndBottomOfDropletCoords(x0arr, y0arr, dxarr, dyarr)
                #TODO testing the 'easy way' of determining top&bottom with only min/max because other method fails sometimes?
                if not FILTERED:    #NOT filtered = all data is there.
                    coordsBottom, coordsTop, coordLeft, coordRight = determineTopBottomLeftRight_DropletCoords(xArrFinal, yArrFinal)
                else:   #Previously filtered = use imported non-filtered data.
                    coordsBottom, coordsTop, coordLeft, coordRight = determineTopBottomLeftRight_DropletCoords(unfilteredCoordsx, unfilteredCoordsy)
                    del unfilteredCoordsx, unfilteredCoordsy
                print(f"Calculated top and bottom coordinates of the droplet to be:\n"
                      f"Top: x={coordsTop[0]}, y={coordsTop[1]}\n"
                      f"Bottom: x={coordsBottom[0]}, y={coordsBottom[1]}")

                resizedimg = cv2.circle(resizedimg, (coordsBottom), 30, (255, 0, 0), -1)    #draw blue circle at calculated bottom/inflection point of droplet
                resizedimg = cv2.circle(resizedimg, (coordsTop), 30, (0, 255, 0), -1)       #green
                resizedimg = cv2.circle(resizedimg, (coordLeft), 30, (255, 0, 255), -1)    #purple
                resizedimg = cv2.circle(resizedimg, (coordRight), 30, (0, 255, 255), -1)       #yellow

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
                        #deselect regions manually, where e.g. a pinning point is.
                        #Filter data in interactive scatter plot
                        fig3, ax3 = plt.subplots(figsize= (15, 9.6))
                        im3 = ax3.scatter(temp_xArrFinal, abs(np.subtract(imgshape[0], temp_yArrFinal)), c=temp_angleDegArr, cmap='jet',
                                          vmin=min(temp_angleDegArr), vmax=max(temp_angleDegArr))

                        #for manual closing of plot - without showing all other figures
                        closed = [False]
                        def on_close(event):
                            closed[0] = True
                        fig3.show()        #show figure
                        # Connect the close event to the figure
                        fig3.canvas.mpl_connect('close_event', on_close)

                        if MANUAL_FILTERING:
                            highlighter = Highlighter(ax3, np.array(temp_xArrFinal), np.array(abs(np.subtract(imgshape[0], temp_yArrFinal))))
                        ax3.set_xlabel("X-coord"); ax3.set_ylabel("Y-Coord"); ax3.set_title(f"Spatial Contact Angles Colormap n = {n}, or t = {deltat_formatted[n]}\n CHOOSE FILTERING IN FIGURE BY DRAWING BOX OVER UNDESIRED DATAPOINTS")
                        ax3.legend([f"Median CA (deg): {(statistics.median(temp_angleDegArr)):.2f}"], loc='center left')
                        fig3.colorbar(im3)

                        # Run a loop to block until the figure is closed
                        while not closed[0]:
                            fig3.canvas.flush_events()
                        #plt.show()
                        if MANUAL_FILTERING:
                            selected_regions = highlighter.mask
                            inverted_selected_regions = [not elem for elem in selected_regions] #invert booleans to 'deselect' the selected regions
                            xrange1, yrange1 = np.array(temp_xArrFinal)[inverted_selected_regions], np.array(temp_yArrFinal)[inverted_selected_regions]
                        #fig3.savefig(os.path.join(analysisFolder, f'Colorplot XYcoord-CA {n:04}.png'), dpi=600)
                        plt.close(fig3)

                        #Show the filtered result, and decide whether done filtering, or more must be performed
                        if MANUAL_FILTERING:
                            filtered_angleDegArr = np.array(temp_angleDegArr)[inverted_selected_regions]
                            fig3, ax3 = plt.subplots(figsize= (15, 9.6))
                            im3 = ax3.scatter(xrange1, abs(np.subtract(imgshape[0], yrange1)), c=filtered_angleDegArr, cmap='jet',
                                              vmin=min(filtered_angleDegArr), vmax=max(filtered_angleDegArr))
                            ax3.set_xlabel("X-coord"); ax3.set_ylabel("Y-Coord"); ax3.set_title(f"Spatial Contact Angles Colormap n = {n}, or t = {deltat_formatted[n]}\n RESULTING FILTERED PROFILE. NEXT: CHOOSE WHETHER THIS IS GOOD (ENOUGH)")
                            ax3.legend([f"Median CA (deg): {(statistics.median(filtered_angleDegArr)):.2f}"], loc='center left')
                            fig3.colorbar(im3)

                            closed = [False]
                            fig3.show()
                            fig3.canvas.mpl_connect('close_event', on_close)         # Connect the close event to the figure
                            #plt.show()
                            fig3.savefig(os.path.join(analysisFolder, f'Colorplot XYcoord-CA {n:04}-filtered.png'), dpi=600)
                            #plt.close()

                            choices = ["Good filtering: use leftover coordinates",
                                       "Bad filtering: filter more in current coordinates",
                                       "Bad filtering: redo entire process",
                                       "Bad filtering: don't filter",
                                       "Bad filtering: filter values above and/or below some contact angle (input values next)"]
                            #myvar = []
                            myvar = easygui.choicebox("What to do next?", choices=choices)

                            # # Run a loop to block until the figure is closed
                            # while not closed[0] and not myvar:
                            #     fig3.canvas.flush_events()

                            if myvar == choices[0]:
                                xArrFinal = xrange1
                                yArrFinal = yrange1
                                angleDegArr = filtered_angleDegArr
                                temp_vectorsFinal = np.array(temp_vectorsFinal)[inverted_selected_regions]
                                vectorsFinal = temp_vectorsFinal
                                DONEFILTERTIING = True
                            elif myvar == choices[1]:
                                temp_xArrFinal = xrange1
                                temp_yArrFinal = yrange1
                                temp_angleDegArr = filtered_angleDegArr
                                temp_vectorsFinal = np.array(temp_vectorsFinal)[inverted_selected_regions]
                            elif myvar == choices[2]:
                                temp_xArrFinal = xArrFinal
                                temp_yArrFinal = yArrFinal
                                temp_angleDegArr = angleDegArr
                                temp_vectorsFinal  = vectorsFinal
                            elif myvar == choices[3]:
                                DONEFILTERTIING = True
                            elif myvar == choices[4]:
                                temp_vectorsFinal = np.array(temp_vectorsFinal)[inverted_selected_regions]
                                validAnswer = False
                                msg = (f"Input intensity float values |below, above| which the CA must be filtered. Input negative value to not filter "
                                       f"\nComma seperated. e.g. 1.17, 4.2. Or 1, -1 to filter below 1, but not filter higher CA's")
                                while not validAnswer:
                                    title = "Filter Contact Angles"
                                    out = easygui.enterbox(msg, title)
                                    try:
                                        CA_low, CA_high = list(map(float, out.split(',')))
                                        if CA_low and CA_high:  # if not empty
                                            validAnswer = True
                                    except:
                                        msg = (
                                            f"Inputted values were incorrect: possibly not a float or not comma seperated. Try again: "
                                            f"\n(comma seperated. Typically e.g. 1.17, 4.2):")
                                if CA_low > 0:
                                    #Next line: filter in x,y & CA list all respective data for a CA<CA_low
                                    temp_xArrFinal, temp_yArrFinal, temp_vectorsFinal, temp_angleDegArr = zip(*((xrange1, yrange1, temp_vectorsFinal, filtered_angleDeg) for xrange1, yrange1, temp_vectorsFinal, filtered_angleDeg in zip(xrange1, yrange1, temp_vectorsFinal, filtered_angleDegArr) if filtered_angleDeg > CA_low))
                                if CA_high > 0:
                                    temp_xArrFinal, temp_yArrFinal, temp_vectorsFinal, temp_angleDegArr = zip(*((temp_xArrFinal, temp_yArrFinal, temp_vectorsFinal, temp_angleDeg) for temp_xArrFinal, temp_yArrFinal, temp_vectorsFinal, temp_angleDeg in zip(temp_xArrFinal, temp_yArrFinal, temp_vectorsFinal, temp_angleDegArr) if temp_angleDeg < CA_high))

                            plt.close(fig3)

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

                logging.info("Plotting intersect of lines for middle of droplet")
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

                logging.info("Fitting CA data with Fourier fits")
                phiCA_fourierFit, phiCA_fourierFit_single, phiCA_N, _, _ = manualFitting(phi_sorted, phiCA_savgol_sorted, analysisFolder, ["Contact angle ", "[deg]"], N_for_fitting, SHOWPLOTS_SHORT)
                tangentF_fourierFit, tangentF_fourierFit_single, tangentF_N, _, _ = manualFitting(phi_sorted, phi_tangentF_savgol_sorted, analysisFolder, ["Horizontal Component Force ", "[mN/m]"], N_for_fitting, SHOWPLOTS_SHORT)
                rFromMiddle_fourierFit, rFromMiddle_fourierFit_single, rFromMiddle_N, _, _ = manualFitting(phi_sorted, rFromMiddle_savgol_sorted, analysisFolder, ["Radius", "[m]"], N_for_fitting, SHOWPLOTS_SHORT)


                logging.info("Plotting CA data w/ Fourier or savgol smoothened")
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
                logging.info("Plotting forces: raw, Fourier fit & savgol smoothened")
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
                logging.info("Calculating forces on droplet")
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
                logging.info("Exporting/saving data & variables")
                wrappedPath = os.path.join(analysisFolder, f"ContactAngleData {n}.csv")
                d = dict({'x-coords': xArrFinal, 'y-coords': yArrFinal, 'contactAngle': angleDegArr})
                df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))  # pad shorter colums with NaN's
                df.to_csv(wrappedPath, index=False)

                if n not in ndata:  # if not already saved median CA, save to txt file.
                    CAfile = open(contactAngleListFilePath, 'a')
                    #Write data to .txt file:
                    #img nr. ; time from start (s); median CA (deg); horizontal component force (mN); middeleX-coord (pixel) ; middleY-coord (pixel)
                    CAfile.write(f"{n}, {usedDeltaTs[-1]:.2f}, {angleDeg_afo_time[-1]}:.4f, {totalForce_afo_time[-1]}, {middleCoord[0]}, {middleCoord[1]}\n")
                    CAfile.close()

                save_measurementdata_to_json(analysedData_folder, coordLeft, coordRight, coordsBottom, coordsTop,
                                             error_quad,
                                             meanmiddleX, meanmiddleY, medianmiddleX, medianmiddleY, middleCoord, n,
                                             stats,
                                             total_force_quad, trapz_intForce_data, trapz_intForce_function,
                                             usedDeltaTs)
                plt.close('all')
                print("------------------------------------Succesfully finished--------------------------------------------\n"
                      "------------------------------------   previous image  --------------------------------------------")

            except Exception:
                logging.critical(f"Some error occured. Still plotting obtained contour & saving data to json file (as much as possible)")
                print(traceback.format_exc())
                save_measurementdata_to_json(analysedData_folder, coordLeft, coordRight, coordsBottom, coordsTop, error_quad,
                                             meanmiddleX, meanmiddleY, medianmiddleX, medianmiddleY, middleCoord, n,
                                             stats,
                                             total_force_quad, trapz_intForce_data, trapz_intForce_function,
                                             usedDeltaTs)

            tstring = str(datetime.now().strftime("%Y_%m_%d"))  # day&timestring, to put into a filename    "%Y_%m_%d_%H_%M_%S"
            resizedimg = cv2.circle(resizedimg, (round(medianmiddleX), round(medianmiddleY)), 30, (0, 255, 0), -1)  # draw median middle. abs(np.subtract(imgshape[0], medianmiddleY))
            cv2.imwrite(os.path.join(analysisFolder, f"rawImage_contourLine_{tstring}_{n}.png"), resizedimg)

            plt.close('all') #close all existing figures

    #once all images are analysed, plot obtained data together. Can also be done separately afterward with the "CA_analysisRoutine()" in this file
    fig2, ax2 = plt.subplots()
    ax2.plot(np.divide(usedDeltaTs, 60), totalForce_afo_time)
    ax2.set_xlabel("Time (minutes)"); ax2.set_ylabel("Horizontal component force (mN)"); ax2.set_title("Horizontal component force over Time")
    fig2.savefig(os.path.join(analysisFolder, f'Horizontal component force vs Time.png'), dpi=600)
    showPlot(SHOWPLOTS_SHORT, [fig2])
    #plt.show()


def save_measurementdata_to_json(analysedData, coordLeft, coordRight, coordsBottom, coordsTop, error_quad, meanmiddleX,
                                 meanmiddleY, medianmiddleX, medianmiddleY, middleCoord, n, stats, total_force_quad,
                                 trapz_intForce_data, trapz_intForce_function, usedDeltaTs):
    # TODO desired outputs:
    # Always write stats to json file
    # Standard measurement specific info
    stats['timeFromStart'] = usedDeltaTs[-1]  # time since image 0 in (s)
    # middle coords 2 ways:
    # [mean surface area X & Y,  intersecting normal vectors: mean &   median X&Y]
    stats['middleCoords-surfaceArea'] = [middleCoord[0], middleCoord[1]]  # pixels
    stats['middleCoords-MeanIntersectingVectors'] = [meanmiddleX, meanmiddleY]  # pixels
    stats['middleCoords-MedianIntersectingVectors'] = [medianmiddleX, medianmiddleY]  # pixels
    # outer pixel locations of top, bottom, left & right
    stats['OuterLeftPixel'] = coordLeft
    stats['OuterRightPixel'] = coordRight
    stats['TopPixel'] = coordsTop
    stats['BottomPixel'] = coordsBottom
    # Forces: Quad integration on function + error, trapz on function, trapz on raw data
    stats['F_hor-quad-fphi'] = [total_force_quad, error_quad]  # force & error      mN
    stats['F-hor-trapz-fphi'] = trapz_intForce_function  # mN
    stats['F-hor-trapz-data'] = trapz_intForce_data  # mN
    with open(os.path.join(analysedData, f"{n}_analysed_data.json"), 'w') as f:
        json.dump(stats, f, indent=4)


def plotPanelFig_I_h_wrapped_CAmap(coef1, heightNearCL, offsetDropHeight, peaks, profile, profileOutwards,
                                   r2, startIndex, unwrapped, wrapped, x, xOutwards, xshift, smallExtraOutwardsVector, endIndex):
    """"
    #for 4-panel plot:  Intensity vs datapoint [0,0],
                        Height vs distance [0,1],
                        Wrapped profile vs datapoint [1,0],
                        CA colormap x,y-coord [1,1]
    """
    #profile = only intensity profile INSIDE DROP
    #profileOutwards = only intensity profile OUTSIDE DROP
    profile_drop_smallExtraOut = profileOutwards[-(smallExtraOutwardsVector-1):] + profile      #Intensity profile: equivalent of the evaluated drop profile + bit outside drop profile in fourier wrapping/unwrapping function

    #TODO check of xshift 0 moet blijven
    xshift = 0
    fig1, ax1 = plt.subplots(2, 2)

    #### Intensity profile
    ax1[0, 0].plot(profileOutwards + profile, 'k');     #intensity profile
    ax1[0, 0].plot(np.array(peaks) + len(profileOutwards) - (smallExtraOutwardsVector-1), np.array(profile_drop_smallExtraOut)[peaks], 'b.')         #plot found 'peaks' or 'minima' from the wrapped profile
    if xOutwards[-1] != 0:
        ax1[0, 0].plot(len(profileOutwards), profileOutwards[-1], 'g.',label='Chosen contour, manual CL')
        ax1[0, 0].plot(startIndex+len(profileOutwards) - (smallExtraOutwardsVector-1), profile[startIndex], 'r.', label='Start linear regime droplet')
        ax1[0, 0].plot(endIndex + len(profileOutwards) - (smallExtraOutwardsVector-1), profile[endIndex], 'r.', label='End linear regime droplet')
        ax1[0, 0].axvspan(0, len(profileOutwards), facecolor='blue', alpha=0.4, label='(Swollen) brush')
    ax1[0, 0].axvspan(len(profileOutwards), len(profileOutwards + profile), facecolor='orange', alpha=0.4, label='droplet')
    ax1[0, 0].legend(loc='upper left')
    ax1[0, 0].set_title(f"Intensity profile");
    ax1[0, 0].set_xlabel("Distance (nr.of datapoints)");
    ax1[0, 0].set_ylabel("Intensity (a.u.)")

    #### Wrapped profile
    ax1[1, 0].plot(wrapped, 'k');
    ax1[1, 0].axvspan(0, len(wrapped)-1, facecolor='orange', alpha=0.4, label='droplet')
    ax1[1, 0].plot(peaks, wrapped[peaks], 'b.')
    ax1[1, 0].set_title("Wrapped profile (drop only)")
    # TODO unit unwrapped was in um, *1000 -> back in nm. unit x in um
    ax1[1, 0].set_xlabel("Distance (nr.of datapoints)");
    ax1[1, 0].set_ylabel("Amplitude (a.u.)")

    #### Height profile
    if xOutwards[-1] != 0:
        #ax1[0, 1].plot(xOutwards, heightNearCL[:len(profileOutwards)], label="Swelling fringe calculation", color='C0');  # plot the swelling ratio outside droplet
        ax1[0, 1].plot(xOutwards, heightNearCL, '--',  label="Swelling fringe calculation", color='C0');  # plot the swelling ratio outside droplet
    ax1[0, 1].plot(x, unwrapped * 1000, '-.', label="Interference fringe calculation", color='C1');
    # '\nCA={angleDeg:.2f} deg. ' Initially had this in label below, but because of code order change angledeg is not defined yet
    ax1[0, 1].plot(x, (np.poly1d(coef1)(x-xshift) - offsetDropHeight) * 1000, '--k', linewidth=1, label=f'Linear fit, R$^2$={r2:.3f}');

    ax1[0, 1].plot(x[startIndex], unwrapped[startIndex] * 1000, 'r.', label='Start linear regime droplet');
    ax1[0, 1].plot(x[endIndex], unwrapped[endIndex] * 1000, 'r.', label='End linear regime droplet');
    ax1[0, 1].plot(x[smallExtraOutwardsVector - 1], unwrapped[smallExtraOutwardsVector - 1] * 1000 , '.', label = 'Stitch location')    #TODO check of deze werkt naar behoren

    ax1[0, 1].legend(loc='best')
    ax1[0, 1].set_title("Brush & drop height vs distance")
    ax1[0, 1].set_xlabel("Distance (um)");
    ax1[0, 1].set_ylabel("Height profile (nm)")
    fig1.set_size_inches(12.8, 9.6)

    #### Spatial CA profile
    #will be put in later when the entire analysis is done

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

    #path = "F:\\2023_11_13_PLMA_Dodecane_Basler5x_Xp_1_24S11los_misschien_WEDGE_v2" #outwardsLengthVector=[590]

    #path = "D:\\2023_07_21_PLMA_Basler2x_dodecane_1_29_S1_WEDGE_1coverslip spacer_____MOVEMENT"
    #path = "D:\\2023_11_27_PLMA_Basler10x_and5x_dodecane_1_28_S2_WEDGE\\10x"
    #path = "D:\\2023_12_08_PLMA_Basler5x_dodecane_1_28_S2_FULLCOVER"
    #path = "E:\\2023_12_12_PLMA_Dodecane_Basler5x_Xp_1_28_S2_FULLCOVER"
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

    #path = "G:\\2024_05_07_PLMA_Basler15uc_Zeiss5x_dodecane_Xp1_31_S2_WEDGE_2coverslip_spacer_V3"
    #path = "D:\\2024_05_17_PLMA_180nm_hexadecane_Basler15uc_Zeiss5x_Xp1_31_S3_v3FLAT_COVERED"
    #path = "D:\\2024_05_17_PLMA_180nm_dodecane_Basler15uc_Zeiss5x_Xp1_31_S3_v1FLAT_COVERED"
    #path = "G:\\2024_02_05_PLMA 160nm_Basler17uc_Zeiss5x_dodecane_FULLCOVER_v3"

    #path = "D:\\2024_05_17_PLMA_180nm_dodecane_Basler15uc_Zeiss5x_Xp1_31_S3_v1FLAT_COVERED"
    #path = "D:\\2023_12_12_PLMA_Dodecane_Basler5x_Xp_1_28_S2_FULLCOVER"

    #P12MA dodecane - tilted stage
    path = "D:\\2024-09-04 PLMA dodecane Xp1_31_2 ZeissBasler15uc 5x M3 tilted drop"
    #path = "D:\\2024-09-04 PLMA dodecane Xp1_31_2 ZeissBasler15uc 5x M2 tilted drop"

    #PODMA on heating stage:
    #path = "E:\\2023_12_21_PODMA_hexadecane_BaslerInNikon10x_Xp2_3_S3_HaloTemp_29_5C_AndBeyond\\40C"
    #path = "E:\\2023_07_31_PODMA_Basler2x_dodecane_2_2_3_WEDGE_1coverslip spacer____MOVEMENT"


    #For Enqing:
    #path = "M:\\Enqing\\Halo_Zeiss20X"

    #For Yi Li
    #path = "M:\\YiLi\\"

    #Zeiss = 520nm, Nikon=533nm
    primaryObtainCARoutine(path, wavelength_laser=520)
    #CA_analysisRoutine(path, wavelength_laser = 533)


if __name__ == "__main__":
    try:
        main()
    except:
        logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')  # configuration for printing logging messages. Can be removed safely

    exit()