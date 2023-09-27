import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
from general_functions import image_resize, conversion_factors
from line_method import coordinates_on_line, normalize_wrappedspace, mov_mean
import numpy as np
import logging
from PIL import Image
from itertools import chain, zip_longest
from sklearn import preprocessing
from configparser import ConfigParser
from scipy.odr import odrpack as odr
from scipy.odr import models
import scipy.signal

from analysis_contactangle import Highlighter

def selectMinimaAndMaxima(y):
    fig, ax = plt.subplots(figsize=(10, 10))
    x = np.arange(0,len(y))
    print(f"len x={len(x)}, len y = {len(y)} ")
    ax.scatter(x, y)
    highlighter = Highlighter(ax, x, y)
    plt.show()
    selected_regions = highlighter.mask
    xrange1, yrange1 = x[selected_regions], y[selected_regions]

    extremaRanges = [xrange1[0]]        #always add first x value into array
    for i in range(1,len(xrange1)):     #check for all x elements if their step increase is 1
        if (xrange1[i] - xrange1[i-1]) > 1: #if the step is larger than 1, a new extremum range occurs.
            extremaRanges.append(xrange1[i-1])    #input last x value to end previous range
            extremaRanges.append(xrange1[i])  #input newest x+1 value to start new extremum range
    extremaRanges.append(xrange1[-1])   #always append last x element to close the last extremum range
    outputExtrema = []
    #next, find maxima and minima for in every extremum range (so between extremaRanges[0-1, 2-3, 4-5] etc..)
    for i in range(0, len(extremaRanges),2):
        lowerlimitRange = extremaRanges[i]
        upperlimitRange = extremaRanges[i+1]
        if y[(round((upperlimitRange + lowerlimitRange) / 2))] > ((y[upperlimitRange] + y[lowerlimitRange]) / 2):  # likely to be a maximum if the middle value in range > the mean of first&last value
            tempPosition = np.argmax(y[lowerlimitRange:upperlimitRange]) + lowerlimitRange  # position of maximum
        else:
            tempPosition = np.argmin(y[lowerlimitRange:upperlimitRange]) + lowerlimitRange  # position of minimum
        outputExtrema.append(tempPosition)
    return outputExtrema

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


def positiontest(source):
    pointa = 5272, 1701
    pointb = 430, 1843
    x_coords, y_coords = zip(*[pointa, pointb])  # unzip coordinates to x and y
    x_coords = np.array(x_coords) / 3.622
    y_coords = np.array(y_coords) / 3.617
    a = (y_coords[1] - y_coords[0]) / (x_coords[1] - x_coords[0])
    b = y_coords[0] - a * x_coords[0]
    offsetx = 465
    offsety = 112
    x = 490
    y = a*(x+offsetx) + b + offsety
    print(f"y is {y}")

    rawImgList = [f for f in glob.glob(os.path.join(source, f"rawslicesimage\\*.png"))]
    im_raw = cv2.imread(rawImgList[0])
    im_temp = image_resize(im_raw, height=1200)
    resize_factor = 1200 / im_raw.shape[0]
    cv2.imshow('image', im_temp)
    plt.plot(x, y, '.', 'ms', 20)
    plt.show()

    print(f"finished")

#Input: a raw slice image, the chosen pixellocation
#Output: a figure with a dot on the chosen pixellocation
def showPixellocationv2(pointa, pointb, source):
    imgblack = Image.open("C:\\Users\\ReuvekampSW\\Documents\\InterferometryPython\\black square.png")
    imgblack.resize((40,40))
    imgblack.show()
    rawImg = Image.open(os.path.join(source, f"rawslicesimage\\rawslicesimage_Basler_a2A5328-15ucBAS__40087133__20230120_162715883_0010_analyzed_.png"))
    rawImg.paste(imgblack,(100,500))
    rawImg.show()
    print(f"this is fine")

#Get images to see where you chose your pixel
def showPixellocation(pointa, pointb, source):
    rawImgList = [f for f in glob.glob(os.path.join(source, f"rawslicesimage\\*.png"))]
    im_raw = cv2.imread(rawImgList[0])
    im_temp = image_resize(im_raw, height=1200)
    resize_factor = 1200 / im_raw.shape[0]
    #cv2.imshow('image', im_temp)
    #cv2.setWindowTitle("image", "Point selection window. Select 1 point.")
    #cv2.setMouseCallback('image', click_eventSingle)
    #cv2.waitKey(0)
    #global right_clicks
    P1 = np.array(right_clicks[0]) / resize_factor
    print(f"Selected coordinates: P1 = [{P1[0]:.0f}, {P1[1]:.0f}]")
    #Obtain scaling factor to correspond chosen pixellocation to new position in raw image
    #Scaling factor = factor by which raw image was made smaller in new image
    P1arr = np.array(pointa)
    P2arr = np.array(pointb)
    BLarr = np.array([im_raw.shape[1], im_raw.shape[0]])

    adjP1 = np.subtract(P1arr, P2arr)
    scaling = np.divide(adjP1, BLarr)
    print(f"Scaling fators are: {scaling}")
    #Read in from config file (selected points on which the line was drawn)
    pointa = 5272, 1701
    pointb = 430, 1843
    x_coords, y_coords = zip(*[pointa, pointb])  # unzip coordinates to x and y
    a = (y_coords[1] - y_coords[0]) / (x_coords[1] - x_coords[0])
    b = y_coords[0] - a * x_coords[0]
    for x in [-2400, -1500]:
        y = a * x + b
        print(f"y is: {y}")
    bn = b
    coordinates = coordinates_on_line(a, bn, [0, im_raw.shape[1], 0, im_raw.shape[0]])
    print(f"The coordinates are: {coordinates}")

def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def normalizeDataV2(data):
    return preprocessing.normalize([data])[0]

def poly_lsq(x,y,n,verbose=False,itmax=20000):
    ''' Performs a polynomial least squares fit to the data,
    with errors! Uses scipy odrpack, but for least squares.

    IN:
       x,y (arrays) - data to fit
       n (int)      - polinomial order
       verbose      - can be 0,1,2 for different levels of output
                      (False or True are the same as 0 or 1)
       itmax (int)  - optional maximum number of iterations

    OUT:
       coeff -  polynomial coefficients, lowest order first
       err   - standard error (1-sigma) on the coefficients

    --Tiago, 20071114
    '''

    # http://www.scipy.org/doc/api_docs/SciPy.odr.odrpack.html
    # see models.py and use ready made models!!!!

    func   = models.polynomial(n)
    mydata = odr.Data(x, y)
    myodr  = odr.ODR(mydata, func,maxit=itmax)

    # Set type of fit to least-squares:
    myodr.set_job(fit_type=2)
    if verbose == 2: myodr.set_iprint(final=2)

    fit = myodr.run()

    # Display results:
    if verbose: fit.pprint()
    if fit.stopreason[0] == 'Iteration limit reached':
        print('(WWW) poly_lsq: Iteration limit reached, result not reliable!')
    # Results and errors
    coeff = fit.beta[::-1]
    err   = fit.sd_beta[::-1]

    return coeff,err



from scipy.optimize import curve_fit
def custom_fit(x,a,b,c,d,e,f,g,h):
    return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**0.67 + g * np.sin(h*x)

def custom_fit2(x,a,b,c,d,e,f,g,h):
    return a * np.sin(b*x + c) + c * np.sin(d*x + e) + f * np.sin(g*x + h)


""""
making an attempt at fitting the intensity vs time curve, in order to then extract data at an equally spaced timeinterval
Doesn't work properly though. 12th order polynomial didnt even fit well
"""
def makeImages(profile, timeFromStart, source, pixelLocation, config):
    conversionFactorXY, conversionFactorZ, unitXY, unitZ = conversion_factors(config)
    if not os.path.exists(os.path.join(source, f"Swellingimages")):
        os.mkdir(os.path.join(source, f"Swellingimages"))
    fig0, ax0 = plt.subplots()
    ax0.plot(timeFromStart, ([profile])[0], label = f'normalized, unfiltered')
    plt.xlabel('Time (h)')
    plt.ylabel('Mean intensity')
    plt.title(f'Intensity profile at pixellocation = {pixelLocation}')

    #order = 10
    #fit, error = poly_lsq(timeFromStart, ([profile])[0], order)

    popt, pcov = curve_fit(custom_fit2, timeFromStart, ([profile])[0])
    print(popt)
    print(np.linalg.cond(pcov))
    xnew = np.linspace(timeFromStart[0], timeFromStart[len(timeFromStart)-1], 1000)
    ynew = custom_fit2(xnew, *popt)
    #ynew = np.polyval(fit, xnew)
    #ax0.plot(xnew, ynew, label=f'fit {order}, unfiltered')
    ax0.plot(xnew, ynew, label=f'fit, unfiltered')

    # plt.show()
    #plt.draw()
    #fig0.savefig(os.path.join(source, f"Swellingimages\\IntensityProfile{pixelLocation}.png"), dpi=300)

    print(f"length of profile = {len(profile)}")
    nrOfDatapoints = len(profile)
    print(f"{nrOfDatapoints}")
    hiR = nrOfDatapoints - round(nrOfDatapoints/18)     #OG = /13
    hiR = 60
    loR = 1
    for i in range(hiR,hiR+1,20):       #removing n highest frequencies
        for j in range(loR, loR+1, 20):        #removing n lowest frequencies
            HIGHPASS_CUTOFF = i
            LOWPASS_CUTOFF = j
            NORMALIZE_WRAPPEDSPACE = False
            NORMALIZE_WRAPPEDSPACE_THRESHOLD = 3.14159265359
            #conversionZ = 0.02885654477258912
            FLIP = False

            profile_fft = np.fft.fft(ynew)  # transform to fourier space
            highPass = HIGHPASS_CUTOFF
            lowPass = LOWPASS_CUTOFF
            mask = np.ones_like(ynew).astype(float)
            mask[0:lowPass] = 0
            if highPass > 0:
                mask[-highPass:] = 0
            profile_fft = profile_fft * mask
            fig3, ax3 = plt.subplots()
            ax3.plot(preprocessing.normalize([profile_fft.real])[0], label=f'hi:{highPass}, lo:{lowPass}')
            #ax3.plot(timeFromStart, normalizeData(profile_fft), label=f'hi:{highPass}, lo:{lowPass}')
            ax3.legend()
            fig3.savefig(os.path.join(source, f"Swellingimages\\FFT at {pixelLocation}, hiFil{i}, lofil{j}.png"),
                         dpi=300)


            #print(f"Size of dataarray: {len(profile_fft)}")

            profile_filtered = np.fft.ifft(profile_fft)
            ax0.plot(xnew, ([profile_filtered.real])[0], label = f'hi:{highPass}, lo:{lowPass}')
            ax0.legend()
            fig0.savefig(os.path.join(source, f"Swellingimages\\IntensityProfile{pixelLocation}, hiFil{i}.png"),
                         dpi=300)

            wrapped = np.arctan2(profile_filtered.imag, profile_filtered.real)
            if NORMALIZE_WRAPPEDSPACE:
                wrapped = normalize_wrappedspace(wrapped, NORMALIZE_WRAPPEDSPACE_THRESHOLD)
            unwrapped = np.unwrap(wrapped)
            if FLIP:
                unwrapped = -unwrapped + np.max(unwrapped)

            fig1, ax1 = plt.subplots()
            # ax.plot(timeFromStart, wrapped)
            ax1.plot(wrapped)
            plt.title(f'Wrapped plot: hi {highPass}, lo {lowPass}, pixelLoc: {pixelLocation}')
            fig2, ax2 = plt.subplots()
            #TODO for even spreading of data (NOT true time!)
            spacedTimeFromStart = np.linspace(timeFromStart[0], timeFromStart[-1:], len(unwrapped))
            ax2.plot(spacedTimeFromStart, unwrapped * conversionFactorZ)
            plt.xlabel('Time (h)')
            plt.ylabel(u"Height (\u03bcm)")
            plt.title(f'Swelling profile: hi {highPass}, lo {lowPass}, pixelLoc: {pixelLocation}')
            #plt.show()

            fig1.savefig(os.path.join(source, f"Swellingimages\\wrapped_pixel{pixelLocation}high{i},lo{j}.png"),
                         dpi=300)
            fig2.savefig(os.path.join(source, f"Swellingimages\\height_pixel{pixelLocation}high{i},lo{j}.png"),
                         dpi=300)
            plt.close(fig0)
            plt.close(fig1)
            plt.close(fig2)

            #Saves data in time vs height profile plot so a csv file.
            wrappedPath = os.path.join(source, f"Swellingimages\\data{pixelLocation}high{i},lo{j}.csv")
            #(np.insert(realProfile, 0, timeelapsed)).tofile(wrappedPath, sep='\n', format='%.2f')
            np.savetxt(wrappedPath, [p for p in zip(timeFromStart, ([profile])[0], unwrapped * conversionFactorZ)], delimiter=',', fmt='%s')
    # now get datapoints we need.
    #unwrapped_um = unwrapped * conversionZ
    #analyzeTimes = np.linspace(0, 57604, 12)
    #analyzeImages = np.array([find_nearest(timeFromStart, t)[1] for t in analyzeTimes])
    #print(analyzeImages)


""""
Do the same as in normal makeImages, but make no attempt at fitting. Just make the data-acquisition timeinterval regular by hand
So e.g. first 40 images every 20 sec, then every 4 minutes -> take image 1 & 13 & 25 & 37 to have images every 4 minutes throughout all data
"""
def makeImagesManualTimeadjust(profile, timeFromStart, source, pixelLocation, config):
    conversionFactorXY, conversionFactorZ, unitXY, unitZ = conversion_factors(config)
    if not os.path.exists(os.path.join(source, f"Swellingimages")):
        os.mkdir(os.path.join(source, f"Swellingimages"))
    fig0, ax0 = plt.subplots()
    ax0.plot(timeFromStart, (profile), label = f'normalized, unfiltered')
    plt.xlabel('Time (h)')
    plt.ylabel('Mean intensity')
    plt.title(f'Intensity profile. Pixellocation = {pixelLocation}')

    #define which values to use for regular timeinterval
    whichValuesToUse1 = [0]
    whichValuesToUse2 = np.arange(1, len(profile),1)
    whichValuesToUseTot = np.append(whichValuesToUse1, whichValuesToUse2)

    #whichValuesToUseTot = np.arange(0, len(profile),1)      #when all values are to be used
    equallySpacedTimeFromStart = []
    equallySpacedProfile = []

    for i in whichValuesToUseTot:
        equallySpacedTimeFromStart = np.append(equallySpacedTimeFromStart, timeFromStart[i])
        equallySpacedProfile = np.append(equallySpacedProfile, profile[i])

    ax0.plot(equallySpacedTimeFromStart, normalizeData(equallySpacedProfile), '.', label=f'equally spaced profile')

    print(f"length of equally spaced profile = {len(equallySpacedProfile)}")
    nrOfDatapoints = len(equallySpacedProfile)
    print(f"{nrOfDatapoints}")
    hiR = nrOfDatapoints - round(nrOfDatapoints/18)     #OG = /13
    hiR = 60
    loR = 1
    for i in range(hiR,hiR+1,1):       #removing n highest frequencies
        for j in range(loR, loR+1, 2):        #removing n lowest frequencies
            HIGHPASS_CUTOFF = i
            LOWPASS_CUTOFF = j
            NORMALIZE_WRAPPEDSPACE = False
            NORMALIZE_WRAPPEDSPACE_THRESHOLD = 3.14159265359
            #conversionZ = 0.02885654477258912
            FLIP = False

            profile_fft = np.fft.fft(equallySpacedProfile)  # transform to fourier space
            highPass = HIGHPASS_CUTOFF
            lowPass = LOWPASS_CUTOFF
            mask = np.ones_like(equallySpacedProfile).astype(float)
            mask[0:lowPass] = 0
            if highPass > 0:
                mask[-highPass:] = 0
            profile_fft = profile_fft * mask
            fig3, ax3 = plt.subplots()
            ax3.plot(preprocessing.normalize([profile_fft.real])[0], label=f'hi:{highPass}, lo:{lowPass}')
            #ax3.plot(timeFromStart, normalizeData(profile_fft), label=f'hi:{highPass}, lo:{lowPass}')
            ax3.legend()
            fig3.savefig(os.path.join(source, f"Swellingimages\\FFT at {pixelLocation}, hiFil{i}, lofil{j}.png"),
                         dpi=300)

            #print(f"Size of dataarray: {len(profile_fft)}")

            profile_filtered = np.fft.ifft(profile_fft)
            #ax0.plot(equallySpacedTimeFromStart, ([profile_filtered.real])[0], label = f'hi:{highPass}, lo:{lowPass}')
            ax0.legend()
            fig0.savefig(os.path.join(source, f"Swellingimages\\IntensityProfile{pixelLocation}, hiFil{i}.png"),
                         dpi=300)

            wrapped = np.arctan2(profile_filtered.imag, profile_filtered.real)
            if NORMALIZE_WRAPPEDSPACE:
                wrapped = normalize_wrappedspace(wrapped, NORMALIZE_WRAPPEDSPACE_THRESHOLD)
            unwrapped = np.unwrap(wrapped)
            if FLIP:
                unwrapped = -unwrapped + np.max(unwrapped)

            fig1, ax1 = plt.subplots()
            # ax.plot(timeFromStart, wrapped)
            ax1.plot(wrapped)
            plt.title(f'Wrapped plot: hi {highPass}, lo {lowPass}, pixelLoc: {pixelLocation}')
            fig2, ax2 = plt.subplots()
            #TODO for even spreading of data (NOT true time!)
            #spacedTimeFromStart = np.linspace(timeFromStart[0], timeFromStart[-1:], len(unwrapped))
            ax2.plot(equallySpacedTimeFromStart, unwrapped * conversionFactorZ)
            plt.xlabel('Time (h)')
            #plt.ylabel(u"Height (\u03bcm)")
            plt.ylabel(f"Height ({config.get('GENERAL', 'UNIT_Z')})")
            plt.title(f'Swelling profile: hi {highPass}, lo {lowPass}, pixelLoc: {pixelLocation}')
            #plt.show()

            fig1.savefig(os.path.join(source, f"Swellingimages\\wrapped_pixel{pixelLocation}high{i},lo{j}.png"),
                         dpi=300)
            fig2.savefig(os.path.join(source, f"Swellingimages\\height_pixel{pixelLocation}high{i},lo{j}.png"),
                         dpi=300)
            plt.close(fig0)
            plt.close(fig1)
            plt.close(fig2)

            #Saves data in time vs height profile plot so a csv file.
            wrappedPath = os.path.join(source, f"Swellingimages\\data{pixelLocation}high{i},lo{j}.csv")
            #(np.insert(realProfile, 0, timeelapsed)).tofile(wrappedPath, sep='\n', format='%.2f')
            np.savetxt(wrappedPath, [p for p in zip(equallySpacedTimeFromStart, equallySpacedProfile, unwrapped * conversionFactorZ)], delimiter=',', fmt='%s')
    # now get datapoints we need.
    #unwrapped_um = unwrapped * conversionZ
    #analyzeTimes = np.linspace(0, 57604, 12)
    #analyzeImages = np.array([find_nearest(timeFromStart, t)[1] for t in analyzeTimes])
    #print(analyzeImages)

""""Out: CrossingValY, CrossingValX
CrossingValY calculated from middle between extrema
CorssingValX calculated from linear interpolation between two indices of closest values to CrassingValY"""
def findMiddleCrossing(indexPeak1, indexPeak2, xdata, ydata):
    extremum1 = ydata[indexPeak1]; extremum2 = ydata[indexPeak2]
    mCrossVal = (extremum1 + extremum2) / 2
    closest_value = min(ydata[indexPeak1:indexPeak2], key=lambda x: abs(mCrossVal - x))
    #mCrossIndex = ydata[indexPeak1:indexPeak2].index(closest_value) + indexPeak1
    mCrossIndex1 = np.where(ydata[indexPeak1:indexPeak2] == closest_value)[0][0] + indexPeak1       #index of value closest to crossing index

    if np.abs(ydata[mCrossIndex1-1] - mCrossVal) > np.abs(ydata[mCrossIndex1+1] - mCrossVal):       #check which index of adjecent values is closer to crossing height
        mCrossIndex2 = mCrossIndex1 + 1
    else:
        mCrossIndex2 = mCrossIndex1 - 1

    ##assume line between mCrossVals can be linearly interpolated with y=ax + b.
    #a = dy/dx = (y2-y1) / (x2 - x1)
    #b = y1 - a*x1
    a = (ydata[mCrossIndex2] - ydata[mCrossIndex1]) / (xdata[mCrossIndex2] - xdata[mCrossIndex1])
    b = ydata[mCrossIndex1] - a * xdata[mCrossIndex1]
    #then x = (y-b)/a       To obtain approximate x value of crossing
    mCrossXval = (mCrossVal-b) / a
    return mCrossVal, mCrossXval

#xdata & ydata of entire profile, and indices of peaks in that entire profile
#index = single index to evaluate the height at
#indices of extrema in ydata
###NOT USED RIGHT NOW
def angleFromExtrumum(xdata, ydata, index, indexExtremum1, indexExtremum2):          #
    I = ydata[index]
    middleCrossingYValue, middleCrossingXval = findMiddleCrossing(indexExtremum1, indexExtremum2, xdata, ydata)
    if ydata[index] > middleCrossingYValue:     #if above crossing
        diffx = xdata[index] - xdata[indexExtremum1]             #difference in x between index & extremum
        diffy = ydata[index] - middleCrossingYValue              #difference in y at index & crossing
    else:                                       # if below crossing
        ###TODO geeft dit issues door gebruik van exacte experimente waarde, waardoor de crossing niet goede x & hoogte heeft?
        ### als het goed is nu niet meer door gebruik te maken van interpolated x value of crossing
        diffx = middleCrossingXval - (xdata[index] - xdata[indexExtremum1])  # difference in x between index & extremum
        diffy = ydata[index] - middleCrossingYValue             # difference in y at index & crossing
    phi = np.tan(diffy / diffx)             #nan issue because of division by 0
    return phi

###NOT USED RIGHT NOW
def relateAngleToHeight(phi, extremumHeight_nm):
    h_nm = np.linspace(0, 90.9, 1000)
    h_rad = np.linspace(0, np.pi, 1000)
    closest_value = min(h_rad, key=lambda x: abs(phi - x))
    closest_index = h_rad.index(closest_value)
    heightFromExtremum = h_nm[closest_index]
    totalheight = heightFromExtremum + extremumHeight_nm
    return totalheight


###3AM thoughts:
"""" MAIN FUNCTION to obtain height profiles from intensity. This function allows for calculation of heights 
IN BETWEEN MINIMA AND MAXIMA, and needs to be accompanied by idkPre & idkPost last extremum for a complete profile.
calculates height from relating experimental intensity to expected intensity at a certain film thickness from a cosine function.

Input: x & corresponding y-data in desired range, with indices of extrema 1 & 2 (min & max or max & min) in that range of 
x & y data. 
Out: height profile in the inputted range of extrema.
"""
def idk(xdata, ydata, indexPeak1, indexPeak2):
    ## I vs h has been shown to fit in a cos. General form: I  =a*cos(x) + b, in which a & b can be fitted to the minium&maximum of profile part, and crossingvalue (half-height)
    ## in which x = (4*pi*h*lambda / n)
    middleCrossingYValue, middleCrossingXval = findMiddleCrossing(indexPeak1, indexPeak2, xdata, ydata)
    a = abs((ydata[indexPeak1] - ydata[indexPeak2])) / 2
    b = middleCrossingYValue
    if ydata[indexPeak1] > middleCrossingYValue:        #if value of extremum1 > crossingval, you start at a maximum: next phase to be evaluated is from 0-pi
        x_range = np.linspace(0, np.pi, 1000)
    else:                                               #Else, start at minimum: next phase is from pi - 2*pi
        x_range = np.linspace(np.pi, 2*np.pi, 1000)
    h_range = np.linspace(0, 90.9, 1000)
    I_modelx = a * np.cos(x_range) + b

    indicesToEvaluate = range(indexPeak1, indexPeak2+1)
    h = []
    for i, index in enumerate(indicesToEvaluate):
        diffInI = np.abs(np.subtract(I_modelx, ydata[index]))
        minIndexInmodelX = np.where(diffInI == np.min(diffInI))[0][0]
        h.append(h_range[minIndexInmodelX])
    return h

""""Determine height profile before first extremum. First index to evaluate is 0, till the index of 1st extremum.
Input xdata & ydata which reach till at least the second extremum (to calculate to middle crossing)"""
def idkPre1stExtremum(xdata, ydata, indexPeak1, indexPeak2):
    ## I vs h has been shown to fit in a cos. General form: I  =a*cos(x) + b, in which a & b can be fitted to the minium&maximum of profile part, and crossingvalue (half-height)
    ## in which x = (4*pi*h*lambda / n)
    middleCrossingYValue, middleCrossingXval = findMiddleCrossing(indexPeak1, indexPeak2, xdata, ydata)
    a = abs((ydata[indexPeak1] - ydata[indexPeak2])) / 2
    b = middleCrossingYValue

    #Evaluate if starting value is in upwards or downwards trend:
    if ydata[indexPeak1] - ydata[0] > 0: #if etrx1 > data[0], we "came from a minimum" and are going to a maximum: we are in upwards trend. We are in somewhere in phase pi - 2*pi
        x_range = np.linspace(np.pi, 2 * np.pi, 1000)
    else:                               #Else, downwards in phase 0-pi
        x_range = np.linspace(0, np.pi, 1000)

    h_range = np.linspace(0, 90.9, 1000)
    I_modelx = a * np.cos(x_range) + b

    indicesToEvaluate = range(0, indexPeak1+1)
    h = []
    for i, index in enumerate(indicesToEvaluate):
        diffInI = np.abs(np.subtract(I_modelx, ydata[index]))
        minIndexInmodelX = np.where(diffInI == np.min(diffInI))[0][0]
        h.append(h_range[minIndexInmodelX])
    return h

def idkPostLastExtremum(xdata, ydata, indexPeak1, indexPeak2):
    ## I vs h has been shown to fit in a cos. General form: I  =a*cos(x) + b, in which a & b can be fitted to the minium&maximum of profile part, and crossingvalue (half-height)
    ## in which x = (4*pi*h*lambda / n)
    middleCrossingYValue, middleCrossingXval = findMiddleCrossing(indexPeak1, indexPeak2, xdata, ydata)
    a = abs((ydata[indexPeak1] - ydata[indexPeak2])) / 2
    b = middleCrossingYValue

    # Evaluate if starting value is in upwards or downwards trend:
    if ydata[indexPeak1] > ydata[indexPeak2] :  # extremum2 was a minimum if intensity of extremum1 > extremum2. Then we are going to a maximum: we are in upwards trend. We are in somewhere in phase pi - 2*pi
        x_range = np.linspace(np.pi, 2 * np.pi, 1000)
        # Check if any of the last intensity data is higher than the last extremum -> then use the max of that data
        # TODO this method is not ideal, because it is unsure if this is actually a maximum, but it's a better approximation than not doing it at all
        if any(ydata[indexPeak1] < ydata[indexPeak2:-1]):
            localmax = max(ydata[indexPeak2:-1])
            a = abs((localmax - ydata[indexPeak2])) / 2
            b = (localmax + ydata[indexPeak2]) / 2
            print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                  f"ATTENTION: Somewhere after the last minimum, higher intensity values were found than the last maximum!\n"
                  f"The reference maximum has therefore been adjusted!\n"
                  f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    else:  # Else, extremum2 was a maximum, now downwards in phase 0-pi
        x_range = np.linspace(0, np.pi, 1000)
        # Check if any of the last intensity data is lower than the last extremum -> then use the min of that data
        # TODO this method is not ideal, because it is unsure if this is actually a maximum, but it's a better approximation than not doing it at all
        if any(ydata[indexPeak1] > ydata[indexPeak2:-1]):
            localmin = min(ydata[indexPeak2:-1])
            a = abs((localmin - ydata[indexPeak2])) / 2
            b = (localmin + ydata[indexPeak2]) / 2
            print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                  f"ATTENTION: Somewhere after the last maximum, lower intensity values were found than the last minimum!\n"
                  f"The reference minimum has therefore been adjusted!\n"
                  f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

    h_range = np.linspace(0, 90.9, 1000)
    I_modelx = a * np.cos(x_range) + b
    indicesToEvaluate = range(indexPeak2, len(xdata)-1)
    h = []
    for i, index in enumerate(indicesToEvaluate):
        diffInI = np.abs(np.subtract(I_modelx, ydata[index]))
        minIndexInmodelX = np.where(diffInI == np.min(diffInI))[0][0]
        h.append(h_range[minIndexInmodelX])
    return h
def makeImagesManualTimeAdjustFromPureIntensity(profile, timeFromStart, source, pixelLocation, config, dry_thickness, approx_startThickness, MANUALPEAKSELECTION):
    #predict intensity as a function of film thickness with simplified model (from Özlems paper). :
    #h = film thickness ; l = wavelength of light ; n = refractive index film
    modelIntensity = lambda h, l, n: (1 / 2) * (np.cos(4 * np.pi * h / (l / n)) + 1)

    normalizeFactor = 256           #256 for the max pixelrange of camera -> profile between 0 -~1. 1 for unadjusted profile
    conversionFactorXY, conversionFactorZ, unitXY, unitZ = conversion_factors(config)
    if not os.path.exists(os.path.join(source, f"Swellingimages")):
        os.mkdir(os.path.join(source, f"Swellingimages"))
    fig0, ax0 = plt.subplots()
    ax0.plot(timeFromStart, np.divide(profile, normalizeFactor), label=f'normalized, unfiltered')
    plt.xlabel('Time (h)')
    plt.ylabel('Mean intensity')
    plt.title(f'Intensity profile at pixellocation = {pixelLocation}')

    # define which values to use for regular timeinterval
    whichValuesToUse1 = [0]
    whichValuesToUse2 = np.arange(1, len(profile), 1)
    whichValuesToUseTot = np.append(whichValuesToUse1, whichValuesToUse2)
    equallySpacedTimeFromStart = []
    equallySpacedProfile = []
    for i in whichValuesToUseTot:
        equallySpacedTimeFromStart = np.append(equallySpacedTimeFromStart, timeFromStart[i])
        equallySpacedProfile = np.append(equallySpacedProfile, profile[i])

    ax0.plot(equallySpacedTimeFromStart, np.divide(equallySpacedProfile, normalizeFactor), '.', label=f'equally spaced profile')

    print(f"Length of total profile: {len(profile)}, length of equally spaced profile = {len(equallySpacedProfile)}")
    nrOfDatapoints = len(equallySpacedProfile)

    peaks, _ = scipy.signal.find_peaks(np.divide(equallySpacedProfile, normalizeFactor),  height=0.60, distance=30, prominence=0.03)  # obtain indeces om maxima
    minima, _ = scipy.signal.find_peaks(-np.divide(np.array(equallySpacedProfile), normalizeFactor), height=-0.35, distance=40, prominence=0.03)  # obtain indices of minima

    ax0.plot(np.array(equallySpacedTimeFromStart)[peaks], np.divide(np.array(equallySpacedProfile)[peaks], normalizeFactor), "x")
    ax0.plot(np.array(equallySpacedTimeFromStart)[minima], np.divide(np.array(equallySpacedProfile)[minima], normalizeFactor), "x")
    print(f"Nr. of maxima found: {len(peaks)}, nr. of minima found: {len(minima)}\n"
          f"Maxima at t= {np.array(equallySpacedTimeFromStart)[peaks]} \n"
          f"With indices= {peaks}\n"
          f"Minima at t= {np.array(equallySpacedTimeFromStart)[minima]}\n"
          f"With indices= {minima}")

    ######################################################################################################
    ############## Below: calculate height profiles making use of the known maxima & minima ##############
    ######################################################################################################
    minAndMax = np.concatenate([peaks, minima])
    minAndMaxOrdered = np.sort(minAndMax)

    if MANUALPEAKSELECTION:
        minAndMaxOrdered = selectMinimaAndMaxima(np.divide(equallySpacedProfile, 256))
    ax0.plot(np.array(equallySpacedTimeFromStart)[minAndMaxOrdered], np.divide(np.array(equallySpacedProfile)[minAndMaxOrdered], normalizeFactor), "o")
    hdry = approx_startThickness
    h = []
    xrange = []
    if len(minAndMaxOrdered) > 1:   #if at least 2 extrema are found
        # TODO Below = temporary: plot part between 2 extrema
        temprange = np.arange(minAndMaxOrdered[0], minAndMaxOrdered[1] + 1, 1)
        ax0.plot(np.array(equallySpacedTimeFromStart)[temprange],
                 np.divide(np.array(equallySpacedProfile)[temprange], normalizeFactor), "-")

        #evaluate before first extremum: before index 0
        #between all extrema: between indices 0 - (len(extrema)-1)
        #after last extremum: after (len(extrema)-1)
        for i in range(0, len(minAndMaxOrdered)-1):
            extremum1 = minAndMaxOrdered[i]
            extremum2 = minAndMaxOrdered[i+1]
            if i == 0:  # calculate profile before first extremum
                dataI = np.divide(np.array(equallySpacedProfile)[0:extremum2], normalizeFactor)  # intensity (y) data
                datax = np.array(equallySpacedTimeFromStart)[0:extremum2]  # time (x) data
                #Below: calculate heights of[0 : Extremum1]. Resulting h will not start at 0, because index=0 does not start at an extremum, so must be corrected for.
                h_newrange = idkPre1stExtremum(datax, dataI, extremum1-1, extremum2-1)      #do some -1 stuff because f how indexes work when parsing
                h_newrange = np.subtract(h_newrange, h_newrange[0]) #substract value at index0 from all heights since the programn assumed the height to start at 0 (but it doesn't since we did not tsart at an extremum)

                # adjust entire h_newrange by stitching last value of h_newrange to height of first extremum
                # estimate from known dry height at what thickness the first extremum is.
                #in case of a maximum: 181.1*N
                #in case of minimum: 90.9 + 181.1*N
                if dataI[extremum1] - dataI[0] > 0: #if etrx1 > data[0], next up is a maximum
                    maximas = np.arange(0,181.1*20, 181.1)
                    diff_maximas = np.abs(np.subtract(maximas, hdry))
                    maxIndex = np.where(diff_maximas == min(diff_maximas))
                    h_1stextremum = maximas[maxIndex]
                else:
                    minima = np.arange(90.9, 90.9 + 181.1 * 20, 181.1)
                    diff_minima = np.abs(np.subtract(minima, hdry))
                    minIndex = np.where(diff_minima == min(diff_minima))
                    h_1stextremum = minima[minIndex]

                print(f"Calculated extremum: {h_1stextremum}")
                diff_hExtremumAndFinalnewRange = np.subtract(h_1stextremum, h_newrange[len(h_newrange)-1])     #substract h of extremum with last value of calculated height
                h_newrange = np.add(h_newrange, diff_hExtremumAndFinalnewRange) #add that difference to all calculated heights to stich profiles together
                xrange = np.concatenate([xrange, np.array(equallySpacedTimeFromStart)[0:extremum1]])
                h = np.concatenate([h, h_newrange])

            dataI = np.divide(np.array(equallySpacedProfile)[extremum1:extremum2], normalizeFactor)  # intensity (y) data
            datax = np.array(equallySpacedTimeFromStart)[extremum1:extremum2]  # time (x) data
            h_newrange = np.add(idk(datax, dataI, 0, len(datax)-1), h_1stextremum + i*90.9)
            xrange = np.concatenate([xrange, datax])
            h = np.concatenate([h, h_newrange])
        if i == len(minAndMaxOrdered)-2: #-2 because this then happens after effectively the last extremum
            #input all data, and allow function adkPostLastExtremum() to do analysis from the last & first-to-last extremum
            dataI = np.divide(np.array(equallySpacedProfile)[0:len(equallySpacedTimeFromStart)-1], normalizeFactor)  # intensity (y) data
            datax = np.array(equallySpacedTimeFromStart)[0:len(equallySpacedTimeFromStart)-1]  # time (x) data
            # Below: calculate heights of[Extremum2:end].
            h_newrange = np.add(idkPostLastExtremum(datax, dataI, extremum1 - 1, extremum2 - 1), h_1stextremum + (i+1)*90.9)  # do some -1 stuff because f how indexes work when parsing
            xrange = np.concatenate([xrange, np.array(equallySpacedTimeFromStart)[extremum2:len(equallySpacedTimeFromStart)-1]])
            h = np.concatenate([h, h_newrange])

        fig1, ax1 = plt.subplots()
        ax1.plot(xrange, h)
        ax1.set_ylabel("Height (nm)")
        ax1.set_xlabel("Time (h)")
        ax1.set_title(f"Height profile at pixellocation = {pixelLocation}")
        ax2 = ax1.twinx()
        ax2.set_ylabel('Swelling ratio (h/h$_{0}$)')
        ax2.plot(xrange, np.divide(h, dry_thickness))
        ax2.tick_params(axis='y')
        ax1.set_ylim(bottom=dry_thickness)
        ax2.set_ylim(bottom=1)
        fig1.show()
        fig1.savefig(os.path.join(source, f"Swellingimages\\HeightProfile{pixelLocation}.png"), dpi=300)
    else:
        print(f"No minimum and maximum were found. Only a single extremum at {equallySpacedTimeFromStart[minAndMaxOrdered]}")
    fig0.savefig(os.path.join(source, f"Swellingimages\\IntensityProfile{pixelLocation}.png"), dpi=300)

    #angleArr = [angleFromExtrumum(datax, dataI, f, 0, len(datax)-1) for f in range(0,(len(datax)-1))]
    #heights = [relateAngleToHeight(f, 180) for f in angleArr]
    #print(f"calculated heights are: {heights}")

    #fig1, ax1 = plt.subplots()
    #ax1.plot(datax, heights, )
    #ax1.set_ylabel("Height (nm)")
    #ax1.set_xlabel("Time (h)")
    #fig1.show()





    # Saves data in time vs height profile plot so a csv file.
    wrappedPath = os.path.join(source, f"Swellingimages\\data{pixelLocation}PureIntensity.csv")
    np.savetxt(wrappedPath, [p for p in zip_longest(equallySpacedTimeFromStart, equallySpacedProfile, xrange, h, fillvalue='')], delimiter=',', fmt='%s')
    return xrange, h

def main():
    """
    Analyzes an input 'pixellocation' on a previously analyzed line (with the main.py file methods), as a function of time
    A CSV file of the intensity along the drawn line at time t is read in, the mean intensity extracted at the pixellocation.
    Then, the same is done at t+1, etc.. In the end, Intensity vs time is obtained.
    This data can then be filtered in the frequency domain (removing high and/or low frequencies after a Fourier Transform),
    after which it is returned to time domain. Due to the filtering, it should contain both a real and an imaginary part.
    The phase difference between the two is determined by an atan2 method, resulting in a 'wrapped' plot, which should resemble a sawtooth profile.
    Unwrapping the wrapped profile results in a swelling profile, after a conversion in Z is performed.

    Main changeables:
        pixeLoc1 & 2.
        source
        rangeLength
        Highpass & lowpass filters
    """
    #TODO elapsedtime now starts at 0, even though first csv file might not be true t=0
    #Required changeables. Note that chosen Pixellocs must have enough datapoints around them to average over. Otherwise code fails.
    pixelLoc1 = 1100
    pixelLoc2 = 1101  # pixelLoc1 + 1
    pixelIV = 100  # interval between the two pixellocations to be taken.
    #timeOutput = [0, 5/60, 10/60, 15/60, 30/60, 45/60] #in hours
    timeOutput = [0, 1, 4, 12, 24]
    #source = "F:\\2023_04_06_PLMA_HexaDecane_Basler2x_Xp1_24_s11_split____GOODHALO-DidntReachSplit\\D_analysis_v2\\PROC_20230612121104" # hexadecane, with filtering in /main.py
    #source = "C:\\Users\\ReuvekampSW\\Documents\\InterferometryPython\\export\\PROC_20230721120624"  # hexadecane, NO filtering in /main.py
    #source = "E:\\2023_04_06_PLMA_HexaDecane_Basler2x_Xp1_24_s11_split____GOODHALO-DidntReachSplit\\D_analysisv4\\PROC_20230724185238"  # hexadecane, NO filtering in /main.py, no contrast enhance
    #source = 'E:\\2023_04_06_PLMA_HexaDecane_Basler2x_Xp1_24_s11_split____GOODHALO-DidntReachSplit\\D_analysisv4\\PROC_20230913122145_condensOnly'  # hexadecane, condens only
    #source = "D:\\2023_02_17_PLMA_DoDecane_Basler2x_Xp1_24_S9_splitv2____DECENT_movedCameraEarly\\B_Analysis_V2\\PROC_20230829105238"  # dodecane
    #source = "E:\\2023_08_30_PLMA_Basler2x_dodecane_1_29_S2_ClosedCell\\B_Analysis2\\PROC_20230905134930"  # dodecane 2d
    source = "E:\\2023_09_22_PLMA_Basler2x_hexadecane_1_29S2_split\\B_Analysis\\PROC_20230927135916_imbed"  # hexadecane, imbed

    dry_thickness = 190     #known dry thickness of the brush (for calculation of swelling ratio)
    approx_startThickness = 190 #approximate thickness of the brush at the desired location. Could be different from dry thickness if already partially swollen
    #source = "E:\\2023_02_17_PLMA_DoDecane_Basler2x_Xp1_24_S9_splitv2____DECENT_movedCameraEarly\\B_Analysis\\PROC_20230710212856"      #The dodecane sample

    MANUALPEAKSELECTION = True

    config = ConfigParser()
    configName = [f for f in glob.glob(os.path.join(source, f"config*"))]
    config.read(os.path.join(source, configName[0]))

    # TODO show where your chosen pixel is actually located
    #positiontest(source)
    #showPixellocationv2(1,2, source)

    csvList = [f for f in glob.glob(os.path.join(source, f"process\\*real.csv"))]
    #Length*2 = range over which the intensity will be taken
    rangeLength = 5

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
            elapsedtime.append(float(rows[0])/3600)

            pixelLocation = i
            range1 = pixelLocation - rangeLength
            range2 = pixelLocation + rangeLength
            if (range1 < 1) or (range2>len(rows)):
                raise Exception(f"There were not enough values to average over. Either lower mean-range, or choose different pixel location. range1 = {range1}, range2 = {range2}, Rows={len(rows)}")

            total = 0

            for idx in range(range1, range2):
                total = total + float(rows[idx])
            meanIntensity.append(total / (range2 - range1))

        #makeImagesManualTimeadjust(meanIntensity, elapsedtime, source, pixelLocation, config)
        time, height = makeImagesManualTimeAdjustFromPureIntensity(meanIntensity, elapsedtime, source, pixelLocation, config, dry_thickness, approx_startThickness, MANUALPEAKSELECTION)
        print(f"Idx     Time(h)    Height(nm)")
        for n in timeOutput:
            fileIndex = np.argmin(abs(time-n))
            print(f"{fileIndex}     {time[fileIndex]:.2f}       {height[fileIndex]:.2f}")

    print(f"Read-in lenght of rows from csv file = {len(rows)}")

if __name__ == "__main__":
    main()
    exit()