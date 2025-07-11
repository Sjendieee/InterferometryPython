"""
Determine the height of a (swelling) thin film from the absolute intensity of said film. By following the fringes /
maxima-minima.

This swellingratio analysis allows for investigation of Intensity vs. Distance, at a single timestep.
This results in a swelling profile for every timestep.
Adapted from SwellingRatioAnalysisv2_testing.py
"""

import csv
from datetime import datetime
import json
import logging
import os
import traceback

import matplotlib.pyplot as plt
import glob
import cv2
import pandas as pd
import scipy.signal

from general_functions import image_resize
from line_method import coordinates_on_line, normalize_wrappedspace, mov_mean, timeFormat
import numpy as np
from PIL import Image
from configparser import ConfigParser
from general_functions import conversion_factors
from itertools import zip_longest

from SwellTest import findMiddleCrossing, idk, idkPre1stExtremum, idkPostLastExtremum

from analysis_contactangle import Highlighter

right_clicks = list()
def click_eventSingle(event, x, y, flags, params):
    '''
    Click event for the setMouseCallback cv2 function. Allows to select 2 points on the image and return it coordinates.
    '''
    if event == cv2.EVENT_LBUTTONDOWN:
        global right_clicks
        right_clicks.append([x, y])
    if len(right_clicks) == 1:
        cv2.destroyAllWindows()


#normalize data of minimum and maximum between 0 and 1 resp.
def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def makeImages(profile, timeFromStart, source, pixelLocation):
    if not os.path.exists(os.path.join(source, f"Swellingimages")):
        os.mkdir(os.path.join(source, f"Swellingimages"))
    fig0, ax0 = plt.subplots()
    ax0.plot(timeFromStart, normalizeData(profile), label = f'normalized, unfiltered')
    plt.xlabel('Time (h)')
    plt.ylabel('Mean intensity')
    plt.title(f'Intensity profile. Pixellocation = {pixelLocation}')
    # plt.show()
    #plt.draw()
    #fig0.savefig(os.path.join(source, f"Swellingimages\\IntensityProfile{pixelLocation}.png"), dpi=300)

    print(f"length of profile = {len(profile)}")
    nrOfDatapoints = len(profile)
    print(f"{nrOfDatapoints}")
    hiR = nrOfDatapoints - round(nrOfDatapoints/18)     #OG = /13
    hiR = 0
    loR = 0
    for i in range(hiR,hiR+1,20):       #removing n highest frequencies
        for j in range(loR, loR+1, 20):        #removing n lowest frequencies
            HIGHPASS_CUTOFF = i
            LOWPASS_CUTOFF = j
            NORMALIZE_WRAPPEDSPACE = False
            NORMALIZE_WRAPPEDSPACE_THRESHOLD = 3.14159265359
            conversionZ = 0.02885654477258912
            FLIP = False

            profile_fft = np.fft.fft(profile)  # transform to fourier space
            highPass = HIGHPASS_CUTOFF
            lowPass = LOWPASS_CUTOFF
            mask = np.ones_like(profile).astype(float)
            mask[0:lowPass] = 0
            if highPass > 0:
                mask[-highPass:] = 0
            profile_fft = profile_fft * mask
            fig3, ax3 = plt.subplots()
            ax3.plot(timeFromStart, normalizeData(profile_fft), label=f'hi:{highPass}, lo:{lowPass}')
            ax3.legend()
            fig3.savefig(os.path.join(source, f"Swellingimages\\FFT at {pixelLocation}, hiFil{i}.png"),
                         dpi=300)


            #print(f"Size of dataarray: {len(profile_fft)}")

            profile_filtered = np.fft.ifft(profile_fft)
            ax0.plot(timeFromStart, normalizeData(profile_filtered), label = f'hi:{highPass}, lo:{lowPass}')
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
            ax2.plot(spacedTimeFromStart, unwrapped * conversionZ)
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
            np.savetxt(wrappedPath, [p for p in zip(timeFromStart, unwrapped * conversionZ)], delimiter=',', fmt='%s')
    # now get datapoints we need.
    #unwrapped_um = unwrapped * conversionZ
    #analyzeTimes = np.linspace(0, 57604, 12)
    #analyzeImages = np.array([find_nearest(timeFromStart, t)[1] for t in analyzeTimes])
    #print(analyzeImages)

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
    print(plt.get_fignums())
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

def selectMinimaAndMaxima(y : np.ndarray, idx) -> list:
    """
    :param y: np.array of y-data
    :param idx:
    :return outputExtrema: list with indices of minima and maxima
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    x = np.arange(0,len(y))
    ax.plot(x, y, '-')
    ax.plot(x, y, '.')
    ax.set(ylim=[min(y)-5, max(y)+5])
    ax.set(title='Select maxima and minima (SwellingFromAbsoluteIntensity.py)')
    fig.show()
    closed = [False]
    def on_close(event):
        closed[0] = True
    # Connect the close event to the figure
    fig.canvas.mpl_connect('close_event', on_close)

    highlighter = Highlighter(ax, x, y) #create highlighter object: allows to select points inside the given ax (corresponding figure)
    # Run a loop to block until the figure is closed
    while not closed[0]:
        fig.canvas.flush_events()

    #plt.show()
    selected_regions = highlighter.mask
    xrange1, yrange1 = x[selected_regions], y[selected_regions]
    outputExtrema = []
    if sum(yrange1) > 0:    #only if extrema are selected (otherwise it'll throw an error)
        extremaRanges = [xrange1[0]]        #always add first x value into array
        for i in range(1,len(xrange1)):     #check for all x elements if their step increase is 1
            if (xrange1[i] - xrange1[i-1]) > 1: #if the step is larger than 1, a new extremum range occurs.
                extremaRanges.append(xrange1[i-1])    #input last x value to end previous range
                extremaRanges.append(xrange1[i])  #input newest x+1 value to start new extremum range
        extremaRanges.append(xrange1[-1])   #always append last x element to close the last extremum range
        #next, find maxima and minima for in every extremum range (so between extremaRanges[0-1, 2-3, 4-5] etc..)
        for i in range(0, len(extremaRanges),2):
            lowerlimitRange = extremaRanges[i]
            upperlimitRange = extremaRanges[i+1]
            if lowerlimitRange == upperlimitRange:      #if only 1 value was selected, just append the index value immideately
                outputExtrema.append(lowerlimitRange)
            else:   #if a range is selected, check if minimum or maximum & append that y value
                if y[(round((upperlimitRange + lowerlimitRange) / 2))] > ((y[upperlimitRange] + y[lowerlimitRange])/2):     #likely to be a maximum if the middle value in range > the mean of first&last value
                    tempPosition = np.argmax(y[lowerlimitRange:upperlimitRange]) + lowerlimitRange  # position of maximum
                else:
                    tempPosition = np.argmin(y[lowerlimitRange:upperlimitRange]) + lowerlimitRange  # position of minimum
                outputExtrema.append(tempPosition)
    else:
        logging.critical("Not enough extrema selected!")
    return outputExtrema

def flipData(data):
    datamax = max(data)
    return [(-x + datamax) for x in data]

def saveDataToFile(data, path, name):           #save a single list of itmens
    if not os.path.exists(os.path.join(path)):
        print(f"ERROR: THIS PATH DOES NOT EXIST YET:\n"
        f"{path}")
    with open(os.path.join(path, f"{name}"), 'w') as fp:
        for item in data:
            # write each item on a new line
            fp.write("%s\n" % item)

def readDataFromfile(path):         #read in a single list of items
    data = []
    with open(path, 'r') as fp:
        for line in fp:
            # remove linebreak from a current name
            # linebreak is the last character of each line
            x = int(line[:-1])
            # add current item to the list
            data.append(x)
    return data

def readDataFromfileV2(file_path):
    """
       Read numbers from a .txt file and convert them to a single array.

       The input data can be written as individual numbers on separate lines
       or on one line divided by a space.

       Parameters:
       - file_path (str): The path to the .txt file.

       Returns:
       - list of int: The array containing the read numbers.
       """
    numbers = []
    with open(file_path, 'r') as file:
        # Read lines from the file
        lines = file.readlines()

        # Check if the data is on separate lines or on one line
        if len(lines) == 1 and ' ' in lines[0]:
            # Data is on one line, split by space
            numbers = [int(num) for num in lines[0].split()]
        else:
            # Data is on separate lines
            numbers = [int(num) for num in lines]
    return numbers

def heightFromIntensityProfileV2(FLIP, MANUALPEAKSELECTION, PLOTSWELLINGRATIO, SAVEFIG, SEPERATEPLOTTING, USESAVEDPEAKS,
                                 ax0, ax1, cmap, colorGradient, dryBrushThickness, elapsedtime, fig0, fig1, idx, idxx,
                                 intensityProfileZoomConverted, knownHeightArr, knownPixelPosition, normalizeFactor,
                                 range1, range2, source, xshifted, vectorNumber, outwardsLengthVector, unitXY= "mm", extraPartIndroplet=0):
    """
    Convert intensity profiles to height profiles:
    For a known light wavelength [lambda] , medium refractive index (n), and order of interference (N), the height (h) of a sample can be calculated.
    h = lambda*N / 2n
    For this, the order of interference is required. In a more general sense however, we can thus say the height difference
    covering ONE fringe (1N) is:
    dh = lambda/2n      , which is a constant (assuming n does not change with varying thickness).

    Experimentally we measure the Intensity (I) of the system, resulting from thin-film interference.
    This means I(h). I vs h has been shown to fit in a cos.
    In this analysis, we use this concept to convert Intensity to height using the general form: I=a*cos(x) + b.
    a & b and b are fitted to the minium&maximum of profile, and half-height respectively. x is in principle equal
    to = (4*pi*h*n / lambda), but the values fall away against each other, leaving only N*2pi.
    covering the range of N=[0, 1], effectively we model Intensity profile of the height of '[0, dh]'

    In this analysis, we do this between each maximum and nearby minium, to best approximate the 'local' a&b values in the 'cos' function.
    So in practice we evaluate half-fringes every time: N=[0, 0.5], N=[0.5, 1], [1, 1.5] etc..

    Intensity profiles in between min- and maxima are then easily evaluated, as both extremum I1 and I2 are known.
    For Intensity profiles outside the first & last extremum no 'true' extremum intensity value is known to determine a&b.
    We therefore approximate those by using the Intensity of the most nearby extremum I3, or the largest/smallest profile Intensity value min/max(I)
    (if that Imax/min > I3)

    TODO's
    -hardcoded height with wavelength: change to vary with input wavelength.   (now. dh=181.1nm for 1 fringe!)

    """

    if not os.path.exists(os.path.join(source, f"Swellingimages")):
        os.mkdir(os.path.join(source, f"Swellingimages"))
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})  # print arrays later with only 2 decimals

    # TODO prominances etc have to be adjusted manually it seems in order to have proper peakfinding
    peaks, _ = scipy.signal.find_peaks(np.divide(intensityProfileZoomConverted, normalizeFactor), height=0.5,
                                       distance=40, prominence=0.05)  # obtain indeces om maxima
    minima, _ = scipy.signal.find_peaks(np.divide(-np.array(intensityProfileZoomConverted), normalizeFactor),
                                        height=-0.35, distance=40, prominence=0.05)  # obtain indices of minima
    print(f"\n------------------- Investigating height profile image number {idx}, vector number {vectorNumber} -------------------\n"
          f"---------------------------------------- HeightFromIntensityProfileV2() --------------------------------------\n")
    print(f"\n\nT = {timeFormat(elapsedtime)}\nMaxima at index: {peaks} \nAt x position: {np.array(xshifted)[peaks]}\nWith Intensity values: {np.array(intensityProfileZoomConverted)[peaks]}")
    print(f"T = {timeFormat(elapsedtime)}\nMinima at index: {minima} \nAt x position: {np.array(xshifted)[minima]}\nWith Intensity values: {np.array(intensityProfileZoomConverted)[minima]}\n")
    # for showing/plotting automatically picked peaks
    # ax0.plot(np.array(xshifted)[peaks], np.array(intensityProfileZoomConverted)[peaks], "x")
    # ax0.plot(np.array(xshifted)[minima], np.array(intensityProfileZoomConverted)[minima], "x")
    print(f"Nr. of maxima found: {len(peaks)}, nr. of minima found: {len(minima)}\n"
          f"Maxima at distance= {np.array(xshifted)[peaks]} \n"
          f"With indices= {peaks}\n"
          f"Minima at distance= {np.array(xshifted)[minima]}\n"
          f"With indices= {minima}")

    ######################################################################################################
    ############## Below: calculate height profiles making use of the known maxima & minima ##############
    ######################################################################################################
    minAndMax = np.concatenate([peaks, minima])
    minAndMaxOrderedUnsorted = np.sort(minAndMax)
    minAndMaxOrdered = []
    ###Below: sort the list of minima and maxima such that minima and maxima are alternating.
    ### this requires all min & maxima to be 'correctly found' beforehand:
    for i, ival in enumerate(minAndMaxOrderedUnsorted):
        if i == 0:  # always input first extremum
            minAndMaxOrdered.append(ival)
        else:
            if minAndMaxOrdered[-1] in minima:  # if last value in new adjust list is a minimum:
                # then next value should be a maximum OR next minimum should be of a lower intensity value (to find absolute minimum)
                if ival in peaks:
                    minAndMaxOrdered.append(ival)
                # Check if next extremum has a lower intensity value (and replace if yes):
                elif intensityProfileZoomConverted[ival] < intensityProfileZoomConverted[minAndMaxOrdered[-1]]:
                    minAndMaxOrdered[-1] = ival
                # #else, find
                # for maximum in peaks:
                #     if maximum > minAndMaxOrdered[-1]:
                #         minAndMaxOrdered.append(maximum)
                #         break
            elif minAndMaxOrdered[-1] in peaks:  # if last value in new adjust list is a maximum:
                # then next value should be a minimum OR next maximum should have a higher intensity value (to find absolute maximum)
                if ival in minima:
                    minAndMaxOrdered.append(ival)
                # Check if next extremum has a lower intensity value (and replace if yes):
                elif intensityProfileZoomConverted[ival] > intensityProfileZoomConverted[minAndMaxOrdered[-1]]:
                    minAndMaxOrdered[-1] = ival

                # for minimum in minima:
                #     if minimum > minAndMaxOrdered[-1]:
                #         minAndMaxOrdered.append(minimum)
                #         break
            else:
                print(f"Skipped {minAndMaxOrderedUnsorted[i]}")

    # TODO select regions in plot to find minima and maxima
    if MANUALPEAKSELECTION:  # use manually selected peaks, either from a previous time or select new ones now
        if USESAVEDPEAKS:  # use peaks from a previous time (if they exist)
            #TODO this way of saving will create A LOT of files for different vectors in same image: optimization desired
            if os.path.exists(os.path.join(source, f"SwellingImages\\MinAndMaximaHandpicked{idx}_{vectorNumber}_{outwardsLengthVector}_{extraPartIndroplet}.txt")):
                minAndMaxOrdered = readDataFromfileV2(
                    os.path.join(source, f"SwellingImages\\MinAndMaximaHandpicked{idx}_{vectorNumber}_{outwardsLengthVector}_{extraPartIndroplet}.txt"))
                print(f">Imported saved peaks from 'MinAndMaximaHandpicked{idx}_{vectorNumber}_{outwardsLengthVector}_{extraPartIndroplet}.txt'")
            else:
                try:
                    print(f"No saved peaks yet. Select them now:")
                    minAndMaxOrdered = selectMinimaAndMaxima(np.divide(intensityProfileZoomConverted, normalizeFactor), idx)
                    saveDataToFile(minAndMaxOrdered, os.path.join(source, f"SwellingImages"), f"MinAndMaximaHandpicked{idx}_{vectorNumber}_{outwardsLengthVector}_{extraPartIndroplet}.txt")
                except:
                    logging.error("Some error occured while trying to manually select peaks!")
                    print(traceback.format_exc())
        else:  # select new peaks now
            try:
                minAndMaxOrdered = selectMinimaAndMaxima(np.divide(intensityProfileZoomConverted, normalizeFactor), idx)
                saveDataToFile(minAndMaxOrdered, os.path.join(source, f"SwellingImages"), f"MinAndMaximaHandpicked{idx}_{vectorNumber}_{outwardsLengthVector}_{extraPartIndroplet}.txt")
            except:
                logging.error("Some error occured while trying to manually select peaks!")
                print(traceback.format_exc())
        print(f"Handpicked extrema at: \n"
              f"Indices: {[minAndMaxOrdered]}\n"
              f"Distance : {np.array(xshifted)[minAndMaxOrdered]} in {unitXY}")

    ax0.plot(np.array(xshifted)[minAndMaxOrdered], np.array(intensityProfileZoomConverted)[minAndMaxOrdered], "or", label='picked max- & minima')
    ax0.plot(xshifted, intensityProfileZoomConverted, '.', label=f'Time={timeFormat(elapsedtime)}', color=cmap(colorGradient[idxx]))  # plot the intensity profile

    # if FLIP:
    #     xshifted.reverse()
    #     np.flip(intensityProfileZoomConverted)
    #     minAndMaxOrdered = np.subtract(len(xshifted)-1,  minAndMaxOrdered)
    #     minAndMaxOrdered = np.sort(minAndMaxOrdered)
    # TODO below was set to 0 before?
    hdry = dryBrushThickness
    h = []
    xrange = []
    if len(minAndMaxOrdered) > 1:  # if at least 2 extrema are found
        # evaluate before first extremum: before index 0
        # between all extrema: between indices 0 - (len(extrema)-1)
        # after last extremum: after (len(extrema)-1)
        for i in range(0, len(minAndMaxOrdered) - 1):  # iterating from the first to the first-to-last extremum
            extremum1 = minAndMaxOrdered[i]
            extremum2 = minAndMaxOrdered[i + 1]
            # to calculate profile before first extremum
            if i == 0:  # calculate profile before first extremum
                dataI = np.divide(np.array(intensityProfileZoomConverted)[0:extremum2+1],
                                  normalizeFactor)  # intensity (y) data
                datax = np.array(xshifted)[0:extremum2+1]  # time (x) data
                # Below: calculate heights of[0 : Extremum1]. Resulting h will not start at 0, because index=0 does not start at an extremum, so must be corrected for.
                h_newrange = idkPre1stExtremum(datax, dataI, extremum1,
                                               extremum2)  #TODO check: removed -1 # do some -1 stuff because f how indexes work when parsing
                h_newrange = np.subtract(h_newrange, h_newrange[
                    0])  # substract value at index0 from all heights since the programn assumed the height to start at 0 (but it doesn't since we did not tsart at an extremum)

                # TODO below: this is not necesairy for this analysis I think. At the left, we don't known/need to know the height in advance
                # TODO: Set height at index 0 just to 0, and later

                # adjust entire h_newrange by stitching last value of h_newrange to height of first extremum
                # estimate from known dry height at what thickness the first extremum is.
                # in case of a maximum: 181.1*N
                # in case of minimum: 90.9 + 181.1*N
                if dataI[extremum1] - dataI[0] > 0:  # if etrx1 > data[0], next up is a maximum
                    maximas = np.arange(0, 181.1 * 20, 181.1)
                    diff_maximas = np.abs(np.subtract(maximas, hdry))
                    maxIndex = np.where(diff_maximas == min(diff_maximas))
                    h_1stextremum = maximas[maxIndex]
                else:
                    minima = np.arange(90.9, 90.9 + 181.1 * 20, 181.1)
                    diff_minima = np.abs(np.subtract(minima, hdry))
                    minIndex = np.where(diff_minima == min(diff_minima))
                    h_1stextremum = minima[minIndex]

                print(f"Calculated extremum: {h_1stextremum}")
                # TODO adusting normal code: set h_1st extremum to the last value of the calculated height profile
                # this just makes for a smooth profile, which should then start at h=0?
                h_1stextremum = h_newrange[-1]
                print(f"But using extremum: {h_1stextremum}")
                diff_hExtremumAndFinalnewRange = np.subtract(h_1stextremum, h_newrange[
                    -1])  # substract h of extremum with last value of calculated height
                h_newrange = np.add(h_newrange,
                                    diff_hExtremumAndFinalnewRange)  # add that difference to all calculated heights to stich profiles together
                xrange = np.concatenate([xrange, np.array(xshifted)[0:extremum1]])
                h = np.concatenate([h, h_newrange[:-1]])  # main output if this part: height profile before first extremum WITHOUT the height of the last extremum (which will be added in the next iteration)

            # to calculate profiles in between extrema
            dataI = np.divide(np.array(intensityProfileZoomConverted)[extremum1:extremum2+1],
                              normalizeFactor)  # intensity (y) data
            datax = np.array(xshifted)[extremum1:extremum2+1]  # time (x) data
            #Calculate the height profile and add it into the array
            h_newrange = np.add(idk(datax, dataI, 0, len(datax) - 1), h_1stextremum + i * 90.9)
            xrange = np.concatenate([xrange, datax[:-1]])        #TODO datax[1:] ??
            h = np.concatenate([h, h_newrange[:-1]])

            # Once the first-to-last maximum is reached, above the profile between first-to-last and last extremum is calculated.
            # Below, the profile after last extremum is calculated
            if i == len(minAndMaxOrdered) - 2:  # -2 because this then happens after effectively the last extremum
                # input data ranging from the first extremum
                dataI = np.divide(np.array(intensityProfileZoomConverted)[0:len(xshifted)],
                                  normalizeFactor)  # intensity (y) data
                datax = np.array(xshifted)[0:len(xshifted)]  # time (x) data
                # Below: calculate heights of[Extremum2:end].
                ###TODO check if (i) or (i+1), beforehand (i+1) worked, now not?
                h_newrange = np.add(idkPostLastExtremum(datax, dataI, extremum1, extremum2),
                                    h_1stextremum + (i + 1) * 90.9)  # TODO removed: "do some -1 stuff with the extremum1&2 because f how indexes work when parsing"
                # xrange = np.concatenate([xrange, datax])
                xrange = np.concatenate(
                    [xrange, np.array(xshifted)[extremum2:len(xshifted)]])
                h = np.concatenate([h, h_newrange])

        # once entire height profile is calculated, convert to 'correct' height profile
        if FLIP:
            # first plot the data upside down, to have the height more swollen on the left
            h = -np.subtract(h, max(h))
        # then, correct height with a 'known' height somewhere. Can be dry height in dry region, or from a known height vs. time curve at a pixellocation
        diffh = knownHeightArr[idxx] - h[knownPixelPosition]
        print(
            f"Correcting height with {diffh} nm, because known height= {knownHeightArr[idxx]}, and calculated height= {h[knownPixelPosition]}")
        h = np.add(h, diffh)

        h_ratio = np.divide(h, hdry)
        ax1.set_xlabel(f"Distance of chosen range ({unitXY})")
        if PLOTSWELLINGRATIO:
            ax1.set_ylabel("Swelling ratio (h/h$_{0}$)")
            ax1.plot(xrange, h_ratio, label=f'Time={timeFormat(elapsedtime)}', color=cmap(colorGradient[idxx]))
            ax1.set_title(f"Swelling profiles in pixelrange {range1}:{range2}\nImgNr={idx}, VectorNr={vectorNumber}")
            ax1.set_title(f"Calibrated swelling profiles")
            ax1.plot(xrange[knownPixelPosition], h_ratio[knownPixelPosition], 'ok', markersize=9)
            ax1.plot(xrange[knownPixelPosition], h_ratio[knownPixelPosition], 'o', color=cmap(colorGradient[idxx]))
        else:
            ax1.set_ylabel("Film thickness (nm)")
            ax1.plot(xrange, h, label=f'Time={timeFormat(elapsedtime)}', color=cmap(colorGradient[idxx]))
            # ax1.set_title(f"Height profile at time: {timeFormat(elapsedtime)} in pixelrange {range1}:{range2}")
            ax1.set_title(f"Calibrated height profiles\nImgNr={idx}, VectorNr={vectorNumber}")
            ax1.plot(xrange[knownPixelPosition], h[knownPixelPosition], 'ok', markersize=9)
            ax1.plot(xrange[knownPixelPosition], h[knownPixelPosition], 'o', color=cmap(colorGradient[idxx]))
        ax1.legend(loc='upper right')
        ax1.autoscale(enable=True, axis='x', tight=True)
        ax1.autoscale(enable=True, axis='y', tight=True)
        if PLOTSWELLINGRATIO:
            ax1.set_ylim(bottom=0.9)
        else:
            ax1.set_ylim(bottom=0)
        fig1.tight_layout()
        print(f"Mean thickness (50 points) far from droplet: {np.mean(h[-50:-1])}")

        # Saves data in time vs height profile plot so a csv file.
        wrappedPath = os.path.join(source,
                                   f"Swellingimages\\data_n={idx}_k={vectorNumber}_t={timeFormat(elapsedtime)}_anchor{knownPixelPosition}_PureIntensity.csv")
        d = dict(
            {f'xshifted ({unitXY})': xshifted, 'Intensity converted (-)': intensityProfileZoomConverted, f'xrange ({unitXY})': xrange,
             'height (nm)': h, 'Swelling ratio (-)': h_ratio})
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))  # pad shorter colums with NaN's
        df.to_csv(wrappedPath, index=False)
        # np.savetxt(wrappedPath, [p for p in zip_longest('xshifted (mm)', 'Insensity converted (-)', 'xrange (mm)', 'height (nm)', 'Swelling ratio (-)', fillvalue='')], delimiter=',', fmt='%s')
        # np.savetxt(wrappedPath, [p for p in zip_longest(xshifted, intensityProfileZoomConverted, xrange, h, h_ratio, fillvalue='')], delimiter=',', fmt='%s')
    else:
        print(f"No minimum and maximum were found. Only a single extremum")
    ax0.legend(loc='upper right')
    ax0.set_ylabel("Intensity (-)")
    ax0.set_xlabel(f"Distance of chosen range ({unitXY})")
    ax0.set_title(f"Intensity profile\n ImgNr={idx}, VectorNr={vectorNumber}")
    if SAVEFIG:
        ax0.autoscale(enable=True, axis='x', tight=True)
        fig0.savefig(os.path.join(source, f"Swellingimages\\n{idx}_k{vectorNumber}_Intensity.png"), dpi=300)
        fig1.savefig(os.path.join(source, f"Swellingimages\\HeightProfile{timeFormat(elapsedtime)}_k{vectorNumber}.png"), dpi=300)
    if SEPERATEPLOTTING:
        plt.close(fig1)
        plt.close(fig0)
    return h, h_ratio

def main():
    """
    Determine the height of a (swelling) thin film from the absolute intensity of said film over a given range by following the fringes /
    maxima-minima.

    This swellingratio analysis allows for investigation of Intensity vs. Distance, at a single timestep.
    This results in a swelling profile for every timestep.
    Adapted from SwellingRatioAnalysisv2_testing.py
    """

    __author__ = 'Sander Reuvekamp'
    __version__ = '1.0'

    """"Changeables: """
    #source = "F:\\2023_04_06_PLMA_HexaDecane_Basler2x_Xp1_24_s11_split____GOODHALO-DidntReachSplit\\D_analysis_v2\\PROC_20230612121104"
    #source = "C:\\Users\\ReuvekampSW\\PycharmProjects\\InterferometryPython\\export\\PROC_20230724185238"  # hexadecane, NO filtering in /main.py, no contrast enhance
    source = "E:\\2023_04_06_PLMA_HexaDecane_Basler2x_Xp1_24_s11_split____GOODHALO-DidntReachSplit\\D_analysisv4\\PROC_20230724185238" # hexadecane, NO filtering in /main.py, no contrast enhance
    #source = 'E:\\2023_04_06_PLMA_HexaDecane_Basler2x_Xp1_24_s11_split____GOODHALO-DidntReachSplit\\D_analysisv4\\PROC_20230913122145_condensOnly'  # hexadecane, condens only
    #source = "F:\\2023_02_17_PLMA_DoDecane_Basler2x_Xp1_24_S9_splitv2____DECENT_movedCameraEarly\\B_Analysis_V2\\PROC_20230829105238"   #dodecane swelling profiles, not filtering no contrast enhance
    #source = "E:\\2023_08_30_PLMA_Basler2x_dodecane_1_29_S2_ClosedCell\\B_Analysis2\\PROC_20230905134930"  # dodecane 2d
    #source = "D:\\2023_09_21_PLMA_Basler2x_tetradecane_1_29S2_split_ClosedCell\\B_Analysis\\PROC_20230922150617"  # tetradecane split, imbed
    source = "M:\\Enqing\\Halo_Zeiss20X\\Img2\\Line1"

    extraRangeInDroplet = 300
    range1 = 1249  - extraRangeInDroplet     #2320       #start x left for plotting
    range2 = 7957     # len(swellingProfile)

    #####Enqing monochromatic images
    knownPixelPosition = range2 - range1 - 1  #index pixellocation at which the brush height is known at various times. integer of intex
    dryBrushThickness = 160  # 160                 # dry brush thickness (measured w/ e.g. ellipsometry)
    idxArrToUse = [0]  # id of csv files to use
    knownHeightArr = [160]  # Known brush swelling at pixellocation in nm for certain timesteps   #in nm
    zeroImage = 0

    ###hexadecane v1
    # knownPixelPosition = 2550 - range1 - 1 #pixellocation at which the bursh height is known at various times
    # dryBrushThickness = 167.4       #160                 # dry brush thickness (measured w/ e.g. ellipsometry)
    # idxArrToUse = [0, 50, 95, 206, 395]         #id of csv files to use
    # knownHeightArr = [260.44, 351.98, 408.30, 443.52]   #Known brush swelling at pixellocation in nm for certain timesteps   #in nm
    # zeroImage = 1
    # ###hexadecane Condens only
    # knownPixelPosition = 2200 - range1 - 1  # pixellocation at which the bursh height is known at various times
    # dryBrushThickness = 172.8  # 160                 # dry brush thickness (measured w/ e.g. ellipsometry)
    # idxArrToUse = [0, 50, 95, 206, 395]  # id of csv files to use
    # knownHeightArr = [ 328.87, 381.28, 445.43, 468.27]  #172.8, Known brush swelling at pixellocation in nm for certain timesteps   #in nm
    # zeroImage = 1

    # ###dodecane
    # knownPixelPosition = 2085 - range1 - 1  # pixellocation at which the bursh height is known at various times
    # dryBrushThickness = 160  # dry brush thickness (measured w/ e.g. ellipsometry)
    # idxArrToUse = [0, 14, 19, 22, 30, 45, 75, 105]  # id of csv files to use
    # knownHeightArr = [252.71, 351.34, 378.37, 395.66, 429.78, 453.8, 493.0, 507.2]  # Total Known brush height at pixellocation in nm for certain timesteps   #in nm
    # zeroImage = 0       #1 to use the first image ONLY as a background reference, 0 to also analyse it.

    # ###dodecane v2
    # knownPixelPosition = 2330 - range1 - 1  # pixellocation at which the bursh height is known at various times
    # dryBrushThickness = 190  # dry brush thickness (measured w/ e.g. ellipsometry)
    # idxArrToUse = [0, 10, 20, 30, 60, 90]  # id of csv files to use
    # knownHeightArr = [203, 325, 404, 492, 535, 538]  # Total Known brush height at pixellocation in nm for certain timesteps   #in nm
    # zeroImage = 0       #1 to use the first image ONLY as a background reference, 0 to also analyse it.

    ### Tetradecane split, imbed
    # knownPixelPosition = 2050 - range1 - 1  # pixellocation at which the bursh height is known at various times
    # dryBrushThickness = 190  # dry brush thickness (measured w/ e.g. ellipsometry)
    # idxArrToUse = [0, 47, 62, 122, 212, 332]  # id of csv files to use
    # knownHeightArr = [181.1, 343.06, 388.29, 470.54, 507.94, 522.23]  # Total Known brush height at pixellocation in nm for certain timesteps   #in nm
    # #knownHeightArr = [181, 584, 610, 611, 631]
    # zeroImage = 0       #1 to use the first image ONLY as a background reference, 0 to also analyse it.

    #Below: evaluation of peaks: change as necessary!
    EVALUATERIGHTTOLEFT = False         #evaluate from left to right, or the other way around    (required for correct conversion of intensity to height profile)
    MANUALPEAKSELECTION = True     #use peaks selected by manual picking (thus not the automatic peakfinder).
    USESAVEDPEAKS = False        #True: use previously manually selected peaks.  False: opens interative plot, in which peak regions can be selected
    REMOVEBACKGROUNDNOISE = False        #Divide by the intensity of 1st image. If this is set to True, set normalizeFactor to 1

    #Below: plotting purposes. Change whether to output & format
    PLOTSWELLINGRATIO = False        #True for swelling ratio, False for height profiles
    SAVEFIG = True
    SHOWFIG = True

    #Below: plotting purposes (output format). Generally no need to change
    normalizeFactor = 1               #normalize intensity by camera intensity range: 256, or use 1 if not normalizing
    FLIP = True                 #True=flip data after h analysis to have the height increase at the left
    MOVMEAN = 5              #average the intensity values to obtain a smoother profile (at a loss of peak intensity)
    SEPERATEPLOTTING = True     #true to plot the intensity profiles in seperate figures
    colorscheme = 'plasma'
    outputFormatXY = 'mm'       #'pix' or 'mm'
    #XLIM - True; Xlim = []
    YLIM = True; Ylim = [-50, 650]  #ylim for swelling profiles (only used when plotting absolute swelling height)

    """"End of changeables"""

    config = ConfigParser()
    configName = [f for f in glob.glob(os.path.join(source, f"config*"))]
    config.read(os.path.join(source, configName[0]))
    conversionFactorXY, conversionFactorZ, unitXY, unitZ = conversion_factors(config)
    if not os.path.exists(os.path.join(source, f"Swellingimages")):
        os.mkdir(os.path.join(source, f"Swellingimages"))
    print(f"With this conversionXY, 1000 pixels = {conversionFactorXY*1000} mm, \n"
          f"and 1 mm = {1/conversionFactorXY} pixels")

    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})        #print arrays later with only 2 decimals
    csvList = [f for f in glob.glob(os.path.join(source, f"process\\*real.csv"))]
    cmap = plt.get_cmap(colorscheme)
    colorGradient = np.linspace(0, 1, len(knownHeightArr))
    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()
    idxx = 0
    try:
        for idx, n in enumerate(csvList):
            if idx in idxArrToUse[:]:    #50, 95, 206 Show intensity profiles from unadjusted/unfiltered intensity profiles from /main.py           # 50, 95, 206,
                file = open(n)
                csvreader = csv.reader(file)
                rows = []
                for row in csvreader:
                    rows.append(float(row[0]))
                file.close()
                elapsedtime = rows[0]
                intensityProfile = rows[1:]
                intensityProfile = mov_mean(intensityProfile, MOVMEAN)  #apply a moving average to the intensity array
                lengthOfData = len(intensityProfile)
                if EVALUATERIGHTTOLEFT:
                    intensityProfile.reverse()
                    if idx == idxArrToUse[0]:
                        #knownPixelPosition = knownPixelPosition + range1       #TODO commented out
                        range2temp = lengthOfData - range1
                        range1 = lengthOfData - range2
                        range2 = range2temp
                        #knownPixelPosition = lengthOfData - knownPixelPosition - range1    #TODO commented out
                        knownPixelPosition = range2 - range1 - 1
                intensityProfileZoom = intensityProfile[range1:range2]      #only look at a certain range in the intensity profile
                if REMOVEBACKGROUNDNOISE:           #divide intensity profile by intensity profile at t=0 to 'remove background noise'
                    if idx == 0:
                        backgroundIntensityZoom = intensityProfileZoom
                        intensityProfileZoom = np.divide(intensityProfileZoom, backgroundIntensityZoom)
                    else:
                        intensityProfileZoom = np.divide(intensityProfileZoom, backgroundIntensityZoom)
                intensityProfileZoomConverted = (intensityProfileZoom)
                plt.ylabel(f"Intensity (-)")
                if outputFormatXY == 'pix':
                    x = np.linspace(range1, range2, range2 - range1)
                    plt.xlabel(f"Distance (pixels)")
                elif outputFormatXY == 'mm':
                    x = np.linspace(range1, range2, range2 - range1) * conversionFactorXY
                    plt.xlabel(f"Distance ({unitXY})")
                else:
                    print("wrong format input")
                plt.title(f"Intensity profile starting at pixel: {range1}")
                xshifted = [q - min(x) for q in x]
                ax0.plot(xshifted, intensityProfileZoomConverted, '.', label=f'Time={timeFormat(elapsedtime)}', color = cmap(colorGradient[idxx]))
                #ax0.plot(xshifted, np.zeros(len(xshifted)), 'w-')        #line at y=0

                ###### Up untill now: only splot intensity profiles in desired range.
                #Below: convert intensity profiles to height profiles
                ######

            if idx in idxArrToUse[zeroImage:]:   #50, 95, 206To make swellingprofiles from the previously shown intensityprofiles
                #TODO prominances etc have to be adjusted manually it seems in order to have proper peakfinding
                peaks, _ = scipy.signal.find_peaks(np.divide(intensityProfileZoomConverted, normalizeFactor), height=0.5, distance=40, prominence=0.05)        #obtain indeces om maxima
                minima, _ = scipy.signal.find_peaks(np.divide(-np.array(intensityProfileZoomConverted), normalizeFactor), height=-0.35, distance=40, prominence=0.05)  #obtain indices of minima

                print(f"\n\nT = {timeFormat(elapsedtime)}\nMaxima at index: {peaks} \nAt x position: {np.array(xshifted)[peaks]}\nWith Intensity values: {np.array(intensityProfileZoomConverted)[peaks]}")
                print(f"T = {timeFormat(elapsedtime)}\nMinima at index: {minima} \nAt x position: {np.array(xshifted)[minima]}\nWith Intensity values: {np.array(intensityProfileZoomConverted)[minima]}\n")

                #for showing/plotting automatically picked peaks
                #ax0.plot(np.array(xshifted)[peaks], np.array(intensityProfileZoomConverted)[peaks], "x")
                #ax0.plot(np.array(xshifted)[minima], np.array(intensityProfileZoomConverted)[minima], "x")

                print(f"Nr. of maxima found: {len(peaks)}, nr. of minima found: {len(minima)}\n"
                      f"Maxima at distance= {np.array(xshifted)[peaks]} \n"
                      f"With indices= {peaks}\n"
                      f"Minima at distance= {np.array(xshifted)[minima]}\n"
                      f"With indices= {minima}")

                ######################################################################################################
                ############## Below: calculate height profiles making use of the known maxima & minima ##############
                ######################################################################################################
                minAndMax = np.concatenate([peaks, minima])
                minAndMaxOrderedUnsorted = np.sort(minAndMax)

                minAndMaxOrdered = []
                ###Below: sort the list of minima and maxima such that minima and maxima are alternating.
                ### this requires all min & maxima to be 'correctly found' beforehand:
                if idx == 22:
                    print("hello")
                for i, ival in enumerate(minAndMaxOrderedUnsorted):
                    if i == 0:  #always input first extremum
                        minAndMaxOrdered.append(ival)
                    else:
                        if minAndMaxOrdered[-1] in minima:   #if last value in new adjust list is a minimum:
                            #then next value should be a maximum OR next minimum should be of a lower intensity value (to find absolute minimum)
                            if ival in peaks:
                                minAndMaxOrdered.append(ival)
                            #Check if next extremum has a lower intensity value (and replace if yes):
                            elif intensityProfileZoomConverted[ival] < intensityProfileZoomConverted[minAndMaxOrdered[-1]]:
                                minAndMaxOrdered[-1] = ival
                            # #else, find
                            # for maximum in peaks:
                            #     if maximum > minAndMaxOrdered[-1]:
                            #         minAndMaxOrdered.append(maximum)
                            #         break
                        elif minAndMaxOrdered[-1] in peaks:   #if last value in new adjust list is a maximum:
                            #then next value should be a minimum OR next maximum should have a higher intensity value (to find absolute maximum)
                            if ival in minima:
                                minAndMaxOrdered.append(ival)
                            #Check if next extremum has a lower intensity value (and replace if yes):
                            elif intensityProfileZoomConverted[ival] > intensityProfileZoomConverted[minAndMaxOrdered[-1]]:
                                minAndMaxOrdered[-1] = ival

                            # for minimum in minima:
                            #     if minimum > minAndMaxOrdered[-1]:
                            #         minAndMaxOrdered.append(minimum)
                            #         break
                        else:
                            print(f"Skipped {minAndMaxOrderedUnsorted[i]}")

                #TODO select regions in plot to find minima and maxima
                if MANUALPEAKSELECTION:     #use manually selected peaks, either from a previous time or select new ones now
                    if USESAVEDPEAKS:       #use peaks from a previous time (if they exist)
                        if os.path.exists(os.path.join(source, f"SwellingImages\\MinAndMaximaHandpicked{idx}.txt")):
                            minAndMaxOrdered = readDataFromfile(os.path.join(source, f"SwellingImages\\MinAndMaximaHandpicked{idx}.txt"))
                        else:
                            print(f"No saved peaks yet. Select them now:")
                            minAndMaxOrdered = selectMinimaAndMaxima(np.divide(intensityProfileZoomConverted, normalizeFactor), idx)
                    else:                   #select new peaks now
                        minAndMaxOrdered = selectMinimaAndMaxima(np.divide(intensityProfileZoomConverted, normalizeFactor), idx)
                    print(f"Handpicked extrema at: \n"
                          f"Indices: {[minAndMaxOrdered]}\n"
                          f"Distance: {np.array(xshifted)[minAndMaxOrdered]}")
                    saveDataToFile(minAndMaxOrdered, os.path.join(source, f"SwellingImages"), f"MinAndMaximaHandpicked{idx}.txt" )

                ax0.plot(np.array(xshifted)[minAndMaxOrdered], np.array(intensityProfileZoomConverted)[minAndMaxOrdered], "ob")

                # if FLIP:
                #     xshifted.reverse()
                #     np.flip(intensityProfileZoomConverted)
                #     minAndMaxOrdered = np.subtract(len(xshifted)-1,  minAndMaxOrdered)
                #     minAndMaxOrdered = np.sort(minAndMaxOrdered)

                #TODO below was set to 0 before?
                hdry = dryBrushThickness
                h = []
                xrange = []
                if len(minAndMaxOrdered) > 1:  # if at least 2 extrema are found
                    # evaluate before first extremum: before index 0
                    # between all extrema: between indices 0 - (len(extrema)-1)
                    # after last extremum: after (len(extrema)-1)
                    for i in range(0, len(minAndMaxOrdered) - 1):   #iterating from the first to the first-to-last extremum
                        extremum1 = minAndMaxOrdered[i]
                        extremum2 = minAndMaxOrdered[i + 1]
                        #to calculate profile before first extremum
                        if i == 0:  # calculate profile before first extremum
                            dataI = np.divide(np.array(intensityProfileZoomConverted)[0:extremum2+1],   #TODO Changed +1
                                              normalizeFactor)  # intensity (y) data
                            datax = np.array(xshifted)[0:extremum2+1]  # time (x) data #TODO Changed +1
                            # Below: calculate heights of[0 : Extremum1]. Resulting h will not start at 0, because index=0 does not start at an extremum, so must be corrected for.
                            h_newrange = idkPre1stExtremum(datax, dataI, extremum1,
                                                           extremum2)  # #TODO Changed extreumum -1 do some -1 stuff because f how indexes work when parsing
                            h_newrange = np.subtract(h_newrange, h_newrange[
                                0])  # substract value at index0 from all heights since the programn assumed the height to start at 0 (but it doesn't since we did not tsart at an extremum)


                            #TODO below: this is not necesairy for this analysis I think. At the left, we don't known/need to know the height in advance
                            #TODO: Set height at index 0 just to 0, and later

                            # adjust entire h_newrange by stitching last value of h_newrange to height of first extremum
                            # estimate from known dry height at what thickness the first extremum is.
                            # in case of a maximum: 181.1*N
                            # in case of minimum: 90.9 + 181.1*N
                            if dataI[extremum1] - dataI[0] > 0:  # if etrx1 > data[0], next up is a maximum
                                maximas = np.arange(0, 181.1 * 20, 181.1)
                                diff_maximas = np.abs(np.subtract(maximas, hdry))
                                maxIndex = np.where(diff_maximas == min(diff_maximas))
                                h_1stextremum = maximas[maxIndex]
                            else:
                                minima = np.arange(90.9, 90.9 + 181.1 * 20, 181.1)
                                diff_minima = np.abs(np.subtract(minima, hdry))
                                minIndex = np.where(diff_minima == min(diff_minima))
                                h_1stextremum = minima[minIndex]

                            print(f"Calculated extremum: {h_1stextremum}")
                            #TODO adusting normal code: set h_1st extremum to the last value of the calculated height profile
                            #this just makes for a smooth profile, which hsould then start at h=0?
                            h_1stextremum = h_newrange[-1]
                            print(f"But using extremum: {h_1stextremum}")
                            diff_hExtremumAndFinalnewRange = np.subtract(h_1stextremum, h_newrange[-1])  # substract h of extremum with last value of calculated height
                            h_newrange = np.add(h_newrange,
                                                diff_hExtremumAndFinalnewRange)  # add that difference to all calculated heights to stich profiles together
                            xrange = np.concatenate([xrange, np.array(xshifted)[0:extremum1]])
                            h = np.concatenate([h, h_newrange[:-1]])     #TODO changed [:-1] #main output if this part: height profile before first extremum.

                        #to calculate profiles in between extrema
                        dataI = np.divide(np.array(intensityProfileZoomConverted)[extremum1:extremum2+1],#TODO changed +1
                                          normalizeFactor)  # intensity (y) data
                        datax = np.array(xshifted)[extremum1:extremum2+1]  #TODO changed +1 # time (x) data
                        h_newrange = np.add(idk(datax, dataI, 0, len(datax) - 1), h_1stextremum + i * 90.9)
                        xrange = np.concatenate([xrange, datax[:-1]])       #TODO Changed [:-1]
                        h = np.concatenate([h, h_newrange[:-1]])#TODO Changed [:-1]

                        #Once the first-to-last maximum is reached, above the profile between first-to-last and last extremum is calculated.
                        #Below, the profile after last extremum is calculated
                        if i == len(minAndMaxOrdered) - 2:  # -2 because this then happens after effectively the last extremum
                            # input data ranging from the first extremum
                            dataI = np.divide(np.array(intensityProfileZoomConverted)[0:len(xshifted) - 1],
                                              normalizeFactor)  # intensity (y) data
                            datax = np.array(xshifted)[0:len(xshifted) - 1]  # time (x) data
                            # Below: calculate heights of[Extremum2:end].
                            ###TODO check if (i) or (i+1), beforehand (i+1) worked, now not?
                            h_newrange = np.add(idkPostLastExtremum(datax, dataI, extremum1 - 1, extremum2 - 1),
                                                h_1stextremum + (i+1) * 90.9)  # do some -1 stuff because f how indexes work when parsing
                            # xrange = np.concatenate([xrange, datax])
                            xrange = np.concatenate([xrange, np.array(xshifted)[extremum2:len(xshifted)]])  #TODO removed len()-1
                            h = np.concatenate([h, h_newrange])

                    #once entire height profile is calculated, convert to 'correct' height profile
                    if FLIP:
                        #first plot the data upside down, to have the height more swollen on the left
                        h = -np.subtract(h, max(h))
                        #then, correct height with a 'known' height somewhere. Can be dry height in dry region, or from a known height vs. time curve at a pixellocation
                        diffh = knownHeightArr[idxx] - h[knownPixelPosition]
                        print(f"Correcting height with {diffh} nm, because known height= {knownHeightArr[idxx]}, and calculated height= {h[knownPixelPosition]}")
                        h = np.add(h, diffh)

                    h_ratio = np.divide(h, hdry)
                    ax1.set_xlabel("Distance of chosen range (mm)")
                    if PLOTSWELLINGRATIO:
                        ax1.set_ylabel("Swelling ratio (h/h$_{0}$)")
                        ax1.plot(xrange, h_ratio, label=f'Time={timeFormat(elapsedtime)}', color=cmap(colorGradient[idxx]))
                        ax1.set_title(f"Swelling profiles in pixelrange {range1}:{range2}")
                        ax1.set_title(f"Calibrated swelling profiles")
                        ax1.plot(xrange[knownPixelPosition], h_ratio[knownPixelPosition], 'ok', markersize=9)
                        ax1.plot(xrange[knownPixelPosition], h_ratio[knownPixelPosition], 'o', color=cmap(colorGradient[idxx]))
                    else:
                        ax1.set_ylabel("Film thickness (nm)")
                        ax1.plot(xrange, h, label=f'Time={timeFormat(elapsedtime)}', color=cmap(colorGradient[idxx]))
                        #ax1.set_title(f"Height profile at time: {timeFormat(elapsedtime)} in pixelrange {range1}:{range2}")
                        ax1.set_title(f"Calibrated height profiles")
                        ax1.plot(xrange[knownPixelPosition], h[knownPixelPosition], 'ok', markersize=9)
                        ax1.plot(xrange[knownPixelPosition], h[knownPixelPosition], 'o', color=cmap(colorGradient[idxx]))
                    print(f"Mean thickness (50 points) far from droplet: {np.mean(h[-50:-1])}")

                    # Saves data in time vs height profile plot so a csv file.
                    wrappedPath = os.path.join(source, f"Swellingimages\\data{timeFormat(elapsedtime)}_anchor{knownPixelPosition}_PureIntensity.csv")
                    d = dict({'xshifted (mm)' : xshifted, 'Insensity converted (-)' :intensityProfileZoomConverted, 'xrange (mm)' : xrange, 'height (nm)' : h, 'Swelling ratio (-)' : h_ratio})
                    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d.items() ]))      #pad shorter colums with NaN's
                    df.to_csv(wrappedPath, index=False)
                    #np.savetxt(wrappedPath, [p for p in zip_longest('xshifted (mm)', 'Insensity converted (-)', 'xrange (mm)', 'height (nm)', 'Swelling ratio (-)', fillvalue='')], delimiter=',', fmt='%s')
                    #np.savetxt(wrappedPath, [p for p in zip_longest(xshifted, intensityProfileZoomConverted, xrange, h, h_ratio, fillvalue='')], delimiter=',', fmt='%s')
                else:
                    print(f"No minimum and maximum were found. Only a single extremum")

                ax0.legend(loc='upper right')
                ax0.set_ylabel("Intensity (-)")
                ax0.set_xlabel("Distance of chosen range (mm)")
                ax0.set_title("Intensity profile")
                ax1.legend(loc='upper right')
                if SAVEFIG:
                    ax0.autoscale(enable=True, axis='x', tight=True)
                    fig0.savefig(os.path.join(source, f"Swellingimages\\{idx}Intensity.png"),dpi=300)
                    ax1.autoscale(enable=True, axis='x', tight=True)
                    ax1.autoscale(enable=True, axis='y', tight=True)
                    ax1.set_ylim(bottom=0.9)
                    fig1.savefig(os.path.join(source, f"Swellingimages\\HeightProfile{timeFormat(elapsedtime)}.png"), dpi=300)
                if SHOWFIG:
                    plt.show()
                if SEPERATEPLOTTING:
                    plt.close(fig1)
                    plt.close(fig0)
                    fig0, ax0 = plt.subplots()
                    fig1, ax1 = plt.subplots()
                idxx = idxx + 1
    except Exception:
        logging.critical(f"Some error occured. Traceback provided")
        print(traceback.format_exc())

    # Save statistics
    proc = f"PROC_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    stats = {}  # save statistics of this analysis
    stats['About'] = {}
    stats['About']['__author__'] = __author__
    stats['About']['__version__'] = __version__
    stats['startDateTime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f %z')
    stats['source'] = source
    stats['range1'] = range1
    stats['range2'] = range2
    stats['datalength'] = lengthOfData
    stats['knownPixelPosition'] = knownPixelPosition
    stats['dryBrushThickness'] = dryBrushThickness
    stats['idxArrToUse'] = idxArrToUse
    stats['knownHeightArr'] = knownHeightArr
    stats['zeroImage'] = zeroImage
    stats['EVALUATERIGHTTOLEFT'] = EVALUATERIGHTTOLEFT
    stats['REMOVEBACKGROUNDNOISE'] = REMOVEBACKGROUNDNOISE
    stats['MOVMEAN'] = MOVMEAN
    stats['conversionFactorXY'] = conversionFactorXY
    stats['conversionFactorZ'] = conversionFactorZ
    stats['unitXY'] = unitXY
    stats['unitZ'] = unitZ
    stats['analysis'] = {}

    with open(os.path.join(source, f"{proc}_h-profile settings.json"), 'w') as f:
        json.dump(stats, f, indent=4)

if __name__ == "__main__":
    main()
    exit()