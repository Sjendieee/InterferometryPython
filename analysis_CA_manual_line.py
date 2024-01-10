import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
import json
import os
import csv
from matplotlib.widgets import RectangleSelector
from datetime import datetime
import glob
from AnalysisCA_spatial import filePathsFunction, getTimeStamps, timeFormat
from general_functions import image_resize
from line_method import coordinates_on_line, align_arrays, mov_mean, normalize_wrappedspace
import logging

from plotting import saveTempPlot

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
def manual_line_chooser(imgpath, conversionFactorXY, conversionFactorZ, unitXY, unitZ, analysisFolder, timeElapsed, **kwargs):
    '''
    Analyzes a 2D slice using 2D fourier transforms.
    User can select 2 points in the image for the slice, or set POINTA and POINTB in the config.
    A linear line is fitted through these 2 points and beyond to the edges of the image. Next, several other slices
    parallel to this line are calculated, a total of 2*PARALLEL_SLICES_EXTENSION+1 in total. These are all averaged to
    obtain an average slice. The slice is transformed to the Fourier domain where low and high frequencies are filtered
    out. Back in spatial domain the atan2 is taken from the imag and real part, to obtain the wrapped space
    distribution. The stepped line is then unwrapped to obtain the final height profile.
    :param: imgpath  full path to the image being investigated
    :param: conversionFactorXY   conversion from pixel to desired unit in XY-plane
    :param: conversionFactorZ    conversion from pixel to desired unit in Z-direction
    :param: unitXY               unit of the XY-plane conversionfactor
    :param: unitZ                unit of the Z-direction conversionFactor
    :param: analysisFolder       Full path to the folder to dump analysis images
    :param: timeElapsed          Time elapsed from 0 image (seconds)
    :return: unwrapped line (non-converted!)
    :return: points A
    :return: points B
    '''
    SAVEIMAGES = True       #save [grey] images or not
    SavingTemp = False       #Save intermediate plots (fourier, wrapped, etc)
    SliceWidth = 0    # get number of extra slices on each side of the center slice from config
    movmeanN = 1    #number of values which should be taken for the moving average of the profile (i.e. 1 for no averaging, or 3, or 100)
    highPass = 2
    lowPass = 2500
    FLIP = True

    savename = os.path.basename(os.path.normpath(imgpath))
    img_raw = cv2.imread(imgpath)
    im_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE()             #enhances contrast - only useable for CA analysis. Even then doesn't seem to work great sometimes
    # im_gray = clahe.apply(im_gray)

    # Save gray image to desired output location
    if SAVEIMAGES:
        if not os.path.exists(os.path.join(analysisFolder, 'GreyImages')):
            os.mkdir(os.path.join(analysisFolder, 'GreyImages'))
        cv2.imwrite(os.path.join(analysisFolder, f"GreyImages\\greyscaled_{savename}.png"), im_gray)

    # get the points for the center linear slice

    print('Select 2 point one-by-one for the slice (points are not shown in the image window).')
    im_temp = image_resize(im_gray, height=1000)
    resize_factor = 1000 / im_gray.shape[0]
    cv2.imshow('image', im_temp)
    cv2.setWindowTitle("image", "Slice selection window. Select 2 points for the slice.")
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    global right_clicks

    P1 = np.array(right_clicks[0]) / resize_factor
    P2 = np.array(right_clicks[1]) / resize_factor
    logging.info(f"Selected coordinates: {P1=}, {P2=}.")
    logging.info(f"Selected coordinates: P1 = [{P1[0]:.0f}, {P1[1]:.0f}], P2 = [{P2[0]:.0f}, {P2[1]:.0f}]")
    logging.info(f"Selection coordinates:\n"
                 f"pointa = {P1[0]:.0f}, {P1[1]:.0f}\n"
                 f"pointb = {P2[0]:.0f}, {P2[1]:.0f}")

    # calculate the linear line coefficients (y=ax+b)
    x_coords, y_coords = zip(*[P1, P2])  # unzip coordinates to x and y
    a = (y_coords[1] - y_coords[0]) / (x_coords[1] - x_coords[0])
    b = y_coords[0] - a * x_coords[0]
    logging.info(f"The selected slice has the formula: y={a}x+{b}.")

    profiles = {}  # empty dict for profiles. profiles are of different sizes of not perfectly hor or vert
    # empty dict for all coordinates on all the slices, like:
    # {[ [x1_1,y1_1],[x1_2,y1_2],...,[x1_n1,y1_n1] ], [x2_1,y2_1],[x2_2,y2_2],...,[x2_n2,y2_n2],
    # ..., [xN_1,yN_1],[xN_2,yN_2],...,[xN_nN,yN_nN] ]}
    # ni is the slice length of slice i, N is the total number of slices
    all_coordinates = {}
    for jdx, n in enumerate(range(-SliceWidth, SliceWidth + 1)):
        bn = b + n * np.sqrt(a ** 2 + 1)
        coordinates, l = coordinates_on_line(a, bn, [0, im_gray.shape[1], 0, im_gray.shape[0]])
        all_coordinates[jdx] = coordinates

        # transpose to account for coordinate system in plotting (reversed y)
        profiles[jdx] = [np.transpose(im_gray)[pnt] for pnt in coordinates]

    logging.info(f"All {SliceWidth * 2 + 1} profiles are extracted from image.")
    # slices may have different lengths, and thus need to be aligned. We take the center of the image as the point to do
    # this. For each slice, we calculate the point (pixel) that is closest to this AlignmentPoint. We then make sure
    # that for all slices these alignments line up.
    AlignmentPoint = (im_gray.shape[0] // 2, im_gray.shape[1] // 2)  # take center of image as alignment
    profiles_aligned = align_arrays(all_coordinates, profiles, AlignmentPoint)

    # TODO filtering here to disregard datapoints (make nan) that have little statistical significance
    def filter_profiles(profiles_aligned, profile):
        nanValues = np.sum(np.isnan(profiles_aligned), axis=0)
        profiles_aligned = profile[np.where(nanValues == 0)]
        return profiles_aligned

    profile = np.nanmean(profiles_aligned, axis=0)

    # Saving temp
    if SavingTemp:
        saveTempPlot(profile, os.path.join(analysisFolder, 'GreyImages', "1profile.png"))

    #disregard datapoints for the average profile that not all profiles have. True is recommended.
    profile = filter_profiles(profiles_aligned, profile)
    # Saving temp
    if SavingTemp:
        saveTempPlot(profile, os.path.join(analysisFolder, 'GreyImages', "2profiles_startEnd.png"))
    logging.info("Profiles are aligned and average profile is determined.")


    profile = mov_mean(profile, movmeanN)
    logging.info("Moving average of profile has been taken")
    # Saving temp
    if SavingTemp:
        saveTempPlot(profile, os.path.join(analysisFolder, 'GreyImages', "3profileMovmeaned.png"))
    profile_fft = np.fft.fft(profile)  # transform to fourier space
    # Saving temp
    if SavingTemp:
        saveTempPlot(profile_fft, os.path.join(analysisFolder, 'GreyImages', "4profilefft.png"))

    mask = np.ones_like(profile).astype(float)

    mask[0:lowPass] = 0         #n highest frequencies in frequency range are removed   (i.e. 2)
    mask[-highPass:] = 0        #n lowest frequencies are removed - Sander
        # mask = smooth_step(mask, highPassBlur)
    profile_fft = profile_fft * mask
    # Saving temp
    if SavingTemp:
        saveTempPlot(profile_fft, os.path.join(analysisFolder, 'GreyImages', "5profilefftmask.png"))

    profile_filtered = np.fft.ifft(profile_fft)
    # Saving temp
    if SavingTemp:
        saveTempPlot(profile_filtered, os.path.join(analysisFolder, 'GreyImages', "6.1profilefiltered.png"))

    logging.info("Average profile is filtered in the Fourier space.")

    # TODO testing to save real part of profile to csv file. First value is the elapsed time from moment 0 in given series of images.
    # if config.getboolean("LINE_METHOD_ADVANCED", "CSV_REAL"):
    #     wrappedPath = os.path.join(Folders['save_process'], f"{savename}_real.csv")
    #     realProfile = profile_filtered.real
    #     (np.insert(realProfile, 0, timeelapsed)).tofile(wrappedPath, sep='\n', format='%.2f')
    # if config.getboolean("LINE_METHOD_ADVANCED", "CSV_IMAG"):
    #     wrappedPath = os.path.join(Folders['save_process'], f"{savename}_imag.csv")
    #     imagProfile = profile_filtered.imag
    #     (np.insert(imagProfile, 0, timeelapsed)).tofile(wrappedPath, sep='\n', format='%.2f')
    # Saving temp
    if SavingTemp:
        saveTempPlot(profile_filtered.real,
                     os.path.join(analysisFolder, 'GreyImages', "6.2profilefiltered_real.png"))
    # Saving temp
    if SavingTemp:
        saveTempPlot(profile_filtered.imag, os.path.join(analysisFolder, 'GreyImages', "6.3profilefiltered_imag.png"))

    # calculate the wrapped space
    wrapped = np.arctan2(profile_filtered.imag, profile_filtered.real)
    # Saving temp
    if SavingTemp:
        saveTempPlot(wrapped, os.path.join(analysisFolder, 'GreyImages', "7wrapped.png"))

    # local normalization of the wrapped space. Since fringe pattern is never ideal (i.e. never runs between 0-1) due
    # to noise and averaging errors, the wrapped space doesn't run from -pi to pi, but somewhere inbetween. By setting
    # this value to True, the wrapped space is normalized from -pi to pi, if the stepsize is above a certain threshold.
    wrapped = normalize_wrappedspace(wrapped, 3.14159265359)
    # Saving temp
    if SavingTemp:
        saveTempPlot(wrapped, os.path.join(analysisFolder, 'GreyImages', "8wrapped_norm.png"))

    unwrapped = np.unwrap(wrapped)
    # Saving temp
    if SavingTemp:
        saveTempPlot(unwrapped, os.path.join(analysisFolder, 'GreyImages', "9unwrapped.png"))

    logging.info("Average slice is wrapped and unwrapped")

    if FLIP:
        unwrapped = -unwrapped + np.max(unwrapped)
        logging.debug('Image surface flipped.')

    unwrapped_converted = unwrapped * conversionFactorZ
    # Saving temp
    if SavingTemp:
        saveTempPlot(unwrapped_converted, os.path.join(analysisFolder, 'GreyImages', "10unwrappedConverted.png"))

    logging.debug('Conversion factor for Z applied.')

    from plotting import plot_lineprocess, plot_profiles, plot_sliceoverlay, plot_unwrappedslice
    fig1 = plot_profiles([], profiles_aligned)
    fig2 = plot_lineprocess([], profile, profile_filtered, wrapped, unwrapped)
    fig3 = plot_sliceoverlay([], all_coordinates, img_raw, timeFormat(timeElapsed))
    fig4 = plot_unwrappedslice([], unwrapped_converted, profiles_aligned, conversionFactorXY, unitXY, unitZ)
    logging.info(f"Plotting done.")

    # Saving
    if SAVEIMAGES:
        fig1.savefig(os.path.join(analysisFolder, f"rawslices_{savename}, t = {timeFormat(timeElapsed)}.png"), dpi=600)
        fig2.savefig(os.path.join(analysisFolder, f"process_{savename}.png"), dpi=600)
        fig3.savefig(os.path.join(analysisFolder, f"rawslicesimage_{savename}.png"), dpi=600)
        fig4.savefig(os.path.join(analysisFolder, f"unwrapped_{savename}.png"), dpi=600)
        logging.debug('PNG saving done.')
    logging.info(f"Saving done.")
    #plt.close()
    return unwrapped, P1, P2

'''
20 periods = 226 pix = 134um
1pi = lambda / 4n = 532 / (4*1.434) = 92.74 um
20 periods = 20*2pi = 40pi = 40*92.74 = 3709.6nm
1.53

237pix = 237/1687 = 0.1405 mm
'''
def main():
    """
    This functions allows for Contact Angle (CA) analysis of an (series of) image(s) along a hand-draw line. Both the
    drawing of the line, and the consecutive CA-analysis happens in this single function (as opposed to the original
    analysis_contactangle.py file).
    """
    path = "E:\\2023_11_13_PLMA_Dodecane_Basler5x_Xp_1_24S11los_misschien_WEDGE_v2"
    imgFolderPath, conversionZ, conversionXY, unitZ, unitXY = filePathsFunction(path, wavelength_laser=520)
    imgList = [f for f in glob.glob(os.path.join(imgFolderPath, f"*tiff"))]
    everyHowManyImages = 3
    #usedImages = np.arange(1, len(imgList), everyHowManyImages)  # 200 is the working one
    usedImages = [46]

    flipData = False
    analysisFolder = os.path.join(imgFolderPath, "Analysis CA Manual Line")
    if not os.path.exists(analysisFolder):
        os.mkdir(analysisFolder)
        print('created path: ', analysisFolder)

    angleDeg_afo_time = []  # for saving median CA's later
    usedDeltaTs = []  # for saving delta t (IN SECONDS) for only the USED IMAGES
    timestamps, deltatseconds, deltatFromZeroSeconds = getTimeStamps(imgList)  # get the timestamps of ALL images in folder, and the delta time of ALL images w/ respect to the 1st image
    deltat_formatted = timeFormat(deltatFromZeroSeconds)  # formatted delta t (seconds, minutes etc..) for ALL IMAGES in folder.

    counter = 0
    try:
        # TODO implement in code that conversion is good also for different units of input (factors 1000) belowfor x & y = ...
        if unitZ != "nm" or unitXY != "mm":
            raise Exception("One of either units is not correct for good conversion. Fix manually or implement in code")
        print(f"unitZ: {unitZ}, conversionZ = {conversionZ}. unitXY: {unitXY},  converzionXY = {conversionXY}")

        for n, img in enumerate(imgList):
            if n in usedImages:
                counter += 1
                print(f'Analyzing image {counter}/{len(usedImages)}.')

                y, P1, P2 = manual_line_chooser(imgList[n], conversionXY, conversionZ, unitXY, unitZ, analysisFolder, [deltatseconds[n]])

                #TODO here the height data is already present.
                #y = np.loadtxt(csvList[imageNumber], delimiter=",") * conversionZ / 1000  # y is in um
                if flipData:
                    y = -y + max(y)

                y *= (conversionZ / 1000)           #height (y) is now in um
                x = np.arange(0, len(y)) * conversionXY * 1000  # x is now in um

                ##TODO: removed first datapoint because for whatever reason it was spuerfar outside the range, making it hard to select the good range in the plot
                x = x[1:]
                y = y[1:]
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.scatter(x, y)
                highlighter = Highlighter(ax, x, y)
                plt.show()
                plt.close()
                selected_regions = highlighter.mask
                xrange1, yrange1 = x[selected_regions], y[selected_regions]
                print(f"x ranges from: [{xrange1[0]} - {xrange1[-1]}]\n"
                      f"y ranges from: [{yrange1[0]} - {yrange1[-1]}]\n"
                      f"Therefore dy/dx = {yrange1[-1] - yrange1[0]} / {xrange1[-1] - xrange1[0]} = {(yrange1[-1] - yrange1[0]) / (xrange1[-1] - xrange1[0])}")

                coef1 = np.polyfit(xrange1, yrange1, 1)
                poly1d_fn1 = np.poly1d(coef1)
                a_horizontal = 0
                angleRad = math.atan((coef1[0] - a_horizontal) / (1 + coef1[0] * a_horizontal))
                angleDeg = math.degrees(angleRad)
                print(f"angledeg = {angleDeg}")
                # Flip measured CA degree if higher than 45.
                if angleDeg > 45:
                    angleDeg = 90 - angleDeg

                fig, ax = plt.subplots()
                ax.scatter(x, y, label=f'Raw data')
                ax.scatter(xrange1, yrange1, color='green', label='Selected data line 1')
                ax.plot(x, poly1d_fn1(x), color='red', linewidth=3, label='Linear fit 1')
                ax.set_title(f"{angleDeg=}")
                ax.set_xlabel("[um]")
                ax.set_ylabel("[um]")
                ax.set_xlim([x[0], x[-1]])
                ax.set_ylim([y[0], y[-1]])

                # foldername = "CA_analysis"
                # newfolder = os.path.join(analysisFolder, foldername)
                # if not os.path.exists(newfolder):
                #     os.mkdir(newfolder)
                #     print('created path: ', newfolder)
                # print(os.path.abspath(foldername))
                # fig.savefig(os.path.join(os.path.dirname(originalPath),  f"angleFitting_{os.path.splitext(os.path.basename(originalPath))[0]}.png"), dpi=300)
                fig.savefig(os.path.join(analysisFolder, f"angleFitting_{n}_.png"), dpi=600)

                angleDeg_afo_time.append(angleDeg)
                usedDeltaTs.append(deltatseconds[n])
                print(f'Contact angle = {angleDeg} degrees.')
                plt.close('all')
    except:
        print("Something went wrong, still saving data.")

    # with open(os.path.join(newfolder, f"angleFittingData.json"), 'w') as f:
    #     json.dump(data, f, indent=4)

    print(f"TimefromStart: {usedDeltaTs}")
    print(f"Angle in deg.: {angleDeg_afo_time}")

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(np.divide(angleDeg_afo_time, 60), angleDeg, '.-')
    ax.set_xlabel(f'[Time from drop creation [min]')
    ax.set_ylabel(f'[Contact angle [deg]')
    fig.tight_layout()
    fig.savefig(os.path.join(analysisFolder, f"angleFittingData.png"), dpi=600)
    #
    # np.savetxt(os.path.join(newfolder, f"angleFittingData{os.path.splitext(os.path.basename(originalPath))[0]}.csv"),
    #            np.vstack((timeFromStart, angleDeg)),
    #            delimiter=',', fmt='%f', header=f'Dataset: {os.path.basename(originalPath)}, row 1 = Time from start '
    #                                            f'(depositing drop) [s], row 2 = contact angle [deg] ')
    exit()

if __name__ == "__main__":
    main()