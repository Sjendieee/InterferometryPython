"""
This swellingratio analysis allows for investigation of Intensity vs. Distance, at a single timestep.
This results in a swelling profile for every timestep.
The "old" swelling analysis
"""

import csv
import os
import matplotlib.pyplot as plt
import glob
import cv2
from general_functions import image_resize
from line_method import coordinates_on_line, normalize_wrappedspace, mov_mean, timeFormat
import numpy as np
from PIL import Image
from configparser import ConfigParser
from general_functions import conversion_factors

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


def flipData(data):
    datamax = max(data)
    return [(-x + datamax) for x in data]

def main():
    """"Changeables: """
    #source = "F:\\2023_04_06_PLMA_HexaDecane_Basler2x_Xp1_24_s11_split____GOODHALO-DidntReachSplit\\D_analysis_v2\\PROC_20230612121104"
    source = "C:\\Users\\ReuvekampSW\\Documents\\InterferometryPython\\export\\PROC_20230724185238"  # hexadecane, NO filtering in /main.py, no contrast enhance

    range1 = 2200#2320       #start x left for plotting
    range2 = 3500  # len(swellingProfile)
    knownPixelPosition = 2550 - range1 - 1 #pixellocation at which the bursh height is known at various times
    dryBrushThickness = 154                 # dry brush thickness (measured w/ e.g. ellipsometry)
    knownHeightArr = [128, 216, 258, 300]       #Known brush swelling at pixellocation in nm for certain timesteps   #in nm
    knownHeightArr = np.add(knownHeightArr, dryBrushThickness)      # true brush thickness = dry thickness + swollen thickness
    outputFormatXY = 'mm'       #'pix' or 'mm'
    #XLIM - True; Xlim = []
    YLIM = True; Ylim = [-50, 650]  #ylim for swelling profiles (only used when plotting absolute swelling height)
    PLOTSWELLINGRATIO = True
    SAVEFIG = True
    INTENSITYPROFILES = True
    REMOVEBACKGROUNDNOISE = False
    """"End of changeables"""

    config = ConfigParser()
    configName = [f for f in glob.glob(os.path.join(source, f"config*"))]
    config.read(os.path.join(source, configName[0]))
    conversionFactorXY, conversionFactorZ, unitXY, unitZ = conversion_factors(config)
    csvList = [f for f in glob.glob(os.path.join(source, f"csv\\*unwrapped.csv"))]
    if not os.path.exists(os.path.join(source, f"Swellingimages")):
        os.mkdir(os.path.join(source, f"Swellingimages"))
    print(f"With this conversionXY, 1000 pixels = {conversionFactorXY*1000} mm, \n"
          f"and 1 mm = {1/conversionFactorXY} pixels")

    idxx = 0
    fig1, ax1 = plt.subplots()
    for idx, n in enumerate(csvList):
        if idx in [50, 95, 206, 395]:               #For hexadecane(0,1,4,24h): 50, 95, 206, 395
            file = open(n)
            csvreader = csv.reader(file)
            rows = []
            for row in csvreader:
                rows.append(float(row[0]))
            file.close()
            elapsedtime = rows[0]
            swellingProfile = rows[1:]
            swellingProfileZoom = swellingProfile[range1:range2]
            #TODO checkout why using flipdata here!
            swellingProfileZoomConverted = flipData([conversionFactorZ * x for x in swellingProfileZoom])

            knownHeight = knownHeightArr[idxx]       #in nm
            swellingProfileZoomConverted = np.subtract(swellingProfileZoomConverted, (swellingProfileZoomConverted[knownPixelPosition] - knownHeight))
            plt.ylabel(f"Height ({unitZ})")
            if outputFormatXY == 'pix':
                x = np.linspace(range1, range2, range2 - range1)
                xshifted = [q - min(x) for q in x]
                if PLOTSWELLINGRATIO:
                    swellingProfileZoomConverted = np.divide(swellingProfileZoomConverted, dryBrushThickness)
                    plt.ylabel(f"Swelling ratio (h/h0)")
                ax1.plot(xshifted, swellingProfileZoomConverted, '.', label=f'time={timeFormat(elapsedtime)}')
                ax1.plot(xshifted, np.zeros(len(xshifted)), '-')
                plt.xlabel(f"Distance (pixels)")
                plt.title(f"Swelling profile at time {timeFormat(elapsedtime)} \n shifted to {range1} pixels")
                ax1.legend()
            elif outputFormatXY == 'mm':
                x = np.linspace(range1, range2, range2 - range1) * conversionFactorXY
                xshifted = [q - min(x) for q in x]
                if PLOTSWELLINGRATIO:
                    swellingProfileZoomConverted = np.divide(swellingProfileZoomConverted, dryBrushThickness)
                    plt.ylabel(f"Swelling ratio (h/h0)")
                ax1.plot(xshifted, swellingProfileZoomConverted, '.', label=f'time={timeFormat(elapsedtime)}')
                ax1.plot(xshifted, np.zeros(len(xshifted)), '-')
                plt.xlabel(f"Distance ({unitXY})")

                plt.title(f"Swelling profile at time {timeFormat(elapsedtime)}")
                ax1.legend()
            else:
                print("wrong format input")
            idxx = idxx + 1

    if INTENSITYPROFILES:
        csvList = [f for f in glob.glob(os.path.join(source, f"process\\*real.csv"))]
        knownHeightArr = [0,0,0,0,0]
        idxx = 0
        fig0, ax0 = plt.subplots()
        for idx, n in enumerate(csvList):
            if idx in [0, 50]:               # 50, 95, 206,
                file = open(n)
                csvreader = csv.reader(file)
                rows = []
                for row in csvreader:
                    rows.append(float(row[0]))
                file.close()
                elapsedtime = rows[0]
                intensityProfile = rows[1:]
                intensityProfileZoom = intensityProfile[range1:range2]
                if REMOVEBACKGROUNDNOISE:
                    if idx == 0:
                        backgroundIntensityZoom = intensityProfileZoom
                    else:
                        intensityProfileZoom = np.subtract(intensityProfileZoom, backgroundIntensityZoom)
                intensityProfileZoomConverted = ([1 * x for x in intensityProfileZoom])       # no conversion required for intensity

                knownHeight = knownHeightArr[idxx]       #in nm
                #swellingProfileZoomConverted = np.subtract(swellingProfileZoomConverted, (swellingProfileZoomConverted[knownPixelPosition] - knownHeight))
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
                ax0.plot(xshifted, intensityProfileZoomConverted, '.', label=f'Time={timeFormat(elapsedtime)}')
                ax0.plot(xshifted, np.zeros(len(xshifted)), '-')
                plt.legend()


                idxx = idxx + 1

    if SAVEFIG:
        if YLIM:    #This part is for the swelling profiles
            if PLOTSWELLINGRATIO:
                Ylim = np.divide(Ylim, dryBrushThickness)
            ax1.set_ylim(Ylim)
        ax1.autoscale(enable=True, axis='x', tight=True)
        fig1.savefig(os.path.join(source, f"Swellingimages\\{idx}Swelling.png"),dpi=300)

        if INTENSITYPROFILES:       #This part is for the intensity profiles
            if REMOVEBACKGROUNDNOISE:
                ax0.set_ylim([-50, 50])
            ax0.autoscale(enable=True, axis='x', tight=True)
            fig0.savefig(os.path.join(source, f"Swellingimages\\{idx}Intensity.png"),dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
    exit()