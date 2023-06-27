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
from itertools import chain
from sklearn import preprocessing
from configparser import ConfigParser
from scipy.odr import odrpack as odr
from scipy.odr import models

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

def custom_fit2(x,a,b,c):
    return a + b*x + c*x**2



def makeImages(profile, timeFromStart, source, pixelLocation, config):
    conversionFactorXY, conversionFactorZ, unitXY, unitZ = conversion_factors(config)
    if not os.path.exists(os.path.join(source, f"Swellingimages")):
        os.mkdir(os.path.join(source, f"Swellingimages"))
    fig0, ax0 = plt.subplots()
    ax0.plot(timeFromStart, ([profile])[0], label = f'normalized, unfiltered')
    plt.xlabel('Time (h)')
    plt.ylabel('Mean intensity')
    plt.title(f'Intensity profile. Pixellocation = {pixelLocation}')

    #order = 10
    #fit, error = poly_lsq(timeFromStart, ([profile])[0], order)

    popt, pcov = curve_fit(custom_fit, timeFromStart, ([profile])[0])
    print(popt)
    print(np.linalg.cond(pcov))
    xnew = np.linspace(timeFromStart[0], timeFromStart[len(timeFromStart)-1], 1000)
    ynew = custom_fit(xnew, *popt)
    #ynew = np.polyval(fit, xnew)
    #ax0.plot(xnew, ynew, label=f'fit {order}, unfiltered')


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
            np.savetxt(wrappedPath, [p for p in zip(timeFromStart, unwrapped * conversionFactorZ)], delimiter=',', fmt='%s')
    # now get datapoints we need.
    #unwrapped_um = unwrapped * conversionZ
    #analyzeTimes = np.linspace(0, 57604, 12)
    #analyzeImages = np.array([find_nearest(timeFromStart, t)[1] for t in analyzeTimes])
    #print(analyzeImages)


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
    #Required changeables. Note that chosen Pixellocs must have enough datapoints around them to average over. Otherwise code fails.
    pixelLoc1 = 2550
    pixelLoc2 = 2551#pixelLoc1 + 1
    pixelIV = 200   #interval between the two pixellocations to be taken.
    #source = "E:\\2023_03_07_Data_for_Swellinganalysis\\export\\PROC_20230306180748"
    #source = "C:\\Users\\ReuvekampSW\\Documents\\InterferometryPython\\export\\PROC_20230327160828_nofilter"
    source = "C:\\Users\\ReuvekampSW\\Documents\\InterferometryPython\\export\\PROC_20230612121104"
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

        #TODO for even spreading of data (NOT true time!)
        #elapsedtime = np.arange(0,len(meanIntensity))

        # for nn in [1]:
        #     makeImages(meanIntensity[0:-nn:], elapsedtime[0:-nn:], source, pixelLocation)
        makeImages(meanIntensity, elapsedtime, source, pixelLocation, config)
    print(f"Read-in lenght of rows from csv file = {len(rows)}")

if __name__ == "__main__":
    main()
    exit()